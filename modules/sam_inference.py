from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import Dict, List, Optional
import torch
import os
from datetime import datetime
import numpy as np
import gradio as gr

from modules.model_downloader import (
    AVAILABLE_MODELS, DEFAULT_MODEL_TYPE, OUTPUT_DIR,
    is_sam_exist,
    download_sam_model_url
)
from modules.paths import SAM2_CONFIGS_DIR, MODELS_DIR, TEMP_OUT_DIR, TEMP_DIR
from modules.constants import BOX_PROMPT_MODE, AUTOMATIC_MODE, COLOR_FILTER, PIXELIZE_FILTER
from modules.mask_utils import (
    save_psd_with_masks,
    create_mask_combined_images,
    create_mask_gallery,
    create_mask_pixelized_image,
    create_solid_color_mask_image
)
from modules.video_utils import get_frames_from_dir
from modules.utils import save_image
from modules.logger_util import get_logger

MODEL_CONFIGS = {
    "sam2_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_l.yaml"),
}
logger = get_logger()


class SamInference:
    def __init__(self,
                 model_dir: str = MODELS_DIR,
                 output_dir: str = OUTPUT_DIR
                 ):
        self.model = None
        self.available_models = list(AVAILABLE_MODELS.keys())
        self.current_model_type = DEFAULT_MODEL_TYPE
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model_path = os.path.join(self.model_dir, AVAILABLE_MODELS[DEFAULT_MODEL_TYPE][0])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.mask_generator = None
        self.image_predictor = None
        self.video_predictor = None
        self.video_inference_state = None

    def load_model(self,
                   model_type: Optional[str] = None,
                   load_video_predictor: bool = False):
        if model_type is None:
            model_type = DEFAULT_MODEL_TYPE

        config = MODEL_CONFIGS[model_type]
        filename, url = AVAILABLE_MODELS[model_type]
        model_path = os.path.join(self.model_dir, filename)

        if not is_sam_exist(model_type):
            logger.info(f"No SAM2 model found, downloading {model_type} model...")
            download_sam_model_url(model_type)
        logger.info(f"Applying configs to {model_type} model..")

        if load_video_predictor:
            try:
                self.model = None
                self.video_predictor = build_sam2_video_predictor(
                    config_file=config,
                    ckpt_path=model_path,
                    device=self.device
                )
            except Exception as e:
                logger.exception("Error while loading SAM2 model for video predictor")
                raise f"Error while loading SAM2 model for video predictor!: {e}"

        try:
            self.model = build_sam2(
                config_file=config,
                ckpt_path=model_path,
                device=self.device
            )
        except Exception as e:
            logger.exception("Error while loading SAM2 model")
            raise f"Error while loading SAM2 model!: {e}"

    def init_video_inference_state(self,
                                   vid_input: str,
                                   model_type: Optional[str] = None):
        if model_type is None:
            model_type = self.current_model_type

        if self.video_predictor is None or model_type != self.current_model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type, load_video_predictor=True)

        if self.video_inference_state is not None:
            self.video_predictor.reset_state(self.video_inference_state)
            self.video_inference_state = None

        self.video_inference_state = self.video_predictor.init_state(video_path=vid_input)

    def generate_mask(self,
                      image: np.ndarray,
                      model_type: str,
                      **params):
        if self.model is None or self.current_model_type != model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            **params
        )
        try:
            generated_masks = self.mask_generator.generate(image)
        except Exception as e:
            logger.exception("Error while auto generating masks")
            raise f"Error while auto generating masks: str({e})"
        return generated_masks

    def predict_image(self,
                      image: np.ndarray,
                      model_type: str,
                      box: Optional[np.ndarray] = None,
                      point_coords: Optional[np.ndarray] = None,
                      point_labels: Optional[np.ndarray] = None,
                      **params):
        if self.model is None or self.current_model_type != model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type)
        self.image_predictor = SAM2ImagePredictor(sam_model=self.model)
        self.image_predictor.set_image(image)

        try:
            masks, scores, logits = self.image_predictor.predict(
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=params["multimask_output"],
            )
        except Exception as e:
            logger.exception(f"Error while predicting image with prompt: {str(e)}")
            raise RuntimeError(f"Error while predicting image with prompt: {str(e)}") from e
        return masks, scores, logits

    def add_prediction_to_frame(self,
                                frame_idx: int,
                                obj_id: int,
                                inference_state: Optional[Dict] = None,
                                points: Optional[np.ndarray] = None,
                                labels: Optional[np.ndarray] = None,
                                box: Optional[np.ndarray] = None):
        if (self.video_predictor is None or
                inference_state is None and self.video_inference_state is None):
            logger.exception("Error while predicting frame from video, load video predictor first")
            raise f"Error while predicting frame from video"

        if inference_state is None:
            inference_state = self.video_inference_state

        try:
            out_frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box
            )
        except Exception as e:
            logger.exception(f"Error while predicting frame with prompt: {str(e)}")
            raise RuntimeError(f"Failed to predicting frame with prompt: {str(e)}") from e

        return out_frame_idx, out_obj_ids, out_mask_logits

    def propagate_in_video(self,
                           inference_state: Optional[Dict] = None,):
        if inference_state is None and self.video_inference_state is None:
            logger.exception("Error while propagating in video, load video predictor first")
            raise f"Error while propagating in video"

        if inference_state is None:
            inference_state = self.video_inference_state

        video_segments = {}

        try:
            generator = self.video_predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0
            )
            cached_images = inference_state["images"]
            images = get_frames_from_dir(vid_dir=TEMP_DIR, as_numpy=True)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                for out_frame_idx, out_obj_ids, out_mask_logits in generator:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    video_segments[out_frame_idx] = {
                        "image": images[out_frame_idx],
                        "mask": mask
                    }
                    print("frame_idx: ", out_frame_idx)
        except Exception as e:
            logger.exception(f"Error while propagating in video: {str(e)}")
            raise RuntimeError(f"Failed to propagate in video: {str(e)}") from e

        return video_segments

    def add_filter_to_preview(self,
                              image_prompt_input_data: Dict,
                              filter_mode: str,
                              frame_idx: int,
                              pixel_size: Optional[int] = None,
                              color_hex: Optional[str] = None,
                              ):
        if self.video_predictor is None or self.video_inference_state is None:
            logger.exception("Error while adding filter to preview, load video predictor first")
            raise f"Error while adding filter to preview"

        if not image_prompt_input_data["points"]:
            error_message = ("No prompt data provided. If this is an incorrect flag, "
                             "Please press the eraser button (on the image prompter) and add your prompts again.")
            logger.error(error_message)
            raise gr.Error(error_message, duration=20)

        image, prompt = image_prompt_input_data["image"], image_prompt_input_data["points"]
        image = np.array(image.convert("RGB"))

        point_labels, point_coords, box = self.handle_prompt_data(prompt)
        obj_id = frame_idx

        self.video_predictor.reset_state(self.video_inference_state)
        idx, scores, logits = self.add_prediction_to_frame(
            frame_idx=frame_idx,
            obj_id=obj_id,
            inference_state=self.video_inference_state,
            points=point_coords,
            labels=point_labels,
            box=box
        )
        masks = (logits[0] > 0.0).cpu().numpy()
        generated_masks = self.format_to_auto_result(masks)

        if filter_mode == COLOR_FILTER:
            image = create_solid_color_mask_image(image, generated_masks, color_hex)

        elif filter_mode == PIXELIZE_FILTER:
            image = create_mask_pixelized_image(image, generated_masks, pixel_size)

        return image

    def add_filter_to_video(self,
                            image_prompt_input_data: Dict,
                            filter_mode: str,
                            frame_idx: int,
                            pixel_size: Optional[int] = None,
                            color_hex: Optional[str] = None,):
        if self.video_predictor is None or self.video_inference_state is None:
            logger.exception("Error while adding filter to preview, load video predictor first")
            raise f"Error while adding filter to preview"

        if not image_prompt_input_data["points"]:
            error_message = ("No prompt data provided. If this is an incorrect flag, "
                             "Please press the eraser button (on the image prompter) and add your prompts again.")
            logger.error(error_message)
            raise gr.Error(error_message, duration=20)

        prompt_frame_image, prompt = image_prompt_input_data["image"], image_prompt_input_data["points"]

        point_labels, point_coords, box = self.handle_prompt_data(prompt)
        obj_id = frame_idx

        self.video_predictor.reset_state(self.video_inference_state)
        idx, scores, logits = self.add_prediction_to_frame(
            frame_idx=frame_idx,
            obj_id=obj_id,
            inference_state=self.video_inference_state,
            points=point_coords,
            labels=point_labels,
            box=box
        )

        video_segments = self.propagate_in_video(inference_state=self.video_inference_state)
        for frame_index, info in video_segments.items():
            orig_image, masks = info["image"], info["mask"]
            masks = self.format_to_auto_result(masks)

            if filter_mode == COLOR_FILTER:
                filtered_image = create_solid_color_mask_image(orig_image, masks, color_hex)

            elif filter_mode == PIXELIZE_FILTER:
                filtered_image = create_mask_pixelized_image(orig_image, masks, pixel_size)

            save_image(filtered_image, os.path.join(TEMP_OUT_DIR, "%05d.jpg"))

    def divide_layer(self,
                     image_input: np.ndarray,
                     image_prompt_input_data: Dict,
                     input_mode: str,
                     model_type: str,
                     *params):
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name = f"result-{timestamp}.psd"
        output_path = os.path.join(self.output_dir, "psd", output_file_name)

        # Pre-processed gradio components
        hparams = {
            'points_per_side': int(params[0]),
            'points_per_batch': int(params[1]),
            'pred_iou_thresh': float(params[2]),
            'stability_score_thresh': float(params[3]),
            'stability_score_offset': float(params[4]),
            'crop_n_layers': int(params[5]),
            'box_nms_thresh': float(params[6]),
            'crop_n_points_downscale_factor': int(params[7]),
            'min_mask_region_area': int(params[8]),
            'use_m2m': bool(params[9]),
            'multimask_output': bool(params[10])
        }

        if input_mode == AUTOMATIC_MODE:
            image = image_input

            generated_masks = self.generate_mask(
                image=image,
                model_type=model_type,
                **hparams
            )

        elif input_mode == BOX_PROMPT_MODE:
            image = image_prompt_input_data["image"]
            image = np.array(image.convert("RGB"))
            prompt = image_prompt_input_data["points"]
            if len(prompt) == 0:
                return [image], []

            point_labels, point_coords, box = self.handle_prompt_data(prompt)

            predicted_masks, scores, logits = self.predict_image(
                image=image,
                model_type=model_type,
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=hparams["multimask_output"]
            )
            generated_masks = self.format_to_auto_result(predicted_masks)

        save_psd_with_masks(image, generated_masks, output_path)
        mask_combined_image = create_mask_combined_images(image, generated_masks)
        gallery = create_mask_gallery(image, generated_masks)
        gallery = [mask_combined_image] + gallery

        return gallery, output_path

    @staticmethod
    def format_to_auto_result(
        masks: np.ndarray
    ):
        place_holder = 0
        if len(masks.shape) <= 3:
            masks = np.expand_dims(masks, axis=0)
        result = [{"segmentation": mask[0], "area": place_holder} for mask in masks]
        return result

    @staticmethod
    def handle_prompt_data(
        prompt_data: List
    ):
        """
        Handle data from ImageInputPrompter.

        Args:
            prompt_data (Dict): A dictionary containing the 'prompt' key with a list of prompts.

        Returns:
            point_labels (List): list of points labels.
            point_coords (List): list of points coords.
            box (List): list of box datas.
        """
        point_labels, point_coords, box = [], [], []

        for x1, y1, left_click_indicator, x2, y2, point_indicator in prompt_data:
            is_point = point_indicator == 4.0
            if is_point:
                point_labels.append(left_click_indicator)
                point_coords.append([x1, y1])
            else:
                box.append([x1, y1, x2, y2])

        point_labels = np.array(point_labels) if point_labels else None
        point_coords = np.array(point_coords) if point_coords else None
        box = np.array(box) if box else None

        return point_labels, point_coords, box
