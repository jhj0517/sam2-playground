from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Dict, List, Optional
import torch
import os
from datetime import datetime
import numpy as np

from modules.model_downloader import (
    AVAILABLE_MODELS, DEFAULT_MODEL_TYPE, OUTPUT_DIR,
    is_sam_exist,
    download_sam_model_url
)
from modules.paths import SAM2_CONFIGS_DIR, MODELS_DIR
from modules.constants import BOX_PROMPT_MODE, AUTOMATIC_MODE
from modules.mask_utils import (
    save_psd_with_masks,
    create_mask_combined_images,
    create_mask_gallery
)

CONFIGS = {
    "sam2_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_l.yaml"),
}


class SamInference:
    def __init__(self,
                 model_dir: str = MODELS_DIR,
                 output_dir: str = OUTPUT_DIR
                 ):
        self.model = None
        self.available_models = list(AVAILABLE_MODELS.keys())
        self.model_type = DEFAULT_MODEL_TYPE
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model_path = os.path.join(self.model_dir, AVAILABLE_MODELS[DEFAULT_MODEL_TYPE][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_generator = None
        self.image_predictor = None
        self.video_predictor = None

    def load_model(self):
        config = CONFIGS[self.model_type]
        filename, url = AVAILABLE_MODELS[self.model_type]
        model_path = os.path.join(self.model_dir, filename)

        if not is_sam_exist(self.model_type):
            print(f"\nNo SAM2 model found, downloading {self.model_type} model...")
            download_sam_model_url(self.model_type)
        print("\nApplying configs to model..")

        try:
            self.model = build_sam2(
                config_file=config,
                ckpt_path=model_path,
                device=self.device
            )
        except Exception as e:
            print(f"Error while Loading SAM2 model! {e}")

    def generate_mask(self,
                      image: np.ndarray,
                      model_type: str,
                      **params):
        if self.model is None or self.model_type != model_type:
            self.model_type = model_type
            self.load_model()
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            **params
        )
        try:
            generated_masks = self.mask_generator.generate(image)
        except Exception as e:
            raise f"Error while auto generating masks: {e}"
        return generated_masks

    def predict_image(self,
                      image: np.ndarray,
                      model_type: str,
                      box: Optional[np.ndarray] = None,
                      point_coords: Optional[np.ndarray] = None,
                      point_labels: Optional[np.ndarray] = None,
                      **params):
        if self.model is None or self.model_type != model_type:
            self.model_type = model_type
            self.load_model()
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
            raise f"Error while predicting image with prompt: {e}"
        return masks, scores, logits

    def divide_layer(self,
                     image_input: np.ndarray,
                     image_prompt_input_data: Dict,
                     input_mode: str,
                     model_type: str,
                     *params):
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name = f"result-{timestamp}.psd"
        output_path = os.path.join(self.output_dir, "psd", output_file_name)

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

            point_labels, point_coords, box = [], [], []

            for x1, y1, left_click_indicator, x2, y2, point_indicator in prompt:
                if point_indicator == 4.0:
                    point_labels.append(left_click_indicator)
                    point_coords.append([x1, y1])
                else:
                    box.append([x1, y1, x2, y2])

            predicted_masks, scores, logits = self.predict_image(
                image=image,
                model_type=model_type,
                box=np.array(box) if box else None,
                point_coords=np.array(point_coords) if point_coords else None,
                point_labels=np.array(point_labels) if point_labels else None,
                multimask_output=hparams["multimask_output"]
            )
            generated_masks = self.format_to_auto_result(predicted_masks)

        save_psd_with_masks(image, generated_masks, output_path)
        mask_combined_image = create_mask_combined_images(image, generated_masks)
        gallery = create_mask_gallery(image, generated_masks)

        return [mask_combined_image] + gallery, output_path

    @staticmethod
    def format_to_auto_result(
        masks: np.ndarray
    ):
        place_holder = 0
        if len(masks.shape) <= 3:
            masks = np.expand_dims(masks, axis=0)
        result = [{"segmentation": mask[0], "area": place_holder} for mask in masks]
        return result
