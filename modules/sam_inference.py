from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os
from datetime import datetime
import numpy as np

from modules.model_downloader import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_TYPE,
    OUTPUT_DIR,
    is_sam_exist,
    download_sam_model_url
)
from modules.paths import SAM2_CONFIGS_DIR, MODELS_DIR
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

        # Tunable Parameters , All default values by https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
        self.maskgen_hparams = {
            "points_per_side": 64,
            "points_per_batch": 128,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.92,
            "stability_score_offset": 0.7,
            "crop_n_layers": 1,
            "box_nms_thresh": 0.7,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 25.0,
            "use_m2m": True,
        }

    def load_model(self):
        config = CONFIGS[self.model_type]
        filename, url = AVAILABLE_MODELS[self.model_type]
        model_path = os.path.join(self.model_dir, filename)

        if not is_sam_exist(self.model_type):
            print(f"\nLayer Divider Extension : No SAM2 model found, downloading {self.model_type} model...")
            download_sam_model_url(self.model_type)
        print("\nLayer Divider Extension : applying configs to model..")

        try:
            self.model = build_sam2(
                config_file=config,
                ckpt_path=model_path,
                device=self.device
            )
            self.image_predictor = SAM2ImagePredictor(sam_model=self.model)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                **self.maskgen_hparams
            )
        except Exception as e:
            print(f"Layer Divider Extension : Error while Loading SAM2 model! {e}")

    def generate_mask(self,
                      image: np.ndarray):
        return self.mask_generator.generate(image)

    def generate_mask_app(self,
                          image: np.ndarray,
                          model_type: str,
                          *params
                          ):
        maskgen_hparams = {
            'points_per_side': int(params[0]),
            'points_per_batch': int(params[1]),
            'pred_iou_thresh': float(params[2]),
            'stability_score_thresh': float(params[3]),
            'stability_score_offset': float(params[4]),
            'crop_n_layers': int(params[5]),
            'box_nms_thresh': float(params[6]),
            'crop_n_points_downscale_factor': int(params[7]),
            'min_mask_region_area': int(params[8]),
            'use_m2m': bool(params[9])
        }
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name = f"result-{timestamp}.psd"
        output_path = os.path.join(self.output_dir, "psd", output_file_name)

        if self.model is None or self.mask_generator is None or self.model_type != model_type or self.maskgen_hparams != maskgen_hparams:
            self.model_type = model_type
            self.maskgen_hparams = maskgen_hparams
            self.load_model()

        masks = self.mask_generator.generate(image)

        save_psd_with_masks(image, masks, output_path)
        combined_image = create_mask_combined_images(image, masks)
        gallery = create_mask_gallery(image, masks)
        return [combined_image] + gallery, output_path
