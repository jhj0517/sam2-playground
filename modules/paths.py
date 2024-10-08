import os

WEBUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
SAM2_CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
MODEL_CONFIGS = {
    "sam2_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_l.yaml"),
    "sam2.1_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2.1_hiera_t.yaml"),
    "sam2.1_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2.1_hiera_l.yaml"),
    "sam2.1_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2.1_hiera_b+.yaml"),
    "sam2.1_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2.1_hiera_l.yaml"),
}
OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")
OUTPUT_PSD_DIR = os.path.join(OUTPUT_DIR, "psd")
OUTPUT_FILTER_DIR = os.path.join(OUTPUT_DIR, "filter")
TEMP_DIR = os.path.join(WEBUI_DIR, "temp")
TEMP_OUT_DIR = os.path.join(TEMP_DIR, "out")

for dir_path in [MODELS_DIR,
                 SAM2_CONFIGS_DIR,
                 OUTPUT_DIR,
                 OUTPUT_PSD_DIR,
                 OUTPUT_FILTER_DIR,
                 TEMP_DIR,
                 TEMP_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

