import os

WEBUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
SAM2_CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")
OUTPUT_PSD_DIR = os.path.join(OUTPUT_DIR, "psd")

for dir_path in [MODELS_DIR,
                 SAM2_CONFIGS_DIR,
                 OUTPUT_DIR,
                 OUTPUT_PSD_DIR]:
    os.makedirs(dir_path, exist_ok=True)
