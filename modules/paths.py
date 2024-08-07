import os

WEBUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
SAM2_CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")

for dir_path in [WEBUI_DIR,
                 MODELS_DIR,
                 SAM2_CONFIGS_DIR,
                 OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
