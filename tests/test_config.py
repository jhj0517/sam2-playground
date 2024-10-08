from modules.paths import *

TEST_MODEL = "sam2.1_hiera_tiny"


def test_download_sam_model_url(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name) + ".pt"
    if os.path.exists(model_path):
        return

    from modules.model_downloader import download_sam_model_url
    download_sam_model_url(model_type=model_name, model_dir=MODELS_DIR)
