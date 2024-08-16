import os
from typing import Optional

from modules.paths import *

DEFAULT_MODEL_TYPE = "sam2_hiera_large"
AVAILABLE_MODELS = {
    "sam2_hiera_tiny": ["sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"],
    "sam2_hiera_small": ["sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"],
    "sam2_hiera_base_plus": ["sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"],
    "sam2_hiera_large": ["sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"],
}


def download_sam_model_url(model_type: str,
                           model_dir: str = MODELS_DIR):
    filename, url = AVAILABLE_MODELS[model_type]
    load_file_from_url(url=url, model_dir=model_dir)


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    filename: Optional[str] = None,
) -> str:
    from urllib.parse import urlparse
    """
    Download a file from `url` into `model_dir`, using the file present if possible.
    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not filename:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def is_sam_exist(
    model_type: str,
    model_dir: Optional[str] = None
):
    if model_dir is None:
        model_dir = MODELS_DIR
    filename, url = AVAILABLE_MODELS[model_type]
    model_path = os.path.join(model_dir, filename)
    return os.path.exists(model_path)
