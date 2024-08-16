import os
from PIL import Image
from typing import Optional, Union
import numpy as np

from modules.constants import IMAGE_FILE_EXT


def open_folder(folder_path: str):
    if os.path.exists(folder_path):
        os.system(f'start "" "{folder_path}"')
    else:
        print(f"The folder '{folder_path}' does not exist.")


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_FILE_EXT


def get_image_files(image_dir: str):
    image_files = []
    for filename in os.listdir(image_dir):
        if is_image_file(filename):
            image_files.append(os.path.join(image_dir, filename))
    return image_files


def save_image(image: Union[np.ndarray, str],
               output_path: Optional[str] = None,
               output_dir: Optional[str] = None):

    if output_dir is None and output_path is None:
        raise ValueError("Either output_path or output_dir should be provided")

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if output_path is not None:
        image.save(output_path, "JPEG")
        return output_path

    os.makedirs(output_dir, exist_ok=True)
    num_images = len(get_image_files(output_dir))
    output_path = os.path.join(output_dir, f"{num_images:05d}.jpg")
    image.save(output_path)

    return output_path
