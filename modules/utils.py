import os
from PIL import Image
import numpy as np


def open_folder(folder_path: str):
    if os.path.exists(folder_path):
        os.system(f'start "" "{folder_path}"')
    else:
        print(f"The folder '{folder_path}' does not exist.")


def save_image(image_numpy: np.ndarray, output_path: str):
    image = Image.fromarray(image_numpy)
    image.save(output_path, "JPEG")