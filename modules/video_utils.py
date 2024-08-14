import subprocess
import os
from typing import List, Optional, Union
from PIL import Image
import numpy as np

from modules.logger_util import get_logger
from modules.paths import TEMP_DIR

logger = get_logger()


def extract_frames(
    vid_input: str,
    output_temp_dir: str = TEMP_DIR,
    quality: int = 2,
    start_number: int = 0
):
    """
    Extract frames as jpg files and save them into output_temp_dir. This needs FFmpeg installed.
    """
    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "%05d.jpg")

    command = [
        'ffmpeg',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-q:v', str(quality),
        '-start_number', str(start_number),
        f'{output_path}'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception("Error occurred while extracting frames from the video")
        raise f"An error occurred: {str(e)}"


def get_frames_from_dir(vid_dir: str,
                        available_extensions: Optional[Union[List, str]] = None,
                        as_numpy: bool = False) -> List:
    """Get image file paths list from the dir"""
    if available_extensions is None:
        available_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]

    if isinstance(available_extensions, str):
        available_extensions = [available_extensions]

    frame_names = [
        p for p in os.listdir(vid_dir)
        if os.path.splitext(p)[-1] in available_extensions
    ]
    if not frame_names:
        return []
    frame_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

    frames = [os.path.join(vid_dir, name) for name in frame_names]
    if as_numpy:
        frames = [np.array(Image.open(frame)) for frame in frames]

    return frames


def clean_image_files(image_dir: str):
    """Removes all image files from the dir"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(image_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                logger.exception("Error while removing image files")
                raise f"Error removing {str(e)}"
