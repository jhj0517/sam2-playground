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
        '-start_number', str(start_number),
        f'{output_path}'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception("Error occurred while extracting frames from the video")
        raise RuntimeError(f"An error occurred: {str(e)}")

    return get_frames_from_dir(output_temp_dir)


def extract_sound(
    vid_input: str,
    output_temp_dir: str = TEMP_DIR,
):
    """
    Extract audio from a video file and save it as a separate sound file. This needs FFmpeg installed.
    """
    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "sound.mp3")

    command = [
        'ffmpeg',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-vn',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception("Error occurred while extracting sound from the video")
        raise RuntimeError(f"An error occurred: {str(e)}")

    return output_path


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


def clean_temp_dir(temp_dir: str = TEMP_DIR):
    """Removes media files from the directory."""
    clean_sound_files(temp_dir)
    clean_image_files(temp_dir)


def clean_sound_files(sound_dir: str):
    """Removes all sound files from the directory."""
    sound_extensions = ('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma')

    for filename in os.listdir(sound_dir):
        if filename.lower().endswith(sound_extensions):
            file_path = os.path.join(sound_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                logger.exception("Error while removing sound files")
                raise RuntimeError(f"Error removing {file_path}: {str(e)}")


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
                raise RuntimeError(f"An error occurred: {str(e)}")
