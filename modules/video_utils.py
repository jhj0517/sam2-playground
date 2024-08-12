import subprocess
import os
from typing import List, Optional

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
    Extract frames and save them into output_temp_dir. This needs FFmpeg installed.
    """
    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "%05d.jpg")

    command = [
        'ffmpeg',
        '-i', vid_input,
        '-q:v', str(quality),
        '-start_number', str(start_number),
        f'{output_path}'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception("Error occured while extracting frames from the video")
        raise f"An error occurred: {str(e)}"


def get_frames_from_dir(vid_dir: str,
                        available_extensions: Optional[List, str] = None) -> List:
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
    return frame_names
