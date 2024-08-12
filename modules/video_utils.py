import subprocess
import os

from modules.logger_util import get_logger

logger = get_logger()


def extract_frames(
    vid_input: str,
    output_temp_dir: str,
    quality: int = 2,
    start_number: int = 0
):
    """Extract frames and save them into output_temp_dir. This needs FFmpeg installed."""
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

