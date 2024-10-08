import os.path
import requests
import numpy as np
import subprocess

from modules.paths import *

TEST_MODEL = "sam2.1_hiera_tiny"
TEST_VIDEO_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
TEST_IMAGE_URL = "https://raw.githubusercontent.com/test-images/png/refs/heads/main/202105/cs-blue-00f.png"
TEST_VIDEO_PATH = os.path.join(WEBUI_DIR, "tests", "test_video.mp4")
TEST_IMAGE_PATH = os.path.join(WEBUI_DIR, "tests", "test_image.png")
TEST_POINTS = np.array([[850., 424.],
                        [810., 510.]])
TEST_LABELS = np.array([1., 1.])
TEST_BOX = np.array([[662., 197., 1056., 711.]])
TEST_GRADIO_PROMPT_BOX = [[653.0, 163.0, 2.0, 1132.0, 702.0, 3.0]]
TEST_GRADIO_PROMPT_POINTS = [[817.0, 366.0, 1.0, 0.0, 0.0, 4.0], [864.0, 533.0, 1.0, 0.0, 0.0, 4.0]]


def download_test_sam_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name) + ".pt"
    if os.path.exists(model_path):
        return

    from modules.model_downloader import download_sam_model_url
    download_sam_model_url(model_type=model_name, model_dir=MODELS_DIR)


def download_test_files():
    if not os.path.exists(TEST_IMAGE_PATH):
        download_file(TEST_IMAGE_URL, TEST_IMAGE_PATH)
    if not os.path.exists(TEST_VIDEO_PATH):
        download_file(TEST_VIDEO_URL, TEST_VIDEO_PATH)
        trim_video(TEST_VIDEO_PATH, seconds=1)


def trim_video(video_path, seconds=1):
    temp_output_path = video_path + ".temp.mp4"

    command = [
        "ffmpeg", "-i", video_path, "-t", f"{seconds}", "-c", "copy", temp_output_path
    ]

    try:
        subprocess.run(command, check=True)
        os.replace(temp_output_path, video_path)
        print(f"Trimmed video to {seconds} seconds and saved to {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e}")


def download_file(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {url} to {path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

