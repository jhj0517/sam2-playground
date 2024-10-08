from test_config import *
from modules.paths import *
from modules.sam_inference import SamInference

import pytest


def test_video_segmentation(
    model_name:str,
):
    inferencer = SamInference()


