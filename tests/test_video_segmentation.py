from test_config import *
import numpy as np
from modules.paths import *
from modules.constants import *
from modules.sam_inference import SamInference

import pytest
from typing import Dict


@pytest.mark.parametrize(
    "model_name,video_path,points,labels,box",
    [
        TEST_MODEL,
        TEST_VIDEO_PATH,
        TEST_POINTS,
        TEST_LABELS,
        TEST_BOX
    ]
)
def test_video_segmentation(
    model_name: str,
    video_path: str,
    points: np.ndarray,
    labels: np.ndarray,
    box: np.ndarray
):
    download_test_files()

    inferencer = SamInference()
    inferencer.init_video_inference_state(
        vid_input=video_path,
        model_type=model_name,
    )

    inferencer.add_prediction_to_frame(
        frame_idx=0,
        obj_id=0,
        points=TEST_POINTS,
        labels=TEST_LABELS,
    )

    inferencer.add_prediction_to_frame(
        frame_idx=1,
        obj_id=1,
        box=TEST_BOX,
    )

    video_segments = inferencer.propagate_in_video()

    assert video_segments and isinstance(video_segments, Dict)

