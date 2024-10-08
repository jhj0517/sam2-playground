from PIL import Image
import pytest
from typing import Dict, Optional

from test_config import *
import numpy as np
from modules.paths import *
from modules.constants import *
from modules.sam_inference import SamInference


@pytest.mark.parametrize(
    "model_name,image_path,points,labels,box,multimask_output",
    [
        (TEST_MODEL, TEST_IMAGE_PATH, TEST_POINTS, TEST_LABELS, TEST_BOX, True)
    ]
)
def test_image_segmentation(
    model_name: str,
    image_path: str,
    points: np.ndarray,
    labels: np.ndarray,
    box: np.ndarray,
    multimask_output: bool
):
    download_test_files()

    inferencer = SamInference()
    image = load_image(image_path)

    hparams = {
        "multimask_output": multimask_output,
    }

    masks, scores, logits = inferencer.predict_image(
        image=image,
        model_type=model_name,
        point_coords=points,
        point_labels=labels,
        **hparams
    )

    assert isinstance(masks, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(logits, np.ndarray)


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    return image_array
