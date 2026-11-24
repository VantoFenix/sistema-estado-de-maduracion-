import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config import CLASS_NAMES
from src.inference import (
    generate_gradcam_base64,
    predict_from_bytes,
    preprocess_image_bytes,
)
from src.model import build_cnn_model


def make_synthetic_image_bytes():
    array = np.zeros((48, 48, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(48, dtype=np.uint16).reshape(48, 1).astype(
        np.uint8
    )
    array[:, :, 1] = np.arange(48, dtype=np.uint16).astype(
        np.uint8
    ).reshape(1, 48)
    array[:, :, 2] = 128

    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocess_image_bytes():
    image_tensor, image_stats = preprocess_image_bytes(
        make_synthetic_image_bytes()
    )

    assert image_tensor.shape == (224, 224, 3)
    assert image_tensor.dtype == np.float32
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) <= 1.0
    assert "mean_brightness" in image_stats
    assert "std_contrast" in image_stats
    assert "mean_rgb" in image_stats
    assert len(image_stats["mean_rgb"]) == 3


def test_predict_from_bytes():
    model = build_cnn_model()
    result = predict_from_bytes(
        make_synthetic_image_bytes(),
        model,
        class_names=CLASS_NAMES,
    )

    assert result["clase_predicha"] in CLASS_NAMES
    assert pytest.approx(1.0, rel=1e-5, abs=1e-5) == sum(
        result["probabilidades"].values()
    )
    assert result["latency_ms"] > 0.0
    assert set(result["probabilidades"]) == set(CLASS_NAMES)


def test_generate_gradcam_base64():
    model = build_cnn_model()
    gradcam_b64, pred_index = generate_gradcam_base64(
        make_synthetic_image_bytes(),
        model,
    )

    assert isinstance(gradcam_b64, str)
    assert 0 <= pred_index < len(CLASS_NAMES)

    png_bytes = base64.b64decode(gradcam_b64)
    decoded = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert decoded is not None
    assert decoded.shape == (224, 224, 3)