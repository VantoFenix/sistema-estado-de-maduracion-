import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    CLASS_NAMES,
    CLASSES_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_CLASSES,
)


def load_inference_model(model_path: Path | str = MODEL_PATH) -> keras.Model:
    model = tf.keras.models.load_model(
        Path(model_path),
        compile=False,
    )
    output = model(
        tf.zeros((1, *IMAGE_SIZE, 3), dtype=tf.float32),
        training=False,
    )
    if not tf.is_tensor(output) or output.shape != (1, NUM_CLASSES):
        raise ValueError("La salida del modelo debe tener un valor por clase")
    return model


def load_class_names(
    classes_path: Path | str = CLASSES_PATH,
    *,
    required: bool = False,
) -> list[str]:
    path = Path(classes_path)
    if not path.is_file():
        if required:
            raise ValueError(f"Falta el catálogo de clases del modelo: {path}")
        return list(CLASS_NAMES)

    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError(f"El catálogo {path} debe contener una lista de clases")

    # El notebook usa «deteriorada» y la API publica «podrida». Cada índice se conserva.
    names = ["podrida" if name == "deteriorada" else name for name in class_names]
    if len(names) != NUM_CLASSES or set(names) != set(CLASS_NAMES):
        raise ValueError(f"El catálogo {path} debe identificar cada clase una sola vez")
    return names


def preprocess_image_bytes(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    if not image_bytes:
        raise ValueError("La imagen esta vacia")

    image_bytes_tensor = tf.convert_to_tensor(image_bytes, dtype=tf.string)
    image = tf.io.decode_image(
        image_bytes_tensor,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)

    image_stats = {
        "mean_brightness": float(tf.reduce_mean(image).numpy()),
        "std_contrast": float(tf.math.reduce_std(image).numpy()),
        "mean_rgb": [
            float(value)
            for value in tf.reduce_mean(image, axis=[0, 1]).numpy()
        ],
    }
    return image.numpy().astype(np.float32), image_stats


def preprocess_image_path(
    image_path: Path | str,
) -> tuple[np.ndarray, dict]:
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"La imagen no existe: {path}")
    return preprocess_image_bytes(path.read_bytes())


def predict_from_bytes(
    image_bytes: bytes,
    model: keras.Model,
    class_names: list[str] | None = None,
) -> dict:
    names = (
        list(class_names)
        if class_names is not None
        else load_class_names()
    )
    if len(names) != model.output_shape[-1]:
        raise ValueError(
            "El numero de clases no coincide con la salida del modelo"
        )

    img_tensor_np, image_stats = preprocess_image_bytes(image_bytes)

    start = time.perf_counter()
    probabilities = np.asarray(
        model(np.expand_dims(img_tensor_np, axis=0), training=False)[0],
        dtype=np.float32,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    pred_index = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_index])

    return {
        "clase_predicha": names[pred_index],
        "confianza": confidence,
        "baja_confianza": confidence < CONFIDENCE_THRESHOLD,
        "probabilidades": {
            name: float(probabilities[index])
            for index, name in enumerate(names)
        },
        "latency_ms": float(latency_ms),
        "image_stats": image_stats,
    }


def generate_gradcam_base64(
    image_bytes: bytes,
    model: keras.Model,
    alpha: float = 0.4,
) -> tuple[str, int]:
    img_tensor_np, _ = preprocess_image_bytes(image_bytes)
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(img_tensor_np, axis=0),
        dtype=tf.float32,
    )

    last_conv_layer = None
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("El modelo no contiene ninguna capa Conv2D")

    if isinstance(model, keras.Sequential):
        # Un Sequential recargado puede tener varios nodos por capa.
        # Construimos ambas salidas en el mismo grafo para conectar el gradiente.
        grad_input = keras.Input(shape=model.input_shape[1:], dtype=model.inputs[0].dtype)
        output = grad_input
        for layer in model.layers:
            output = layer(output)
            if layer is last_conv_layer:
                conv_output = output
    else:
        grad_input = model.inputs
        conv_output = last_conv_layer.output
        output = model.outputs[0]
    grad_model = keras.Model(inputs=grad_input, outputs=[conv_output, output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        pred_index = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, pred_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise ValueError("No fue posible calcular los gradientes de Grad-CAM")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(
        conv_outputs[0] * pooled_gradients,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap_max = tf.reduce_max(heatmap)
    heatmap = tf.where(
        heatmap_max > 0,
        heatmap / heatmap_max,
        tf.zeros_like(heatmap),
    )

    heatmap_np = heatmap.numpy()
    heatmap_np = cv2.resize(
        heatmap_np,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
    )
    heatmap_uint8 = np.uint8(np.clip(heatmap_np, 0.0, 1.0) * 255)
    jet_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

    base_image = np.uint8(np.clip(img_tensor_np, 0.0, 1.0) * 255)
    overlay = cv2.addWeighted(
        base_image,
        1.0 - alpha,
        jet_heatmap,
        alpha,
        0,
    )

    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise ValueError("No fue posible codificar el Grad-CAM como PNG")

    gradcam_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return gradcam_b64, pred_index
