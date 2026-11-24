from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    PROPORCION_PARTE1,
    SEED,
    VALIDATION_SPLIT,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(dataset_dir: Path | str = DATA_DIR):
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise ValueError(f"La carpeta de datos no existe: {dataset_path}")

    paths: list[str] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_path / class_name
        if not class_dir.is_dir():
            raise ValueError(
                f"Falta la subcarpeta de clase '{class_name}' en {dataset_path}"
            )

        class_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(
                f"La subcarpeta '{class_name}' no contiene imagenes soportadas"
            )

        paths.extend(str(path) for path in class_paths)
        labels.extend([class_index] * len(class_paths))

    paths_np = np.asarray(paths, dtype=np.str_)
    labels_np = np.asarray(labels, dtype=np.int32)
    return paths_np, labels_np


def split_global_and_val(paths_np, labels_np):
    part1_paths, part2_paths, part1_labels, part2_labels = train_test_split(
        paths_np,
        labels_np,
        train_size=PROPORCION_PARTE1,
        stratify=labels_np,
        random_state=SEED,
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        part1_paths,
        part1_labels,
        test_size=VALIDATION_SPLIT,
        stratify=part1_labels,
        random_state=SEED,
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (part2_paths, part2_labels),
    )


def decode_and_resize_tensor(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def preprocess_image_tensor(path, label):
    return decode_and_resize_tensor(path, label)


def build_balanced_train_dataset(train_paths, train_labels):
    train_paths = np.asarray(train_paths)
    train_labels = np.asarray(train_labels)

    class_datasets = []
    max_class_count = 0

    for class_index in range(NUM_CLASSES):
        class_mask = train_labels == class_index
        class_paths = train_paths[class_mask]
        class_count = len(class_paths)

        if class_count == 0:
            raise ValueError(
                f"No hay muestras de '{CLASS_NAMES[class_index]}' en el conjunto train"
            )

        max_class_count = max(max_class_count, class_count)
        class_labels = train_labels[class_mask]

        ds = tf.data.Dataset.from_tensor_slices((class_paths, class_labels))
        ds = ds.shuffle(class_count, reshuffle_each_iteration=True, seed=SEED)
        ds = ds.repeat()
        ds = ds.map(decode_and_resize_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        class_datasets.append(ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / NUM_CLASSES] * NUM_CLASSES,
        seed=SEED,
    )
    balanced_ds = balanced_ds.batch(BATCH_SIZE)

    steps_per_epoch = max(
        1,
        (max_class_count * NUM_CLASSES) // BATCH_SIZE,
    )
    return balanced_ds, steps_per_epoch


def build_val_dataset(val_paths, val_labels):
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(val_paths), np.asarray(val_labels))
    )
    ds = ds.map(decode_and_resize_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )


def prepare_train_pipeline(balanced_ds, augment: bool = True):
    if augment:
        augmentation = build_augmentation_layer()

        def augment_batch(images, labels):
            return augmentation(images, training=True), labels

        balanced_ds = balanced_ds.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return balanced_ds.prefetch(tf.data.AUTOTUNE)