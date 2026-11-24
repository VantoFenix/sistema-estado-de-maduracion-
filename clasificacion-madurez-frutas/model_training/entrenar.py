import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ['madura', 'podrida', 'verde']
ROTTEN_INDEX = CLASS_NAMES.index('podrida')
RIPE_INDEX = CLASS_NAMES.index('madura')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def read_dataset(root):
    roles = {'train': [], 'val': [], 'test': []}
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            roles[row['split']].append(row)
    return roles


def load_images(root, rows):
    paths = [str(root / row['file']) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def decode(path):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.ensure_shape(image, (*IMAGE_SIZE, 3))

    options = tf.data.Options()
    options.threading.private_threadpool_size = 3
    return dataset.map(decode, num_parallel_calls=2).batch(BATCH_SIZE).prefetch(1).with_options(options)


def indices(rows):
    return np.asarray([CLASS_NAMES.index(row['label']) for row in rows], dtype=np.int64)


def summarize(real, predictions):
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(real, predictions, labels=labels)
    report = classification_report(real, predictions, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    damaged_support = int(matrix[ROTTEN_INDEX].sum())
    critical_count = int(matrix[ROTTEN_INDEX, RIPE_INDEX])
    return {
        'n': len(real),
        'accuracy': float(np.trace(matrix) / len(real)),
        'macro_f1': report['macro avg']['f1-score'],
        'damaged_recall': report['podrida']['recall'],
        'damaged_as_ripe': critical_count,
        'damaged_support': damaged_support,
        'critical_error_rate': critical_count / damaged_support if damaged_support else None,
        'class_report': {key: report[key] for key in CLASS_NAMES},
        'confusion': matrix.tolist(),
    }


def evaluate(real, probabilities, rows):
    predictions = probabilities.argmax(axis=1)
    breakdown = {}
    for col in ('source', 'species'):
        for group in sorted({row[col] for row in rows}):
            subset = np.array([row[col] == group for row in rows])
            breakdown[group] = summarize(real[subset], predictions[subset])
    return {'overall': summarize(real, predictions), 'groups': breakdown}


def classifier():
    input_tensor = keras.Input(shape=(*IMAGE_SIZE, 3), name='imagen_rgb')
    resized = keras.layers.Resizing(128, 128)(input_tensor)
    scaled = keras.layers.Rescaling(2, offset=-1)(resized)
    base = keras.applications.MobileNetV3Small(
        input_shape=(128, 128, 3), include_top=False, weights='imagenet',
        include_preprocessing=False,
    )
    base.trainable = False
    features = base(scaled, training=False)
    extractor = keras.Model(input_tensor, features, name='extractor_mobilenet_v3')
    head = keras.Sequential([
        keras.Input(shape=extractor.output_shape[1:]),
        keras.layers.Conv2D(80, 1, activation='relu', name='maturity_features'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name='maturity_classes'),
    ])
    return extractor, head


def fit_classifier(dataset_root, destination):
    keras.utils.set_random_seed(42)
    roles = read_dataset(dataset_root)
    labels = {role: indices(rows) for role, rows in roles.items()}
    print('Particiones:', {role: len(rows) for role, rows in roles.items()}, flush=True)
    extractor, head = classifier()

    feature_sets = {}
    for role in ('train', 'val'):
        images = load_images(dataset_root, roles[role])
        feature_sets[role] = extractor.predict(images, verbose=0)
        print('Características', role, feature_sets[role].shape, flush=True)

    train_images = load_images(dataset_root, roles['train'])
    flipped = train_images.map(tf.image.flip_left_right, num_parallel_calls=2)
    flipped_features = extractor.predict(flipped, verbose=0)
    x_train = np.concatenate([feature_sets['train'], flipped_features])
    y_train = np.tile(labels['train'], 2)
    counts = Counter(labels['train'])
    weights = np.array([
        (len(roles['train']) / (3 * counts[label])) ** 0.5
        * (3.0 if label == ROTTEN_INDEX and row['source'] == 'vegnet' else 1.0)
        for label, row in zip(labels['train'], roles['train'])
    ], dtype=np.float32)
    head.compile(optimizer=keras.optimizers.Adam(0.0015),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def batches(features, labels, weights=None):
        parts = (features, labels) if weights is None else (features, labels, weights)
        dataset = tf.data.Dataset.from_tensor_slices(parts)
        if weights is not None:
            dataset = dataset.shuffle(len(labels), seed=42)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return dataset.batch(48).prefetch(1).with_options(options)

    history = head.fit(
        batches(x_train, y_train, np.tile(weights, 2)),
        validation_data=batches(feature_sets['val'], labels['val']),
        epochs=25, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
    )

    features = extractor.output
    for layer in head.layers:
        features = layer(features)
    model = keras.Model(extractor.input, features, name='madurez_mobilenet_v3')
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / 'modelo_clasificacion_frutas.keras'
    model.save(checkpoint)
    (destination / 'clases.json').write_text(json.dumps(CLASS_NAMES) + '\n', encoding='utf-8')

    loaded = keras.models.load_model(checkpoint, compile=False)
    details = {'architecture': model.name, 'epochs_run': len(history.epoch),
               'train_n': len(labels['train']), 'parameters': model.count_params()}
    for role in ('val', 'test'):
        dataset = load_images(dataset_root, roles[role])
        probabilities = loaded.predict(dataset, verbose=0)
        details[role] = evaluate(labels[role], probabilities, roles[role])
        for group, result in [('all', details[role]['overall']), *details[role]['groups'].items()]:
            metrics = {key: result[key] for key in ('n','accuracy','macro_f1','damaged_recall','damaged_as_ripe','damaged_support')}
            print(role, group, json.dumps(metrics), flush=True)
    (destination / 'evaluacion.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    print('Checkpoint:', checkpoint, checkpoint.stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    fit_classifier(arguments.dataset, arguments.destination)
