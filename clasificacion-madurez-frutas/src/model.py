import tensorflow as tf
from tensorflow import keras

from src.config import (
    CLASS_NAMES,
    COST_MATRIX,
    IMAGE_SIZE,
    LAMBDA_COSTO,
    LEARNING_RATE,
    NUM_CLASSES,
)


@tf.keras.utils.register_keras_serializable(package="frutas_ml")
def perdida_sensible_a_costos(y_true, y_pred):
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)

    cost_matrix = tf.constant(COST_MATRIX, dtype=tf.float32)
    row_costs = tf.gather(cost_matrix, y_true_int)
    costo_esperado = tf.reduce_sum(row_costs * y_pred, axis=-1)

    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_int,
        y_pred,
    )
    return cross_entropy + LAMBDA_COSTO * costo_esperado


@tf.keras.utils.register_keras_serializable(package="frutas_ml")
class RecallClasePodrida(tf.keras.metrics.Metric):
    def __init__(self, name="recall_podrida", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives",
            initializer="zeros",
        )
        self.false_negatives = self.add_weight(
            name="false_negatives",
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        rotten_index = CLASS_NAMES.index("podrida")
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_rot = tf.equal(y_true, rotten_index)
        y_pred_rot = tf.equal(y_pred, rotten_index)

        true_positives = tf.logical_and(y_true_rot, y_pred_rot)
        false_negatives = tf.logical_and(y_true_rot, tf.logical_not(y_pred_rot))

        tp = tf.reduce_sum(tf.cast(true_positives, self.dtype))
        fn = tf.reduce_sum(tf.cast(false_negatives, self.dtype))
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="frutas_ml")
class PrecisionClasePodrida(tf.keras.metrics.Metric):
    def __init__(self, name="precision_podrida", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives",
            initializer="zeros",
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        rotten_index = CLASS_NAMES.index("podrida")
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_rot = tf.equal(y_true, rotten_index)
        y_pred_rot = tf.equal(y_pred, rotten_index)

        true_positives = tf.logical_and(y_true_rot, y_pred_rot)
        false_positives = tf.logical_and(tf.logical_not(y_true_rot), y_pred_rot)

        tp = tf.reduce_sum(tf.cast(true_positives, self.dtype))
        fp = tf.reduce_sum(tf.cast(false_positives, self.dtype))
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="frutas_ml")
class F1ClasePodrida(tf.keras.metrics.Metric):
    def __init__(self, name="f1_podrida", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives",
            initializer="zeros",
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            initializer="zeros",
        )
        self.false_negatives = self.add_weight(
            name="false_negatives",
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        rotten_index = CLASS_NAMES.index("podrida")
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_rot = tf.equal(y_true, rotten_index)
        y_pred_rot = tf.equal(y_pred, rotten_index)

        true_positives = tf.logical_and(y_true_rot, y_pred_rot)
        false_positives = tf.logical_and(tf.logical_not(y_true_rot), y_pred_rot)
        false_negatives = tf.logical_and(y_true_rot, tf.logical_not(y_pred_rot))

        tp = tf.reduce_sum(tf.cast(true_positives, self.dtype))
        fp = tf.reduce_sum(tf.cast(false_positives, self.dtype))
        fn = tf.reduce_sum(tf.cast(false_negatives, self.dtype))
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "perdida_sensible_a_costos": perdida_sensible_a_costos,
    "RecallClasePodrida": RecallClasePodrida,
    "PrecisionClasePodrida": PrecisionClasePodrida,
    "F1ClasePodrida": F1ClasePodrida,
}


def build_cnn_model(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in (32, 64, 128):
        x = keras.layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Clasificacion_Mejorada",
    )


def compile_cnn_model(
    model: keras.Model,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=perdida_sensible_a_costos,
        metrics=[
            "accuracy",
            RecallClasePodrida(),
            PrecisionClasePodrida(),
            F1ClasePodrida(),
        ],
    )
    return model