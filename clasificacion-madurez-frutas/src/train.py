import json
from pathlib import Path

import tensorflow as tf

from src.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    CLASSES_PATH,
    DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
)
from src.dataset import (
    build_balanced_train_dataset,
    build_val_dataset,
    discover_images,
    prepare_train_pipeline,
    split_global_and_val,
)
from src.model import (
    CUSTOM_OBJECTS,
    build_cnn_model,
    compile_cnn_model,
)


def _load_checkpoint_metrics(checkpoint_path: Path, val_ds):
    checkpoint_model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects=CUSTOM_OBJECTS,
    )
    evaluation = checkpoint_model.evaluate(val_ds, verbose=0, return_dict=True)

    return {
        "model": checkpoint_model,
        "accuracy": float(evaluation.get("accuracy", 0.0)),
        "recall_podrida": float(
            evaluation.get(
                "recall_podrida",
                evaluation.get("recall_clase_podrida", 0.0),
            )
        ),
        "precision_podrida": float(
            evaluation.get(
                "precision_podrida",
                evaluation.get("precision_clase_podrida", 0.0),
            )
        ),
        "f1_podrida": float(
            evaluation.get(
                "f1_podrida",
                evaluation.get("f1_clase_podrida", 0.0),
            )
        ),
    }


def run_training(
    dataset_dir: Path | str = DATA_DIR,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    max_acc_drop: float = 0.03,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    paths_np, labels_np = discover_images(dataset_dir)
    (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (_, _),
    ) = split_global_and_val(paths_np, labels_np)

    balanced_ds, steps_per_epoch = build_balanced_train_dataset(
        train_paths,
        train_labels,
    )
    train_ds = prepare_train_pipeline(balanced_ds, augment=True)
    val_ds = build_val_dataset(val_paths, val_labels)

    model = build_cnn_model()
    model = compile_cnn_model(model, learning_rate=learning_rate)

    accuracy_checkpoint = ARTIFACTS_DIR / "mejor_modelo.keras"
    safety_checkpoint = (
        ARTIFACTS_DIR / "mejor_modelo_seguridad_alimentaria.keras"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(accuracy_checkpoint),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(safety_checkpoint),
            monitor="val_f1_podrida",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    acc_metrics = _load_checkpoint_metrics(accuracy_checkpoint, val_ds)
    seg_metrics = _load_checkpoint_metrics(safety_checkpoint, val_ds)

    use_safety_checkpoint = (
        seg_metrics["f1_podrida"] > acc_metrics["f1_podrida"]
        and (
            acc_metrics["accuracy"] - seg_metrics["accuracy"]
        ) <= max_acc_drop
    )
    selected_metrics = seg_metrics if use_safety_checkpoint else acc_metrics
    selected_checkpoint = (
        "mejor_modelo_seguridad_alimentaria.keras"
        if use_safety_checkpoint
        else "mejor_modelo.keras"
    )

    final_model = selected_metrics["model"]
    final_model.save(MODEL_PATH)

    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLASSES_PATH.open("w", encoding="utf-8") as classes_file:
        json.dump(CLASS_NAMES, classes_file, ensure_ascii=False)

    return {
        "model": final_model,
        "history": history,
        "accuracy_checkpoint": acc_metrics,
        "safety_checkpoint": seg_metrics,
        "selected_checkpoint": selected_checkpoint,
        "selected_metrics": selected_metrics,
    }


if __name__ == "__main__":
    result = run_training()
    print(
        "Entrenamiento finalizado:",
        result["selected_checkpoint"],
        result["selected_metrics"],
    )