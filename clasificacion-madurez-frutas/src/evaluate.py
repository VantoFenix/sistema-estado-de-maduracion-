from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES, LOGS_DIR


REFERENCE_CRITICAL_ERROR = 0.2241


def evaluate_model_on_dataset(
    model,
    val_ds,
    class_names=CLASS_NAMES,
    save_plots: bool = True,
    output_dir: Path | str = LOGS_DIR,
    history: Optional[object] = None,
):
    y_pred_probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_real = np.concatenate(
        [np.asarray(labels).reshape(-1) for _, labels in val_ds],
        axis=0,
    )

    y_real = np.asarray(y_real, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)

    confusion = confusion_matrix(
        y_real,
        y_pred,
        labels=list(range(len(class_names))),
    )
    report = classification_report(
        y_real,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    try:
        rotten_index = list(class_names).index("podrida")
        mature_index = list(class_names).index("madura")
    except ValueError as exc:
        raise ValueError(
            "class_names debe contener 'madura' y 'podrida'"
        ) from exc

    total_podrida_real = int(np.sum(y_real == rotten_index))
    podrida_como_madura = int(
        confusion[rotten_index, mature_index]
    )
    tasa_error_critico = (
        podrida_como_madura / total_podrida_real
        if total_podrida_real > 0
        else 0.0
    )
    recall_podrida = float(
        report["podrida"]["recall"]
        if "podrida" in report
        else 0.0
    )

    output_path = Path(output_dir)
    if save_plots:
        output_path.mkdir(parents=True, exist_ok=True)

        if history is not None:
            history_dict = getattr(history, "history", history)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(history_dict.get("loss", []), label="train_loss")
            axes[0].plot(history_dict.get("val_loss", []), label="val_loss")
            axes[0].set_title("Loss")
            axes[0].legend()

            axes[1].plot(
                history_dict.get("accuracy", []),
                label="train_accuracy",
            )
            axes[1].plot(
                history_dict.get("val_accuracy", []),
                label="val_accuracy",
            )
            axes[1].set_title("Accuracy")
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(
                output_path / "curvas_entrenamiento.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title("Matriz de confusion")
        fig.tight_layout()
        fig.savefig(
            output_path / "matriz_confusion.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        confusion_percent = np.zeros_like(confusion, dtype=np.float64)
        row_totals = confusion.sum(axis=1, keepdims=True)
        np.divide(
            confusion,
            row_totals,
            out=confusion_percent,
            where=row_totals != 0,
        )
        confusion_percent *= 100.0

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_percent,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title("Matriz de confusion (%)")
        fig.tight_layout()
        fig.savefig(
            output_path / "matriz_confusion_porcentaje.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return {
        "confusion_matrix": confusion,
        "classification_report": report,
        "y_pred_probs": y_pred_probs,
        "y_pred": y_pred,
        "y_real": y_real,
        "total_podrida_real": total_podrida_real,
        "podrida_como_madura": podrida_como_madura,
        "tasa_error_critico": tasa_error_critico,
        "referencia_error_critico": REFERENCE_CRITICAL_ERROR,
        "recall_podrida": recall_podrida,
    }