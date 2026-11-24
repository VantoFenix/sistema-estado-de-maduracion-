import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    CONFIDENCE_THRESHOLD,
    DRIFT_BRIGHTNESS_DELTA,
    LOGS_DIR,
)


PREDICTIONS_LOG = LOGS_DIR / "predictions_log.jsonl"
FEEDBACK_LOG = LOGS_DIR / "feedback_log.jsonl"
BASELINE_STATS = LOGS_DIR / "baseline_stats.json"

_prediction_lock = threading.Lock()
_feedback_lock = threading.Lock()


def _read_json_lines(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _ensure_baseline(baseline_path: Path) -> dict:
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.is_file():
        try:
            with baseline_path.open("r", encoding="utf-8") as file:
                baseline = json.load(file)
            if "expected_brightness" not in baseline:
                baseline["expected_brightness"] = 0.5
            if "expected_contrast" not in baseline:
                baseline["expected_contrast"] = 0.2
            return baseline
        except (json.JSONDecodeError, OSError):
            pass

    baseline = {
        "expected_brightness": 0.5,
        "expected_contrast": 0.2,
    }
    with baseline_path.open("w", encoding="utf-8") as file:
        json.dump(baseline, file)
    return baseline


def initialize_monitoring_storage() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for log_path in (PREDICTIONS_LOG, FEEDBACK_LOG):
        log_path.touch(exist_ok=True)
    _ensure_baseline(BASELINE_STATS)


def evaluate_drift_and_confidence(
    confianza: float,
    image_stats: dict,
    baseline_path: Path | str = BASELINE_STATS,
) -> dict:
    baseline = _ensure_baseline(Path(baseline_path))
    expected_brightness = float(baseline["expected_brightness"])
    expected_contrast = float(baseline["expected_contrast"])

    mean_brightness = float(image_stats["mean_brightness"])
    std_contrast = float(image_stats["std_contrast"])
    brightness_delta = abs(mean_brightness - expected_brightness)
    low_confidence = confianza < CONFIDENCE_THRESHOLD

    reasons = []
    if brightness_delta > DRIFT_BRIGHTNESS_DELTA:
        reasons.append("desviacion_brightness")
    if std_contrast < expected_contrast * 0.5:
        reasons.append("caida_contraste")
    if low_confidence:
        reasons.append("baja_confianza")

    return {
        "drift_detected": bool(reasons),
        "reasons": reasons,
        "brightness_delta": float(brightness_delta),
        "low_confidence": low_confidence,
    }


def record_prediction(
    payload: dict,
    log_path: Path | str = PREDICTIONS_LOG,
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _prediction_lock:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_feedback(
    prediction_id: str,
    etiqueta_real: str,
    comentario: str | None = None,
    feedback_path: Path | str = FEEDBACK_LOG,
    predictions_path: Path | str = PREDICTIONS_LOG,
) -> dict:
    previous_prediction = None
    for record in reversed(_read_json_lines(Path(predictions_path))):
        if record.get("prediction_id") == prediction_id:
            previous_prediction = record
            break

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        "etiqueta_real": etiqueta_real,
        "comentario": comentario,
    }
    if previous_prediction is not None:
        predicted_class = previous_prediction.get("clase_predicha")
        record["clase_predicha"] = predicted_class
        record["acuerdo"] = predicted_class == etiqueta_real

    feedback_path = Path(feedback_path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock:
        with feedback_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "etiqueta_real": etiqueta_real,
        "total_feedbacks": len(_read_json_lines(feedback_path)),
    }


def get_monitoring_summary(
    predictions_path: Path | str = PREDICTIONS_LOG,
    feedback_path: Path | str = FEEDBACK_LOG,
) -> dict:
    predictions = _read_json_lines(Path(predictions_path))
    feedbacks = _read_json_lines(Path(feedback_path))

    total_predictions = len(predictions)
    if total_predictions:
        avg_latency_ms = sum(
            float(record.get("latency_ms", 0.0))
            for record in predictions
        ) / total_predictions
        avg_confidence = sum(
            float(record.get("confianza", 0.0))
            for record in predictions
        ) / total_predictions
        low_confidence_rate = sum(
            bool(record.get("baja_confianza", False))
            for record in predictions
        ) / total_predictions
        drift_events_count = sum(
            bool(record.get("drift_detected", False))
            for record in predictions
        )
    else:
        avg_latency_ms = 0.0
        avg_confidence = 0.0
        low_confidence_rate = 0.0
        drift_events_count = 0

    class_distribution = {}
    for record in predictions:
        predicted_class = record.get("clase_predicha")
        if predicted_class:
            class_distribution[predicted_class] = (
                class_distribution.get(predicted_class, 0) + 1
            )

    agreement_values = [
        record["acuerdo"]
        for record in feedbacks
        if "acuerdo" in record
    ]
    feedback_agreement_rate = (
        sum(bool(value) for value in agreement_values)
        / len(agreement_values)
        if agreement_values
        else None
    )

    return {
        "total_predictions": total_predictions,
        "avg_latency_ms": float(avg_latency_ms),
        "avg_confidence": float(avg_confidence),
        "low_confidence_rate": float(low_confidence_rate),
        "drift_events_count": int(drift_events_count),
        "class_distribution": class_distribution,
        "total_feedbacks": len(feedbacks),
        "feedback_agreement_rate": feedback_agreement_rate,
    }