import json
import statistics
from pathlib import Path

from src.config import (
    DRIFT_BRIGHTNESS_DELTA,
    LOGS_DIR,
)


PREDICTIONS_PATH = LOGS_DIR / "predictions_log.jsonl"
FEEDBACK_PATH = LOGS_DIR / "feedback_log.jsonl"
BASELINE_PATH = LOGS_DIR / "baseline_stats.json"
REPORT_PATH = LOGS_DIR / "drift_report.json"


def _read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _load_baseline(path: Path) -> dict:
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as file:
                baseline = json.load(file)
            if isinstance(baseline, dict):
                return baseline
        except (OSError, json.JSONDecodeError):
            pass
    return {"expected_brightness": 0.5, "expected_contrast": 0.2}


def run_drift_audit(
    predictions_path: Path | str = PREDICTIONS_PATH,
    feedback_path: Path | str = FEEDBACK_PATH,
    baseline_path: Path | str = BASELINE_PATH,
) -> dict:
    predictions = _read_jsonl(predictions_path)
    feedbacks = _read_jsonl(feedback_path)
    baseline = _load_baseline(Path(baseline_path))

    image_stats = [
        record.get("image_stats")
        for record in predictions
        if isinstance(record.get("image_stats"), dict)
    ]
    brightness = [
        float(stats["mean_brightness"])
        for stats in image_stats
        if isinstance(stats.get("mean_brightness"), (int, float))
    ]
    contrast = [
        float(stats["std_contrast"])
        for stats in image_stats
        if isinstance(stats.get("std_contrast"), (int, float))
    ]

    confidence = [
        float(record["confianza"])
        for record in predictions
        if isinstance(record.get("confianza"), (int, float))
    ]
    latency = [
        float(record["latency_ms"])
        for record in predictions
        if isinstance(record.get("latency_ms"), (int, float))
    ]

    total_predictions = len(predictions)
    drift_count = sum(bool(record.get("drift_detected")) for record in predictions)
    low_confidence_count = sum(
        bool(record.get("baja_confianza")) for record in predictions
    )
    drift_rate = drift_count / total_predictions if total_predictions else 0.0
    low_confidence_rate = (
        low_confidence_count / total_predictions if total_predictions else 0.0
    )

    evaluated_feedback = [
        record for record in feedbacks if isinstance(record.get("acuerdo"), bool)
    ]
    disagreement_count = sum(
        record["acuerdo"] is False for record in evaluated_feedback
    )
    disagreement_rate = (
        disagreement_count / len(evaluated_feedback)
        if evaluated_feedback
        else 0.0
    )
    critical_errors = [
        {
            "prediction_id": record.get("prediction_id"),
            "timestamp": record.get("timestamp"),
        }
        for record in feedbacks
        if record.get("etiqueta_real") == "podrida"
        and record.get("clase_predicha") == "madura"
    ]

    expected_brightness = float(baseline.get("expected_brightness", 0.5))
    brightness_delta = (
        abs(statistics.fmean(brightness) - expected_brightness)
        if brightness
        else 0.0
    )

    if critical_errors or drift_rate >= 0.20 or low_confidence_rate >= 0.30:
        recommendation = "RETRAIN_RECOMMENDED"
    elif (
        drift_rate > 0.0
        or low_confidence_rate >= 0.10
        or disagreement_rate >= 0.15
        or brightness_delta > DRIFT_BRIGHTNESS_DELTA
    ):
        recommendation = "MONITOR_CLOSELY"
    else:
        recommendation = "OK"

    summary = {
        "total_predictions": total_predictions,
        "total_feedbacks": len(feedbacks),
        "statistics": {
            "mean_brightness": _distribution(brightness),
            "std_contrast": _distribution(contrast),
            "confianza": _distribution(confidence),
            "latency_ms": _distribution(latency),
        },
        "drift_detected_count": drift_count,
        "drift_detected_rate": float(drift_rate),
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": float(low_confidence_rate),
        "feedback_disagreement_count": disagreement_count,
        "feedback_disagreement_rate": float(disagreement_rate),
        "feedback_evaluated_count": len(evaluated_feedback),
        "critical_rotten_as_mature_errors": critical_errors,
        "brightness_delta_from_baseline": float(brightness_delta),
        "recommendation": recommendation,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "drift_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return summary


if __name__ == "__main__":
    report = run_drift_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))