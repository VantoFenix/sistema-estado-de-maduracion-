from pydantic import BaseModel, field_validator

from src.config import CLASS_NAMES


class ImageStatsSchema(BaseModel):
    mean_brightness: float
    std_contrast: float
    mean_rgb: list[float]


class DriftAlertSchema(BaseModel):
    drift_detected: bool
    reasons: list[str]
    brightness_delta: float
    low_confidence: bool


class PredictionResponse(BaseModel):
    prediction_id: str
    clase_predicha: str
    confianza: float
    baja_confianza: bool
    probabilidades: dict[str, float]
    latency_ms: float
    image_stats: ImageStatsSchema
    drift: DriftAlertSchema
    gradcam_base64: str | None = None


class FeedbackRequest(BaseModel):
    prediction_id: str
    etiqueta_real: str
    comentario: str | None = None

    @field_validator("etiqueta_real")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        if value not in CLASS_NAMES:
            raise ValueError(
                f"Etiqueta invalida. Debe ser una de: {', '.join(CLASS_NAMES)}"
            )
        return value


class FeedbackResponse(BaseModel):
    status: str
    prediction_id: str
    etiqueta_real: str
    total_feedbacks: int


class MetricsSummaryResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    avg_confidence: float
    low_confidence_rate: float
    drift_events_count: int
    class_distribution: dict[str, int]
    total_feedbacks: int
    feedback_agreement_rate: float | None