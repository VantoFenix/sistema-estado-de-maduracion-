from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.monitoring import (
    evaluate_drift_and_confidence,
    get_monitoring_summary,
    initialize_monitoring_storage,
    record_feedback,
    record_prediction,
)
from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    MetricsSummaryResponse,
    PredictionResponse,
)
from src.config import BASE_DIR, CLASS_NAMES, MODEL_PATH
from src.inference import (
    generate_gradcam_base64,
    load_class_names,
    load_inference_model,
    predict_from_bytes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_monitoring_storage()
    app.state.model = None
    has_model = MODEL_PATH.is_file()
    app.state.class_names = load_class_names(required=has_model)
    if has_model:
        app.state.model = load_inference_model()
    else:
        logging.getLogger(__name__).warning(
            "No se encontró el modelo en %s; /predict responderá HTTP 503 "
            "hasta que se despliegue un modelo entrenado.",
            MODEL_PATH,
        )
    yield


app = FastAPI(title="Clasificacion de Madurez", lifespan=lifespan)
app.state.model = None
app.state.class_names = list(CLASS_NAMES)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
if not cors_origins:
    cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "api" / "static"
static_file = static_dir / "index.html"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "classes": app.state.class_names,
    }


@app.get("/ready")
def readiness():
    if app.state.model is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen esta vacia",
        )
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible",
        )

    try:
        prediction = predict_from_bytes(
            image_bytes,
            app.state.model,
            app.state.class_names,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"No fue posible procesar la imagen: {exc}",
        ) from exc

    drift = evaluate_drift_and_confidence(
        prediction["confianza"],
        prediction["image_stats"],
    )

    gradcam_base64 = None
    if include_gradcam:
        try:
            gradcam_base64, _ = generate_gradcam_base64(
                image_bytes,
                app.state.model,
            )
        except Exception:
            gradcam_base64 = None

    prediction_id = str(uuid4())
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        "clase_predicha": prediction["clase_predicha"],
        "confianza": prediction["confianza"],
        "baja_confianza": prediction["baja_confianza"],
        "latency_ms": prediction["latency_ms"],
        "image_stats": prediction["image_stats"],
        "drift_detected": drift["drift_detected"],
        "drift_reasons": drift["reasons"],
    }
    record_prediction(payload)

    return PredictionResponse(
        prediction_id=prediction_id,
        clase_predicha=prediction["clase_predicha"],
        confianza=prediction["confianza"],
        baja_confianza=prediction["baja_confianza"],
        probabilidades=prediction["probabilidades"],
        latency_ms=prediction["latency_ms"],
        image_stats=prediction["image_stats"],
        drift=drift,
        gradcam_base64=gradcam_base64,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    if request.etiqueta_real not in app.state.class_names:
        raise HTTPException(
            status_code=422,
            detail="Etiqueta real invalida",
        )
    return FeedbackResponse(
        **record_feedback(
            prediction_id=request.prediction_id,
            etiqueta_real=request.etiqueta_real,
            comentario=request.comentario,
        )
    )


@app.get("/metrics", response_model=MetricsSummaryResponse)
def metrics():
    return MetricsSummaryResponse(**get_monitoring_summary())


@app.get("/")
def index():
    if static_file.is_file() and static_file.stat().st_size > 0:
        return FileResponse(static_file)
    return JSONResponse(
        {
            "message": "Servicio de clasificacion de madurez",
            "docs": "/docs",
        }
    )
