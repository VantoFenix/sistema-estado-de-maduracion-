from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"
MODEL_PATH = ARTIFACTS_DIR / "modelo_madurez.keras"
CLASSES_PATH = ARTIFACTS_DIR / "clases.json"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
CLASS_NAMES = ["madura", "podrida", "verde"]

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.2
PROPORCION_PARTE1 = 0.40
SEED = 42

COST_MATRIX = [
    [0.0, 2.0, 0.0],
    [4.0, 0.0, 1.5],
    [0.0, 1.5, 0.0],
]
LAMBDA_COSTO = 1.0

CONFIDENCE_THRESHOLD = 0.70
DRIFT_BRIGHTNESS_DELTA = 0.15