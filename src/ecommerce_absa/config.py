from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "vietnamese_ecommerce_review.csv"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TEXT_COLUMN = "content"
SCORE_COLUMN = "score"
THUMBS_COLUMN = "thumbsupcount"
REPLY_COLUMN = "replycontent"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
ABSA_ASPECT_COLUMNS = ["product", "price", "delivery", "service", "app"]
ABSA_ANNOTATION_COLUMNS = ["reviewid", "content", "sentiment", *ABSA_ASPECT_COLUMNS]
ABSA_ANNOTATION_SAMPLE_PATH = DATA_DIR / "absa_annotation_sample.csv"
RACE_STAGES = ["Reach", "Act", "Convert", "Engage"]

DEFAULT_ALERT_THUMBS_THRESHOLD = 10
DEFAULT_RANDOM_STATE = 42
