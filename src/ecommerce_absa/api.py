from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import ABSA_ASPECT_COLUMNS, DEFAULT_ALERT_THUMBS_THRESHOLD, MODELS_DIR
from .data import score_to_sentiment
from .models.baselines import load_model_artifact
from .preprocessing import TextPreprocessor
from .race import analyze_business_signal, detect_aspects


class PredictionService:
    def __init__(
        self,
        model_path: str | Path | None = None,
        aspect_model_path: str | Path | None = None,
    ):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.aspect_model = None
        self.metadata: dict[str, Any] = {}
        self.aspect_metadata: dict[str, Any] = {}
        if model_path:
            self.model, self.metadata = load_model_artifact(model_path)
        resolved_aspect_model_path = aspect_model_path or (MODELS_DIR / "absa_aspect_baseline.joblib")
        if Path(resolved_aspect_model_path).exists():
            self.aspect_model, self.aspect_metadata = load_model_artifact(resolved_aspect_model_path)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload.get("content", "")
        processed_text = self.preprocessor.transform(content)
        sentiment = self._predict_sentiment(processed_text, payload)
        aspects = self._predict_aspects(content, processed_text)
        business = analyze_business_signal(
            content=f"{content} {processed_text}",
            sentiment=sentiment,
            aspects=aspects,
            thumbsupcount=payload.get("thumbsupcount", 0),
            replycontent=payload.get("replycontent", ""),
        )
        return {
            "content": content,
            "processed_text": processed_text,
            "sentiment": sentiment,
            "aspects": business["aspects"],
            "race": business["race"],
            "alert": business["alert"],
            "metadata": {
                "sentiment_model_loaded": self.model is not None,
                "aspect_model_loaded": self.aspect_model is not None,
                "aspect_source": "model" if self.aspect_model is not None else "rule_based",
                "alert_thumbsup_threshold": DEFAULT_ALERT_THUMBS_THRESHOLD,
            },
        }

    def _predict_sentiment(self, processed_text: str, payload: dict[str, Any]) -> str | None:
        if self.model is not None:
            return str(self.model.predict([processed_text])[0])
        return score_to_sentiment(payload.get("score"))

    def _predict_aspects(self, content: str, processed_text: str) -> list[str]:
        if self.aspect_model is None:
            return detect_aspects(f"{content} {processed_text}")

        prediction = self.aspect_model.predict([processed_text])[0]
        aspects = [
            aspect
            for index, aspect in enumerate(ABSA_ASPECT_COLUMNS)
            if int(prediction[index]) == 1
        ]
        return aspects or ["general"]


def create_app(
    model_path: str | Path | None = None,
    aspect_model_path: str | Path | None = None,
):
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:
        raise RuntimeError("Flask is required. Install dependencies with: pip install -r requirements.txt") from exc

    app = Flask(__name__)
    service = PredictionService(model_path, aspect_model_path)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "sentiment_model_loaded": service.model is not None,
                "aspect_model_loaded": service.aspect_model is not None,
            }
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if "content" not in payload:
            return jsonify({"error": "content is required"}), 400
        return jsonify(service.predict(payload))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABSA Flask JSON API")
    parser.add_argument("--model", default=None, help="Path to a joblib model artifact")
    parser.add_argument(
        "--aspect-model",
        default=str(MODELS_DIR / "absa_aspect_baseline.joblib"),
        help="Path to a joblib aspect model artifact",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(args.model, args.aspect_model)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
