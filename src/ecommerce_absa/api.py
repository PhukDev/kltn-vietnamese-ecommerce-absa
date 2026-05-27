from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import ABSA_ASPECT_COLUMNS, DEFAULT_ALERT_THUMBS_THRESHOLD, MODELS_DIR, SENTIMENT_LABELS
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
        self.is_pytorch = False
        self.pytorch_model = None
        self.tokenizer = None
        self.device = "cpu"
        self.max_len = 128

        # Check if either path is a PyTorch checkpoint (.pt)
        pt_path = None
        if model_path and str(model_path).endswith(".pt"):
            pt_path = model_path
        elif aspect_model_path and str(aspect_model_path).endswith(".pt"):
            pt_path = aspect_model_path

        if pt_path and Path(pt_path).exists():
            self._load_pytorch_model(pt_path)
        else:
            if model_path and Path(model_path).exists():
                self.model, self.metadata = load_model_artifact(model_path)
            resolved_aspect_model_path = aspect_model_path or (MODELS_DIR / "absa_aspect_baseline.joblib")
            if Path(resolved_aspect_model_path).exists():
                self.aspect_model, self.aspect_metadata = load_model_artifact(resolved_aspect_model_path)

    def _load_pytorch_model(self, pt_path: str | Path) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from .models.deep_learning import build_phobert_multitask_classifier
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for PyTorch models. "
                "Install deep learning dependencies with: pip install -r requirements-dl.txt"
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(pt_path, map_location=self.device)
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", "vinai/phobert-base")
        self.max_len = config.get("max_len", 128)

        self.pytorch_model = build_phobert_multitask_classifier(
            model_name=model_name,
            num_sentiment_labels=3,
            num_aspect_labels=5,
            dropout=config.get("dropout", 0.2),
        )
        self.pytorch_model.load_state_dict(checkpoint["model_state"])
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()

        tokenizer_name = checkpoint.get("tokenizer_name") or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.is_pytorch = True

    def _predict_pytorch(self, text: str) -> tuple[list[str], str]:
        import torch

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.pytorch_model(input_ids, attention_mask=attention_mask)
            aspect_logits = outputs["aspect_logits"]
            sentiment_logits = outputs["sentiment_logits"]

            aspect_preds = (torch.sigmoid(aspect_logits) >= 0.5).int().cpu().numpy()[0]
            aspects = [
                aspect
                for index, aspect in enumerate(ABSA_ASPECT_COLUMNS)
                if int(aspect_preds[index]) == 1
            ]
            if not aspects:
                aspects = ["general"]

            sentiment_pred = sentiment_logits.argmax(dim=-1).cpu().item()
            sentiment = SENTIMENT_LABELS[sentiment_pred]

        return aspects, sentiment

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload.get("content", "")
        processed_text = self.preprocessor.transform(content)
        if self.is_pytorch:
            aspects, sentiment = self._predict_pytorch(processed_text)
        else:
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
                "sentiment_model_loaded": self.model is not None or self.is_pytorch,
                "aspect_model_loaded": self.aspect_model is not None or self.is_pytorch,
                "aspect_source": "phobert_multitask" if self.is_pytorch else ("model" if self.aspect_model is not None else "rule_based"),
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
