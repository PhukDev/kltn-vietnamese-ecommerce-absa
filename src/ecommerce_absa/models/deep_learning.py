from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torch is required for deep learning. Install: pip install -r requirements-dl.txt") from exc
    return torch


class FocalLoss:
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
        torch = require_torch()
        self.torch = torch
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def __call__(self, logits, targets):
        torch = self.torch
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_bilstm_classifier(
    *,
    vocab_size: int,
    num_classes: int,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    pos_vocab_size: int | None = None,
    pos_embedding_dim: int = 32,
):
    torch = require_torch()

    class BiLSTMClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.pos_embedding = (
                torch.nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)
                if pos_vocab_size
                else None
            )
            lstm_input_dim = embedding_dim + (pos_embedding_dim if pos_vocab_size else 0)
            self.lstm = torch.nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, input_ids, attention_mask=None, pos_ids=None):
            embeddings = [self.word_embedding(input_ids)]
            if self.pos_embedding is not None:
                if pos_ids is None:
                    raise ValueError("pos_ids is required when pos_vocab_size is configured")
                embeddings.append(self.pos_embedding(pos_ids))
            x = torch.cat(embeddings, dim=-1)
            output, _ = self.lstm(x)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled = output.mean(dim=1)
            return self.classifier(self.dropout(pooled))

    return BiLSTMClassifier()


def build_phobert_multitask_classifier(
    *,
    model_name: str = "vinai/phobert-base",
    num_sentiment_labels: int = 3,
    num_aspect_labels: int = 5,
    dropout: float = 0.2,
):
    try:
        import torch  # type: ignore
        from transformers import AutoModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required. Install: pip install -r requirements-dl.txt") from exc

    class PhoBERTMultiTaskClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout)
            self.aspect_head = torch.nn.Linear(hidden_size, num_aspect_labels)
            self.sentiment_head = torch.nn.Linear(hidden_size, num_sentiment_labels)

        def forward(
            self,
            input_ids,
            attention_mask=None,
            aspect_labels=None,
            sentiment_labels=None,
            aspect_pos_weight=None,
            sentiment_class_weight=None,
        ):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = output.last_hidden_state[:, 0, :]
            cls_embedding = self.dropout(cls_embedding)
            aspect_logits = self.aspect_head(cls_embedding)
            sentiment_logits = self.sentiment_head(cls_embedding)
            loss = None
            if aspect_labels is not None:
                aspect_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    aspect_logits,
                    aspect_labels.float(),
                    pos_weight=aspect_pos_weight,
                )
                loss = aspect_loss
            if sentiment_labels is not None:
                sentiment_loss = torch.nn.functional.cross_entropy(
                    sentiment_logits,
                    sentiment_labels,
                    weight=sentiment_class_weight,
                )
                loss = sentiment_loss if loss is None else loss + sentiment_loss
            return {
                "loss": loss,
                "aspect_logits": aspect_logits,
                "sentiment_logits": sentiment_logits,
            }

    return PhoBERTMultiTaskClassifier()
