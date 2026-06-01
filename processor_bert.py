from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_models():
    if os.getenv("DISABLE_BERT", "").strip() in {"1", "true", "True", "yes", "YES"}:
        raise RuntimeError("BERT classifier disabled via DISABLE_BERT")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Optional dependency missing: sentence-transformers (and its backend, e.g. torch)."
        ) from exc

    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Optional dependency missing: joblib") from exc

    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    model_path = Path(__file__).resolve().parent / "models" / "log_classifier.joblib"
    model_classification = joblib.load(model_path)
    return model_embedding, model_classification


def classify_with_bert(log_message: str) -> str:
    model_embedding, model_classification = _load_models()

    embeddings = model_embedding.encode([log_message])
    probabilities = model_classification.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = model_classification.predict(embeddings)[0]
    
    return predicted_label


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
