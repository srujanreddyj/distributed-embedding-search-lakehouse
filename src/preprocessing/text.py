# src/preprocessing/text.py

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.preprocessing.base import BasePreprocessor
from src.preprocessing.device import default_device
from src.preprocessing.payload import payload_values


class TextPreprocessor(BasePreprocessor):
    """Clean, tokenize, and embed text."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = default_device()
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=str(self.device),
        )
        self.model.eval()

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        texts = payload_values(batch, "caption")
        cleaned = [" ".join(t.split())[:2_000] for t in texts]
        with torch.no_grad():
            embeddings = self.model.encode(
                cleaned,
                convert_to_tensor=True,
                device=str(self.device),
            )
        batch["embedding"] = embeddings.cpu().numpy().tolist()
        return batch
