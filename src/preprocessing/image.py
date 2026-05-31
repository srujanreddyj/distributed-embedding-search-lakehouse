# src/preprocessing/image.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.preprocessing.base import BasePreprocessor
from src.preprocessing.clip import image_features
from src.preprocessing.device import default_device, move_to_device
from src.preprocessing.payload import payload_values


class ImagePreprocessor(BasePreprocessor):
    """Load image, preprocess, and embed with CLIP."""

    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = default_device()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.model.eval()

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        paths = payload_values(batch, "content")
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = move_to_device(inputs, self.device)
        with torch.no_grad():
            embeddings = image_features(self.model, inputs)
        batch["embedding"] = embeddings.cpu().numpy().tolist()
        return batch
