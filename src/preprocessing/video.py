# src/preprocessing/video.py

import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.preprocessing.base import BasePreprocessor
from src.preprocessing.clip import image_features
from src.preprocessing.device import default_device, move_to_device
from src.preprocessing.payload import payload_values


class VideoPreprocessor(BasePreprocessor):
    """Load keyframes, embed each with CLIP, mean-pool to video embedding."""

    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = default_device()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.model.eval()

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        all_embeddings = []

        for keyframe_json in payload_values(batch, "metadata.keyframe_paths", "[]"):
            paths = json.loads(keyframe_json)
            if not paths:
                all_embeddings.append(None)
                continue

            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = move_to_device(inputs, self.device)
            with torch.no_grad():
                frame_embeds = image_features(self.model, inputs)
            video_embed = frame_embeds.mean(dim=0)  # mean-pool
            all_embeddings.append(video_embed.cpu().numpy())

        batch["embedding"] = [
            e.tolist() if e is not None else [] for e in all_embeddings
        ]
        return batch
