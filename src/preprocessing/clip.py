"""Compatibility helpers for CLIP embeddings across Transformers versions."""

from typing import Any

import torch


def image_features(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return 512-d CLIP image features regardless of model output wrapper shape."""
    output = model.get_image_features(**inputs)
    if isinstance(output, torch.Tensor):
        return output

    image_embeds = getattr(output, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds

    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        projection = getattr(model, "visual_projection", None)
        if (
            projection is not None
            and hasattr(projection, "in_features")
            and pooled.shape[-1] == projection.in_features
        ):
            return projection(pooled)
        return pooled

    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden.mean(dim=1)

    raise TypeError(f"Unsupported CLIP image feature output: {type(output).__name__}")
