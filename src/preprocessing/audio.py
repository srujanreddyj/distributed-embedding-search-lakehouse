# src/preprocessing/audio.py

import torch
from transformers import AutoProcessor, AutoModel
from src.preprocessing.base import BasePreprocessor
from src.preprocessing.device import default_device, move_to_device
from src.preprocessing.payload import payload_values


class AudioPreprocessor(BasePreprocessor):
    """Load audio, extract Whisper-style features, embed with text model."""

    def __init__(self) -> None:
        # Using Whisper encoder for audio features, then project to embedding space
        self.processor = AutoProcessor.from_pretrained("openai/whisper-small")
        self.device = default_device()
        self.model = AutoModel.from_pretrained("openai/whisper-small").to(self.device)
        self.model.eval()
        # Project to same dim as text embeddings (384)
        self.projector = torch.nn.Linear(768, 384).to(self.device)

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        paths = payload_values(batch, "content")

        # Load audio arrays - using soundfile like your connector
        import soundfile as sf

        audio_arrays = [sf.read(p)[0] for p in paths]

        inputs = self.processor(audio_arrays, sampling_rate=16000, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)
        input_features = (
            inputs["input_features"]
            if isinstance(inputs, dict)
            else inputs.input_features
        )

        with torch.no_grad():
            audio_features = self.model.encoder(input_features).last_hidden_state
            # Mean pool over time dimension
            audio_embeds = audio_features.mean(dim=1)
            # Project to 384-dim to match text embeddings
            embeddings = self.projector(audio_embeds)

        batch["embedding"] = embeddings.cpu().numpy().tolist()
        return batch
