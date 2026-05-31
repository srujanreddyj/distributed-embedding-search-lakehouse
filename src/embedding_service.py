# Embedding Services
# A reusable layer that turns raw content into vector embeddings. Right now your preprocessors each load their own models (MiniLM, CLIP, Whisper) —
# Component 5 extracts this into a shared service that any pipeline can call.
# Precompute + Cache Architecture

# Concept: Embeddings are stored keyed by content_hash. Before running inference, check if we already have it. This prevents redundant GPU work when the same content appears multiple times.

# Storage: LanceDB is perfect for this — vector search + metadata in one place.

import lancedb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
import torch
from PIL import Image
from typing import Literal


class EmbeddingService:
    """Compute and cache embeddings keyed by content_hash

    Model version stamping ensures cached embeddings from different
    model version are never mixed in the same dedup/search pass.
    """

    MODEL_IDS = {
        "text": "sentence-transformers/all-MiniLM-L6-v2",
        "image": "openai/clip-vit-base-patch32",
        "video": "openai/clip-vit-base-patch32",
        "audio": "openai/whisper-small",
    }

    DIMS = {
        "text": 384,
        "image": 512,
        "video": 512,
        "audio": 384,  # projected from 768
    }

    def __init__(
        self, db_path: Path, modality: Literal["text", "image", "video", "audio"]
    ) -> None:
        self.modality = modality
        self.model_name = self.MODEL_IDS[modality]
        self.model_version = f"{modality}_v001"
        self.dim = self.DIMS[modality]

        self.db = lancedb.connect(str(db_path))

        # Open or create table with schema
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.table = self.db.create_table(
                self.table_name,
                data=[
                    {
                        "content_hash": "placeholder",
                        "embedding": np.zeros(self.dim, dtype=np.float32).tolist(),
                        "modality": modality,
                        "model_version": self.model_version,
                    }
                ],
                mode="overwrite",
            )
            # Remove placeholder
            self.table.delete("content_hash = 'placeholder'")

            # Load model
            if modality == "text":
                self.model = SentenceTransformer(self.model_name)
            elif modality in ("image", "video"):
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.model.eval()
            elif modality == "audio":
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.projector = torch.nn.Linear(768, 384)

    def get(
        self, content_hash: str, model_version: str | None = None
    ) -> np.ndarray | None:
        """Retrieve cached embedding by hash. Optionally filter by model version."""
        mv = model_version or self.model_version
        result = (
            self.table.search()
            .where(f"content_hash = f'{content_hash}' AND model_version = '{mv}'")
            .to_pandas()
        )
        if len(result) > 0:
            return np.array(result.iloc[0]["embedding"], dtype=np.float32)
        return None

    def compute_and_cache(self, content_hash: str, content: str | Path) -> np.ndarray:
        """Get from cache or compute, store, and return."""
        cached = self.get(content_hash)
        if cached is not None:
            return cached

        embedding = self._compute(content)
        self.table.add(
            [
                {
                    "content_hash": content_hash,
                    "embedding": embedding.tolist(),
                    "modality": self.modality,
                    "model_version": self.model_version,
                }
            ]
        )
        return embedding

    def _compute(self, content: str | Path) -> np.ndarray:
        """Dispatch to modality-specific computation."""
        if self.modality == "text":
            vec = self.model.encode(str(content), normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        elif self.modality == "image":
            return self._compute_image(content)
        elif self.modality == "video":
            return self._compute_video(content)
        elif self.modality == "audio":
            return self._compute_audio(content)
        raise ValueError(f"Unknown modality: {self.modality}")

    def _compute_image(self, image_path: str | Path) -> np.ndarray:
        """Clip image embedding"""
        img = Image.open(str(image_path)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        """
        CLIP has two encoders: a text encoder and an image encoder
        .get_image_features() runs the image through the vision transformer (ViT)
        Output is a 512-dimensional vector (for ViT-B/32)
        **inputs unpacks the dict into keyword arguments
        """
        with torch.no_grad():
            vec = self.model.get_image_features(**inputs)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze(0).cpu().numpy().astype(np.float32)

    def _compute_video(self, manifest_path: str | Path) -> np.ndarray:
        """CLIP on keyframes → mean-pool → normalized video embedding."""
        import json

        data = (
            json.loads(Path(manifest_path).read_text())
            if isinstance(manifest_path, (str, Path))
            and Path(manifest_path).suffix == ".json"
            else {}
        )
        keyframe_paths = data.get("keyframe_paths", [])

        if not keyframe_paths:
            return np.zeros(self.dim, dtype=np.float32)

        """
        A video is just a sequence of images (frames)
        We pick 4 keyframes sampled evenly across the video (from ffmpeg in the connector)
        padding=True — if frames are different resolutions after processing, pad to the same size for batching
        """

        images = [Image.open(p).convert("RGB") for p in keyframe_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            frame_embeds = self.model.get_image_features(**inputs)

        """
        Mean pooling across the 4 keyframes → single [512] vector
        Why not concatenate? Concatenation would make vectors different sizes for different videos. Mean pooling preserves 512-dim.
        Why not max pooling? Mean works better for video representation — captures the "average" visual content, not just the most salient moment.
        """
        vec = frame_embeds.mean(dim=0)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().astype(np.float32)

    def _compute_audio(self, audio_path: str | Path) -> np.ndarray:
        """Whisper encoder → mean pool → project to 384-dim."""
        import soundfile as sf

        audio_array, sr = sf.read(str(audio_path))
        """
        Whisper was trained on 16kHz audio. If LibriSpeech provides 44.1kHz audio, it won't fit Whisper's expected input
        librosa.resample() converts to 16kHz — standard audio preprocessing step
        """

        """
        Whisper processor: converts raw audio waveform into a mel spectrogram (an image-like representation of frequency over time)
        This is the audio equivalent of CLIP's image preprocessing
        """
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            features = self.model.encoder(inputs.input_features).last_hidden_state

            # Mean pool over the time dimension → single [768] vector
            # Same logic as video: need a fixed-size representation regardless of clip length

            pooled = features.mean(dim=1)

            # 768-dim is too large for our ANN index (all other modalities are 384 or 512)
            # A simple linear projection 768 → 384 keeps dimensions compatible
            # Why 384? Same as the text embedding model — makes text-to-audio search possible

            vec = self.projector(pooled)

        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze(0).cpu().numpy().astype(np.float32)
