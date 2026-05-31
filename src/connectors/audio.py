# src/connectors/audio.py

from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset

from src.cas import ContentAddressedStore
from src.connectors.base import SourceConnector


class LibriSpeechConnector(SourceConnector):

    def __init__(
        self,
        output_dir: Path,
        audio_dir: Path,
        cas: ContentAddressedStore | None = None,
    ) -> None:
        super().__init__("librispeech", output_dir, cas=cas)
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def connect(self, **kwargs) -> Iterator:
        return load_dataset(
            "openslr/librispeech_asr",
            "clean",  # clean subset — highest quality speech
            split="train.100",  # 100-hour subset — avoids downloading 1000 hours
            streaming=True,
        )

    @staticmethod
    def _to_numpy(value: Any):
        """Convert torch/NumPy/list audio payloads to a NumPy array."""
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value.cpu(), "numpy"):
            value = value.cpu().numpy()
        elif hasattr(value, "numpy"):
            value = value.numpy()
        return np.array(value)

    @staticmethod
    def _first_attr(obj: Any, names: list[str], default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _write_audio(self, audio_data: Any, audio_path: Path) -> int | None:
        """Write old dict-style or new torchcodec decoder audio to WAV."""
        import soundfile as sf

        if isinstance(audio_data, dict):
            sampling_rate = int(audio_data.get("sampling_rate") or 16000)
            if audio_data.get("array") is not None:
                array = self._to_numpy(audio_data["array"])
                sf.write(str(audio_path), array, sampling_rate)
                return sampling_rate
            if audio_data.get("bytes"):
                audio_path.write_bytes(audio_data["bytes"])
                return sampling_rate
            if audio_data.get("path") and Path(audio_data["path"]).exists():
                import shutil

                shutil.copy2(audio_data["path"], audio_path)
                return sampling_rate
            return None

        if hasattr(audio_data, "get_all_samples"):
            samples = audio_data.get_all_samples()
            sampling_rate = self._first_attr(
                samples,
                ["sample_rate", "sampling_rate"],
                self._first_attr(audio_data, ["sample_rate", "sampling_rate"], 16000),
            )
            array = self._first_attr(samples, ["data", "samples", "values"])
            if array is None:
                raise ValueError(
                    "AudioDecoder samples did not expose data/samples/values"
                )
            array = self._to_numpy(array)
            if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
                array = array.T
            sf.write(str(audio_path), array, int(sampling_rate))
            return int(sampling_rate)

        raise TypeError(f"Unsupported audio payload type: {type(audio_data).__name__}")

    def transform(self, raw_row: dict, idx: int) -> dict:
        transcript = " ".join(str(raw_row.get("text", "")).split())
        if not transcript:
            return None

        audio_id = f"librispeech_{idx:08d}"
        audio_path = self.audio_dir / f"{audio_id}.wav"

        audio_data = raw_row.get("audio", {})
        try:
            sampling_rate = self._write_audio(audio_data, audio_path)
        except Exception as exc:
            print(f"Skipping {audio_id}: could not write audio payload: {exc}")
            return None

        if sampling_rate is None:
            print(
                f"Skipping {audio_id}: audio payload has no array, bytes, path, "
                "or decoder samples"
            )
            return None

        return {
            "id": audio_id,
            "source": "openslr/librispeech_asr/clean",
            "modality": "audio",
            "content_hash": self.hash_content(audio_id + transcript),
            "payload": {
                "type": "audio",
                "content": str(audio_path),
                "caption": transcript,
                "metadata": {
                    "sampling_rate": sampling_rate,
                    "speaker_id": str(raw_row.get("speaker_id", "")),
                    "chapter_id": str(raw_row.get("chapter_id", "")),
                },
            },
        }
