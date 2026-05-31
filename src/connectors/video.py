# src/connectors/video.py

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from typing import Iterator

from datasets import load_dataset

from src.cas import ContentAddressedStore
from src.connectors.base import SourceConnector


def _nested_get(value: dict[str, Any], path: str, default: Any = "") -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part, default)
    return current


def _compact_caption(raw_row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = _nested_get(raw_row, field)
        if value:
            return " ".join(str(value).split())[:2_000]
    return ""


def _ffprobe_duration(video_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


class MSRVTTConnector(SourceConnector):

    def __init__(
        self,
        output_dir: Path,
        video_dir: Path,
        keyframe_dir: Path,
        cas: ContentAddressedStore | None = None,
    ) -> None:
        super().__init__("msrvtt", output_dir, cas=cas)
        self.video_dir = video_dir
        self.keyframe_dir = keyframe_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

    def connect(self, **kwargs) -> Iterator:
        return load_dataset(
            "AlexZigma/msr-vtt",
            split="train",
            streaming=True,
        )

    def extract_keyframes(
        self, video_path: Path, video_id: str, n_frames: int = 4
    ) -> list[str]:
        """Extract n evenly-spaced keyframes using ffmpeg."""
        frame_dir = self.keyframe_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        """ 
        Take an input video file, 
        filter the video by frame numbers. Pick every nth frame 
        n is the frame number. mod(n, 4) gives the remainder when divided by 4 — so it equals 0 at frames 0, 4, 8, 12... not(0) = True, so those frames get selected
        variable frame rate — only write selected frames, skip the rest
        # JPEG quality (1=best, 31=worst), 2 is near-lossless
        and suppress ffmpeg's noisy stdout


        ffmpeg is a system binary — no Python dependency, handles every video codec
        OpenCV would work too but adds a heavy install and is overkill for just frame extraction
        """
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{n_frames}))",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(frame_dir / "frame_%03d.jpg"),
            "-y",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=False)

        return [str(p) for p in sorted(frame_dir.glob("*.jpg"))]

    def transform(self, raw_row: dict, idx: int) -> dict:
        caption = " ".join(str(raw_row.get("caption", "")).split())
        if not caption:
            return None

        video_id = f"msrvtt_{idx:08d}"
        video_path = self.video_dir / f"{video_id}.mp4"

        video_obj = raw_row.get("video", {})
        if isinstance(video_obj, dict):
            video_bytes = video_obj.get("bytes")
            source_path = video_obj.get("path")
        else:
            video_bytes = None
            source_path = str(video_obj) if video_obj else None

        if video_bytes:
            video_path.write_bytes(video_bytes)
        elif source_path and Path(source_path).exists():
            shutil.copy2(source_path, video_path)
        else:
            print(
                f"Skipping {video_id}: video payload has no bytes or local path. "
                f"Available keys: {sorted(video_obj.keys()) if isinstance(video_obj, dict) else type(video_obj).__name__}"
            )
            return None

        keyframe_paths = self.extract_keyframes(video_path, video_id)
        if not keyframe_paths:
            print(f"Skipping {video_id}: ffmpeg extracted no keyframes")
            return None

        return {
            "id": video_id,
            "source": "AlexZigma/msr-vtt",
            "modality": "video",
            "content_hash": self.hash_content(video_id + caption),
            "payload": {
                "type": "video",
                "content": str(video_path),
                "caption": caption,
                "metadata": {
                    "keyframe_paths": json.dumps(keyframe_paths),
                    "n_keyframes": len(keyframe_paths),
                    "category": raw_row.get("category", ""),
                },
            },
        }


class FineVideoConnector(SourceConnector):
    """Stream FineVideo mp4 bytes and split each source video into short clips."""

    def __init__(
        self,
        output_dir: Path,
        video_dir: Path,
        clip_dir: Path,
        keyframe_dir: Path,
        cas: ContentAddressedStore | None = None,
        clips_per_video: int = 2,
        clip_seconds: float = 8.0,
        keyframe_stride: int = 24,
    ) -> None:
        super().__init__("finevideo", output_dir, cas=cas)
        self.video_dir = video_dir
        self.clip_dir = clip_dir
        self.keyframe_dir = keyframe_dir
        self.clips_per_video = clips_per_video
        self.clip_seconds = clip_seconds
        self.keyframe_stride = keyframe_stride

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

    def connect(self, **kwargs) -> Iterator:
        token = (
            kwargs.get("token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        return load_dataset(
            "HuggingFaceFV/finevideo",
            split=kwargs.get("split", "train"),
            streaming=True,
            token=token if token else None,
        )

    def extract_keyframes(self, video_path: Path, clip_id: str) -> list[str]:
        frame_dir = self.keyframe_dir / clip_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{self.keyframe_stride}))",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(frame_dir / "frame_%03d.jpg"),
            "-y",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=False)
        return [str(p) for p in sorted(frame_dir.glob("*.jpg"))]

    def make_clip(self, source_path: Path, clip_path: Path, start: float) -> bool:
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            str(source_path),
            "-t",
            str(self.clip_seconds),
            "-c",
            "copy",
            str(clip_path),
            "-y",
            "-loglevel",
            "error",
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (
            result.returncode == 0
            and clip_path.exists()
            and clip_path.stat().st_size > 0
        )

    def transform(self, raw_row: dict, idx: int) -> list[dict] | None:
        caption = _compact_caption(
            raw_row,
            [
                "json.content_metadata.description",
                "json.youtube_title",
                "json.text_to_speech",
                "json.content_fine_category",
            ],
        )
        if not caption:
            print(f"Skipping finevideo_{idx:08d}: no caption metadata")
            return None

        video_bytes = raw_row.get("mp4")
        if not video_bytes:
            print(f"Skipping finevideo_{idx:08d}: no mp4 bytes")
            return None

        source_id = f"finevideo_{idx:08d}"
        source_path = self.video_dir / f"{source_id}.mp4"
        source_path.write_bytes(video_bytes)

        duration = _ffprobe_duration(source_path)
        if duration is None:
            print(f"Skipping {source_id}: ffprobe could not read duration")
            return None

        metadata = raw_row.get("json", {})
        if not isinstance(metadata, dict):
            metadata = {}

        max_clips = max(
            1, min(self.clips_per_video, int(duration // self.clip_seconds) or 1)
        )
        records = []
        for clip_idx in range(max_clips):
            clip_start = clip_idx * self.clip_seconds
            clip_id = f"{source_id}_clip_{clip_idx:03d}"
            clip_path = self.clip_dir / f"{clip_id}.mp4"

            if not self.make_clip(source_path, clip_path, clip_start):
                print(f"Skipping {clip_id}: ffmpeg could not create clip")
                continue

            keyframe_paths = self.extract_keyframes(clip_path, clip_id)
            if not keyframe_paths:
                print(f"Skipping {clip_id}: ffmpeg extracted no keyframes")
                continue

            records.append(
                {
                    "id": clip_id,
                    "source": "HuggingFaceFV/finevideo",
                    "modality": "video",
                    "content_hash": self.hash_content(clip_id + caption),
                    "payload": {
                        "type": "video",
                        "content": str(clip_path),
                        "caption": caption,
                        "metadata": {
                            "keyframe_paths": json.dumps(keyframe_paths),
                            "n_keyframes": len(keyframe_paths),
                            "source_duration_seconds": duration,
                            "clip_start_seconds": clip_start,
                            "clip_duration_seconds": self.clip_seconds,
                            "content_parent_category": metadata.get(
                                "content_parent_category", ""
                            ),
                            "content_fine_category": metadata.get(
                                "content_fine_category", ""
                            ),
                            "original_video_filename": metadata.get(
                                "original_video_filename", ""
                            ),
                            "original_json_filename": metadata.get(
                                "original_json_filename", ""
                            ),
                            "youtube_title": metadata.get("youtube_title", ""),
                            "youtube_channel": metadata.get("youtube_channel", ""),
                            "youtube_upload_date": metadata.get(
                                "youtube_upload_date", ""
                            ),
                        },
                    },
                }
            )

        return records or None
