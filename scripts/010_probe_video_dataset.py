"""Probe a Hugging Face video dataset before wiring it into the pipeline.

The probe answers three practical questions:

1. Do streamed rows expose a usable video payload?
2. Can ffmpeg decode that payload and extract keyframes?
3. Can we write a small manifest-shaped parquet for downstream tests?
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from datasets.exceptions import DatasetNotFoundError

PRESETS = {
    "finevideo": {
        "dataset": "HuggingFaceFV/finevideo",
        "split": "train",
        "caption_fields": [
            "json.content_metadata.description",
            "json.youtube_title",
            "json.text_to_speech",
            "json.content_fine_category",
        ],
        "video_fields": ["mp4"],
    },
    "alexzigma": {
        "dataset": "AlexZigma/msr-vtt",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "chengxiang-archive": {
        "dataset": "Chengxiang1122/MSRVTT",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "chengxiang-mcl": {
        "dataset": "Chengxiang1122/mcl-mmcl-msrvtt",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "friedrichor": {
        "dataset": "friedrichor/MSR-VTT",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "vlm2vec": {
        "dataset": "VLM2Vec/MSR-VTT",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "ntqai": {
        "dataset": "NTQAI/MSR-VTT-Video-Captioning-Vi",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "arushijain": {
        "dataset": "arushijain45/my-msr-vtt-video-dataset",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
    "arushijain-zip": {
        "dataset": "arushijain45/my-msr-vtt-video-dataset-zip",
        "split": "train",
        "caption_fields": ["caption", "text", "sentence"],
        "video_fields": ["video", "video_path", "path", "file"],
    },
}


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def compact(value: Any, max_len: int = 160) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def first_present(row: dict[str, Any], fields: list[str]) -> tuple[str | None, Any]:
    for field in fields:
        value = nested_field(row, field)
        if value not in (None, ""):
            return field, value
    return None, None


def nested_field(row: dict[str, Any], field: str, default: Any = None) -> Any:
    current = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def flatten_keys(row: dict[str, Any]) -> dict[str, str]:
    summary = {}
    for key, value in row.items():
        if isinstance(value, dict):
            summary[key] = f"dict keys={sorted(value.keys())}"
        elif isinstance(value, list):
            summary[key] = f"list len={len(value)}"
        else:
            summary[key] = type(value).__name__
    return summary


def caption_from_row(row: dict[str, Any], fields: list[str]) -> str:
    _, value = first_present(row, fields)
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item), "")
    if value is None:
        return ""
    return " ".join(str(value).split())[:2_000]


def safe_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("json", {})
    return metadata if isinstance(metadata, dict) else {}


def materialize_video(
    value: Any,
    output_path: Path,
) -> tuple[bool, str]:
    if isinstance(value, dict):
        if value.get("bytes"):
            output_path.write_bytes(value["bytes"])
            return True, "dict.bytes"
        source_path = value.get("path") or value.get("filename")
        if source_path and Path(source_path).exists():
            shutil.copy2(source_path, output_path)
            return True, f"dict.path:{source_path}"
        return False, f"dict keys={sorted(value.keys())}"

    if isinstance(value, (bytes, bytearray)):
        output_path.write_bytes(bytes(value))
        return True, "bytes"

    if isinstance(value, str):
        source_path = Path(value)
        if source_path.exists():
            shutil.copy2(source_path, output_path)
            return True, f"path:{value}"
        return False, f"string but not local path:{compact(value)}"

    if hasattr(value, "path") and Path(str(value.path)).exists():
        shutil.copy2(str(value.path), output_path)
        return True, f"object.path:{value.path}"

    return False, type(value).__name__


def extract_keyframes(video_path: Path, frame_dir: Path, n_frames: int) -> list[str]:
    frame_dir.mkdir(parents=True, exist_ok=True)
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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    frames = [str(path) for path in sorted(frame_dir.glob("*.jpg"))]
    if not frames and result.stderr:
        print(f"    ffmpeg stderr: {compact(result.stderr, 300)}")
    return frames


def ffprobe_duration(video_path: Path) -> float | None:
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


def make_clip(source_path: Path, clip_path: Path, start: float, seconds: float) -> bool:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-i",
        str(source_path),
        "-t",
        str(seconds),
        "-c",
        "copy",
        str(clip_path),
        "-y",
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return (
        result.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0
    )


def load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": not args.no_streaming,
    }
    if args.token:
        kwargs["token"] = args.token if isinstance(args.token, str) else True
    if args.config:
        kwargs["name"] = args.config
    if args.data_files:
        kwargs["data_files"] = args.data_files
    return load_dataset(**kwargs)


def print_structure(dataset: str) -> None:
    try:
        configs = get_dataset_config_names(dataset)
    except Exception as exc:
        print(f"Could not read configs for {dataset}: {exc}")
        return
    print(f"Configs for {dataset}: {configs}")
    for config in configs[:10]:
        try:
            splits = get_dataset_split_names(dataset, config)
        except Exception as exc:
            print(f"  {config}: could not read splits: {exc}")
            continue
        print(f"  {config}: splits={splits}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--dataset")
    parser.add_argument("--config")
    parser.add_argument("--split")
    parser.add_argument("--data-files")
    parser.add_argument("--caption-field", action="append", dest="caption_fields")
    parser.add_argument("--video-field", action="append", dest="video_fields")
    parser.add_argument("--accepted-target", type=int, default=2)
    parser.add_argument("--max-scan", type=int, default=100)
    parser.add_argument("--keyframe-stride", type=int, default=24)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--max-clips-per-video", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("data/video_probe"))
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--list-structure", action="store_true")
    parser.add_argument("--fail-under", type=int, default=1)
    parser.add_argument(
        "--token",
        nargs="?",
        const=True,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Use a Hugging Face token for gated datasets. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN if set.",
    )
    args = parser.parse_args()

    preset = PRESETS.get(args.preset or "", {})
    args.dataset = args.dataset or preset.get("dataset")
    args.split = args.split or preset.get("split") or "train"
    args.caption_fields = (
        args.caption_fields
        or preset.get("caption_fields")
        or [
            "caption",
            "text",
            "sentence",
        ]
    )
    args.video_fields = (
        args.video_fields
        or preset.get("video_fields")
        or [
            "video",
            "video_path",
            "path",
            "file",
        ]
    )

    if not args.dataset:
        parser.error("--dataset is required unless --preset is provided")
    return args


def main() -> int:
    args = parse_args()

    if args.list_structure:
        print_structure(args.dataset)

    videos_dir = args.output_dir / "videos"
    clips_dir = args.output_dir / "clips"
    keyframes_dir = args.output_dir / "keyframes"
    manifests_dir = args.output_dir / "manifests"
    for directory in [videos_dir, clips_dir, keyframes_dir, manifests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config or '<default>'}")
    print(f"Split: {args.split}")
    print(f"Streaming: {not args.no_streaming}")
    print(f"Video fields: {args.video_fields}")
    print(f"Caption fields: {args.caption_fields}")
    print(f"Token: {'yes' if args.token else 'no'}")

    accepted = []
    scanned = 0
    try:
        rows = load_rows(args)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "\nFineVideo is gated. Visit the dataset page, accept the terms, "
            "then rerun with HF_TOKEN/HUGGING_FACE_HUB_TOKEN set or pass --token.",
            file=sys.stderr,
        )
        return 3

    for idx, row in enumerate(rows):
        if scanned >= args.max_scan or len(accepted) >= args.accepted_target:
            break
        scanned += 1

        if idx == 0:
            print(f"First row schema: {json.dumps(flatten_keys(row), indent=2)}")

        caption = caption_from_row(row, args.caption_fields)
        video_field, video_value = first_present(row, args.video_fields)
        if not caption:
            print(f"[{idx}] skip: no caption")
            continue
        if video_value is None:
            print(f"[{idx}] skip: no video field among {args.video_fields}")
            continue

        video_id = f"probe_{idx:08d}"
        video_path = videos_dir / f"{video_id}.mp4"
        ok, reason = materialize_video(video_value, video_path)
        if not ok:
            print(f"[{idx}] skip: unusable {video_field} payload ({reason})")
            continue

        duration = ffprobe_duration(video_path)
        if duration is None:
            print(f"[{idx}] skip: ffprobe could not read duration from {reason}")
            continue

        if args.clip_seconds > 0:
            clip_count = max(
                1,
                min(args.max_clips_per_video, int(duration // args.clip_seconds) or 1),
            )
        else:
            clip_count = 1
        source_metadata = safe_metadata(row)
        source_category = source_metadata.get("content_parent_category", "")
        source_fine_category = source_metadata.get("content_fine_category", "")

        accepted_from_video = 0
        for clip_idx in range(clip_count):
            if len(accepted) >= args.accepted_target:
                break

            clip_start = clip_idx * args.clip_seconds
            clip_id = f"{video_id}_clip_{clip_idx:03d}"
            clip_path = clips_dir / f"{clip_id}.mp4"

            if args.clip_seconds > 0:
                made_clip = make_clip(
                    video_path, clip_path, clip_start, args.clip_seconds
                )
                if not made_clip:
                    print(f"[{idx}] skip clip {clip_idx}: ffmpeg could not create clip")
                    continue
            else:
                clip_path = video_path

            keyframes = extract_keyframes(
                clip_path,
                keyframes_dir / clip_id,
                args.keyframe_stride,
            )
            if not keyframes:
                print(f"[{idx}] skip clip {clip_idx}: ffmpeg extracted no keyframes")
                continue

            record = {
                "id": clip_id,
                "source": args.dataset,
                "modality": "video",
                "content_hash": sha256_text(clip_id + caption),
                "payload": {
                    "type": "video",
                    "content": str(clip_path),
                    "caption": caption,
                    "metadata": {
                        "keyframe_paths": json.dumps(keyframes),
                        "n_keyframes": len(keyframes),
                        "source_video_field": video_field,
                        "source_video_shape": reason,
                        "source_duration_seconds": duration,
                        "clip_start_seconds": clip_start,
                        "clip_duration_seconds": args.clip_seconds,
                        "content_parent_category": source_category,
                        "content_fine_category": source_fine_category,
                    },
                },
            }
            accepted.append(record)
            accepted_from_video += 1
            print(
                f"[{idx}] accept clip {clip_idx}: "
                f"{len(keyframes)} keyframes from {reason}"
            )

        if accepted_from_video == 0:
            print(f"[{idx}] skip: decoded video but produced no valid clips")

    manifest_path = manifests_dir / "video_probe_manifest.parquet"
    pd.DataFrame(accepted).to_parquet(manifest_path, index=False)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "scanned": scanned,
                "accepted": len(accepted),
                "manifest": str(manifest_path),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )

    if len(accepted) < args.fail_under:
        print(
            f"FAILED: accepted {len(accepted)} records, below --fail-under {args.fail_under}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
