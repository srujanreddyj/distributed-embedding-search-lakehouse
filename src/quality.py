# src/quality.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class QualityGate:
    """Tier 2: Rule-based quality checks per modality.

    Soft filtering — marks rows pass/fail with reasons,
    never deletes raw data.
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality_status and quality_reason columns."""
        df["quality_status"] = "pass"
        df["quality_reason"] = ""

        for idx, row in df.iterrows():
            modality = row.get("modality", "")
            reasons = []

            if modality == "text":
                reasons = self._check_text(row)
            elif modality == "image":
                reasons = self._check_image(row)
            elif modality == "video":
                reasons = self._check_video(row)
            elif modality == "audio":
                reasons = self._check_audio(row)

            if reasons:
                df.at[idx, "quality_status"] = "fail"
                df.at[idx, "quality_reason"] = "|".join(reasons)

        return df

    def _check_text(self, row: dict) -> list[str]:
        reasons = []
        text = str(row.get("payload", {}).get("caption", ""))
        if len(text.split()) < 10:
            reasons.append("too_short")
        if len(text) > 0 and sum(c.isalpha() for c in text) / len(text) < 0.5:
            reasons.append("low_alpha_ratio")
        return reasons

    def _check_image(self, row: dict) -> list[str]:
        reasons = []
        path = row.get("payload", {}).get("content", "")
        if not Path(path).exists():
            reasons.append("missing_file")
            return reasons
        try:
            img = Image.open(path)
            img.verify()
        except:
            reasons.append("corrupt_image")
        caption = row.get("payload", {}).get("caption", "")
        if not caption.strip():
            reasons.append("empty_caption")
        return reasons

    def _check_video(self, row: dict) -> list[str]:
        reasons = []
        path = row.get("payload", {}).get("content", "")
        if not Path(path).exists():
            reasons.append("missing_file")
        metadata = row.get("payload", {}).get("metadata", {})
        n_frames = metadata.get("n_keyframes", 0)
        if n_frames == 0:
            reasons.append("no_keyframes")
        return reasons

    def _check_audio(self, row: dict) -> list[str]:
        reasons = []
        path = row.get("payload", {}).get("content", "")
        if not Path(path).exists():
            reasons.append("missing_file")
            return reasons
        try:
            import soundfile as sf

            data, sr = sf.read(path)
            if np.abs(data).max() < 0.01:
                reasons.append("silent_audio")
            if len(data) / sr < 0.5:
                reasons.append("too_short")
        except:
            reasons.append("corrupt_audio")
        return reasons

    def report(self, df: pd.DataFrame) -> dict:
        """Generate quality report summary."""
        total = len(df)
        passed = len(df[df["quality_status"] == "pass"])
        failed = len(df[df["quality_status"] == "fail"])

        reasons = {}
        for r in df[df["quality_status"] == "fail"]["quality_reason"]:
            for reason in r.split("|"):
                reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0,
            "drop_reasons": reasons,
        }
