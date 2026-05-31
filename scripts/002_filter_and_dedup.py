"""
Pipeline script that wires QualityGate → preprocessing → EmbeddingDeduplicator

manifests (Component 1)
    → quality.py (Tier 2 — rule checks)
    → 01_run_preprocessing.py (Component 3 — embed)
    → dedup.py (Tier 3 — ANN on embeddings)
    → filtered output
"""

import json
import time
from pathlib import Path

import pandas as pd
import ray
import ray.data

from src.quality import QualityGate
from src.dedup import EmbeddingDeduplicator

DATA_DIR = Path("data")
MANIFEST_DIR = DATA_DIR / "manifests"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
OUTPUT_DIR = DATA_DIR / "filtered"


def filter_modality(modality: str) -> dict:
    """Run quality gates, then ANN dedup on embeddings, for one modality."""

    manifest = MANIFEST_DIR / f"{modality}_manifest.parquet"
    if not manifest.exists():
        return {"modality": modality, "status": "skipped", "reason": "no manifest"}

    # Step 1: Quality gates (Tier 2)
    print(f"  Quality gates...")
    df = pd.read_parquet(manifest)
    gate = QualityGate()
    df = gate.run(df)

    passed = df[df["quality_status"] == "pass"]
    report = gate.report(df)

    # Step 2: ANN dedup (Tier 3) — only on items that passed quality
    print(f"  ANN dedup on {len(passed)} items...")
    output_dir = OUTPUT_DIR / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{modality}_filtered.parquet"

    if len(passed) == 0:
        df.to_parquet(output_path)
        return {**report, "near_duplicates_removed": 0}

    # Dummy embeddings until Component 5 is wired — skips dedup for now
    # Actually run dedup if embedding column exists
    if "embedding" in passed.columns:
        deduper = EmbeddingDeduplicator(dim=len(passed.iloc[0]["embedding"]))
        keep_ids = deduper.deduplicate(
            ids=passed["id"].tolist(),
            embeddings=passed["embedding"].tolist(),
        )
        kept = passed[passed["id"].isin(keep_ids)]
        near_dups = len(passed) - len(kept)
    else:
        kept = passed
        near_dups = 0

    kept.to_parquet(output_path, index=False)

    return {
        **report,
        "rows_after_quality": len(passed),
        "near_duplicates_removed": near_dups,
        "rows_final": len(kept),
    }


def main() -> None:
    modalities = ["fineweb_edu", "coco_captions", "finevideo", "librispeech"]

    for mod in modalities:
        print(f"\n--- {mod.upper()} ---")
        result = filter_modality(mod)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
