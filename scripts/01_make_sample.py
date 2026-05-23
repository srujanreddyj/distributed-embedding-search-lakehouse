from itertools import islice
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = Path("data/fineweb-edu-sample.parquet")
ENV_PATH = Path(".env")


def load_local_env() -> None:
    if not ENV_PATH.exists():
        return

    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_text(text: str, max_chars: int = 2_000) -> str:
    if not text:
        return ""

    text = " ".join(text.split())
    return text[:max_chars]

def main(limit: int = 2_000) -> None:
    load_local_env()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {limit} documents from FineWeb-Edu...")
    hf_token = os.environ.get("HF_TOKEN")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        token=hf_token,
    )

    rows = []

    for idx, row in enumerate(islice(ds, limit)):
        text = clean_text(row.get("text", ""))

        if len(text.split()) < 30:
            continue

        rows.append({
            "id": str(idx),
            "text": text,
            "url": row.get("url", ""),
            "token_count": int(row.get("token_count") or 0),
            "source": "fineweb-edu/sample-10BT",
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    os._exit(0)

if __name__ == "__main__":
    files_pulled = main()
    print(files_pulled)
