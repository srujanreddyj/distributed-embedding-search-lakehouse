from pathlib import Path
from src.loader_benchmark import ShardLoaderBenchmark

DATA_DIR = Path("data")
SHARDS_DIR = DATA_DIR / "shards"


def main(version_name: str = "multimodal-demo-v001") -> None:
    shard_dir = SHARDS_DIR / version_name
    if not shard_dir.exists():
        print(f"No shards found at {shard_dir}")
        return

    benchmark = ShardLoaderBenchmark()
    metrics = benchmark.benchmark(shard_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import json

    main()
