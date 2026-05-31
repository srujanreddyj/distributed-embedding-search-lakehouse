import time
import tarfile
import json
from pathlib import Path


class ShardLoaderBenchmark:
    """
    streaming shards
    and measure throughput
    """

    def benchmark(self, shard_dir: Path) -> dict:
        start = time.time()

        total_items = 0
        total_bytes = 0

        for shard_path in sorted(shard_dir.glob("shard-*.tar")):
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if member.name.endswith(".json"):
                        tar.extractfile(member).read()  # simulate loading
                        total_items += 1
                    total_bytes += member.size

        elapsed = time.time() - start

        return {
            "shard_dir": str(shard_dir),
            "items": total_items,
            "seconds": round(elapsed, 2),
            "items_per_second": round(total_items / elapsed, 2),
            "mb_per_second": round(total_bytes / elapsed / 1e6, 2),
        }
