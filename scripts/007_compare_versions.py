# A lightweight script that takes two dataset versions and compares them on a small eval set.

from pathlib import Path
from src.versioning import DatasetVersion

DATA_DIR = Path("data")


def main() -> None:
    versions = DatasetVersion.load_versions(DATA_DIR / "dataset_versions")

    for version_name in versions:
        v = DatasetVersion.load(version_name)
        print(
            f"{v['version']}: {v['total_items']} items from {list(v['modalities'].keys())}"
        )


if __name__ == "__main__":
    main()
