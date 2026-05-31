from pathlib import Path
from src.catalog import MetadataCatalog
from src.versioning import DatasetVersion

DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "catalog"
VERSIONS_DIR = DATA_DIR / "dataset_versions"


def main() -> None:
    catalog = MetadataCatalog(CATALOG_PATH)
    versions = DatasetVersion(catalog, VERSIONS_DIR)

    # Create a multimodal version
    manifest = versions.create(
        version_name="multimodal-demo-v001",
        modalities=["text", "image", "video", "audio"],
        limit_per_modality=200,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import json

    main()
