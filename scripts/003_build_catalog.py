from pathlib import Path
from src.catalog import MetadataCatalog

DATA_DIR = Path("data")
MANIFEST_DIR = DATA_DIR / "manifests"
CATALOG_PATH = DATA_DIR / "catalog"


def main() -> None:
    catalog = MetadataCatalog(CATALOG_PATH)

    for manifest in MANIFEST_DIR.glob("*_manifest.parquet"):
        count = catalog.ingest_manifest(manifest)
        print(f"Ingested {count} rows from {manifest.name}")

    # Quick test
    print("\n--- Sample queries ---")
    for mod in ["text", "image"]:
        results = catalog.query(modality=mod, limit=3)
        print(f"\n{mod.upper()} (3 rows):")
        for _, r in results.iterrows():
            print(f"  {r['id']} | {r['caption'][:60]}")


if __name__ == "__main__":
    main()
