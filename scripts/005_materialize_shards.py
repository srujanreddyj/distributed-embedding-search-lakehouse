from pathlib import Path
import pandas as pd

from src.versioning import DatasetVersion
from src.sharding import ShardWriter
from src.cas import ContentAddressedStore
from src.catalog import MetadataCatalog

DATA_DIR = Path("data")
VERSIONS_DIR = DATA_DIR / "dataset_versions"
CATALOG_PATH = DATA_DIR / "catalog"
SHARDS_DIR = DATA_DIR / "shards"
CAS_ROOT = DATA_DIR / "cas"


def main(version_name: str = "multimodal-demo-v001") -> None:

    ## create an instance of the CAS Store pointing at data/cas this gives access to all stored asset files by content hash.
    cas = ContentAddressedStore(CAS_ROOT)

    ## Creates a ShardWriter that will write tar files into data/shards/multimodal-demo-v001/.
    # Passes the CAS store so it can find asset files by hash.
    writer = ShardWriter(cas=cas, output_dir=SHARDS_DIR / version_name)

    version = DatasetVersion.load(version_name)  # from versioning module
    item_ids = version["item_ids"]

    # Load records from catalog for these item_ids from lancedb
    catalog = MetadataCatalog(CATALOG_PATH)
    all_records = catalog.query(limit=10_000)
    records_df = all_records[all_records["id"].isin(item_ids)]

    ## Filters the catalog records to only those that belong to this dataset version. The version might be a subset of what's in the catalog.
    shards = writer.materialize(version, records_df=records_df)

    print(f"Created {len(shards)} shards for {version_name}")
    for s in shards:
        print(f"  {s['shard']}: {s['item_count']} items")


if __name__ == "__main__":
    main()
