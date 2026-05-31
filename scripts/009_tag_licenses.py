# Every item in the catalog gets a license and datasource field — so you can trace any training sample back to its origin.

licenses = {
    "fineweb_edu": "mit",
    "coco_captions": "cc-by-4.0",
    "finevideo": "cc-by-4.0",
    "msrvtt": "research-only",
    "librispeech": "cc-by-4.0",
}


def tag_catalog(catalog_path: Path) -> None:
    df = pd.read_parquet(catalog_path)
    df["license"] = df["source"].map(licenses)
    df.to_parquet(catalog_path, index=False)
