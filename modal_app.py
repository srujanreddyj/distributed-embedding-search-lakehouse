# modal_app.py
"""Modal search endpoints and browser UI for the multimodal lakehouse demo.

Search responsibilities only:
- /search_text, /search_images, /search_all (legacy compatibility)
- /search_all_modalities (unified 4-modality search via catalog)
- Browser UI
- /asset/{item_id} and /image/{image_id} serving

Pipeline orchestration lives in modal_pipeline.py.
"""

from pathlib import Path
import modal

APP_NAME = "multimodal-lakehouse-search"
VOLUME_NAME = "multimodal-lakehouse-volume"
DATA_DIR = "/data"
STATIC_DIR = Path(__file__).parent / "static"
HTML_PATH = STATIC_DIR / "index.html"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0.0",
        "lancedb>=0.17.0",
        "pyarrow>=15.0.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "fastapi[standard]",
        "Pillow>=10.0.0",
        "torch",
    )
    .add_local_dir("static", remote_path="/root/static")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME, image=image)

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
CATALOG_TABLE = "item_catalog"


def _search_all_modalities_impl(item: dict) -> dict:
    """Search the unified catalog across text, image, video, and audio."""
    from pathlib import Path

    import lancedb
    from sentence_transformers import SentenceTransformer

    volume.reload()

    query = str(item.get("query", "")).strip()
    k = int(item.get("k", 5))

    if not query:
        return {"error": "Please provide a non-empty query."}

    results = {"query": query, "k": k}
    catalog_path = Path(DATA_DIR) / "catalog"

    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    text_vector = (
        text_model.encode([query], normalize_embeddings=True)[0]
        .astype("float32")
        .tolist()
    )

    image_model = SentenceTransformer(IMAGE_MODEL_NAME)
    clip_vector = (
        image_model.encode([query], normalize_embeddings=True)[0]
        .astype("float32")
        .tolist()
    )

    try:
        db = lancedb.connect(str(catalog_path))
        table = db.open_table(CATALOG_TABLE)
    except Exception as e:
        return {
            "query": query,
            "k": k,
            "error": f"Unified catalog is not available: {e}",
            "hint": "Run modal_pipeline.py::stage_04_catalog after preprocessing.",
        }

    modality_config = {
        "text": {
            "vector_col": "text_vector",
            "query_vector": text_vector,
        },
        "image": {
            "vector_col": "clip_vector",
            "query_vector": clip_vector,
        },
        "video": {
            "vector_col": "clip_vector",
            "query_vector": clip_vector,
        },
        "audio": {
            "vector_col": "text_vector",
            "query_vector": text_vector,
        },
    }

    fields = [
        "id",
        "caption",
        "content_path",
        "source",
        "metadata_json",
        "quality_status",
        "license",
        "content_hash",
    ]

    for modality, config in modality_config.items():
        try:
            search_results = (
                table.search(
                    config["query_vector"],
                    vector_column_name=config["vector_col"],
                )
                .where(f"modality = '{modality}' AND has_embedding = true")
                .limit(k)
                .to_pandas()
            )

            matches = []
            for _, row in search_results.iterrows():
                match = {field: row.get(field, "") for field in fields}
                match["modality"] = modality
                match["distance"] = float(row.get("_distance", 0.0))
                if isinstance(match.get("caption"), str):
                    match["caption"] = match["caption"][:700]
                if modality == "text":
                    match["text"] = match.get("caption", "")
                if modality == "image" and match.get("id"):
                    match["image_url"] = f"/asset/{match['id']}"
                if modality in {"video", "audio"} and match.get("id"):
                    match["asset_url"] = f"/asset/{match['id']}"
                matches.append(match)

            results[f"{modality}_matches"] = matches

        except Exception as e:
            results[f"{modality}_matches"] = []
            results[f"{modality}_error"] = str(e)

    results["note"] = (
        "Results are ranked separately per modality because they use "
        "different embedding models and distance scales."
    )

    return results


@app.function(
    cpu=2, memory=12_288, timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_all_modalities(item: dict) -> dict:
    """Unified search across all 4 modalities via the metadata catalog.

    Queries the unified `item_catalog` for each modality using the appropriate
    vector column: `text_vector` for text/audio and `clip_vector` for image/video.
    Results are returned as separate ranked lists — distances across different
    embedding spaces are not comparable.
    """
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    return _search_all_modalities_impl(item)


@app.function(
    cpu=2, memory=12_288, timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_all(item: dict) -> dict:
    """Legacy-compatible all-search endpoint backed by the unified catalog."""
    return _search_all_modalities_impl(item)


# --- Keep existing endpoints for backward compatibility ---

@app.function(
    cpu=2, memory=8192, timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_text(item: dict) -> dict:
    """Search text documents (legacy endpoint)."""
    from sentence_transformers import SentenceTransformer
    import lancedb

    volume.reload()
    query = str(item.get("query", "")).strip()
    k = int(item.get("k", 5))
    if not query:
        return {"error": "Please provide a non-empty query."}

    db = lancedb.connect(str(Path(DATA_DIR) / "lancedb_text"))
    table = db.open_table("text_documents")
    model = SentenceTransformer(TEXT_MODEL_NAME)
    qv = model.encode([query], normalize_embeddings=True)[0].astype("float32").tolist()
    results = table.search(qv, vector_column_name="text_vector").limit(k).to_pandas()

    return {
        "query": query, "k": k,
        "matches": [
            {"id": r.get("id",""), "text": r.get("text","")[:700],
             "url": r.get("url",""), "source": r.get("source",""),
             "distance": float(r.get("_distance",0))}
            for _, r in results.iterrows()
        ],
    }


@app.function(
    cpu=2, memory=8192, timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_images(item: dict) -> dict:
    """Search image documents (legacy endpoint)."""
    from sentence_transformers import SentenceTransformer
    import lancedb

    volume.reload()
    query = str(item.get("query", "")).strip()
    k = int(item.get("k", 5))
    if not query:
        return {"error": "Please provide a non-empty query."}

    db = lancedb.connect(str(Path(DATA_DIR) / "lancedb"))
    table = db.open_table("image_documents")
    model = SentenceTransformer(IMAGE_MODEL_NAME)
    qv = model.encode([query], normalize_embeddings=True)[0].astype("float32").tolist()
    results = table.search(qv, vector_column_name="image_vector").limit(k).to_pandas()

    return {
        "query": query, "k": k,
        "matches": [
            {"image_id": r.get("image_id",""), "caption": r.get("caption",""),
             "filename": r.get("filename",""), "image_url": f"/image/{r.get('image_id','')}",
             "distance": float(r.get("_distance",0))}
            for _, r in results.iterrows()
        ],
    }


@app.function(
    cpu=2, memory=12_288, timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.asgi_app()
def web_ui():
    """Browser UI + API routes."""
    import mimetypes
    import re
    import lancedb
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    fastapi_app = FastAPI(title="Multimodal Lakehouse Search")

    @fastapi_app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PATH.read_text()

    @fastapi_app.post("/api/search_all")
    def api_search_all(item: dict):
        """Proxy to the unified 4-modality search."""
        result = _search_all_modalities_impl(item)
        return JSONResponse(result)

    @fastapi_app.get("/asset/{item_id}")
    def serve_asset(item_id: str):
        """Serve a catalog-backed image/video/audio asset by item id."""
        volume.reload()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item_id):
            return JSONResponse({"error": "Invalid item_id"}, status_code=400)

        try:
            db = lancedb.connect(str(Path(DATA_DIR) / "catalog"))
            table = db.open_table(CATALOG_TABLE)
            rows = table.search().where(f"id = '{item_id}'").limit(1).to_pandas()
        except Exception as e:
            return JSONResponse({"error": f"Catalog lookup failed: {e}"}, status_code=404)

        if len(rows) == 0:
            return JSONResponse({"error": "Asset not found"}, status_code=404)

        content_path = str(rows.iloc[0].get("content_path", ""))
        path = Path(content_path)
        data_root = Path(DATA_DIR).resolve()

        try:
            resolved = path.resolve()
        except Exception:
            return JSONResponse({"error": "Invalid asset path"}, status_code=404)

        if not str(resolved).startswith(str(data_root)) or not resolved.exists():
            return JSONResponse({"error": "Asset file not found"}, status_code=404)

        media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        return FileResponse(str(resolved), media_type=media_type)

    @fastapi_app.get("/image/{image_id}")
    def serve_image(image_id: str):
        volume.reload()
        if not re.fullmatch(r"coco_[0-9]{12}", image_id):
            return JSONResponse({"error": "Invalid image_id"}, status_code=400)
        legacy_path = Path(DATA_DIR) / "coco_images" / f"{image_id}.jpg"
        if legacy_path.exists():
            return FileResponse(str(legacy_path), media_type="image/jpeg")
        return serve_asset(image_id)

    return fastapi_app
