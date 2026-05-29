# Executable Plan: Runnable Multimodal Vertical Slice

## Goal

Extend the existing serverless multimodal embedding lakehouse into a runnable vertical slice that demonstrates the production architecture without building a large repo rewrite.

The slice should prove this flow across text, image, video, and audio:

```text
public multimodal samples
  -> local/Modal ingestion manifests
  -> Ray Data map_batches pipelines
  -> stateful embedding actors
  -> LanceDB vector tables on local disk or Modal Volume
  -> manifest-based dataset version
  -> training-ready shards
  -> loader benchmark
  -> Modal search endpoints + small UI
```

The implementation should reuse the current text and image spine:

- FineWeb-Edu text pipeline.
- COCO image-caption pipeline.
- Ray Data `map_batches`.
- Stateful embedding actor classes.
- Modal L4 GPU batch jobs.
- LanceDB tables on Modal Volume.
- `/search_text`, `/search_images`, `/search_all`, and the browser UI.

## Target Demo Scale

Keep the demo small enough to run locally or with one Modal L4 GPU.

| Modality | Dataset | Demo size | Primary model | Output table |
| --- | --- | ---: | --- | --- |
| Text | FineWeb-Edu sample | 2K-10K docs | `sentence-transformers/all-MiniLM-L6-v2` | `text_documents` |
| Image | COCO captions | 500-2K images | `sentence-transformers/clip-ViT-B-32` | `image_documents` |
| Video | short public clips sampled from a small HF video dataset or local fixtures | 100-500 clips | CLIP frame embeddings + optional caption text embeddings | `video_documents` |
| Audio | small public speech/audio-caption dataset or local fixtures | 100-500 clips | Whisper transcript + text embedding, optional CLAP later | `audio_documents` |

Default demo target:

```text
2K text docs
500 images
100 videos
100 audio clips
```

Portfolio run target:

```text
10K text docs
2K images
500 videos
500 audio clips
```

## Non-Goals

- Do not build a full production data lake.
- Do not copy large datasets into the repo.
- Do not train a foundation model.
- Do not claim cross-modality scores are globally calibrated.
- Do not rewrite the current working text/image code unless needed to create shared abstractions.

## Final Repo Shape

Add only the pieces needed to make the vertical slice complete:

```text
distributed-embedding-search-lakehouse/
├── docker-compose.yml
├── Dockerfile
├── EXECUTABLE_PLAN.md
├── modal_app.py
├── requirements.txt
├── data/
├── scripts/
│   ├── 01_make_sample.py
│   ├── 03_local_ray_text_embed.py
│   ├── 05_make_coco_sample.py
│   ├── 07_local_ray_image_embed.py
│   ├── 09_make_video_sample.py
│   ├── 10_local_ray_video_embed.py
│   ├── 11_make_audio_sample.py
│   ├── 12_local_ray_audio_embed.py
│   ├── 13_build_dataset_manifest.py
│   ├── 14_materialize_webdataset_shards.py
│   ├── 15_loader_benchmark.py
│   ├── 16_generate_datasheet.py
│   ├── 17_dedup_and_quality_filter.py
│   ├── 18_create_train_eval_splits.py
│   ├── 19_run_dataset_ablation_eval.py
│   └── 20_compare_dataset_versions.py
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── manifests.py
│   ├── lancedb_io.py
│   ├── embeddings/
│   │   ├── text.py
│   │   ├── image.py
│   │   ├── video.py
│   │   └── audio.py
│   └── shards/
│       ├── webdataset_writer.py
│       └── loader_benchmark.py
└── static/
    └── index.html
```

## Phase 1: Stabilize Current Baseline

### 1.1 Verify existing text pipeline

Run:

```bash
cd distributed-embedding-search-lakehouse
source .venv/bin/activate
python scripts/01_make_sample.py
python scripts/03_local_ray_text_embed.py
python scripts/04_local_ray_text_search.py
```

Acceptance checks:

- `data/fineweb-edu-sample.parquet` exists.
- `data/lancedb_ray/text_documents` exists.
- `data/metrics_local_ray_text.json` contains row count, runtime, model, actor count, and batch size.
- A query like `reinforcement learning` returns non-empty text results.

### 1.2 Verify existing image pipeline

Run:

```bash
python scripts/05_make_coco_sample.py
python scripts/07_local_ray_image_embed.py
python scripts/08_local_ray_image_search.py
```

Acceptance checks:

- `data/coco_dog_sample.parquet` or renamed generic COCO sample exists.
- `data/lancedb_ray_images/image_documents` exists.
- `data/metrics_local_ray_image.json` exists.
- A query like `a dog playing outside` returns image-caption rows.

### 1.3 Normalize table roots

Current local roots are separate:

```text
data/lancedb_ray
data/lancedb_ray_images
```

Target local root:

```text
data/lancedb/
  text_documents
  image_documents
  video_documents
  audio_documents
```

Modal root should remain:

```text
/data/lancedb_text
/data/lancedb
```

Then converge later to:

```text
/data/lancedb/
  text_documents
  image_documents
  video_documents
  audio_documents
```

Acceptance check:

- Local and Modal table names match exactly.
- Search code opens tables by name, not by hard-coded modality-specific database roots.

## Phase 2: Add Docker Compose Local Runtime

### 2.1 Add `Dockerfile`

Purpose:

- Provide reproducible local CPU runtime.
- Avoid requiring the evaluator to hand-build the Python environment.
- Keep Modal as the GPU/serverless path.

Contents:

- Python 3.11 slim image.
- Install ffmpeg system package for video/audio decoding.
- Install `requirements.txt`.
- Set working directory to `/app`.
- Default command should run a shell, not automatically execute the pipeline.

Acceptance check:

```bash
docker build -t multimodal-lakehouse .
```

### 2.2 Add `docker-compose.yml`

Services:

- `ray-head`: single-node Ray runtime for local demos.
- `pipeline`: runs scripts against mounted repo data.
- Optional `ui`: simple static/dev server only if needed for local UI.

Volumes:

```text
./data:/app/data
./static:/app/static
```

Environment:

```text
RAY_ADDRESS=ray://ray-head:10001
HF_HOME=/app/data/.hf-cache
```

Acceptance check:

```bash
docker compose up -d ray-head
docker compose run --rm pipeline python scripts/01_make_sample.py
docker compose run --rm pipeline python scripts/03_local_ray_text_embed.py
```

## Phase 3: Add Shared Schemas And Manifests

### 3.1 Create `src/schemas.py`

Define one normalized record shape per modality.

Text record:

```text
id
modality = text
source
source_id
text
url
license
metadata_json
content_hash
```

Image record:

```text
id
modality = image
source
source_id
asset_path
caption
license
metadata_json
content_hash
```

Video record:

```text
id
modality = video
source
source_id
asset_path
caption
transcript
duration_seconds
fps_sampled
keyframe_paths_json
license
metadata_json
content_hash
```

Audio record:

```text
id
modality = audio
source
source_id
asset_path
transcript
duration_seconds
sample_rate
license
metadata_json
content_hash
```

Acceptance check:

- Every sample script writes a parquet manifest with these core fields.
- Every row has a stable `id`, `source`, `source_id`, `modality`, and `content_hash`.

### 3.2 Create `src/manifests.py`

Functions:

- `hash_file(path) -> str`
- `hash_text(text) -> str`
- `write_manifest(df, path)`
- `load_manifest(path)`
- `build_dataset_version(manifest_paths, transform_config, output_path)`

Dataset version JSON:

```json
{
  "dataset_version": "multimodal-demo-v001",
  "created_at": "ISO_TIMESTAMP",
  "inputs": [
    "data/fineweb-edu-sample.parquet",
    "data/coco_sample.parquet",
    "data/video_sample.parquet",
    "data/audio_sample.parquet"
  ],
  "tables": {
    "text": "text_documents",
    "image": "image_documents",
    "video": "video_documents",
    "audio": "audio_documents"
  },
  "models": {
    "text": "sentence-transformers/all-MiniLM-L6-v2",
    "image": "sentence-transformers/clip-ViT-B-32",
    "video": "sentence-transformers/clip-ViT-B-32",
    "audio_transcript": "openai/whisper-tiny",
    "audio_text": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "transforms": {
    "text_max_chars": 2000,
    "image_resize": null,
    "video_keyframes": 4,
    "audio_transcribe": true
  }
}
```

Acceptance check:

```bash
python scripts/13_build_dataset_manifest.py
```

Writes:

```text
data/dataset_versions/multimodal-demo-v001.json
```

## Phase 4: Add Video Pipeline

### 4.1 Add video sample creation

Create:

```text
scripts/09_make_video_sample.py
```

Preferred data sources, in order:

1. Small Hugging Face dataset with short videos and captions if streaming works reliably.
2. A tiny checked-in metadata fixture that downloads a few public-domain/sample videos.
3. Local fixture mode: read videos from `data/raw_videos/`.

Implementation shape:

- Download or copy videos into `data/video_clips/`.
- Keep clips short, ideally under 15 seconds.
- Sample 4 keyframes per clip with ffmpeg or OpenCV.
- Save keyframes under `data/video_keyframes/{video_id}/frame_000.jpg`.
- Write `data/video_sample.parquet`.

Acceptance check:

```bash
python scripts/09_make_video_sample.py --limit 100
```

Writes:

```text
data/video_sample.parquet
data/video_clips/
data/video_keyframes/
```

### 4.2 Add Ray video embedding

Create:

```text
scripts/10_local_ray_video_embed.py
src/embeddings/video.py
```

Stateful actor:

```text
VideoEmbedderActor
  __init__: load CLIP model once
  __call__: load sampled keyframes, embed frames, mean-pool frame vectors, embed caption/transcript if present
```

Table:

```text
video_documents
```

Columns:

```text
video_id
asset_path
caption
transcript
duration_seconds
keyframe_paths_json
frame_vector
caption_vector
source
split
content_hash
```

Acceptance check:

```bash
python scripts/10_local_ray_video_embed.py --actor-count 1 --batch-size 8
```

Writes:

```text
data/lancedb/video_documents
data/metrics_local_ray_video.json
```

Search check:

- Text query is embedded with CLIP text encoder.
- Search uses `frame_vector`.
- Return video metadata plus representative keyframe paths.

## Phase 5: Add Audio Pipeline

### 5.1 Add audio sample creation

Create:

```text
scripts/11_make_audio_sample.py
```

Preferred data sources, in order:

1. Small public speech dataset from Hugging Face with transcripts.
2. Small public audio-caption dataset if easy to stream.
3. Local fixture mode: read files from `data/raw_audio/`.

Implementation shape:

- Save audio clips under `data/audio_clips/`.
- Normalize to short clips if needed.
- Prefer existing transcripts to avoid unnecessary local Whisper cost.
- Write `data/audio_sample.parquet`.

Acceptance check:

```bash
python scripts/11_make_audio_sample.py --limit 100
```

Writes:

```text
data/audio_sample.parquet
data/audio_clips/
```

### 5.2 Add Ray audio embedding

Create:

```text
scripts/12_local_ray_audio_embed.py
src/embeddings/audio.py
```

Default path:

- Use transcript text embedding with `sentence-transformers/all-MiniLM-L6-v2`.
- If transcript is missing, run Whisper tiny to create one.
- Store both transcript and vector.

Optional later path:

- Add CLAP audio embeddings for audio-to-text or text-to-audio retrieval.

Stateful actor:

```text
AudioEmbedderActor
  __init__: load text embedding model once, optionally load Whisper once
  __call__: transcribe missing transcript, embed transcript, return audio_text_vector
```

Table:

```text
audio_documents
```

Columns:

```text
audio_id
asset_path
transcript
duration_seconds
sample_rate
audio_text_vector
source
split
content_hash
```

Acceptance check:

```bash
python scripts/12_local_ray_audio_embed.py --actor-count 1 --batch-size 16
```

Writes:

```text
data/lancedb/audio_documents
data/metrics_local_ray_audio.json
```

Search check:

- Text query embeds with MiniLM.
- Search uses `audio_text_vector`.
- Return transcript snippets and audio metadata.

## Phase 6: Extend Modal GPU Jobs

### 6.1 Add Modal builders

Extend `modal_app.py` with:

```text
build_video_table(limit: int = 100, batch_size: int = 8)
build_audio_table(limit: int = 100, batch_size: int = 16)
build_all_tables(text_limit, image_limit, video_limit, audio_limit)
```

Rules:

- Keep each modality independently runnable.
- Use one L4 GPU for image/video/audio where useful.
- Build LanceDB tables under `/tmp` first.
- Copy completed tables into Modal Volume.
- Call `volume.commit()`.

Acceptance check:

```bash
modal run modal_app.py::health_check
modal run modal_app.py::build_text_table --limit 2000
modal run modal_app.py::build_image_table --limit 500
modal run modal_app.py::build_video_table --limit 100
modal run modal_app.py::build_audio_table --limit 100
```

### 6.2 Extend Modal search endpoints

Add:

```text
/search_videos
/search_audio
```

Update:

```text
/search_all
```

Response shape:

```json
{
  "query": "...",
  "k": 5,
  "text_matches": [],
  "image_matches": [],
  "video_matches": [],
  "audio_matches": []
}
```

Important:

- Return separate ranked lists.
- Do not merge scores across MiniLM and CLIP spaces.
- Include model name and vector column in debug metadata.

Acceptance check:

```bash
curl -X POST "$SEARCH_ALL_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "children playing outside", "k": 3}'
```

Returns non-empty lists for at least text, image, and any built video/audio tables.

## Phase 7: Add Training-Ready Materialization

### 7.1 Choose shard format

Use WebDataset for the vertical slice.

Reason:

- Simple tar-based format.
- Easy to inspect.
- Works for image, text metadata, video keyframes, and audio files.
- Better for a portfolio slice than introducing Mosaic MDS immediately.

Mention in docs:

- Production alternative: MosaicML StreamingDataset/MDS for deterministic elastic resume.

### 7.2 Add materializer

Create:

```text
scripts/14_materialize_webdataset_shards.py
src/shards/webdataset_writer.py
```

Inputs:

```text
data/dataset_versions/multimodal-demo-v001.json
data/*.parquet
```

Outputs:

```text
data/shards/multimodal-demo-v001/
  text-000000.tar
  image-000000.tar
  video-000000.tar
  audio-000000.tar
  shard_manifest.json
```

Shard contents:

Text:

```text
{id}.txt
{id}.json
```

Image:

```text
{id}.jpg
{id}.json
```

Video:

```text
{id}.mp4
{id}.json
{id}.frames.json
```

Audio:

```text
{id}.wav or {id}.mp3
{id}.json
```

Acceptance check:

```bash
python scripts/14_materialize_webdataset_shards.py \
  --version data/dataset_versions/multimodal-demo-v001.json \
  --output data/shards/multimodal-demo-v001
```

Writes tar shards and `shard_manifest.json`.

## Phase 8: Add Loader Benchmark

### 8.1 Add benchmark script

Create:

```text
scripts/15_loader_benchmark.py
src/shards/loader_benchmark.py
```

Benchmark modes:

```text
small-files
webdataset-shards
```

Metrics:

```text
records_read
seconds
records_per_second
bytes_read
mb_per_second
num_workers
batch_size
modality
format
```

Default benchmark:

```bash
python scripts/15_loader_benchmark.py \
  --shards data/shards/multimodal-demo-v001 \
  --batch-size 32 \
  --num-workers 2
```

Acceptance checks:

- Writes `data/metrics_loader_benchmark.json`.
- Shows WebDataset shard reads are at least functional for all modalities.
- Prints a clear comparison table.

Portfolio claim:

- "The loader benchmark demonstrates why training systems materialize sequential shards instead of reading millions of small files directly."

## Phase 9: Add Datasheet Generation

### 9.1 Add datasheet script

Create:

```text
scripts/16_generate_datasheet.py
```

Input:

```text
data/dataset_versions/multimodal-demo-v001.json
data/metrics_*.json
```

Output:

```text
data/datasheets/multimodal-demo-v001.md
```

Sections:

- Dataset version hash/name.
- Modality counts.
- Source datasets.
- Models used.
- Transform config.
- Vector table names.
- Loader benchmark.
- Known limitations.
- Reproducibility commands.

Acceptance check:

```bash
python scripts/16_generate_datasheet.py \
  --version data/dataset_versions/multimodal-demo-v001.json
```

Writes a readable markdown datasheet.

## Phase 10: Add Quality, Splits, And Evaluation Feedback

This phase makes the project read like training-data infrastructure, not only multimodal search. It connects dataset properties to downstream behavior through versioned manifests, quality-filtered variants, train/eval splits, and small ablation reports.

### 10.1 Add dedup and quality filtering

Create:

```text
scripts/17_dedup_and_quality_filter.py
```

Inputs:

```text
data/fineweb-edu-sample.parquet
data/coco_sample.parquet
data/video_sample.parquet
data/audio_sample.parquet
data/dataset_versions/multimodal-demo-v001.json
```

Filtering rules:

- Text: drop exact duplicate `content_hash` values, very short documents, extreme whitespace, and low alphabetic ratio rows.
- Image: drop exact duplicate image hashes, missing or unreadable assets, and empty captions.
- Video: drop duplicate clip hashes, missing files, clips with no sampled keyframes, and clips outside the demo duration range.
- Audio: drop duplicate audio hashes, missing files, clips with no transcript after transcription, and clips outside the demo duration range.

Important:

- Do not delete raw rows.
- Represent filtering as metadata plus a new dataset version.
- Preserve a `quality_reason` or `drop_reason` for excluded rows.

Outputs:

```text
data/filtered/text_documents_filtered.parquet
data/filtered/image_documents_filtered.parquet
data/filtered/video_documents_filtered.parquet
data/filtered/audio_documents_filtered.parquet
data/reports/quality_filter_report.json
data/dataset_versions/multimodal-demo-v002_quality_filtered.json
```

Report fields:

```text
modality
rows_in
rows_kept
rows_dropped
drop_reasons
duplicate_count
missing_asset_count
empty_text_or_caption_count
duration_filtered_count
```

Acceptance check:

```bash
python scripts/17_dedup_and_quality_filter.py \
  --input-version data/dataset_versions/multimodal-demo-v001.json \
  --output-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json
```

The command writes filtered parquet files, a quality report, and a new dataset version manifest.

### 10.2 Add train/eval split generation

Create:

```text
scripts/18_create_train_eval_splits.py
```

Inputs:

```text
data/dataset_versions/multimodal-demo-v002_quality_filtered.json
```

Split rules:

- Deterministic hash-based split by stable `id`.
- Default: 90% train, 10% eval.
- Keep split assignment stable across reruns.
- Split each modality independently so small modalities do not disappear from eval.
- Write split metadata back into parquet and into the dataset version manifest.

Outputs:

```text
data/splits/train_text.parquet
data/splits/eval_text.parquet
data/splits/train_image.parquet
data/splits/eval_image.parquet
data/splits/train_video.parquet
data/splits/eval_video.parquet
data/splits/train_audio.parquet
data/splits/eval_audio.parquet
data/reports/split_report.json
data/dataset_versions/multimodal-demo-v003_splits.json
```

Acceptance check:

```bash
python scripts/18_create_train_eval_splits.py \
  --input-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json \
  --output-version data/dataset_versions/multimodal-demo-v003_splits.json \
  --eval-ratio 0.10
```

The command writes stable train/eval split files and a report with per-modality counts.

### 10.3 Add a small ablation eval

Create:

```text
scripts/19_run_dataset_ablation_eval.py
```

Purpose:

- Compare retrieval quality across dataset versions.
- Demonstrate the feedback loop between data changes and model/system behavior.
- Keep this lightweight; the eval is not a benchmark claim.

Inputs:

```text
data/dataset_versions/multimodal-demo-v001.json
data/dataset_versions/multimodal-demo-v002_quality_filtered.json
data/dataset_versions/multimodal-demo-v003_splits.json
```

Eval set:

- A small checked-in JSON file of query probes, grouped by modality.
- Each probe contains a query, target modality, and expected keywords or source ids when available.

Example:

```json
{
  "query": "a dog running outside",
  "target_modality": "image",
  "expected_terms": ["dog", "outside", "grass"]
}
```

Metrics:

```text
version
modality
query_count
hit_rate_at_5
mean_keyword_overlap_at_5
mean_distance_at_5
empty_result_count
```

Outputs:

```text
data/eval/query_probes.json
data/reports/eval_multimodal-demo-v001.json
data/reports/eval_multimodal-demo-v002_quality_filtered.json
data/reports/eval_multimodal-demo-v003_splits.json
```

Acceptance check:

```bash
python scripts/19_run_dataset_ablation_eval.py \
  --versions \
    data/dataset_versions/multimodal-demo-v001.json \
    data/dataset_versions/multimodal-demo-v002_quality_filtered.json \
    data/dataset_versions/multimodal-demo-v003_splits.json \
  --queries data/eval/query_probes.json
```

If a modality table is missing, the report should mark that modality as unavailable instead of failing the whole run.

### 10.4 Add dataset version comparison

Create:

```text
scripts/20_compare_dataset_versions.py
```

Inputs:

```text
data/dataset_versions/multimodal-demo-v001.json
data/dataset_versions/multimodal-demo-v002_quality_filtered.json
data/dataset_versions/multimodal-demo-v003_splits.json
data/reports/quality_filter_report.json
data/reports/split_report.json
data/reports/eval_*.json
data/metrics_*.json
```

Outputs:

```text
data/reports/dataset_comparison.md
```

Report sections:

- Version lineage.
- Per-modality row counts.
- Quality filter drop rates.
- Train/eval split counts.
- Loader benchmark metrics.
- Retrieval eval deltas.
- Known limitations.

Acceptance check:

```bash
python scripts/20_compare_dataset_versions.py \
  --versions \
    data/dataset_versions/multimodal-demo-v001.json \
    data/dataset_versions/multimodal-demo-v002_quality_filtered.json \
    data/dataset_versions/multimodal-demo-v003_splits.json \
  --output data/reports/dataset_comparison.md
```

The markdown report should make one thing obvious: a dataset change creates a new manifest, changes the composition metrics, and can be compared against retrieval/eval behavior.

## Phase 11: Extend UI

### 11.1 Update `static/index.html`

Add result sections:

```text
Text
Images
Videos
Audio
```

Rules:

- Keep scores separate by modality.
- For videos, show representative keyframe plus caption/transcript.
- For audio, show transcript snippet and source metadata.
- Do not make the UI look like a marketing landing page.

Acceptance check:

```bash
modal serve modal_app.py
```

Open the Modal UI URL and query:

```text
children playing outside
people speaking about science
a dog running on grass
```

The page renders without layout overlap and returns separate modality sections.

## Phase 12: One-Command Demo Paths

### 12.1 Local CPU path

Add `Makefile` or documented commands:

```bash
make local-samples
make local-embed
make local-version
make local-shards
make local-benchmark
make local-datasheet
```

Expanded commands:

```bash
python scripts/01_make_sample.py
python scripts/05_make_coco_sample.py
python scripts/09_make_video_sample.py --limit 100
python scripts/11_make_audio_sample.py --limit 100

python scripts/03_local_ray_text_embed.py
python scripts/07_local_ray_image_embed.py
python scripts/10_local_ray_video_embed.py
python scripts/12_local_ray_audio_embed.py

python scripts/13_build_dataset_manifest.py
python scripts/14_materialize_webdataset_shards.py --version data/dataset_versions/multimodal-demo-v001.json
python scripts/15_loader_benchmark.py --shards data/shards/multimodal-demo-v001
python scripts/16_generate_datasheet.py --version data/dataset_versions/multimodal-demo-v001.json
python scripts/17_dedup_and_quality_filter.py --input-version data/dataset_versions/multimodal-demo-v001.json --output-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json
python scripts/18_create_train_eval_splits.py --input-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json --output-version data/dataset_versions/multimodal-demo-v003_splits.json
python scripts/19_run_dataset_ablation_eval.py --versions data/dataset_versions/multimodal-demo-v001.json data/dataset_versions/multimodal-demo-v002_quality_filtered.json data/dataset_versions/multimodal-demo-v003_splits.json --queries data/eval/query_probes.json
python scripts/20_compare_dataset_versions.py --versions data/dataset_versions/multimodal-demo-v001.json data/dataset_versions/multimodal-demo-v002_quality_filtered.json data/dataset_versions/multimodal-demo-v003_splits.json --output data/reports/dataset_comparison.md
```

### 12.2 Docker path

Commands:

```bash
docker compose build
docker compose up -d ray-head
docker compose run --rm pipeline python scripts/01_make_sample.py
docker compose run --rm pipeline python scripts/05_make_coco_sample.py
docker compose run --rm pipeline python scripts/09_make_video_sample.py --limit 100
docker compose run --rm pipeline python scripts/11_make_audio_sample.py --limit 100
docker compose run --rm pipeline python scripts/13_build_dataset_manifest.py
docker compose run --rm pipeline python scripts/14_materialize_webdataset_shards.py --version data/dataset_versions/multimodal-demo-v001.json
docker compose run --rm pipeline python scripts/15_loader_benchmark.py --shards data/shards/multimodal-demo-v001
docker compose run --rm pipeline python scripts/17_dedup_and_quality_filter.py --input-version data/dataset_versions/multimodal-demo-v001.json --output-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json
docker compose run --rm pipeline python scripts/18_create_train_eval_splits.py --input-version data/dataset_versions/multimodal-demo-v002_quality_filtered.json --output-version data/dataset_versions/multimodal-demo-v003_splits.json
docker compose run --rm pipeline python scripts/20_compare_dataset_versions.py --versions data/dataset_versions/multimodal-demo-v001.json data/dataset_versions/multimodal-demo-v002_quality_filtered.json data/dataset_versions/multimodal-demo-v003_splits.json --output data/reports/dataset_comparison.md
```

### 12.3 Modal path

Commands:

```bash
modal run modal_app.py::health_check
modal run modal_app.py::gpu_smoke_test
modal run modal_app.py::build_text_table --limit 2000
modal run modal_app.py::build_image_table --limit 500
modal run modal_app.py::build_video_table --limit 100
modal run modal_app.py::build_audio_table --limit 100
modal deploy modal_app.py
```

## Phase 13: Portfolio Acceptance Criteria

The vertical slice is complete when all of these are true:

- Text, image, video, and audio sample manifests can be generated.
- Each modality has a local Ray Data `map_batches` embedding pipeline.
- Each embedding pipeline uses a stateful actor that loads models once per actor.
- LanceDB has four tables: `text_documents`, `image_documents`, `video_documents`, `audio_documents`.
- Modal can build at least text and image tables, plus video/audio if runtime and dependency size stay reasonable.
- `/search_all` returns separate ranked lists for all built modalities.
- Dataset versions are manifests, not copied datasets.
- WebDataset shards can be materialized from a dataset version.
- Loader benchmark runs and writes metrics.
- Datasheet generation summarizes the dataset version and pipeline metrics.
- Quality filtering creates a new dataset version without deleting raw records.
- Train/eval split generation is deterministic and recorded in manifests.
- Ablation eval reports compare behavior across dataset versions.
- Dataset comparison report links row counts, filter decisions, split counts, loader metrics, and eval deltas.
- README documents local, Docker, and Modal execution paths.

## Recommended Build Order

1. Normalize local LanceDB table paths.
2. Add shared schema and manifest helpers.
3. Add video sample creation.
4. Add local video Ray embedding.
5. Add audio sample creation.
6. Add local audio Ray embedding.
7. Add dataset version manifest.
8. Add WebDataset materialization.
9. Add loader benchmark.
10. Add datasheet generation.
11. Add quality/dedup filtering.
12. Add deterministic train/eval split generation.
13. Add ablation eval and dataset comparison reports.
14. Add Dockerfile and docker-compose.
15. Extend Modal video/audio builders.
16. Extend search endpoints.
17. Extend UI.
18. Update README with the final demo script.

## Interview Story After Completion

Use this concise story:

> I built a runnable vertical slice of a multimodal training-data lakehouse. It ingests text, image-caption, video, and audio samples; processes them through Ray Data batch pipelines with stateful embedding actors; persists embeddings and metadata in LanceDB; creates immutable dataset-version manifests; applies quality and dedup filters as new dataset versions; creates deterministic train/eval splits; materializes training-ready WebDataset shards; benchmarks loader throughput; compares lightweight eval behavior across dataset versions; and serves multimodal search through Modal endpoints. The demo is intentionally small, but each interface mirrors the production-scale architecture: source manifests, reusable embeddings, vector tables, versioned datasets, quality gates, shard materialization, loader metrics, and evaluation-ready datasheets.
