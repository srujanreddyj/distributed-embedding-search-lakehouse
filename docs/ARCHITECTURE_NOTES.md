# Architecture Notes

This is the living explanation document for the project. It records why each step exists, how the architecture was chosen, what trade-offs were made, and how to explain the system in interviews.

## Project Title

Serverless Multimodal Lakehouse Search with Ray, Modal, and LanceDB

## Project Goal

Build a demo-scale multimodal lakehouse search system that shows the architecture behind large-scale AI data pipelines:

```text
FineWeb-Edu text sample
  -> text embedding model
  -> Ray Data batch inference
  -> LanceDB text_documents table

COCO image-caption sample
  -> CLIP/OpenCLIP-style embedding model
  -> Ray Data batch inference
  -> LanceDB image_documents table

Modal endpoint
  -> /search_text
  -> /search_images
  -> /search_all
```

The goal is not to process the full FineWeb-Edu or COCO image-caption datasets. The goal is to demonstrate the engineering pattern behind production-scale multimodal systems while keeping cost, complexity, and runtime controlled.

## Core Interview Story

I initially considered a text-only embedding pipeline, but realized that at demo scale, Ray and Modal would be overkill for text alone. I expanded the project into a multimodal lakehouse demo with both text and image-caption data. This made the architecture more realistic: Ray handles batch preprocessing and inference, Modal provides ephemeral GPU execution, and LanceDB stores raw records, metadata, and embeddings together for semantic retrieval.

This project is inspired by Netflix-style multimodal lakehouse architecture, but it is not a production replica. I implemented the same architectural pattern at demo scale: raw assets, metadata, and embeddings stored together, with distributed embedding generation and searchable retrieval.

## Why This Architecture?

### Why Multimodal?

A text-only demo can be useful, but at small scale it does not fully justify distributed batch inference or GPU execution. A laptop can process a few thousand short text records with a small embedding model.

Adding images makes the workload more realistic:

- Image decoding is heavier than text parsing.
- CLIP-style preprocessing has more moving parts than plain text tokenization.
- Image embedding benefits more obviously from GPU execution.
- Multimodal retrieval better matches AI-native lakehouse patterns.

The trade-off is more complexity. The project now needs two data pipelines, two embedding paths, and separate search endpoints. That complexity is acceptable because it makes the architecture more credible and closer to real media-search systems.

### Why Batch Inference?

Embedding generation is naturally an offline batch inference problem. During ingestion, we do not need low-latency per-record serving. We need to process many text records and image-caption records efficiently, usually in batches, and persist the resulting vectors for later search.

This is different from online inference:

- Online inference optimizes for request latency.
- Batch inference optimizes for throughput, cost, and reliable processing of many records.

For this project, batch inference is the right shape because we embed the corpora before search happens.

### Why Ray Data?

Ray Data gives us a distributed data pipeline abstraction:

- Read data from parquet or other sources.
- Partition it into blocks.
- Apply transformations and model inference with `map_batches`.
- Collect or write the output.

The important design decision is using callable classes with `map_batches`.

In Ray Data:

- A plain function behaves like stateless tasks.
- A callable class behaves like stateful actors.

For embedding, stateful actors are useful because the model can be loaded once per actor and reused across many batches. That avoids repeatedly loading the model for every row or every small task.

Ray remains valuable in the multimodal version because the project has two batch inference flows:

- Text embedding pipeline.
- Image/CLIP embedding pipeline.

The image pipeline also gives us a better reason to discuss heterogeneous CPU/GPU work: CPU-side loading/decoding/preprocessing and GPU-side model inference.

### Why Modal?

Modal gives us GPU compute without managing infrastructure. Instead of setting up EC2, Kubernetes, Docker registries, autoscaling, or a persistent Ray cluster, we define the environment and GPU requirements in Python.

For this demo, Modal is useful because:

- GPU jobs can run only when needed.
- The environment is reproducible through `modal.Image`.
- We can attach persistent storage through Modal Volumes.
- The same app can expose a search endpoint.

The trade-off is that this is not a fully managed production platform with custom autoscaling, observability, auth, and workload orchestration already designed by us. For a portfolio demo, that is acceptable and intentional.

### Why LanceDB?

LanceDB stores embeddings and metadata together, which makes it a good fit for vector lakehouse-style workflows. It is especially relevant for this project because LanceDB positions itself around multimodal data and vector search over raw data, metadata, and embeddings.

For this project, LanceDB gives us:

- Local or filesystem-backed vector storage.
- Simple vector search.
- Metadata stored alongside vectors.
- Separate tables for text records and image-caption records.
- A path toward larger Lance-style storage patterns later.

At demo scale, brute-force kNN search is acceptable. If the dataset grows, the next step would be vector indexing and recall/latency tuning.

### Why FineWeb-Edu?

FineWeb-Edu is a realistic web-scale text dataset, which makes the project feel credible. But the full dataset is far larger than needed for a 2-day demo.

We use a streamed sample because:

- It avoids downloading the full dataset.
- It keeps runtime and cost controlled.
- It still demonstrates a realistic ingestion pattern.

The trade-off is that metrics from a few thousand documents should not be presented as production-scale performance. They are demo-scale metrics for validating the architecture.

### Why COCO Image-Caption Data?

COCO-style image-caption data is a strong fit for a text-to-image demo because the images are paired with natural-language captions. That lets us build CLIP-style retrieval without inventing labels or scraping assets.

We use a small sample because:

- The goal is architecture validation, not benchmark-scale training or evaluation.
- Downloading and embedding all images is unnecessary for the first demo.
- Image workloads are more expensive than text workloads, so cost control matters.

The trade-off is that retrieval quality will depend heavily on sample size and caption coverage. That is acceptable because the demo is about the pipeline architecture, not claiming state-of-the-art retrieval.

### How Do FineWeb-Edu And COCO Relate?

FineWeb-Edu and COCO do not need to correlate at the row level. The project is not trying to match a specific text document to a specific image during ingestion.

Instead, the lakehouse contains two independent collections:

```text
text_documents  -> educational web text
image_documents -> image-caption records
```

The correlation happens at query time:

```text
query -> search_text  -> related educational text
query -> search_images -> related images and captions
```

This is closer to a real multimodal lakehouse, where articles, images, videos, captions, metadata, and embeddings may come from different sources but become searchable through semantic retrieval.

Important trade-off:

- Text documents use a text embedding model.
- Images use a CLIP-style multimodal embedding model.
- Their distances are not directly comparable.

Because of that, `/search_all` should return separate ranked lists instead of pretending one global ranking is meaningful:

```json
{
  "query": "...",
  "text_matches": [],
  "image_matches": []
}
```

A production system could calibrate scores, use a single shared multimodal embedding model for all content, or add a reranker.

## Storage Design

Use two LanceDB tables.

### `text_documents`

```text
id
text
url
source
token_count
text_vector
```

This table supports text-to-text semantic search.

### `image_documents`

```text
image_id
image_path_or_url
caption
source
split
image_vector
caption_vector
```

This table supports text-to-image search and caption search. The first implementation should store local cached image paths plus metadata. On Modal, those paths should live on a Modal Volume. Later, the same design can move to S3-backed Lance storage.

## Endpoint Design

### `/search_text`

Searches the `text_documents` table using a text query.

Input:

```json
{
  "query": "What is reinforcement learning?",
  "k": 5
}
```

### `/search_images`

Searches the `image_documents` table using a text query embedded into the same CLIP-style vector space as images.

Input:

```json
{
  "query": "a dog playing in a grassy field",
  "k": 5
}
```

### `/search_all`

Runs both text search and image search from one query.

Input:

```json
{
  "query": "children playing soccer outside",
  "k": 5
}
```

## Updated Execution Order

Follow this order so each new moving part is isolated:

1. Finish local FineWeb-Edu sample extraction.
2. Build local LanceDB text smoke test.
3. Build local Ray text embedding pipeline.
4. Add COCO image-caption sample extraction.
5. Build local CLIP/OpenCLIP-style image embedding smoke test.
6. Build local Ray image embedding pipeline.
7. Move both pipelines into a Modal GPU batch job.
8. Store both LanceDB tables on a Modal Volume.
9. Add `/search_text`.
10. Add `/search_images`.
11. Add `/search_all`.

Why this order:

- Text first proves LanceDB and Ray with a simpler modality.
- Image second proves multimodal inference and GPU-oriented workloads.
- Modal last proves remote execution after the local logic is known-good.

## Compute Plan

Start with:

```text
GPU: L4 or A10
Ray actors: 1
Batch size: 64-128
Storage: Modal Volume
```

Optional stronger run:

```text
GPU: 2x L4
Ray actors: 2
Each actor uses 1 GPU
```

Do not start with two GPUs. First prove one actor on one GPU, then scale.

## Cost Plan

Recommended hard cap: `$100` for the full two-day demo.

| Version | Text docs | Images | Expected cost |
| --- | ---: | ---: | ---: |
| Text-only demo | 5K-25K | 0 | $5-$25 |
| Small multimodal demo | 5K | 2K | $10-$35 |
| Good multimodal demo | 10K | 5K | $20-$60 |
| Strong portfolio run | 25K | 10K | $40-$120 |

The first full multimodal run should be the small version.

## Step Notes

### Step 0 - Virtual Environment

We use a repo-local Python 3.11 virtual environment managed by `uv`.

Why:

- Python 3.11 is broadly compatible with Ray, Modal, sentence-transformers, and LanceDB.
- `uv` gives faster, more reproducible package installation than plain `pip`.
- A repo-local `.venv` keeps dependencies isolated from system Python.

Trade-off:

- The local virtual environment is only for development and local tests.
- Modal will build its own remote container image separately.

### Step 1 - Project Scaffold

The repo is organized as:

```text
README.md
requirements.txt
modal_app.py
scripts/
src/
data/
notebooks/
```

Why:

- `scripts/` keeps each pipeline milestone runnable and easy to explain.
- `src/` is for shared config or reusable code once duplication appears.
- `data/` is generated and ignored by git.
- `modal_app.py` keeps the Modal deployment surface obvious.

Trade-off:

- This is intentionally simple. A larger production project might use a package layout, typed config system, CI, and separate deployment modules.

### Step 2 - Dependencies

Key dependencies:

- `ray[data]`: distributed batch processing.
- `modal`: serverless GPU jobs and endpoint serving.
- `datasets`: streaming FineWeb-Edu from Hugging Face.
- `sentence-transformers`: text embeddings and a simple CLIP-style first path.
- `lancedb`: vector storage and search.
- `pandas`, `numpy`, `pyarrow`: local tabular/parquet handling.
- `Pillow`: image loading and conversion for the image pipeline.

Why not start with heavier infrastructure:

- S3, multi-node clusters, orchestration tools, and dashboards are useful later, but they slow down the first working version.
- The first milestone should prove the pipeline end to end.

### Step 3 - Model Choice

For text, start with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Why:

- Small and fast.
- Good enough for educational text semantic search.
- Works locally on CPU for smoke tests.

For images, start with a CLIP-style model through `sentence-transformers`, such as:

```text
clip-ViT-B-32
```

Why:

- It keeps embedding logic explicit in our Ray pipeline.
- It supports text and image embeddings in a shared space.
- It avoids hiding the important learning step inside LanceDB's embedding registry.

Later, LanceDB's OpenCLIP integration can be added as an alternative implementation path.

Trade-off:

- Using `sentence-transformers` for CLIP may be less directly LanceDB-native than its OpenCLIP integration.
- It is better for learning because the model loading, batching, and actor reuse stay visible.

## Updated Out Of Scope

The project will not include:

- Production authentication.
- Large-scale retries.
- Multi-region deployment.
- Full FineWeb-Edu processing.
- Full COCO image-caption processing.
- Production monitoring.
- Streaming ingestion.
- Production-grade indexing strategy.

The goal is a clean, explainable multimodal demo.

## References

- [LanceDB documentation](https://docs.lancedb.com/)
- [LanceDB OpenCLIP integration](https://docs.lancedb.com/integrations/embedding/openclip)
- [COCO Image Captioning dataset on Hugging Face](https://hf.co/datasets/MagiBoss/COCO-Image-Captioning)
- [Ray Data working with images](https://docs.ray.io/en/master/data/working-with-images.html)
- [Ray end-to-end multimodal AI workloads](https://docs.ray.io/en/master/ray-overview/examples/e2e-multimodal-ai-workloads/index.html)

## Open Questions And Decisions

Use this section to capture decisions as we make them.

| Question | Decision | Why |
| --- | --- | --- |
| What Python version? | Python 3.11 | Better compatibility with ML/distributed packages than newer Python versions. |
| What storage first? | Modal Volume + local LanceDB | Fastest path to working end-to-end demo. |
| What GPU first? | L4 | Good cost/performance for demo embedding workloads. |
| Full dataset? | No | Too expensive and unnecessary for the learning objective. |
| Text-only or multimodal? | Multimodal | Images make Ray, Modal GPU execution, and LanceDB's lakehouse story more justified. |
| LanceDB OpenCLIP first? | No | Use explicit CLIP-style embedding in Ray first for learning; evaluate LanceDB OpenCLIP later. |

## Questions Asked During Build

Add Q&A here as the project evolves.

### How do we use `HF_TOKEN`?

Hugging Face tokens can be used in two common ways:

1. Export the token into the shell before running Python:

   ```bash
   export HF_TOKEN=...
   python scripts/01_make_sample.py
   ```

2. Store it in a local `.env` file and have the script load it.

For this project, `scripts/01_make_sample.py` reads `.env` if it exists and passes `HF_TOKEN` explicitly to `datasets.load_dataset(..., token=hf_token)`.

Why this matters:

- A `.env` file is just a text file; Python does not automatically read it.
- Passing the token avoids anonymous Hugging Face Hub requests.
- Authenticated requests usually get higher rate limits and smoother downloads.

Trade-off:

- We avoid adding `python-dotenv` for now because the `.env` parsing we need is tiny.
- If the project grows, replacing the helper with `python-dotenv` would be reasonable.

Important:

- `.env` must stay out of git.
- The token should never be printed in logs or committed.

## Build Log And Architecture Decisions

This section records decisions made during implementation. The purpose is to preserve the reasoning trail, not just the final code.

### Why Did We Move From Text-Only To Multimodal?

Initial direction:

```text
FineWeb-Edu -> text embeddings -> Ray -> Modal -> LanceDB
```

Updated direction:

```text
FineWeb-Edu text records -> text embeddings -> LanceDB text table
COCO image-caption records -> CLIP embeddings -> LanceDB image table
```

Why:

- A small text-only demo can run comfortably on a laptop, which makes Ray and Modal feel less justified.
- Image workloads are more realistic for distributed/GPU infrastructure because they involve image decoding, preprocessing, larger model inference, and text-to-image retrieval.
- Multimodal search better matches AI-native lakehouse patterns where raw assets, metadata, embeddings, and derived features are stored together.

Trade-off:

- Multimodal adds complexity: more datasets, more schemas, more models, and more endpoint design.
- The added complexity is worthwhile because it makes the architecture easier to defend in interviews.

### Why FineWeb-Edu And COCO Do Not Need Row-Level Correlation

The project has two independent collections:

```text
text_documents  -> educational web text
image_documents -> COCO image-caption records
```

The records do not need to match each other during ingestion. Instead, the user's query is the bridge:

```text
query -> /search_text   -> text matches
query -> /search_images -> image matches
query -> /search_all    -> both result sets
```

Why:

- Real lakehouses often contain heterogeneous assets from different sources.
- Semantic retrieval can make those assets searchable from one interface even if they do not share row-level keys.

Trade-off:

- Text embeddings and CLIP image embeddings live in different vector spaces.
- Distances from the two spaces are not directly comparable.

Decision:

- `/search_all` should return separate ranked lists:

  ```json
  {
    "query": "...",
    "text_matches": [],
    "image_matches": []
  }
  ```

- Do not merge text and image results into one global ranking unless we add calibration, a shared embedding model, or a reranker.

### Dataset Decision: Flickr30k Rejected For First Version

We first considered:

```text
nlphuji/flickr30k
```

What happened:

- `datasets.load_dataset("nlphuji/flickr30k", ...)` failed because the repo contains an old-style dataset script.
- The installed `datasets` version no longer supports loading dataset scripts in that path.

Why not fight this immediately:

- We could load converted Parquet shards manually.
- But that would add dataset-specific workaround complexity before the main Modal/Ray/LanceDB architecture was stable.

Decision:

- Do not use Flickr30k on the critical path.
- Prefer a dataset that loads cleanly through standard Hugging Face dataset/parquet paths.

Trade-off:

- Flickr30k is a recognized image-caption dataset.
- Avoiding it keeps implementation smoother and lets us focus on the infrastructure goal.

### Dataset Decision: MagiBoss COCO Rejected For Main Demo

We inspected:

```text
MagiBoss/COCO-Image-Captioning
```

Schema:

```text
image: PIL image
text: str
```

What happened:

- The first sampled caption was Thai, not English.

Why this matters:

- Our demo queries are English.
- For an interview demo, query behavior should be predictable and easy to inspect.

Decision:

- Do not use this dataset as the main image dataset.

Trade-off:

- The schema was simple and attractive.
- Non-English captions make the demo less clear unless we intentionally build multilingual retrieval.

### Dataset Decision: COCO Dog Dataset As Smoke Test

We inspected:

```text
ArkaMukherjee/coco_dog_images_with_captions
```

Schema:

```text
image: PIL image
captions: str
```

Why it was useful:

- It loaded cleanly.
- Captions were English.
- Schema was simple.
- It was enough to prove the local image pipeline:

  ```text
  local image files -> CLIP embeddings -> Ray -> LanceDB -> image search
  ```

Trade-off:

- Domain is narrow: mostly dogs.
- Good for a smoke test, weaker for portfolio positioning.

Decision:

- Keep it as a local learning checkpoint/fallback.
- Do not use it as the final Modal demo dataset.

### Dataset Decision: Multimodal-Fatima COCO Captions As Main Image Dataset

We inspected:

```text
Multimodal-Fatima/COCO_captions_train
```

Schema:

```text
image: PIL image
filepath: str
sentids: list
filename: str
imgid: int
split: str
sentences_tokens: list
sentences_raw: list
sentences_sentid: list
cocoid: int
id: int
```

Why this is the best main image dataset:

- General COCO image-caption coverage.
- English captions.
- Multiple human captions per image.
- Stable metadata: `cocoid`, `imgid`, `filename`, `split`.
- Rich enough to look like a real multimodal table.

Implementation decision:

- Use the first caption as the display/search caption:

  ```python
  caption = captions[0]
  ```

- Preserve all captions as serialized metadata:

  ```python
  captions_json = json.dumps(captions)
  ```

Trade-off:

- More complex schema than the dog dataset.
- Worth it because the final demo is more general and more credible.

### Why Serialize Caption Lists As JSON?

Ray warned when passing a list/object column through Arrow:

```text
Failed to convert column 'captions' into pyarrow array
```

Why:

- `captions` is a Python list per row.
- Ray Data and Arrow prefer columns with stable, simple types.

Decision:

- Store the primary caption as a string column:

  ```text
  caption
  ```

- Store all captions as JSON text:

  ```text
  captions_json
  ```

Trade-off:

- JSON text is less queryable than a native nested array.
- It is simpler and more robust for Ray -> pandas -> LanceDB in the first version.

Future improvement:

- Use a proper Arrow list schema or a separate captions table if we need structured caption-level retrieval.

### Why Modal Health Check Before GPU Work?

We added a small Modal `health_check` before running GPU jobs.

Why:

- Modal failures can come from image builds, dependency versions, auth, Volume mounts, or Python imports.
- A cheap CPU function isolates the infrastructure foundation before spending GPU money.

Trade-off:

- It is an extra step before the "real" demo.
- It saves time by narrowing future failures.

What it proved:

- Modal image builds.
- Core dependencies import remotely.
- Modal Volume can be written and committed.

### Why GPU Smoke Test Before Ray Batch Job?

We added `gpu_smoke_test`.

Why:

- It verifies Modal can allocate an L4 GPU.
- It verifies `torch.cuda.is_available()`.
- It verifies `sentence-transformers/clip-ViT-B-32` can run on the GPU.
- It checks embedding shape before introducing Ray, datasets, and LanceDB.

Trade-off:

- It downloads/loads a model separately from the batch job.
- It gives a clean, cheap checkpoint for GPU readiness.

### Why Use The Fully Qualified CLIP Model Name?

Short model name:

```text
clip-ViT-B-32
```

Safer model name:

```text
sentence-transformers/clip-ViT-B-32
```

Why:

- Modal/Hugging Face resolution can be stricter than local cached aliases.
- The fully qualified repo ID avoids casing/alias issues such as `clip-Vit_b-32`.

Decision:

- Use `sentence-transformers/clip-ViT-B-32` everywhere.

### Why Use Modal Secrets For `HF_TOKEN`?

Local `.env` files do not automatically exist inside Modal containers.

Decision:

- Create a Modal Secret named:

  ```text
  hf-token
  ```

- Add it to functions that call Hugging Face:

  ```python
  secrets=[modal.Secret.from_name("hf-token")]
  ```

Why:

- Hugging Face libraries read `HF_TOKEN` from the environment.
- Authenticated requests get better rate limits and fewer flaky downloads.
- Secrets keep credentials out of code and git.

Trade-off:

- Slightly more setup in Modal.
- Correct approach for remote execution.

### Why Write LanceDB To `/tmp` Before Copying To Modal Volume?

Observed error:

```text
Unable to rename file: Operation not permitted
```

Context:

- LanceDB/Lance commits data using filesystem operations such as atomic renames.
- Modal Volume is persistent, but it does not behave exactly like a normal local POSIX disk for all operations.

Decision:

Build LanceDB on ephemeral container disk:

```text
/tmp/lancedb_build
```

Then copy the completed artifact into the persistent Modal Volume:

```text
/data/lancedb
```

Why:

- `/tmp` is normal local disk inside the Modal container.
- LanceDB can safely create and rename files there.
- `/data` persists across Modal function runs and can be read by endpoints.

Trade-off:

- Requires a copy step.
- Uses temporary container storage during the build.
- Acceptable for demo-scale tables.

Future improvement:

- Use object storage/S3 directly if LanceDB support and filesystem semantics fit the production target.
- Or validate LanceDB's recommended pattern for Modal Volumes if available.

### Why Save Images To Disk Before Ray Embedding?

The Modal image function streams image rows from Hugging Face. We save each image to a file before Ray reads the manifest.

Why:

- Ray Data can read a simple parquet manifest with paths.
- Actors can load images from stable local paths.
- It avoids passing PIL objects through Ray/Arrow.

Trade-off:

- Uses extra disk space.
- Adds an image serialization step.

Decision:

- Save images under the Modal Volume during remote builds.
- For LanceDB construction, use local `/tmp` for the database itself and copy the final DB to Volume.

### Is Ray Pointless At The Current Dataset Size?

Short answer: at the current demo scale, Ray is technically overkill.

Current validated runs:

```text
Text: 500 FineWeb-Edu records
Images: 100 COCO image-caption records
```

A single Python process could handle these sizes. That is not something to hide. It is an important part of the explanation.

Interview framing:

> At the initial 500/100 scale, Ray is not necessary. I used that run for correctness. Ray is included to demonstrate the production batch-inference pattern: partitioned reads, `map_batches`, and stateful actors that load models once and process many batches. The same code path can scale by increasing dataset size, actor count, and GPU resources.

Why this is still a useful project:

- The demo validates the end-to-end architecture without spending heavily.
- Ray Data keeps the pipeline shape close to a larger production ingestion job.
- The code uses stateful actors rather than repeatedly loading models.
- The image pipeline is meaningfully heavier than text-only embedding.
- Modal lets larger GPU-backed runs happen without a persistent cluster.

Trade-off:

- Small limits make the first version cheap and reliable.
- Small limits also mean the current metrics are validation metrics, not evidence that distributed compute was required.

Decision:

- Be explicit in the README/interview: tiny runs prove correctness; larger runs prove scaling behavior.

### Scaling Experiment Plan And Cost Framing

Modal's pricing page listed these relevant rates during the build:

```text
Nvidia L4:       $0.000222 / sec
Nvidia A10:      $0.000306 / sec
CPU physical core: $0.0000131 / core / sec
Memory:          $0.00000222 / GiB / sec
```

Our batch functions roughly use:

```text
1x L4 + 2 CPU + 16 GiB memory
```

Approximate runtime cost:

```text
0.000222 + 2*0.0000131 + 16*0.00000222
~= $0.000284 / sec
~= $1.02 / hour
```

Measured baseline:

```text
Image: 100 rows / 21.84 sec = 4.58 images/sec
Text:  500 rows / 16.48 sec = 30.34 docs/sec
```

First scale-check run:

```text
Image: 250 rows / 21.95 sec = 11.39 images/sec
Text:  1,000 rows / 19.78 sec = 50.55 docs/sec
```

Demo-scale run:

```text
Image: 1,000 rows / 33.08 sec = 30.23 images/sec
Text:  5,000 rows / 21.59 sec = 231.58 docs/sec
```

Strong demo run:

```text
Image: 2,500 rows / 39.80 sec = 62.82 images/sec
Text:  10,000 rows / 28.49 sec = 351.06 docs/sec
Observed billed cost: ~$0.04 for image run, ~$0.02 for text run
```

Portfolio run:

```text
Image: 5,000 rows / 72.10 sec = 69.34 images/sec
Text:  25,000 rows / 52.89 sec = 472.71 docs/sec
```

Interpretation:

- Throughput improved as limits increased.
- This is expected because fixed overheads such as Modal startup, model loading, Hugging Face dataset setup, Ray initialization, LanceDB table creation, and Volume copy/commit are amortized over more records.
- Tiny runs are useful for correctness, but larger runs are better for performance discussion.
- The text pipeline improves especially sharply because MiniLM inference is lightweight relative to the fixed setup cost.
- The image pipeline remains slower because CLIP image embedding includes image loading, decoding, preprocessing, and heavier model inference.

Suggested experiment ladder:

| Run | Text docs | Images | Runtime | Cost | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Smoke | 500 | 100 | done | not recorded | correctness |
| Small scale | 1K | 250 | text 19.78 sec, image 21.95 sec | not recorded | quick scaling check |
| Demo | 5K | 1K | text 21.59 sec, image 33.08 sec | not recorded | useful README metrics |
| Strong demo | 10K | 2.5K | text 28.49 sec, image 39.80 sec | text ~$0.02, image ~$0.04 | better Ray talking point |
| Portfolio run | 25K | 5K | text 52.89 sec, image 72.10 sec | observed cost TBD | strong screenshot |
| Stretch | 50K | 10K | not run yet | estimate TBD | only if a longer run is acceptable |

Measured scaling table:

| Modality | Records | Runtime | Throughput |
| --- | ---: | ---: | ---: |
| Text | 500 | 16.48 sec | 30.34 docs/sec |
| Text | 1,000 | 19.78 sec | 50.55 docs/sec |
| Text | 5,000 | 21.59 sec | 231.58 docs/sec |
| Text | 10,000 | 28.49 sec | 351.06 docs/sec |
| Text | 25,000 | 52.89 sec | 472.71 docs/sec |
| Image | 100 | 21.84 sec | 4.58 images/sec |
| Image | 250 | 21.95 sec | 11.39 images/sec |
| Image | 1,000 | 33.08 sec | 30.23 images/sec |
| Image | 2,500 | 39.80 sec | 62.82 images/sec |
| Image | 5,000 | 72.10 sec | 69.34 images/sec |

Cost interpretation:

- The measured strong demo costs were low: about two cents for 10K text rows and four cents for 2.5K image rows.
- Image cost was higher despite fewer records because CLIP image embedding is heavier than MiniLM text embedding.
- This supports the project argument that image workloads make GPU/serverless infrastructure more justified than text-only smoke tests.
- Cost should still be monitored when increasing limits because dataset streaming, model loading, retries, and larger image volumes can change the bill.
- The 25K/5K run is now the strongest portfolio-scale evidence: same architecture, larger limits, still completing in minutes on one L4.

Important caveat:

- Real cost can be higher than this estimate because of cold starts, model loading, dataset streaming, Volume operations, web calls, retries, and imperfect scaling.
- A reasonable safety multiplier is 2-5x for planning small demo experiments.

Recommended test order:

```text
1. 1K text + 250 images
2. 5K text + 1K images
3. 10K text + 2.5K images
4. Stop and compare metrics
```

Only run `25K text + 5K images` if the previous run is stable.

Interview framing:

> I started with tiny limits to validate correctness, then increased the limits to observe runtime and cost. The architecture does not change as the workload grows: Ray partitions data, actors keep models warm, Modal provides ephemeral GPU compute, and LanceDB persists searchable embeddings.

### Why Keep Text And Image Metrics Separate?

Text and image embedding have different performance profiles.

Text pipeline:

- Uses `sentence-transformers/all-MiniLM-L6-v2`.
- Processes short normalized text.
- Throughput is higher.

Image pipeline:

- Uses `sentence-transformers/clip-ViT-B-32`.
- Loads and decodes image files.
- Performs CLIP image preprocessing and GPU inference.
- Throughput is lower.

Decision:

- Report text docs/sec and image records/sec separately.
- Do not blend them into one "records/sec" number for the whole system unless the workload mix is explicitly defined.

Trade-off:

- Separate metrics are less tidy.
- They are more honest and more useful.

### Ray Warnings Observed During Scaling Runs

The larger Modal runs surfaced a few Ray/Lance warnings. None blocked the demo, but each is a useful operational talking point.

Observed warning: actor constructor arguments in object store

```text
Actor with class name: 'MapWorker(MapBatches(...))' has constructor arguments
in the object store and max_restarts > 0. If the arguments in the object store
go out of scope or are lost, the actor restart will fail.
```

Interpretation:

- Ray is warning about actor restart robustness.
- We pass model names through `fn_constructor_args`.
- For this demo, the argument is tiny and stable, so it is not a practical blocker.

Production discussion:

- Avoid large constructor args.
- Keep model/config identifiers small and reconstruct state inside actors.
- Use durable model artifacts or environment config rather than passing big objects.

Observed warning: object store memory proportion

```text
Ray's object store is configured to use only ~19% or ~5.5% of available memory.
For optimal Ray Data performance, set object_store_memory or
RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION.
```

Interpretation:

- Ray Data uses an object store to move blocks between operators.
- Modal containers can expose a large memory number, and Ray chooses a conservative object-store fraction.
- The current run uses tiny parquet manifests and small vector tables, so this is not a blocker.

Production discussion:

- Tune `ray.init(object_store_memory=...)`.
- Watch object spilling and memory pressure as image counts grow.
- Configure memory explicitly for predictable larger runs.

Observed warning: high memory detector for image actor

```text
Operator 'MapBatches(ImageEmbedderActor)' uses ~5.1GiB of memory per task
on average, but Ray only requests 0.0B per task at the start of the pipeline.
Consider setting memory=5.1GiB.
```

Interpretation:

- The CLIP actor and image batches use several GiB of memory.
- Ray noticed memory use that is not declared in scheduling requirements.
- With one actor and one GPU, concurrency is low, so the run completed.

Production discussion:

- Add explicit memory requirements to `map_batches` if scaling actor count.
- Keep image batch size under control.
- Monitor GPU memory and CPU memory separately.
- Increase Modal memory or reduce batch size if OOM appears.

Observed warning: LanceDB fork safety

```text
lance is not fork-safe. If you are using multiprocessing, use spawn instead.
```

Interpretation:

- LanceDB warns when imported in a process that also starts Ray/multiprocessing.
- In this project, LanceDB writes happen after Ray materializes the DataFrame.
- The run completed, so the warning is not blocking the demo.

Production discussion:

- Separate embedding workers from database-writing processes.
- Import/use LanceDB after Ray work completes.
- Consider a dedicated writer function/job for larger pipelines.

### Why Add A Browser UI?

The JSON endpoints prove the API, but a browser UI makes the project easier to demo to other people.

Decision:

- Add a Modal-hosted FastAPI/ASGI UI.
- Keep the HTML in `static/index.html` instead of a giant Python string.
- Use `image.add_local_dir("static", remote_path="/root/static")` at the end of the Modal image chain so Modal does not rebuild dependency layers on every HTML change.

Why:

- The UI can be opened from a deployed Modal URL.
- No separate frontend host is needed.
- It shows text snippets and image cards side by side.

Trade-off:

- The UI is intentionally simple.
- It is not a production frontend with routing, auth, caching, or state management.

### Why Serve Images Through `/image/{image_id}`?

The image table stores paths such as:

```text
/data/coco_images/coco_000000522418.jpg
```

Those are internal Modal Volume paths. A browser cannot load them directly.

Decision:

- Add an image-serving route:

  ```text
  GET /image/{image_id}
  ```

- Validate `image_id` before building the file path.
- Return JPEG bytes with `FileResponse`.

Why:

- The UI can display real images from Modal Volume.
- The LanceDB table can keep internal storage paths as metadata.
- The browser only sees a safe HTTP route.

Trade-off:

- Serving images from Modal Volume is fine for a demo.
- Production systems would typically store images in S3/object storage and serve them through a CDN or signed URLs.

### Why Keep HTML In A Separate File?

The UI started as an inline `HTML_PAGE` Python string, but that made `modal_app.py` noisy.

Decision:

- Move HTML/CSS/JS to:

  ```text
  static/index.html
  ```

- Read it in the UI route.

Why:

- UI edits are easier.
- Python backend code stays focused on Modal/Ray/LanceDB.
- The repo is easier for interviewers to scan.

Trade-off:

- Modal must explicitly include the `static/` directory.
- We fixed that by adding local static files to the Modal image.

Important Modal-specific lesson:

- `image.add_local_dir(...)` should be last in the image chain.
- If it appears before `pip_install(...)`, Modal warns because local file changes would cause unnecessary rebuild behavior.

### Why Use `modal serve`, `modal deploy`, And `modal run` Differently?

Each command has a different role.

`modal run modal_app.py`:

- Runs the local entrypoint.
- Good for health checks and batch jobs.
- Temporary app lifecycle.

`modal serve modal_app.py`:

- Keeps dev web endpoints alive.
- Good for testing `/search_*` and the UI locally during development.

`modal deploy modal_app.py`:

- Creates persistent deployed endpoints.
- Good for portfolio/demo use.

Trade-off:

- `modal run` may print endpoint URLs, but those URLs stop working when the run finishes.
- Use `modal serve` or `modal deploy` for browser/curl endpoint testing.

### Known Limitations To Say Out Loud

- The current dataset limits are intentionally small.
- Result quality is sample-limited.
- Metrics are demo validation metrics, not production benchmarks.
- Ray is not strictly required for the smallest runs.
- Endpoint functions load embedding models in request containers, so cold starts can be slow.
- Text and image distances are not directly comparable.
- There is no authentication, authorization, rate limiting, monitoring dashboard, retry queue, or production observability.
- There is no production index tuning or recall/latency evaluation.
- LanceDB is copied from `/tmp` into Modal Volume as a demo workaround.
- Images are served from Modal Volume for convenience; production would likely use object storage and CDN delivery.
- The project does not process full FineWeb-Edu or full COCO.

Interview framing:

> This is not a production replica. It is a demo-scale implementation of the architecture pattern. I intentionally kept the first runs small, documented the limitations, and left clear extension points for scaling, indexing, S3 storage, authentication, and observability.

### Strongest Interview Talking Points

Use these concise points:

- I started text-only, then expanded to multimodal because images better justify GPU and distributed batch inference.
- Ray Data is used for the batch inference shape: read, partition, `map_batches`, stateful actors, write results.
- Callable classes in `map_batches` create stateful actors, so models load once per actor and process many batches.
- Modal gives GPU execution without managing EC2, Kubernetes, Docker registries, or a persistent Ray cluster.
- LanceDB stores vectors and metadata together, matching a lakehouse-style retrieval pattern.
- I kept text and image retrieval in separate embedding spaces and separate rankings because their scores are not calibrated.
- I used Modal Volume for persistence, but had to build LanceDB under `/tmp` first because direct Volume writes hit filesystem rename limitations.
- I added a browser UI and image-serving route so the demo is easy to show, not just query with curl.
- I kept the demo cost-controlled and documented how to scale limits for more meaningful metrics.

### Extension: Adding New Embedding Columns For Training

A natural next step is to show iterative feature generation: take an existing LanceDB table, run another model over existing rows, and persist the result as a new feature column.

Example:

```text
image_documents
  image_id
  image_path
  caption
  image_vector          # CLIP image embedding
  caption_vector        # CLIP text embedding
  caption_minilm_vector # new text embedding feature
```

Why this matters:

- It demonstrates repeated transformation of the same media records.
- It gets closer to the Lance/LanceDB lakehouse story from the talk.
- It creates a training-ready feature table for downstream models.

Trade-off:

- The current demo mostly rebuilds tables with `mode="overwrite"`.
- True in-place Lance schema evolution is a more advanced extension.
- For a safe demo, create an enriched table/root first, then optionally promote it.

Practical implementation path for this repo:

```text
1. Read existing image_documents from Modal Volume.
2. Use Ray Data + a new embedding actor to compute caption_minilm_vector.
3. Write an enriched LanceDB table under /tmp.
4. Copy the enriched table to /data/lancedb_image_features.
5. Expose a training export function that writes selected columns to Parquet.
```

Why not mutate the production image table immediately:

- The existing `/search_images` endpoint depends on the current table.
- Writing an enriched copy avoids breaking the serving path.
- Once validated, the enriched table can become the new default.

Downstream training shape:

```text
training rows:
  image_path
  caption
  image_vector
  caption_minilm_vector
  split
```

Training access options:

- Read the enriched LanceDB table directly in a Modal/Ray training job.
- Export selected columns to Parquet on Modal Volume.
- Later, move the Lance data or Parquet export to S3 for external training jobs.

### What Has Been Proven So Far?

Local:

- FineWeb-Edu text sample extraction.
- Local text embedding and LanceDB search.
- Local Ray text embedding with stateful actors.
- Local image-caption sample extraction.
- Local CLIP image/caption embedding and LanceDB search.
- Local Ray image embedding with stateful actors.

Modal:

- Remote dependency health check.
- Modal Volume write and commit.
- GPU allocation.
- CLIP model inference on Modal GPU.
- COCO caption streaming.
- Ray starts inside Modal.
- Ray image embedding job completes.
- LanceDB image table persistence works by building under `/tmp` and copying to Modal Volume.
- `/search_images` endpoint works against persisted Modal Volume LanceDB data.
- Modal FineWeb-Edu text table build works with Ray and LanceDB persistence.
- `/search_text` endpoint works against persisted Modal Volume LanceDB data.
- `/search_all` endpoint works and returns separate text/image ranked lists.
- Modal-hosted browser UI exists for portfolio/demo use.

Current unresolved implementation item:

- Deploy and visually verify the browser UI, then commit the working milestone.

## Current Project Status

### Completed

- [x] Repo scaffold and Python 3.11 `uv` environment.
- [x] Requirements setup.
- [x] Modal authentication.
- [x] Local `HF_TOKEN` handling discussed.
- [x] Modal Secret direction for `HF_TOKEN` chosen.
- [x] FineWeb-Edu text sample extraction.
- [x] Local LanceDB text smoke test.
- [x] Local Ray text embedding pipeline.
- [x] Local Ray text search validation.
- [x] Multimodal scope update.
- [x] Dataset exploration and trade-off decisions:
  - [x] Rejected Flickr30k for old dataset-script issue.
  - [x] Rejected MagiBoss COCO because sampled captions were non-English.
  - [x] Used COCO dog dataset for local image smoke testing.
  - [x] Chose `Multimodal-Fatima/COCO_captions_train` for main Modal image pipeline.
- [x] Local image-caption sample extraction.
- [x] Local CLIP image embedding smoke test.
- [x] Local Ray image embedding pipeline.
- [x] Local Ray image search validation.
- [x] Modal image builder upgrade.
- [x] Modal health check passed.
- [x] Modal GPU smoke test passed.
- [x] Modal COCO image streaming works.
- [x] Modal Ray image embedding starts and completes.
- [x] Modal LanceDB image table persistence works through `/tmp -> Modal Volume`.
- [x] Modal `/search_images` endpoint works.
- [x] Modal text table build works through `/tmp -> Modal Volume`.
- [x] Modal `/search_text` endpoint works.
- [x] Modal `/search_all` endpoint works.
- [x] Modal-hosted browser UI implemented.
- [x] Architecture notes updated with decisions and trade-offs.

### In Progress

- [ ] Modal search endpoints.

Latest successful Modal image run:

```json
{
  "rows_requested": 100,
  "rows_written": 100,
  "seconds": 21.84,
  "rows_per_second": 4.58,
  "gpu": "L4",
  "ray_actor_count": 1,
  "batch_size": 32,
  "model": "sentence-transformers/clip-ViT-B-32",
  "storage": "Modal Volume + LanceDB"
}
```

Latest successful Modal text run:

```json
{
  "rows_requested": 500,
  "rows_written": 500,
  "seconds": 16.48,
  "rows_per_second": 30.34,
  "gpu": "L4",
  "ray_actor_count": 1,
  "batch_size": 128,
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "storage": "Modal Volume + LanceDB"
}
```

Notes:

- Modal health check passed.
- Modal GPU smoke test passed on NVIDIA L4.
- Ray emitted warnings about object store sizing and actor constructor args in object store. These are not blockers for the demo-scale run, but they are worth mentioning as operational tuning topics.
- LanceDB emitted a fork-safety warning when used in a process that also starts Ray. The current workaround is acceptable for this demo because LanceDB table creation happens after Ray embedding output is materialized. For production, we would separate these phases more carefully or validate process start methods.
- `/search_images` was tested with text queries including `"a woman cutting a cake"` and `"people riding horses"`.
- The cake query returned `COCO_val2014_000000522418.jpg` with caption `"A woman wearing a net on her head cutting a cake."` as the top result.
- The horses query returned horse-related and riding-related matches. This is useful demo evidence, but not a benchmark-quality retrieval evaluation.
- `/search_text` was tested with `"What is reinforcement learning?"`.
- The top text result discussed learning, motivation, and observational learning. This validates the endpoint and persisted text vector table, but relevance is limited by the 500-document sample size.
- `/search_all` was tested with `"children playing outside"`.
- `/search_all` returned separate `text_matches` and `image_matches` plus an explicit note that the two result lists are not globally ranked against each other.
- A browser UI was added as an ASGI app. It calls `/api/search_all`, renders text and image matches side by side, and serves images through `/image/{image_id}`.
- The image-serving route is necessary because `/data/coco_images/...` paths are internal Modal Volume paths, not public browser URLs.
- README now includes exact run/deploy commands, interview talking points, known limitations, and a hand-drawn style architecture diagram.

### Remaining TODO

- [x] Fix `build_image_table` LanceDB persistence using `/tmp -> Modal Volume`.
- [x] Serialize `captions` list as `captions_json`.
- [ ] Add clear comments and docstrings throughout `modal_app.py`.
- [x] Rerun Modal image build with a small limit such as `25` or `100`.
- [x] Add Modal text table build function using FineWeb-Edu.
- [ ] Add combined `build_lakehouse` function.
- [x] Add `/search_images` endpoint.
- [x] Add `/search_text` endpoint.
- [x] Add `/search_all` endpoint.
- [x] Add Modal-hosted browser UI.
- [x] Add exact run and deploy commands to README.
- [x] Add interview talking points to README.
- [x] Add known limitations to README.
- [x] Add hand-drawn architecture diagram asset.
- [x] Run `modal serve modal_app.py`.
- [x] Test endpoints with `curl`.
- [ ] Capture metrics:
  - [ ] Local text Ray.
  - [ ] Local image Ray.
  - [x] Modal GPU image.
  - [x] Modal GPU text.
- [ ] Update README status and metrics table.
- [ ] Commit stable milestones to git.

### Optional Later

- [ ] Add S3 storage.
- [ ] Add LanceDB indexing.
- [ ] Add 2-GPU Modal run.
- [ ] Add a simple UI.
