# Battle Scars: Building the Multimodal Lakehouse

This project looks clean in the final architecture diagram, but several of the most useful lessons came from mistakes at the boundaries between stages. These are the issues that made the system more realistic and easier to explain in interviews.

## 1. The catalog initially bypassed quality filtering

The intended flow was:

```text
connectors -> manifests -> quality/dedup -> filtered -> catalog
```

The first implementation had the catalog reading directly from `manifests/`. That meant rows could fail quality checks and still enter the searchable and training-ready catalog.

This matters because the catalog is the trust boundary. If bad rows enter the catalog, every downstream consumer treats them as valid: search, dataset versions, sharding, loaders, and evals.

The fix was to make `stage_04_catalog` ingest only from:

```text
filtered/*_filtered.parquet
```

Now quality gates control what enters the dataset.

## 2. I claimed unified catalog search while querying separate tables

The first `search_all_modalities` implementation opened separate LanceDB roots:

```text
lancedb_text
lancedb
lancedb_video
lancedb_audio
```

That contradicted the architecture. Component 6 was supposed to be one unified metadata catalog.

The fix was to query one table:

```text
/data/catalog/item_catalog
```

The unified catalog now carries metadata, provenance, quality status, license, and vector columns for search.

## 3. One embedding column was the wrong abstraction

Text/audio and image/video embeddings do not live in the same vector space:

```text
MiniLM text vectors: 384 dims
CLIP image/video vectors: 512 dims
```

A single `embedding` column would mix incompatible dimensions and distance scales.

The better design was:

```text
item_catalog
  id
  modality
  metadata
  text_vector
  clip_vector
```

Search uses `text_vector` for text/audio and `clip_vector` for image/video. Results are returned separately by modality because distances are not directly comparable across embedding models.

## 4. `modal_app.py` was becoming a monolith

At one point, `modal_app.py` was absorbing search endpoints, UI routes, pipeline orchestration, status checks, and stage wrappers.

That made the deployment surface too large and blurred responsibilities.

The cleaner split became:

```text
src/              reusable pipeline logic
scripts/          local runnable stages
modal_pipeline.py cloud orchestration
modal_app.py      serving, search, and UI
```

This keeps reusable data logic testable outside Modal and makes the system easier to explain.

## 5. Connector constructors drifted from the base class

`SourceConnector` owned the CAS integration, but subclasses did not forward `cas`. The base class also exposed `hash_connect`, while connector implementations called `hash_content`.

That was interface drift.

The fix was to make every connector accept:

```python
cas: ContentAddressedStore | None = None
```

and forward it:

```python
super().__init__("source_name", output_dir, cas=cas)
```

The base helper was also renamed to `hash_content`, matching the subclasses.

## 6. A tiny COCO field bug blocked ingestion

The COCO connector had this line:

```python
raw_row.get()
```

It should have been:

```python
raw_row.get("cocoid") or idx
```

This was a small bug with a big blast radius: the entire ingestion stage would fail before preprocessing or cataloging could start.

The lesson is that source connectors need tiny smoke tests. Dataset schemas are brittle, and one bad field access can stop the pipeline.

## 7. The UI lagged behind the backend architecture

The backend moved toward four modalities, but the browser UI still rendered only text and image results.

For a portfolio demo, the UI is part of the architecture explanation. If it shows only two modalities, the project looks like a two-modality search demo even if the backend has four-modality plumbing.

The UI now renders:

```text
Text Matches
Image Matches
Video Matches
Audio Matches
```

Image results show previews. Video and audio results expose catalog-backed asset links.

## 8. Asset serving had to become catalog-backed

The old UI assumed images lived at:

```text
/data/coco_images/{image_id}.jpg
```

The expanded pipeline stores assets through connectors and CAS/raw paths. Hardcoded modality-specific paths became technical debt.

The fix was:

```text
/asset/{item_id}
```

That route looks up the item in `item_catalog` and serves its `content_path`. The old `/image/{image_id}` route remains as a compatibility wrapper.

## 9. Empty filtered output could leave stale catalog data

If `filtered/` was empty, catalog rebuild returned `0` but left the previous `item_catalog` in place.

That is dangerous because the system appears to work while serving old data.

The fix was to drop the old catalog table when there are no records. Empty input should produce empty output, not stale output.

## 10. The quality/dedup stage originally did quality only

The stage name promised quality plus dedup, but the implementation only wrote quality-passed rows. FAISS near-dedup was imported but unused.

The fix was to run near-dedup against `preprocessed/{modality}_embedded.parquet` when embeddings are available, then write the final rows into `filtered/`.

The stage now reports:

```text
near_duplicates_removed
rows_final
```

## 11. A zero limit still touched remote datasets

During Modal smoke testing, `audio_limit=0` still triggered LibriSpeech loading. The base connector loop started iterating over the remote dataset before it checked the limit.

That caused an avoidable failure:

```text
ImportError: To support decoding audio data, please install 'torchcodec'.
```

The real issue was not only the missing dependency. The pipeline should not open or decode a remote dataset when the requested limit is zero.

The fix was to make `SourceConnector.run()` return an empty manifest immediately when:

```python
limit <= 0
```

Now tiny smoke tests can intentionally skip modalities without paying dataset startup cost or triggering decoder dependencies.

## 12. Audio decoding changed under the dataset library

LibriSpeech used to work with the older Hugging Face audio decoding path, but the current `datasets` stack requires:

```text
torchcodec
```

The dataset also warned that:

```text
trust_remote_code is not supported anymore
```

The fix was to remove `trust_remote_code=True` from the LibriSpeech connector and add `torchcodec` to both local and Modal dependencies.

This is a good example of why ML data pipelines need dependency pinning or regular smoke tests. Dataset decoding behavior can change even when your own source code did not.

## 13. `video_limit=2` did not mean two usable videos

The video connector originally treated `limit` as "scan this many raw rows." That is not the same as "produce this many valid records."

For MSR-VTT, the first few rows can have captions but no inline video bytes in the shape the connector expected. The connector silently returned `None`, wrote a 0-row manifest, and then preprocessing had no `video_embedded.parquet` to produce.

The fix had three parts:

1. `SourceConnector.run()` now counts accepted records, not just scanned rows.
2. The video connector logs why a row was skipped.
3. The video connector accepts either `video["bytes"]` or a local `video["path"]`, and rejects rows where ffmpeg extracts no keyframes.

The better mental model is:

```text
requested limit = accepted records target
max_scan = bounded search through imperfect raw rows
```

This keeps small demos from silently producing empty outputs.

## 14. Empty manifests should be visible, not quietly embedded

When the video connector wrote an empty manifest, preprocessing did not make the reason obvious. The stage simply had no useful output.

The fix was to make `stage_02_preprocessing` read the manifest row count before launching Ray. If the manifest is empty, it records:

```text
status: skipped
reason: empty manifest
```

This makes Modal logs easier to interpret. In distributed data pipelines, "nothing happened" is often worse than a clear failure.

## 15. LibriSpeech stopped returning a dict

After adding `torchcodec`, LibriSpeech no longer returned audio as a simple dictionary with:

```python
audio["array"]
audio["sampling_rate"]
```

Instead, Modal logs showed:

```text
AttributeError: 'AudioDecoder' object has no attribute 'get'
```

That meant the connector was written against the old Hugging Face `Audio` decode shape, while the current stack returns a `torchcodec` `AudioDecoder`.

The fix was to make the audio connector accept both shapes:

```text
old path: dict with array / sampling_rate / bytes / path
new path: AudioDecoder.get_all_samples()
```

This is the broader lesson: multimodal dataset rows are not stable plain JSON. Images, audio, and video may arrive as decoder objects, local cache paths, bytes, or arrays depending on library versions and dataset format. Connectors need adapter code and loud skip reasons.

## 16. Hugging Face `modality:video` does not guarantee usable video bytes

The MSR-VTT connector looked correct on paper because `AlexZigma/msr-vtt` is an MSR-VTT dataset and the project only needed a tiny demo sample. In Modal, however, the first 100 scanned rows produced logs like:

```text
video payload has no bytes or local path. Available keys: []
```

The important distinction is that a Hugging Face dataset can be tagged or described as video while still exposing only captions, IDs, metadata, URLs, or benchmark annotations. That is useful for retrieval experiments, but not enough for this pipeline because the video stage needs an actual media payload or a resolvable local path so ffmpeg can extract keyframes.

The fix is not to special-case `AlexZigma/msr-vtt`; it is to treat dataset selection as part of connector validation:

```text
does the row contain bytes, path, or an archive entry?
can ffmpeg decode it?
does a small accepted-record smoke test produce video_embedded.parquet?
```

For future runs, prefer an MSR-VTT repo that explicitly ships an archive or inspectable media files, such as `Chengxiang1122/MSRVTT`, before falling back to annotation-only benchmark repos like `friedrichor/MSR-VTT` or `VLM2Vec/MSR-VTT`.

## 17. Ray Data did not flatten the payload the way preprocessing expected

The text preprocessing stage failed with:

```text
KeyError: 'payload.caption'
```

The connector wrote a uniform manifest with a nested `payload` object:

```text
payload.caption
payload.content
payload.metadata.*
```

But the physical Ray batch did not expose `payload.caption` as a top-level flattened column. It exposed the original nested `payload` column instead. The preprocessor was coupled to one Parquet/Ray projection shape rather than the actual manifest contract.

The fix was to add a small payload reader used by all preprocessors. It supports both forms:

```text
flattened: batch["payload.caption"]
nested:    batch["payload"][i]["caption"]
```

This also fixed the same latent bug in image, video, and audio preprocessing, which expected `payload.content` or `payload.metadata.keyframe_paths`.

The second lesson from this run was about honest stage reporting. Stage 1 printed `video: 5` even after the video connector accepted 0 records. The pipeline now reports the row count from the written manifest instead of echoing the requested limit.

## 18. FineVideo is the right shape, but gated access is part of the test

After MSR-VTT failed to expose usable media, FineVideo became a better video-source candidate because it streams real mp4 bytes from Parquet:

```text
sample["mp4"]   -> video bytes
sample["json"]  -> rich metadata and captions
```

The local probe confirmed the next boundary:

```text
Dataset 'HuggingFaceFV/finevideo' is a gated dataset on the Hub.
```

That is not a code bug. It is a data-access precondition. Before testing FineVideo locally or in Modal, the dataset terms must be accepted on Hugging Face, and the runtime must receive a token through `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or a Modal secret.

The probe script now has a `finevideo` preset that checks the real contract before the expensive pipeline runs:

```text
mp4 bytes -> local source video -> short clips -> keyframes -> manifest parquet
```

Once access was granted and an HF token was present, the smoke test passed:

```text
First row schema: {"mp4": "bytes", "json": "dict ..."}
[0] accept clip 0: 9 keyframes from bytes
[0] accept clip 1: 9 keyframes from bytes
[1] accept clip 0: 11 keyframes from bytes
[1] accept clip 1: 11 keyframes from bytes
accepted: 4
```

That result changed the implementation plan. The default video connector moved from MSR-VTT to FineVideo, and `SourceConnector.run()` now supports one raw source row producing multiple accepted records. That is necessary because the unit of ingestion is a full video, while the unit of training/search is a shorter clip.

The follow-up connector smoke test validated the real pipeline path too:

```text
FineVideoConnector.run(limit=2, max_scan=1)
Wrote 2 records to finevideo_manifest.parquet
CAS paths exist: true
n_keyframes: 9, 9
```

The broader lesson is that "local smoke test" for cloud data should validate access, row schema, media decode, clip generation, and manifest shape. Testing only one of those layers can still leave Modal to discover the next failure.

## 19. CLIP image features changed return shape across Transformers versions

Text preprocessing passed, but image preprocessing failed in Modal with:

```text
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'cpu'
```

The code expected:

```python
model.get_image_features(**inputs) -> torch.Tensor
```

In that Modal image, the call returned a wrapped model output instead. The conceptual operation was still correct, but the code assumed one library return shape. Since image and video both use CLIP, this was a shared preprocessing bug.

The fix was to add a CLIP compatibility adapter that accepts:

```text
torch.Tensor
output.image_embeds
output.pooler_output, projected only when its width matches visual_projection.in_features
output.last_hidden_state mean pool
```

and always returns a tensor before writing embeddings. This keeps the catalog contract stable at 512-d vectors even if the lower-level Transformers object shape changes.

The first adapter attempt still projected a 512-d pooled tensor through a 768-to-512 projection layer, causing:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x512 and 768x512)
```

That clarified the real rule: do not infer "pooler output" means "pre-projection." Inspect the tensor width before applying the projection layer.

## 20. A dedup stage can do all the work and still return `None`

After the CLIP fix, the pipeline finally completed preprocessing for all four modalities:

```text
text: 20 rows embedded
image: 5 rows embedded
video: 4 rows embedded
audio: 2 rows embedded
```

The next failure happened in quality/dedup:

```text
TypeError: 'NoneType' object is not iterable
duplicate_ids = embedded_ids - set(keep_ids)
```

The FAISS deduplicator built `keep_ids` internally but forgot to return it. That made the call site receive `None`, even though the dedup logic itself had run.

The fix was one line:

```python
return keep_ids
```

The local sanity check now proves the behavior:

```text
near-identical vectors -> keep first only
orthogonal vector -> keep
empty input -> []
```

The broader lesson is to test helper contracts, not only stage-level outcomes. Pipeline stages often fail several minutes after the actual bug because a small utility returned the wrong shape or nothing at all.

## 21. LanceDB writes can fail on Modal Volumes because atomic rename is not guaranteed

After preprocessing and quality/dedup passed, catalog creation failed:

```text
RuntimeError: lance error: Unable to rename file: Operation not permitted
```

The failing operation was `create_table(..., mode="overwrite")` under `/data/catalog`, which is a Modal Volume mount. LanceDB/Lance uses atomic rename during commits. That is normal on a regular local filesystem, but Modal Volume semantics can reject that rename.

The fix was to separate write semantics from persistence:

```text
1. Build LanceDB catalog on /tmp/catalog_build
2. Let Lance complete its normal local atomic commit
3. Copy the finished read-ready catalog directory into /data/catalog
4. Commit the Modal Volume
```

This preserves the serving contract (`/data/catalog/item_catalog`) while avoiding write-time filesystem limitations.

The same run also showed a stale `msrvtt` manifest from older experiments being swept up by globbing. Stage 3 and Stage 4 now explicitly process only the current four manifest names:

```text
fineweb_edu
coco_captions
finevideo
librispeech
```

The broader lesson is that persistent volumes are not clean build directories. Pipeline stages should either clean known outputs intentionally or enumerate the exact current inputs they accept.

## 22. Reserving a GPU does not automatically move model tensors onto it

Modal and Ray were configured to reserve an L4 GPU for preprocessing:

```python
@app.function(gpu="L4")
...
map_batches(..., num_gpus=1.0)
```

That only controls scheduling. It does not automatically move PyTorch models or processor outputs to CUDA. Without explicit device handling, the actor can reserve a GPU and still execute model math on CPU.

The fix was to add a shared device helper:

```text
device = cuda if torch.cuda.is_available() else cpu
model.to(device)
inputs.to(device)
```

Text uses SentenceTransformers' `device` argument, while image/video/audio move PyTorch models and tensor inputs directly. Local development still falls back to CPU, and Modal actors should now use the reserved GPU.

## 23. Pandas rows are not plain records in sharding

After catalog and versioning passed, sharding failed with:

```text
ValueError: The truth value of a Series is ambiguous
```

The shard writer did:

```python
record = df_indexed.loc[item_id]
if not record:
    continue
```

But `df_indexed.loc[item_id]` returns a Pandas `Series`, not a plain dictionary. Pandas refuses boolean truth checks because it is ambiguous whether "truthy" means any value, all values, non-empty, or something else.

The fix was to normalize the catalog row at the boundary:

```text
DataFrame -> first row Series
Series -> dict
dict -> dict
None/empty -> skip
```

The same pass fixed the shard metadata to use the catalog's real vector columns:

```text
text/audio -> text_vector
image/video -> clip_vector
```

The broader lesson is that catalog rows, manifests, and shards should have explicit interchange shapes. If a stage accepts a DataFrame, it should convert rows immediately rather than letting Pandas semantics leak into business logic.

## 24. Moving processor outputs to a device can change attribute-style access

After adding explicit CUDA movement, audio preprocessing failed with:

```text
AttributeError: 'dict' object has no attribute 'input_features'
```

The Whisper processor originally returned an object that supported:

```python
inputs.input_features
```

The shared `move_to_device()` helper moved tensor values recursively through mapping keys. That was fine for the tensors, but it normalized the processor output into a plain dictionary. After that, attribute access no longer worked.

The fix was to read audio features in a shape-tolerant way:

```python
input_features = (
    inputs["input_features"]
    if isinstance(inputs, dict)
    else inputs.input_features
)
```

The broader lesson is similar to the CLIP output issue: model and processor wrappers are conveniences, not contracts. At stage boundaries, normalize deliberately and then read normalized structures consistently.

## 25. Text content is not an asset path

Sharding next failed with:

```text
OSError: [Errno 36] File name too long
```

The shard writer treated every `content_path` as a filesystem path:

```python
asset_path = Path(record["content_path"])
asset_path.exists()
```

For image, video, and audio rows, `content_path` points to a CAS file. For text rows, the same catalog field contains inline article text. The writer tried to stat a long article as if it were a filename.

The fix was to make sharding modality-aware:

```text
text -> store inline text in metadata JSON
image/video/audio -> add asset file only if content_path exists and is a real file
invalid/non-file paths -> skip asset, keep metadata
```

The broader lesson is that a shared catalog column can have modality-specific meaning. A field named `content_path` was not honest for text, so downstream code must either normalize it into separate `text_content` / `asset_path` fields or handle modality explicitly.

## 26. Ray scaling is actor math, not just "turn on GPU"

A 2,500-image run showed the image stage moving slowly even though a GPU was present:

```text
Actors: 1
Resources: 1/1 GPU
Queued blocks: 30+
ImagePreprocessor uses ~4.6-4.8 GiB per task
Ray requested 0.0B memory per task
```

The bottleneck was not Parquet read; it was one CLIP actor serializing all image batches. The first scaling step was to make the Ray settings explicit:

```text
batch_size
actor_count
num_gpus per actor
memory per actor
```

The conservative single-GPU setup is best for debugging:

```text
1 x L4
1 actor
1 GPU per actor
batch_size 64 for images
lower cost
simpler logs
```

The two-GPU setup is better for larger image/video runs:

```text
2 x L4
2 actors
1 GPU per actor
batch_size 64 for images
declared memory per actor
higher cost
higher throughput
```

The pipeline now uses two L4 GPUs for preprocessing and two GPU actors per modality. The detailed tuning notes live in `docs/RAY_PREPROCESSING_SCALING.md`.

## Interview framing

The hardest part was not embedding text or images. The hard part was making the pipeline semantics honest.

Each stage had to consume the previous stage's approved output. The catalog had to become the trust boundary. Search and UI behavior had to match the architecture. Most of the bugs were boundary bugs: constructor contracts, stale catalog state, embedding-space mismatch, quality bypasses, and hardcoded asset paths.

That is the difference between a diagram and a system.
