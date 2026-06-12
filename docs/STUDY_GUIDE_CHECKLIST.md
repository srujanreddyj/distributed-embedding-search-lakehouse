# Multimodal Lakehouse Study Guide Checklist

This is the running checklist for learning the project end to end. The goal is not to memorize files. The goal is to deeply understand the problem, the design choices, the implementation, the edge cases, and why this system matters in ML/data infrastructure.

Use this doc during study sessions. A section should only be marked complete after the learner can explain it in her own words and answer follow-up questions.

## How We Will Study

- Start each stage with the learner restating her current understanding.
- Identify gaps, misconceptions, and fuzzy words.
- Explain the problem before the solution.
- Drill down through why, what, and how.
- Connect local code changes to broader ML infrastructure concepts.
- Confirm mastery before moving to the next stage.
- Use open-ended questions and multiple-choice checks.
- Add battle scars and debugging lessons as they come up.

## Stage 1: Project Motivation And Problem Shape

The learner should understand:

- [ ] What problem this project is solving.
- [ ] Why multimodal training/search data needs more infrastructure than a folder of files.
- [ ] Why text, image, video, and audio create different ingestion and preprocessing problems.
- [ ] Why a lakehouse-style catalog is useful for ML datasets.
- [ ] Why demo-scale correctness is different from production-scale reliability.
- [ ] What makes this project an ML/data infrastructure project rather than just an embedding demo.

She should be able to explain:

- [ ] The end-to-end dataflow in plain English.
- [ ] The difference between raw data, manifests, preprocessed embeddings, filtered rows, catalog rows, dataset versions, and shards.
- [ ] Which parts are implemented strongly, partially implemented, and future-facing.

Mastery check:

- [ ] Can explain why the system exists without naming any specific library.
- [ ] Can explain why each modality adds complexity.
- [ ] Can describe what would go wrong if the project only stored vectors without metadata/provenance.

## Stage 2: Source Connectors And CAS

The learner should understand:

- [ ] What source connectors do.
- [ ] Why every connector must output a uniform manifest schema.
- [ ] Why constructor signatures drifted and how `cas` forwarding fixed it.
- [ ] What a content-addressed store is.
- [ ] Why SHA256 hashes make deduplication, provenance, and versioning easier.
- [ ] Why two-level prefix layout matters for large object stores.

Edge cases:

- [ ] Empty manifests.
- [ ] Missing or unexpected Hugging Face fields.
- [ ] Dataset rows containing bytes, local paths, decoder objects, or nested payloads.
- [ ] Why `limit=0` is unsafe as a resume strategy.

## Stage 3: Ray Preprocessing And Embeddings

The learner should understand:

- [ ] Why Ray Data is used.
- [ ] What `map_batches` does.
- [ ] Why stateful actors matter for model loading.
- [ ] Why batch size, actor count, GPU count, and memory must be tuned together.
- [ ] How text, image, video, and audio embeddings differ.
- [ ] Why CLIP is used for image/video and MiniLM/Whisper-style features for text/audio.
- [ ] Why declaring `gpu="L4:2"` is not enough unless tensors/models move to CUDA.

Edge cases:

- [ ] Nested payloads in Ray batches.
- [ ] CLIP output shape differences.
- [ ] Partial Parquet output after cancellation.
- [ ] Timeout vs true code crash.
- [ ] Video is slower because one row can become many frames/keyframes.

## Stage 4: Quality, Dedup, Safety

The learner should understand:

- [ ] Why quality gates happen before catalog creation.
- [ ] Why the catalog must ingest from `filtered/`, not `manifests/`.
- [ ] Difference between hash dedup and embedding near-dedup.
- [ ] How FAISS near-dedup works conceptually.
- [ ] Why safety classifiers are a design target.
- [ ] Why Stage 4 needed per-modality resume.

Edge cases:

- [ ] Completed Stage 3 can still fail in Stage 4.
- [ ] Existing filtered output should skip unless forced.
- [ ] Missing embeddings should not crash quality filtering.

## Stage 5: Unified Catalog And Search

The learner should understand:

- [ ] Why the unified catalog is the trust boundary.
- [ ] Why text/audio and image/video use different vector columns.
- [ ] Why distances across embedding spaces are not directly comparable.
- [ ] Why the webpage should display text, image, video, and audio separately.
- [ ] Why catalog-based search is more honest than separate hardcoded LanceDB roots.

Edge cases:

- [ ] Stale manifests from old experiments.
- [ ] Missing catalog.
- [ ] Missing asset file for an item.
- [ ] Search result exists but media cannot be served.

## Stage 6: Dataset Versioning

The learner should understand:

- [ ] Why dataset versions are immutable manifests rather than copied data.
- [ ] How version manifests reference catalog item IDs.
- [ ] Why model config, transform spec, schema hash, and code version should be pinned.
- [ ] Why zero-copy branching matters.
- [ ] How rollback works by selecting an older manifest.

Current limitations:

- [ ] Schema hash not fully implemented.
- [ ] Git commit/code version not fully pinned.
- [ ] Experiment run linkage not fully implemented.

## Stage 7: Materialization And Loader Benchmark

The learner should understand:

- [ ] Why training systems prefer sequential shards over many small random files.
- [ ] What WebDataset-style tar shards contain.
- [ ] What the current loader benchmark measures.
- [ ] What a GPU-aware loader benchmark would add.
- [ ] Why batch size, worker count, prefetch, and pinned memory matter.

Future concepts:

- [ ] PyTorch `Dataset`, `IterableDataset`, and `DataLoader`.
- [ ] `num_workers`, `prefetch_factor`, `persistent_workers`.
- [ ] GPU idle time due to input stalls.
- [ ] Host-to-device transfer bottlenecks.

## Stage 8: Observability, Resumability, And Operations

The learner should understand:

- [ ] Why long-running pipelines need restart semantics.
- [ ] How artifact-aware resume works.
- [ ] Difference between `--force` and normal skip behavior.
- [ ] Why cancellation logs are not always Python errors.
- [ ] What stage result dicts do and do not prove.
- [ ] What production observability would add.

Future concepts:

- [ ] Freshness checks.
- [ ] Drift/skew checks.
- [ ] Failure alerts.
- [ ] OpenLineage events.
- [ ] Grafana dashboards.
- [ ] Dead-letter queues and retry policies.

## Stage 9: Future ML Infrastructure Extensions

The learner should understand:

- [ ] GPU-aware data loading benchmark.
- [ ] Transform pipeline for resampling, clipping, segmentation, augmentation, and feature extraction.
- [ ] Batch inference service with safe retries.
- [ ] Idempotency keys using `content_hash + model_version + transform_version`.
- [ ] Exactly-once vs at-least-once processing.
- [ ] Why offline embedding is a batch inference system that feeds training/search.

## Session Progress

| Stage | Status | Evidence Of Mastery |
| --- | --- | --- |
| 1. Motivation and problem shape | Not started | |
| 2. Source connectors and CAS | Not started | |
| 3. Ray preprocessing and embeddings | Not started | |
| 4. Quality, dedup, safety | Not started | |
| 5. Unified catalog and search | Not started | |
| 6. Dataset versioning | Not started | |
| 7. Materialization and loader benchmark | Not started | |
| 8. Observability and operations | Not started | |
| 9. Future ML infra extensions | Not started | |

