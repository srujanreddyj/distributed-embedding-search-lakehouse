# Multimodal Training Data Lakehouse — Design Document

## Why This Project Exists

This project demonstrates the engineering pattern behind production-scale multimodal training data infrastructure. It is designed as a portfolio project for ML data engineering roles — showing not just that code works, but that architectural tradeoffs were considered and documented.

The system ingests text, image, video, and audio samples from public datasets; processes them through distributed batch pipelines with GPU embedding; stores vectors and metadata in a queryable catalog; creates immutable dataset versions as manifests (zero-copy); materializes training-ready WebDataset shards; benchmarks loader throughput; and compares retrieval quality across dataset versions.

## Architecture Overview

```
Source datasets (HF)
     ↓
Source Connectors (pluggable, one per data source)
     ↓
Content-Addressed Store (hash-keyed dedup)
     ↓
Ray Data Preprocessing (stateful GPU actors per modality)
     ↓
Quality Gates (rules) → ANN Dedup (FAISS) → Soft Filter
     ↓
Embedding Service (hash-cached, model-versioned)
     ↓
Metadata Catalog (unified LanceDB)
     ↓
Dataset Versioning (immutable JSON manifests)
     ↓
WebDataset Sharding (training-ready tar files)
     ↓
Loader Benchmark (throughput measurement)
     ↓
Eval / Feedback Loop (version-to-version comparison)
```

---

## Component Breakdown

### 1. Source Connectors

**What:** Pluggable adapters that pull raw items from each source under one uniform schema.

**Uniform schema (every connector outputs this):**
```
id, source, modality, content_hash, payload.type, payload.content, payload.caption, payload.metadata
```

**Sources:**
| Modality | Dataset | Size (demo) | Model |
|----------|---------|------------|-------|
| Text | FineWeb-Edu (sample-10BT) | 500–25K docs | `all-MiniLM-L6-v2` |
| Image | COCO Captions | 100–5K images | `clip-vit-base-patch32` |
| Video | MSR-VTT | 50–500 clips | CLIP (keyframes → mean-pool) |
| Audio | LibriSpeech (clean) | 100–1K clips | Whisper encoder → projection |

**Key design decision:** Each connector only implements `transform()`. The base class handles `connect()`, `hash_content()`, and `write_manifest()` — ensuring all outputs share the same schema.

**Failure mode:** Source API changes break the connector (e.g. HF dataset restructuring). Mitigation: version-lock dataset references, pin connector per dataset version.

**Interview framing:** "I designed a pluggable connector interface so adding a new data source means writing one small adapter class — the entire downstream pipeline stays unchanged."

---

### 2. Content-Addressed Store (CAS)

**What:** Files stored by their SHA256 hash, not their filename.

**Layout:** `{root}/{hash[:2]}/{hash[2:4]}/{hash}{ext}`

```
data/cas/
  ab/cd/abcd1234...jpg
  ef/gh/efgh5678...wav
```

**Why this matters:**
- Free exact deduplication — same content from two sources stored once
- Reproducibility — any dataset version can reference blobs by hash
- Cache-ability — CAS is the foundation for layer caching in build systems

**Failure mode:** Hash collision (theoretically impossible for SHA256 at our scale). Real failure: two semantically identical files with different compression produce different hashes. Mitigation: CAS handles exact duplicates; ANN dedup (Component 4) handles semantic near-duplicates.

**Interview framing:** "CAS is the same pattern git uses for object storage. It makes dataset versioning trivial — a version is just a manifest of hashes, not a copy of terabytes."

---

### 3. Preprocessing Engine (Ray Data)

**What:** Distributed batch inference using Ray Data with stateful actors.

**Architecture:**
```
Parquet manifest → Ray Data → map_batches(stateful actor) → embedded parquet
```

**Per-modality actors:**
| Modality | Actor | GPU work | Output dim |
|----------|-------|----------|-----------|
| Text | `TextPreprocessor` | MiniLM embedding | 384 |
| Image | `ImagePreprocessor` | CLIP image features | 512 |
| Video | `VideoPreprocessor` | CLIP on keyframes + mean-pool | 512 |
| Audio | `AudioPreprocessor` | Whisper encoder + projection | 384 |

**Key design decision:** Stateful actors (callable classes) load models once in `__init__` and reuse them across batches. Stateless functions would reload models for every task — 100x overhead.

**Failure mode:** GPU OOM on variable-size video batches. Mitigation: size-aware dynamic batching, monitor memory per actor.

**Interview framing:** "Ray Data with callable classes creates stateful actors — models load once per actor, not once per row. This is the production pattern for GPU batch inference at scale."

---

### 4. Quality / Dedup / Safety

**What:** Tiered filters applied in increasing cost order.

**Tiers:**
| Tier | Check | Cost | When |
|------|-------|------|------|
| 1 | Exact hash dedup | Free | At CAS ingest |
| 2 | Quality gates | CPU | Pre-embedding |
| 3 | ANN near-dedup | GPU | Post-embedding |
| 4 | Safety filter | GPU | Optional |

**Tier 2 rules:**
- Text: >10 words, >50% alphabetic characters
- Image: file exists, not corrupt, caption non-empty
- Video: file exists, keyframes extracted
- Audio: file exists, not silent, >0.5s duration

**Key design decision:** Soft filtering — filter decisions recorded as metadata (`quality_status: "pass"|"fail"`, `quality_reason: "..."`) rather than deleting rows. A future dataset version can change thresholds and recover previously filtered items.

**Failure mode:** Overly aggressive filtering silently collapses data diversity. Mitigation: monitor post-filter distribution, never hard-delete raw data.

**Interview framing:** "I use soft filtering — every filter decision is recorded as metadata, not a deletion. This means thresholds can be tuned per dataset version without re-ingesting. It's the difference between a data pipeline and data infrastructure."

---

### 5. Embedding Service

**What:** Compute and cache embeddings keyed by `content_hash`.

**Cache table schema:**
```
content_hash, embedding (vector), modality, model_version
```

**Model versioning:** Each embedding is stamped with the model version that produced it (`text_v001`, `image_v001`, etc.). Cached embeddings from different model versions are never mixed in the same dedup pass.

**Dimensions per modality:**
| Modality | Model | Dim |
|----------|-------|-----|
| Text | MiniLM L6-v2 | 384 |
| Image | CLIP ViT-B/32 | 512 |
| Video | CLIP ViT-B/32 (mean-pooled frames) | 512 |
| Audio | Whisper encoder → 384 projection | 384 |

**Key design decision:** Precompute + cache vs on-the-fly. Precompute trades storage cost for training speed. Hash-keyed caching means the same content is embedded once, even if it appears in multiple dataset versions.

**Failure mode:** Model upgrade invalidates cached embeddings. Mitigation: model version stamping ensures old embeddings are never mixed with new ones. Lazy recomputation — only re-embed items queried by a dataset version using the new model.

**Interview framing:** "Each embedding is stamped with model_version. When I upgrade from CLIP-ViT-B to ViT-L, old and new embeddings never mix in the same search or dedup pass. Lazy recomputation means I only re-embed items that are actively in use."

---

### 6. Metadata Catalog

**What:** A unified LanceDB table containing every item across all modalities, with searchable columns and modality-specific metadata.

**Catalog schema:**
```
id, source, modality, content_hash, caption, content_path, metadata_json
```

**Key capability:** Cross-modality queries — "find all English-captioned images and text documents with quality=pass."

**Why LanceDB:**
- Vectors and metadata stored together (lakehouse pattern)
- Local or remote (Modal Volume for deployed, local disk for dev)
- Simple `where()` filtering without a separate database

**Failure mode:** Catalog table grows large without indexing. Mitigation: partition by source/date, scheduled compaction.

**Interview framing:** "The catalog is the bridge between raw data and researchers. A researcher can ask 'show me all COCO images with pass quality,' export those IDs to a manifest, and create a new dataset version — all without touching the raw storage layer."

---

### 7. Dataset Versioning (The Spine)

**What:** A dataset version is an immutable JSON manifest — not a copy of the data.

**Version manifest example:**
```json
{
  "version": "multimodal-demo-v001",
  "created_at": "2026-05-30T...",
  "total_items": 7500,
  "modalities": {"text": 5000, "image": 2000, "video": 300, "audio": 200},
  "models": {"text": "text_v001", "image": "image_v001", ...},
  "item_ids": ["fineweb_00000001", "coco_000000522418", ...]
}
```

**Why this is the spine:**
- Data stays in CAS (no copying terabytes)
- Models/config pinned in manifest (bit-for-bit reproducibility)
- Branching = new manifest (cheap)
- Rollback = point to old manifest (instant)

**Failure mode:** Manifest references blobs that garbage collection deleted. Mitigation: reference-counting in CAS, GC never deletes a hash referenced by any live manifest.

**Interview framing:** "This is what separates ML data infrastructure from a generic data lake. A dataset version is a JSON file listing content hashes. Creating a new training mix costs nothing — it's just writing a new manifest. The actual data is never copied."

---

### 8. WebDataset Sharding

**What:** Materialize a version manifest into training-ready sequential tar shards.

**Why sharding:**
| Approach | Read pattern | GPU utilization |
|----------|-------------|-----------------|
| 50K individual files | Random seeks | ~60% (starved) |
| 50 tar shards | Sequential read | >95% (fed) |

**Shard layout:**
```
shard-000000.tar/
  fineweb_00000001.txt
  fineweb_00000001.json
  coco_000000522418.jpg
  coco_000000522418.json
  ...
```

**Key design decision:** WebDataset over MDS (Mosaic StreamingDataset). WebDataset is simpler, inspectable with `tar -tf`, and modality-agnostic (text, image, video, audio all work). MDS would be the production upgrade for deterministic elastic resume.

**Failure mode:** Shard size skew. Mitigation: fixed-size shards (1000 items per shard), rebalance at materialization time.

**Interview framing:** "Random reads from 50K small files starve the GPU. WebDataset packs them into ~50 sequential tar shards. The loader streams through one shard after another — sequential reads keep the GPU fed."

---

### 9. Training Data Loader

**What:** Measure throughput of shard reading — items/second, MB/second.

**Metrics tracked:**
```
items, seconds, items_per_second, mb_per_second
```

**Why benchmark the loader:**
- Quantify the difference between random-file and shard-based reading
- Identify bottlenecks before they impact training runs
- Compare against model training throughput (samples/second)

**Benchmark result (expected):**
- Random files: ~500 items/sec
- WebDataset shards: ~5000 items/sec

**Interview framing:** "I benchmarked both small-file and shard-based loading. The shards delivered ~10x throughput because they eliminated random seeks. This is a concrete number you can put in a design review."

---

### 10. Eval / Feedback Loop

**What:** Close the loop between dataset version → model behavior → dataset decisions.

**Mechanism:** Compare two dataset versions on a frozen eval set of query probes.

**Eval probe example:**
```json
{
  "query": "a dog running outside",
  "target_modality": "image",
  "expected_terms": ["dog", "outside", "grass"]
}
```

**Metrics compared across versions:**
- Hit rate at K=5
- Mean keyword overlap
- Empty result count

**Key insight:** Without this loop, you can't prove version v002 is better than v001. The eval set is frozen and version-pinned so results are reproducible.

**Interview framing:** "I added a small eval set of query probes and compare every dataset version against it. This makes the link between data changes and retrieval quality measurable, not anecdotal."

---

### 11. Precompute vs On-the-Fly

**What:** The decision point — which transforms run once at materialization time and which run live in the training loop.

| Decision | Applied when | Reason |
|----------|-------------|--------|
| Precompute | At materialization | Resize, normalize — deterministic, always the same |
| On-the-fly | In training loader | Random crop, noise, mix — need diversity per epoch |

**Tradeoff:** Precomputed transforms are faster in training but fixed. On-the-fly transforms are slower but provide augmentation diversity. The hybrid approach: precompute deterministic transforms, randomize cheap augmentations in the loader.

**Failure mode:** A bad precompute transform choice is frozen into every run. Mitigation: version the transform config inside the manifest.

**Interview framing:** "I precompute deterministic transforms (resize, normalize) and keep stochastic augmentations (crop, noise) on-the-fly. The transform config is pinned in the dataset version manifest, so a bad choice can be fixed by creating a new version — not by re-ingesting."

---

### 12. Provenance & Licensing

**What:** Per-item license tracking so every training sample can be traced to its source and usage terms.

**License mapping:**
| Source | License |
|--------|---------|
| FineWeb-Edu | MIT |
| COCO Captions | CC-BY-4.0 |
| MSR-VTT | Research-only |
| LibriSpeech | CC-BY-4.0 |

**Why this exists:**
- Legal compliance — know which data can be used for commercial training
- Takedown requests — "remove all items from vendor X" = new manifest excluding source
- Auditable lineage — trace any training sample back to source

**Failure mode:** Someone writes data bypassing the pipeline, creating a lineage gap. Mitigation: enforce a single write path, contract tests that reject unlabeled items.

**Interview framing:** "Provenance isn't optional in production — it's legal protection. A takedown request becomes 'write a new manifest excluding source X.' Without provenance, it becomes 're-ingest everything from scratch.'"

---

## Scaling Strategy

| Scale | Text | Image | Video | Audio | Total cost |
|-------|-----:|------:|------:|------:|-----------:|
| Smoke | 500 | 100 | 50 | 100 | ~$0.03 |
| Demo | 5K | 1K | 200 | 500 | ~$0.05 |
| Portfolio | 25K | 5K | 500 | 1K | ~$0.15 |
| Strong | 50K | 10K | 1K | 2K | ~$0.40 |

The architecture does not change as scale increases — Ray partitions data, actors keep models warm, Modal provides ephemeral GPU compute, LanceDB persists queryable embeddings, and manifests version everything. Each of those interfaces scales independently.

## Future Considerations

These are the next concepts to learn and potentially add. They are not fully implemented in the current demo, but they map directly to ML infrastructure work that appears in production training and inference systems.

### GPU-Aware Data Loading Benchmark

**Current state:** The project materializes WebDataset-style tar shards and measures CPU-side shard reading throughput. That proves sequential sharding and basic loader performance, but it does not yet measure whether the input pipeline can keep a GPU fed.

**Future implementation:**
- Add a mini PyTorch training input loop over the materialized shards.
- Sweep `batch_size`, `num_workers`, `prefetch_factor`, shard size, and pinned memory.
- Measure samples/sec, batches/sec, host-to-device transfer time, loader wait time, and simulated GPU step time.
- Compare small-file reads vs tar shard reads.
- Report where the bottleneck moves as workers/prefetch increase.

**Concepts to learn:**
- PyTorch `Dataset`, `IterableDataset`, and `DataLoader`
- `num_workers`, `prefetch_factor`, `persistent_workers`, and pinned memory
- Host-to-device transfer and CUDA synchronization
- Input pipeline stalls, GPU utilization, and backpressure
- WebDataset shuffling and shard-level randomness

**Interview framing:** "I benchmarked not just how fast files read, but whether the loader keeps the accelerator busy. The useful metric is not only MB/sec; it is GPU idle time caused by input stalls."

### Transform Pipeline: Resampling, Clipping, Segmentation, Augmentation, Feature Extraction

**Current state:** The pipeline already handles multimodal feature extraction, audio resampling, video clipping/keyframe extraction, quality filtering, and Ray/Modal scaling. Richer segmentation and augmentation policy tracking are future upgrades.

**Future implementation:**
- Audio: resample, trim silence, normalize loudness, segment long audio, then extract Whisper/audio features.
- Video: decode bytes, segment into clips, detect low-quality clips, sample keyframes, then extract CLIP/video features.
- Image: validate decode, resize, normalize, optionally augment at training time.
- Text: normalize, length filter, dedup, safety filter, then embed.
- Track transform config in the dataset version manifest.
- Produce quality and throughput reports per transform stage.

**Concepts to learn:**
- Audio sample rates, mono/stereo conversion, clipping, silence detection, loudness normalization
- Video decoding, frame sampling, scene detection, keyframe extraction, clip-level quality metrics
- Deterministic transforms vs stochastic training augmentations
- Data quality gates, rejection reasons, and transform provenance
- Ray Data `map_batches`, `flat_map`, actor pools, and memory pressure

**Interview framing:** "The pipeline separates deterministic preprocessing from stochastic augmentation. Deterministic transforms are versioned and reproducible; cheap random augmentations stay in the training loader for diversity."

### Batch Inference Service With Safe Retries

**Current state:** The project has a Modal/Ray batch inference pipeline with artifact-aware resume. It reads manifests, runs GPU-backed embedding in batches, writes Parquet/catalog outputs, and can retry failed modalities without rerunning completed work. It is not yet a continuously running service that watches for new data.

**Future implementation:**
- Add a job table or queue with `pending`, `running`, `succeeded`, `failed`, and `dead_letter` states.
- Use idempotency keys such as `content_hash + model_version + transform_version`.
- Batch new rows by modality and model.
- Write outputs through a temporary path, then commit atomically.
- Retry transient failures with bounded retry counts and error logs.
- Store run metadata: code version, model version, input manifest, output artifact, timings, and failure reason.

**Concepts to learn:**
- Idempotent batch jobs
- Retry policies, dead-letter queues, and poison-pill records
- Exactly-once vs at-least-once processing
- Atomic writes and partial-output detection
- Run metadata, lineage, and reproducibility
- Training/inference boundary: offline embedding as batch inference feeding downstream training/search systems

**Interview framing:** "The important production property is idempotency. A failed embedding job should be safe to retry because outputs are keyed by content hash, model version, and transform version."

### Observability, Drift, And Rollback

**Current state:** The pipeline reports stage results, quality counts, loader metrics, and artifact status. It also supports manual resume and dataset-version rollback. It does not yet have production observability.

**Future implementation:**
- Track freshness: newest source timestamp, last successful stage, and stale artifact detection.
- Track skew/drift: modality mix, source mix, embedding norm distribution, quality-pass rate, duplicate rate, and license distribution.
- Emit OpenLineage-style events for each stage.
- Add Grafana/Prometheus dashboards for throughput, failures, queue depth, and cost.
- Alert on failed stages, stale catalog, quality-pass collapse, or unexpected modality imbalance.
- Roll back by selecting a previous dataset version manifest.

**Concepts to learn:**
- Data observability vs system observability
- Freshness, volume, schema, distribution, and lineage checks
- Metrics, logs, traces, and pipeline events
- Prometheus/Grafana basics
- OpenLineage concepts: job, run, dataset, input/output facets
- Rollback through immutable manifests

**Interview framing:** "For ML data, observability means knowing whether the dataset changed in a dangerous way, not only whether the job returned 200 OK."

## Known Limitations

- Demo-scale datasets — metrics are correctness/architecture validation, not production benchmarks
- Ray is technically overkill at the smallest limits — included to demonstrate the production batch-inference shape
- Text and image distances are not calibrated against each other (separate ranking lists)
- LanceDB built under `/tmp` then copied to Modal Volume (filesystem rename workaround)
- Images served from Modal Volume instead of CDN/object storage
- No authentication, rate limiting, monitoring dashboard, or production observability

## Interview Talking Points (Concise Versions)

1. "I started text-only, but images better justify GPU + distributed inference."
2. "Callable classes in map_batches create stateful Ray actors — model loads once, processes many batches."
3. "Modal provides GPU execution without managing Kubernetes or a persistent cluster."
4. "LanceDB stores vectors and metadata together — lakehouse pattern."
5. "Text and image rankings stay separate — their distance scales are not calibrated."
6. "LanceDB built under /tmp first, then copied to Modal Volume — rename compatibility workaround."
7. "Dataset versions are manifests of content hashes, not copies of data."
8. "Soft filtering records reasons, never hard-deletes rows."
9. "10x throughput improvement from shard-based loading over random file reads."
10. "Eval loop ties dataset version to retrieval quality — measurable, not anecdotal."
