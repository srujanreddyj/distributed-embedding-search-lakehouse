# Ray Preprocessing Scaling Notes

This project uses Ray Data inside Modal for Stage 3 preprocessing:

```text
manifest parquet -> ray.data.read_parquet()
                 -> map_batches(StatefulPreprocessor)
                 -> write_parquet(preprocessed embeddings)
```

The important detail is that the preprocessors are stateful actors. Each actor loads a model once, keeps it in memory, receives batches, computes embeddings, and returns rows with an `embedding` column.

## What the Slow Image Log Meant

For 2,500 images, the Ray log showed:

```text
ReadParquet->SplitBlocks(40): 2500/2500
MapBatches(ImagePreprocessor): 63/2520
Actors: 1
Resources: 1/1 GPU
Queued blocks: 30+
Operator uses ~4.6-4.8 GiB memory per task
Ray requests 0.0B memory per task
```

That tells us several things:

1. Reading Parquet was not the bottleneck.
   The manifest was fully read quickly.

2. Work was queued behind one CLIP actor.
   `Actors: 1` means only one `ImagePreprocessor` actor was doing model inference.

3. The GPU was reserved, but not multiplied.
   `1/1 GPU` means the job had one GPU and one GPU actor using it.

4. Ray did not know the actor's memory needs.
   Ray observed roughly 4.6-4.8 GiB per image preprocessing task, but the code had not declared `memory=...`.

5. Batch size was too small for larger image runs.
   `batch_size=16` creates more batches and more overhead. CLIP can usually handle larger image batches on an L4.

## The Old Single-GPU Configuration

The original configuration was effectively:

```python
@app.function(gpu="L4", cpu=4, memory=24_576)

embedded = ds.map_batches(
    ImagePreprocessor,
    batch_size=16,
    compute=ray.data.ActorPoolStrategy(size=1),
    num_gpus=1.0,
)
```

This is a safe configuration for proving the pipeline works.

Characteristics:

```text
Modal GPUs: 1 x L4
Ray actors: 1 image actor
GPU per actor: 1.0
Image batch size: 16
Parallel CLIP inference: no
Risk of OOM: low
Cost: lower
Throughput: limited
```

Why it was recommended first:

1. It reduces debugging noise.
   When the system was still failing on schema, dataset, CLIP, audio, and catalog issues, a single actor made logs easier to understand.

2. It avoids two actors loading the same model while the implementation is unstable.

3. It minimizes GPU cost while the code is still proving correctness.

Why it becomes slow:

1. One actor serializes all image batches.
2. Batch size 16 underuses the GPU for CLIP.
3. PIL image decode and preprocessing happen inside the same actor, so CPU-ish work can also block the actor.
4. Ray has no declared memory reservation, so it warns and may schedule too optimistically in larger runs.

## The New Two-GPU Configuration

The preprocessing function now reserves two L4 GPUs:

```python
@app.function(gpu="L4:2", cpu=8, memory=49_152)
```

The image preprocessing settings are now:

```python
{
    "modality": "image",
    "batch_size": 64,
    "actor_count": 2,
    "num_gpus": 1.0,
    "memory": 6 * 1024**3,
}
```

This means:

```text
Modal GPUs: 2 x L4
Ray actors: 2 image actors
GPU per actor: 1.0
Image batch size: 64
Parallel CLIP inference: yes, one actor per GPU
Declared memory per actor: 6 GiB
Cost: higher
Throughput: higher
```

The goal is to turn this:

```text
one CLIP model -> one GPU -> one stream of batches
```

into this:

```text
CLIP actor A -> GPU 0 -> half the batches
CLIP actor B -> GPU 1 -> half the batches
```

## Single GPU vs Two GPUs

| Setting | Conservative single GPU | Current two GPU |
|---|---:|---:|
| Modal GPU request | `gpu="L4"` | `gpu="L4:2"` |
| Modal CPU | `4` | `8` |
| Modal memory | `24 GiB` | `48 GiB` |
| Image actor count | `1` | `2` |
| GPU per actor | `1.0` | `1.0` |
| Image batch size | `16`, recommended `64` | `64` |
| Memory per image actor | not declared | `6 GiB` |
| Model copies | 1 CLIP copy | 2 CLIP copies |
| Throughput | baseline | roughly up to 2x, workload-dependent |
| Cost | lower | about 2x GPU cost while running |
| Failure isolation | simpler logs | more concurrent logs |

The two-GPU setting should help most when:

1. There are enough rows to keep both actors busy.
2. The dataset has many Ray blocks or enough batches.
3. Model inference is the bottleneck.
4. Modal startup/model load time is small compared with total processing time.

The two-GPU setting helps less when:

1. The run is tiny.
   For 5 or 20 images, two actors only add startup overhead.

2. The bottleneck is remote dataset ingestion or writing files.
   More GPUs do not make Hugging Face streaming or filesystem writes infinitely faster.

3. CPU image decode is the bottleneck inside each actor.
   If PIL decode dominates, more GPU may sit idle unless actor count and CPU are also increased.

4. Quality/dedup/catalog/sharding become the bottleneck.
   Those stages are CPU/file/catalog heavy, not CLIP inference heavy.

## Knob 1: Batch Size

Batch size controls how many rows one actor processes per call.

For images:

```text
16  -> safe, but high overhead
64  -> good first scaling value on L4
128 -> possible, but watch memory
```

Higher batch size usually improves GPU utilization because each model call has more work. But it also increases peak memory.

Failure mode when too high:

```text
CUDA out of memory
container memory warnings
slow spill/retry behavior
```

How to respond:

1. Lower image batch size from 128 to 64.
2. If still failing, lower to 32.
3. Keep `memory=6 * 1024**3` or increase it if Ray warns.

## Knob 2: Actor Count

Actor count controls how many model replicas Ray creates.

```python
compute=ray.data.ActorPoolStrategy(size=2)
```

For two GPUs with `num_gpus=1.0`, use:

```text
actor_count = 2
num_gpus = 1.0
```

That gives one actor per GPU.

Trade-off:

```text
more actors -> more parallelism
more actors -> more model copies in memory
more actors -> more startup/model-loading overhead
```

If you use two actors on one GPU with `num_gpus=0.5`, you are sharing a GPU:

```text
actor A -> 0.5 GPU
actor B -> 0.5 GPU
```

That can help if preprocessing or CPU decode is the bottleneck, but it can hurt if CLIP inference already saturates the GPU.

For this project, the cleaner two-GPU setup is:

```text
2 GPUs, 2 actors, 1 GPU each
```

## Knob 3: `num_gpus`

`num_gpus` is a Ray scheduling reservation per actor, not a PyTorch device move by itself.

This reserves GPU resources:

```python
num_gpus=1.0
```

This actually moves model math to CUDA:

```python
model.to(device)
inputs.to(device)
```

Both are required.

This project now does both:

```text
Modal reserves GPUs
Ray assigns GPU resources to actors
Preprocessors move models and tensors to cuda when available
```

## Knob 4: `memory`

Ray warned:

```text
Operator 'MapBatches(ImagePreprocessor)' uses 4.6-4.8 GiB
Ray only requests 0.0B per task
```

The fix is:

```python
memory=6 * 1024**3
```

This does not give the actor more memory by magic. It tells Ray's scheduler how much memory the task is expected to need, so it avoids overscheduling memory-heavy work.

Trade-off:

```text
higher memory request -> safer scheduling
higher memory request -> fewer actors can run concurrently on the same node
```

For image/video/audio model actors, a 6 GiB declaration is a reasonable starting point.

## Knob 5: Blocks And Queues

The log showed:

```text
ReadParquet->SplitBlocks(40)
Queued blocks: 30+
```

That is not automatically bad. It means Ray has read input blocks and is waiting for map actors to consume them.

If actors are the bottleneck, queued blocks are expected.

If you have many GPUs but too few blocks, workers can sit idle. For larger runs, 40 blocks is fine. For very large runs, Ray will usually create enough work naturally from Parquet input, but you can repartition if needed.

## Current Stage 3 Settings

The current settings are:

```text
text:
  batch_size: 128
  actors: 2
  gpu per actor: 1.0
  memory per actor: 4 GiB

image:
  batch_size: 64
  actors: 2
  gpu per actor: 1.0
  memory per actor: 6 GiB

video:
  batch_size: 8
  actors: 2
  gpu per actor: 1.0
  memory per actor: 6 GiB

audio:
  batch_size: 16
  actors: 2
  gpu per actor: 1.0
  memory per actor: 6 GiB
```

Video batch size stays lower because each video row can contain multiple keyframes. A batch of 8 video clips might mean many more image frames inside the actor.

Audio batch size stays moderate because Whisper features are larger and sequence lengths can vary.

## What To Try Next

For 2,500 images:

1. Run the two-GPU config as-is.
2. Compare image stage seconds against the one-GPU run.
3. If memory is stable, try image batch size `128`.
4. If GPU utilization is low but CPU is busy, try actor count `4` with `num_gpus=0.5`.
5. If GPU utilization is high, stay with `2` actors and increase dataset size.

Good experiment matrix:

```text
single GPU:
  actor_count=1, num_gpus=1.0, batch_size=64

two GPU:
  actor_count=2, num_gpus=1.0, batch_size=64

two GPU larger batch:
  actor_count=2, num_gpus=1.0, batch_size=128

one GPU shared:
  actor_count=2, num_gpus=0.5, batch_size=64
```

Compare:

```text
image stage seconds
rows per second
GPU utilization
memory warnings
failure/retry behavior
```

## Cost Notes

Two GPUs do not make the job free just because it finishes faster.

Roughly:

```text
cost = GPU count * runtime * GPU price
```

If two GPUs cut runtime nearly in half, cost may be similar and developer time improves.

If two GPUs only improve runtime by 10-20%, cost is higher and the bottleneck is probably not GPU inference.

That is why the recommended path is:

```text
prove correctness on one GPU
measure 2,500 image run
move to two GPUs
compare seconds and cost
then scale dataset size
```

