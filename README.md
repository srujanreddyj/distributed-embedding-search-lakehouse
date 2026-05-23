# Serverless Multimodal Lakehouse Search with Ray, Modal, and LanceDB

This project is a demo-scale multimodal lakehouse search system. It embeds both educational text documents and image-caption records, stores vectors and metadata in LanceDB, and exposes Modal endpoints for semantic search.

The goal is to demonstrate the architecture and trade-offs behind distributed AI data pipelines:

- Ray Data for batch inference.
- Modal for serverless GPU execution.
- LanceDB for vector storage and semantic retrieval.
- FineWeb-Edu for educational text search.
- COCO image-caption data for text-to-image search.

## Architecture

```text
FineWeb-Edu text sample
        ↓
Text embedding model
        ↓
Ray Data batch inference
        ↓
LanceDB text_documents table


COCO image-caption sample
        ↓
CLIP/OpenCLIP-style embedding model
        ↓
Ray Data batch inference
        ↓
LanceDB image_documents table


Modal endpoint
        ↓
/search_text
/search_images
/search_all
browser UI
```

## Why This Project Exists

This is not production-grade and does not try to process full web-scale datasets. It is a portfolio/interview project designed to show the engineering pattern behind multimodal embedding pipelines: batch inference, GPU execution, vector storage, and semantic retrieval.

I initially considered a text-only embedding pipeline, but at demo scale that can make Ray and Modal feel over-engineered. Adding images makes the workload more realistic because image decoding, CLIP preprocessing, GPU inference, and multimodal retrieval are heavier than text-only embedding.

## Storage Design

The project uses two LanceDB tables.

### `text_documents`

```text
id
text
url
source
token_count
text_vector
```

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

## Endpoint Design

### `/search_text`

Search text documents by semantic meaning.

```json
{
  "query": "What is reinforcement learning?",
  "k": 5
}
```

### `/search_images`

Use a text query to retrieve matching images and captions.

```json
{
  "query": "a dog playing in a grassy field",
  "k": 5
}
```

### `/search_all`

Search both text and image tables from one query.

```json
{
  "query": "children playing soccer outside",
  "k": 5
}
```

### Browser UI

The Modal app also serves a small browser UI. It calls `/api/search_all`, renders FineWeb-Edu text matches beside COCO image matches, and serves persisted COCO images through `/image/{image_id}`.

Images are served through Modal because stored paths like `/data/coco_images/...` are internal Modal Volume paths and are not directly accessible by a browser.

## Execution Order

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

## Current Status

- [x] Python 3.11 virtual environment with `uv`
- [x] Project scaffold
- [x] FineWeb-Edu local sample extraction
- [x] Local LanceDB text smoke test
- [x] Local Ray text embedding pipeline
- [x] COCO image-caption sample extraction
- [x] Local image embedding smoke test
- [x] Local Ray image embedding pipeline
- [x] Modal GPU image batch job
- [x] Modal `/search_images` endpoint
- [x] Modal text batch job
- [x] Modal `/search_text` endpoint
- [x] Modal `/search_all` endpoint
- [x] Modal-hosted browser UI

## Current Metrics

| Run | Records | GPU | Ray actors | Batch size | Runtime | Records/sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Modal image batch | 100 | L4 | 1 | 32 | 21.84 sec | 4.58 |
| Modal text batch | 500 | L4 | 1 | 128 | 16.48 sec | 30.34 |

## Endpoint Evidence

`/search_images` validates the serving path: a text query is embedded with CLIP and searched against persisted image vectors in LanceDB on Modal Volume.

Example query:

```json
{
  "query": "a woman cutting a cake",
  "k": 5
}
```

Top result:

```json
{
  "image_id": "coco_000000522418",
  "filename": "COCO_val2014_000000522418.jpg",
  "caption": "A woman wearing a net on her head cutting a cake.",
  "distance": 1.3600707054138184
}
```

Another query, `"people riding horses"`, returned horse-related matches, including:

```json
{
  "image_id": "coco_000000037675",
  "caption": "Horses graze in front of a large building amid snow.",
  "distance": 1.4331679344177246
}
```

These results are demo-scale retrieval examples, not benchmark claims.

`/search_text` validates the text serving path: a text query is embedded with MiniLM and searched against persisted FineWeb-Edu text vectors in LanceDB on Modal Volume.

Example query:

```json
{
  "query": "What is reinforcement learning?",
  "k": 5
}
```

Top result:

```json
{
  "id": "21",
  "url": "http://www.funderstanding.com/category/child-development/brain-child-development/",
  "source": "HuggingFaceFW/fineweb-edu/sample-10BT",
  "token_count": 1062,
  "distance": 1.2111157178878784
}
```

The top result discusses learning, motivation, and observational learning. Because this run indexed only 500 sampled documents, the search result proves the endpoint and vector table work, but it should not be treated as a high-quality domain search benchmark for reinforcement learning.

`/search_all` validates the final multimodal serving shape: one query searches both tables and returns separate ranked lists.

Example query:

```json
{
  "query": "children playing outside",
  "k": 3
}
```

The endpoint returned `text_matches` from FineWeb-Edu and `image_matches` from COCO captions. The response intentionally includes this note:

```text
Text and image matches are ranked separately because they use different embedding models and distance scales.
```

That design avoids pretending MiniLM text distances and CLIP image distances are directly comparable.

## References

- [LanceDB documentation](https://docs.lancedb.com/)
- [LanceDB OpenCLIP integration](https://docs.lancedb.com/integrations/embedding/openclip)
- [COCO Image Captioning dataset on Hugging Face](https://hf.co/datasets/MagiBoss/COCO-Image-Captioning)
- [Ray Data working with images](https://docs.ray.io/en/master/data/working-with-images.html)
- [Ray end-to-end multimodal AI workloads](https://docs.ray.io/en/master/ray-overview/examples/e2e-multimodal-ai-workloads/index.html)
