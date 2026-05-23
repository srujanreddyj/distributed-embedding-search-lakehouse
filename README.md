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
- [ ] Local LanceDB text smoke test
- [ ] Local Ray text embedding pipeline
- [ ] COCO image-caption sample extraction
- [ ] Local image embedding smoke test
- [ ] Modal GPU batch job
- [ ] Modal search endpoints

## References

- [LanceDB documentation](https://docs.lancedb.com/)
- [LanceDB OpenCLIP integration](https://docs.lancedb.com/integrations/embedding/openclip)
- [COCO Image Captioning dataset on Hugging Face](https://hf.co/datasets/MagiBoss/COCO-Image-Captioning)
- [Ray Data working with images](https://docs.ray.io/en/master/data/working-with-images.html)
- [Ray end-to-end multimodal AI workloads](https://docs.ray.io/en/master/ray-overview/examples/e2e-multimodal-ai-workloads/index.html)
