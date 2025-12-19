# cv-duplicate-detection-mdpi

Detect and analyze near-duplicate images using computer vision approaches.  
Project to be applied to any image dataset where one or more near-duplicates may exist.  

**Dependencies:**  
Managed with UV. Environment installations before running the scripts.

## Task 1 - Basic Image Processing
Loads a sample image, converts it to grayscale, and applies a Gaussian blur (5x5 kernel).  

**Run via CLI:**
```
bash
mdpi_assesment task1 --random
```
Arguments:

```--random```: processes a random image from ```data/raw```

Output: processed images are saved to disk and paths are printed in the console

## Task 2 — Duplicate Detection
Detects near-duplicate images in a dataset. Currently implemented strategy: embedding_nn, which extracts global embeddings using MobileNetV2 (pretrained on ImageNet) and compares them using cosine similarity, for a tuning threshold value


**Run via CLI:**
```
bash
mdpi_assesment task2 \
  --src data/raw \
  --out data/results/task2_embedding_nn.csv \
  --strategy embedding_nn
```


Arguments:

```--src```: directory containing input images

```--out```: CSV file where candidate duplicate pairs are written

```--strategy```: which duplicate detection strategy to use (embedding_nn currently)

Output:

CSV columns: ```image_a,image_b,score```

Multiple candidate pairs may be returned if they exceed the similarity threshold

One true duplicate should be existing in the data; other returned pairs are potential matches for manual inspection

Notes:
```work in progress```
