# Dataset: HAM10000

This study uses **HAM10000** ("Human Against Machine with 10,000 training images"), a public collection of skin-lesion photographs that have already been diagnosed. It is one of the most widely used dermatology datasets in machine-learning research.

## Source and licence

- **Published by:** Tschandl P., Rosendahl C., Kittler H. — *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Scientific Data 5:180161 (2018).
- **Download:** Harvard Dataverse, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
- **Licence:** free for non-commercial, academic research (you accept the terms on the Dataverse page before downloading).

## What kind of images these are

They are **dermatoscopic** images. A dermatoscope is a handheld magnifier with its own light source that a dermatologist presses against the skin. Every photo is therefore a clean, evenly lit, in-focus, zoomed-in close-up of a single lesion.

This matters for the study. HAM10000 is the "clean hospital" condition. These are **not** the messy phone photos the project is about; the degradation steps (blur, low light, noise, compression, rotation) simulate real-world phone photos on top of this clean data.

## Size and structure

- **10,015 images**, standard JPG files, roughly 600 × 450 pixels each (about 2.6 GB total).
- They come from only **7,470 distinct lesions**, so some lesions were photographed more than once.

That last point drives an important design choice: the data is split **by lesion, not by photo**, so two pictures of the same mole never end up in both the training set and the test set. Splitting by photo would leak information and inflate the scores.

## The seven lesion types

The label we predict is the `dx` column, one of seven classes:

| code | count | what it is | nature |
|------|-------|-----------|--------|
| nv | 6,705 | melanocytic nevus (an ordinary mole) | benign |
| mel | 1,113 | melanoma | dangerous cancer |
| bkl | 1,099 | benign keratosis-like lesion | benign |
| bcc | 514 | basal cell carcinoma | cancer (rarely fatal) |
| akiec | 327 | actinic keratosis / intraepithelial carcinoma | pre-cancer |
| vasc | 142 | vascular lesion | benign |
| df | 115 | dermatofibroma | benign |

## Class imbalance

The data is very uneven: about two-thirds of it is the ordinary mole (`nv`), and the rarest class has only 115 examples. A lazy model that always answered "mole" would score about 67% on plain accuracy while learning nothing. For that reason the study reports **balanced accuracy** and trains with **class weights**, and it pays special attention to melanoma, which is both rare and the most important to catch.

## The metadata (the answer sheet)

Each image has a row in `HAM10000_metadata` (a normal CSV) with these fields:

| column | meaning |
|--------|---------|
| `lesion_id` | ID of the physical lesion; one lesion may have several photos |
| `image_id` | the photo's filename, e.g. `ISIC_0027419` |
| `dx` | the diagnosis label the study predicts (one of the 7 above) |
| `dx_type` | how that diagnosis was confirmed |
| `age` | patient age in years |
| `sex` | male / female / unknown |
| `localization` | body site of the lesion |
| `dataset` | which source the image came from |

## How reliable the labels are

The `dx_type` field records how each diagnosis was verified. Over half are biopsy-confirmed, the strongest possible ground truth:

| confirmation method | count |
|---------------------|-------|
| histopathology (biopsy) | 5,340 |
| clinical follow-up over time | 3,704 |
| expert consensus | 902 |
| confocal microscopy | 69 |

## Who the patients are

- **Age:** 0 to 85 years, average about 52 (57 images have no recorded age).
- **Sex:** 5,406 male, 4,552 female, 57 unknown.
- **Body site:** most common are back, lower limb, trunk, upper limb, and abdomen, followed by face, chest, and foot.

So the images come from real patients across a wide age range, both sexes, and many parts of the body.

## Where it came from

The images were gathered from four sources across Austria and Australia:

| source | images |
|--------|--------|
| vidir_molemax | 3,954 |
| vidir_modern | 3,363 |
| rosendahl | 2,259 |
| vienna_dias | 439 |

Being multi-source is a strength: the data is not from a single camera or clinic, so a model is less able to cheat by learning one machine's quirks.

## How this study splits the data

Split once, by lesion, stratified by diagnosis, and saved to `data/split.csv` so every step uses the exact same piles:

| pile | photos | purpose |
|------|--------|---------|
| train | 6,981 | the study material (and the retrieval library) |
| val | 1,532 | tuning choices (e.g. picking k, calibrating confidence) |
| test | 1,502 | the hidden set, scored only at the end |

## Why the shape of this data matters here

- It is **clean clinical data**, so it is the fair "before" picture. The whole study is about what happens when photos are no longer this clean.
- Multiple photos per lesion forced the **lesion-level split** that keeps the test honest.
- The **imbalance** is why plain accuracy is not trusted on its own, and why melanoma is tracked separately.
- **Trustworthy labels** (half biopsy-confirmed) mean the "correct answers" the models are graded against are dependable.
