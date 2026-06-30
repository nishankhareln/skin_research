# Skin-lesion screening: how reliable is it when the photo isn't perfect?

When a skin photo is low quality, the kind a real person takes on a phone (blurry, dark, slightly tilted), does an AI screening model still give trustworthy answers? And does a model that works by comparing against past cases do any better at *knowing when it is unsure*, instead of being confidently wrong?

This README is the working book : what has actually been done, in order, so anyone (including me, a month from now) can follow along or reproduce it.

## What kind of data this is

We use **HAM10000**, a public collection of dermatoscopic skin-lesion images that doctors have already diagnosed. Because every photo comes with a confirmed answer, we can measure how often a model is right.

- **Source:** Harvard Dataverse, the official release by Tschandl et al. Free for academic, non-commercial use. https://doi.org/10.7910/DVN/DBW86T
- **Size:** 10,015 photos, about 2.6 GB.
- **Answer sheet:** a CSV where each row describes one photo. The column we care about is `dx`, the diagnosis. The `image_id` column matches the photo's filename (for example `ISIC_0027419.jpg`).

Each photo is one of seven lesion types:

| code | count | what it is |
|------|-------|-----------|
| nv | 6,705 | melanocytic nevi (a harmless mole) |
| mel | 1,113 | melanoma (the dangerous skin cancer) |
| bkl | 1,099 | benign keratosis |
| bcc | 514 | basal cell carcinoma |
| akiec | 327 | actinic keratoses (a pre-cancer) |
| vasc | 142 | vascular lesion |
| df | 115 | dermatofibroma |

**One thing to watch:** the data is very uneven. Two thirds of it is the harmless mole, and the rarest type has only 115 examples. A lazy model could score 67% just by answering "harmless mole" every time, without learning anything. So later we will not trust plain accuracy on its own; we will also measure performance on the rare-but-important classes like melanoma.

---

## Folder layout

```
skin_research\
├── research_proposal.md          the motivation and full plan
├── README.md                     this file (the running log)
├── venv\                         the Python environment
├── data\
│   ├── images\                   10,015 .jpg skin photos
│   ├── HAM10000_images_part_1.zip
│   ├── HAM10000_images_part_2.zip
│   ├── HAM10000_metadata         the label CSV
│   └── split.csv                 train/val/test assignment (Step 3)
├── src\
│   ├── 02_view_samples.py        Step 2: show sample photos
│   └── 03_train_baseline.py      Step 3: train the baseline classifier
├── models\
│   └── baseline_efficientnet_b0.pt   the trained baseline (Step 3)
└── results\
    ├── sample_photos.png         Step 2 output: the labeled sample grid
    ├── baseline_metrics.txt      Step 3 output: the scores
    └── baseline_confusion.png    Step 3 output: confusion matrix
```

---

## Steps

### Step 1 — Get the data and unzip it  

1. Downloaded three files from Harvard Dataverse into `data\`: the two image archives and the metadata file. (The metadata downloaded without a `.csv` ending, which does not matter; it is still a normal comma-separated file.)

2. Unzipped both archives into `data\images\`. On Windows this was done with PowerShell:

   ```powershell
   Add-Type -AssemblyName System.IO.Compression.FileSystem
   $d = "D:\skin_research\data"
   $out = Join-Path $d "images"
   New-Item -ItemType Directory -Force -Path $out | Out-Null
   foreach ($z in "HAM10000_images_part_1.zip","HAM10000_images_part_2.zip") {
     $zip = [System.IO.Compression.ZipFile]::OpenRead((Join-Path $d $z))
     foreach ($e in $zip.Entries) {
       if ($e.Name -match '\.(jpg|jpeg)$') {
         [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $out $e.Name), $true)
       }
     }
     $zip.Dispose()
   }
   ```

3. Checked nothing was lost. Part 1 held 5,000 images and part 2 held 5,015, so 10,015 photos landed in `data\images\`. That matches the 10,015 rows in the metadata file exactly.

### Step 2 — Look at the data 

Set up a Python environment in `venv\` (pandas, pillow, matplotlib) and wrote `src/02_view_samples.py`. It reads the metadata, prints how many photos there are of each type, and saves a labeled grid of example photos to `results/sample_photos.png` (four samples of each of the 7 lesion types).

What it showed: the photos are clean, close-up dermatoscope images, well lit and centered. That is the easy condition. The whole point of this study is how models behave when the photo is not this clean, so it helps to see the polished version first.

Run:

```
venv\Scripts\python.exe src\02_view_samples.py
```

### Step 3 — Train the baseline classifier (the "Memorizer")

This builds the first of the two models: an ordinary EfficientNet-B0 classifier, trained on the clean photos. It is meant to be the standard, what-everyone-does approach. Its score on clean photos is the baseline that every later result is compared against.

`src/03_train_baseline.py` does three things:

1. Splits the photos into train / val / test, grouped by lesion_id so the same lesion never appears in two piles (no leakage). The split is saved to `data/split.csv` so later steps reuse the exact same hidden test set. Sizes: train 6,981 photos, val 1,532, test 1,502.
2. Fine-tunes EfficientNet-B0 (pretrained on general images) on the 7 classes for 5 epochs, with class weights so the lopsided data does not push it to just answer "mole".
3. Grades the best version on the hidden test set.

**Results on the clean test set (1,502 photos it never saw):**

- Overall accuracy: **76.2%**
- Balanced accuracy, the fair score: **74.5%**

The fair score matters here. A lazy model that always guessed "mole" would score about 67% on plain accuracy but would fall apart on the balanced score. Sitting at 74.5% means the model is genuinely working across all seven types, not gaming the common one.

Per class, it is strong on the harmless mole and weak exactly where it matters most:

| type | recall (share it caught) |
|------|------|
| nv (harmless mole) | 81% |
| mel (melanoma) | 56% |
| bkl | 65% |
| bcc | 73% |
| akiec | 81% |
| vasc | 81% |
| df | 85% |

**The finding that sets up the whole study:** even on these clean, easy photos, the baseline catches only 56% of melanomas. Of the 167 real melanomas in the test set, it called 35 of them a harmless mole — the exact mistake that would tell a sick person they are fine. It also made the opposite, milder error: 95 harmless moles flagged as melanoma.

So before we touch image quality at all, the standard model is already shaky on the dangerous class. That is the starting point. The rest of the study asks what happens to this when photos get worse, and whether a retrieval approach is more honest about its own uncertainty.

Outputs: `models/baseline_efficientnet_b0.pt`, `results/baseline_metrics.txt`, `results/baseline_confusion.png`.

Run:

```
venv\Scripts\python.exe src\03_train_baseline.py
```

Next: Step 4 — build the second model (the case-retrieval "Lookup") and measure it on the same clean test set.




