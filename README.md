# Skin-lesion screening: how reliable is it when the photo isn't perfect?

When a skin photo is low quality, the kind a real person takes on a phone (blurry, dark, slightly tilted), does an AI screening model still give trustworthy answers? And does a model that works by comparing against past cases do any better at *knowing when it is unsure*, instead of being confidently wrong?

This README is the working book : what has actually been done, in order, so anyone (including me, a month from now) can follow along or reproduce it.

# How this differs from other research
None of the individual ingredients is new: skin classification, corruption robustness, calibration, and image retrieval have all been studied. The contribution is the combination and two specific findings:

Trust as the target, not accuracy. We measure robustness and calibration together, on the same degraded images, for the dangerous class specifically.
Aggregate accuracy hides the melanoma collapse. A model can look fine on the headline number while silently failing on the class that matters. We show this directly.
A retrieval model's honesty under distribution shift depends on its confidence signal. Vote share is miscalibrated; a distance-based, shift-calibrated confidence stays well-calibrated (ECE ≈ 0.03) even under heavy corruption. This is the least-explored point and the sharpest candidate for genuine novelty.
An honest correction is part of the result. The first confidence signal failed; we diagnosed why and fixed it, rather than reporting only the version that worked.

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
│   ├── 03_train_baseline.py      Step 3: train the baseline classifier
│   ├── 04_retrieval_dinov2.py    Step 4: the DINOv2 retrieval model
│   ├── 05_crash_test.py          Step 5: degrade photos, re-run both models
│   ├── 06_honesty_test.py        Step 6: is the confidence honest (calibration)?
│   ├── 07_lookup_distance_confidence.py  Step 7: distance-based confidence
│   ├── 08_selective_prediction.py  Step 8: refer-when-unsure (honest negative)
│   └── 09_realworld_pad.py        Step 9: test on real smartphone photos
├── models\
│   ├── baseline_efficientnet_b0.pt   the trained baseline (Step 3)
│   └── dino_library.npz              train fingerprints (Step 4)
└── results\
    ├── sample_photos.png         Step 2 output: the labeled sample grid
    ├── baseline_metrics.txt      Step 3 output: the scores
    ├── baseline_confusion.png    Step 3 output: confusion matrix
    ├── retrieval_metrics.txt     Step 4 output: the scores
    ├── retrieval_confusion.png   Step 4 output: confusion matrix
    ├── degradation_results.csv            Step 5 output: every score at every level
    ├── degradation_balanced_accuracy.png  Step 5 output: accuracy-vs-damage curves
    ├── degradation_melanoma_recall.png    Step 5 output: melanoma-vs-damage curves
    ├── calibration_results.csv            Step 6 output: confidence vs accuracy
    ├── calibration_overconfidence.png     Step 6 output: overconfidence curves
    ├── calibration_reliability.png        Step 6 output: reliability diagrams
    ├── calibration_lookup_compare.csv     Step 7 output: the three confidences
    ├── calibration_lookup_reliability.png Step 7 output: honesty comparison
    ├── selective_prediction.csv           Step 8 output: refer-when-unsure
    ├── selective_prediction.png           Step 8 output: refer-when-unsure curves
    ├── pad_realworld_metrics.txt          Step 9 output: real-photo scores + OOD
    └── pad_ood_detection.png              Step 9 output: knows-it-is-unfamiliar
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

### Step 4 — The case-retrieval model (the "Lookup")

This builds the second model. Instead of learning a single decision rule, it decides by comparison, like a doctor flipping through an album of past cases. `src/04_retrieval_dinov2.py`:

1. Uses DINOv2 (a pretrained vision model) to turn every training photo into a "fingerprint" — a list of numbers describing how it looks — and stores them as the library.
2. Fingerprints each test photo, finds its most similar cases in the library, and votes, weighted so the common "mole" class does not dominate.
3. Uses the val set to pick k (how many neighbours to consult); k=5 won. Then grades on the same 1,502 hidden test photos as the baseline.

**Results on the clean test set (same photos as Step 3):**

- Overall accuracy: **60.9%**
- Balanced accuracy, the fair score: **52.5%**

Side by side on clean photos:

| model | accuracy | balanced accuracy | melanoma caught |
|-------|----------|-------------------|-----------------|
| Memorizer (classifier, Step 3) | 76.2% | 74.5% | 56% |
| Lookup (DINOv2 retrieval, Step 4) | 60.9% | 52.5% | 47% |

So on clean, easy photos the classifier clearly wins. That is expected, and it is fine. The reason: DINOv2 was trained on everyday photos, not skin, so its fingerprints do not separate the seven lesion types as sharply as a model fine-tuned on skin. Two honest takeaways:

- On clean photos, the specialised classifier is stronger. Both models now have a clean-photo starting line.
- The gap hints that a skin-specific fingerprint-maker (Google's Derm Foundation) would likely lift the Lookup a lot. That is the optional upgrade noted earlier.

But clean-photo accuracy is not the point of this study. The real question is Step 5: when photos degrade, which model falls apart faster, and which one stays honest about being unsure. A model can start lower and still be the more trustworthy one under pressure.

Outputs: `results/retrieval_metrics.txt`, `results/retrieval_confusion.png`, `models/dino_library.npz`.

Run:

```
venv\Scripts\python.exe src\04_retrieval_dinov2.py
```

### Step 5 — The crash test (how each model holds up on bad photos)

This is the core experiment. `src/05_crash_test.py` takes the same 1,502 hidden test photos and damages them on purpose — blur, JPEG compression, low light, noise, and rotation — each at three increasing strengths, then re-runs BOTH models on every damaged version. Both models see the same damaged photo; each then applies its own preprocessing.

**Overall (balanced) accuracy as photos degrade:**

- Under low light, noise, and rotation, the Memorizer nosedives while the Lookup stays nearly flat and overtakes it. Under severe low light, for example, the Memorizer falls from 0.745 to 0.096 while the Lookup holds around 0.45.
- Under blur and JPEG, the Memorizer keeps its lead; both drop, but the classifier stays ahead. So on aggregate accuracy the picture is mixed, not a clean win. This is stated honestly rather than oversold.

**Melanoma detection as photos degrade — the finding that matters:**

This part is not mixed. The Memorizer's ability to catch melanoma collapses toward zero under every kind of damage, while the Lookup stays roughly flat.

| damage (most severe level) | Memorizer catches melanoma | Lookup catches melanoma |
|-----------------------------|----------------------------|-------------------------|
| low light | 0% | 50% |
| noise | 0% | 49% |
| blur | 4% | 40% |
| jpeg | 1% | 34% |
| rotation | 8% | 50% |

Even under blur and JPEG, where the Memorizer wins on overall accuracy, it is missing 96–99% of melanomas while the Lookup still catches about 40%. Overall accuracy hides this; you only see it by looking at melanoma on its own.

**What this means.** The standard classifier looks stronger on clean photos, but on the messy photos people actually take it fails on the one class that can kill someone, while the retrieval model degrades gracefully. The retrieval model started lower yet is far more trustworthy under real-world conditions. That is the study's main result.

Outputs: `results/degradation_results.csv`, `results/degradation_balanced_accuracy.png`, `results/degradation_melanoma_recall.png`.

Run:

```
venv\Scripts\python.exe src\05_crash_test.py
```

### Step 6 — The honesty test (calibration), and a surprise

Being *right* and being *honest* are different things. A safe screening tool should get unsure when a photo is bad, not stay confident while wrong. `src/06_honesty_test.py` records each guess's confidence (the classifier's own probability; the Lookup's neighbour vote share) and measures calibration with ECE (0 = perfectly honest).

The result was the opposite of what we expected:

| ECE (lower = more honest) | Memorizer | Lookup (vote share) |
|---|-----------|---------------------|
| clean photos | 0.08 | 0.23 |
| heavily damaged | 0.14 | 0.31 |

As first built, the classifier was the *better*-calibrated model and the Lookup looked badly overconfident. The reason turned out to be how we measured the Lookup's confidence: with only 5 neighbours, one class almost always dominates the vote, so the vote share sits near 0.84 and cannot say "I'm unsure". That is a measurement flaw, not a property of retrieval, which set up Step 7.

Outputs: `results/calibration_results.csv`, `results/calibration_overconfidence.png`, `results/calibration_reliability.png`.

Run:

```
venv\Scripts\python.exe src\06_honesty_test.py
```

### Step 7 — A real uncertainty signal makes the Lookup honest

The right uncertainty signal for a retrieval model is distance: if a photo's nearest stored cases are far away, the model is in unfamiliar territory and should be unsure. `src/07_lookup_distance_confidence.py` uses the mean similarity to the 5 nearest cases, calibrated into a probability on a degraded held-out val set, then re-measures honesty. The predictions do not change, only the confidence, so accuracy and melanoma recall stay exactly as in Steps 4-5.

| ECE (lower = more honest) | Memorizer | Lookup (vote share) | Lookup (distance, new) |
|---|-----------|---------------------|------------------------|
| clean photos | 0.08 | 0.23 | **0.05** |
| heavily damaged | 0.14 | 0.31 | **0.03** |

The distance-based confidence took the Lookup from the worst calibrated to the best. It drops from about 0.64 on clean photos to 0.45-0.54 on severe damage, tracking its real accuracy, so it now knows when it is unsure. And it stays honest exactly where it matters: under heavy damage the classifier gets worse (0.08 to 0.14) while the Lookup gets better (0.05 to 0.03).

Outputs: `results/calibration_lookup_compare.csv`, `results/calibration_lookup_reliability.png`.

Run:

```
venv\Scripts\python.exe src\07_lookup_distance_confidence.py
```

### Step 8 — Can it refer the hard cases to a doctor? (an honest negative)

`src/08_selective_prediction.py` tests letting each model abstain on its least-confident photos and send them to a human. On overall accuracy this works as expected — referring the unsure cases raises accuracy on the rest. But it did NOT boost melanoma recall for the retrieval model, and the classifier's confidence actually triaged overall accuracy better. The honest lesson: a well-calibrated confidence (Step 7) is not automatically a good ranking signal for which case you are about to get wrong. Kept as a limitation, not a headline.

Outputs: `results/selective_prediction.csv`, `results/selective_prediction.png`.

### Step 9 — The real-world test: real smartphone photos (PAD-UFES-20)

The strongest, most real part of the study. Instead of simulated damage, both HAM-trained models were run — with no retraining — on PAD-UFES-20: 2,106 real smartphone photos of skin lesions (shared classes; details in `datasets.md`). HAM is dermatoscopic; PAD is macroscopic phone photos, so this is a large, genuine real-world shift.

| on real phone photos | Memorizer | Lookup |
|-----------------------|-----------|--------|
| melanoma caught | 11.5% | 40.4% |
| knows the photos are unfamiliar (OOD AUROC) | 0.72 | 0.94 |

1. The retrieval model catches about 3.5x more melanomas on real phone photos (40% vs 11%) — the robustness result survives a real dataset shift, not just simulated corruption.
2. The retrieval model knows it is out of its depth (AUROC 0.94): its similarity to known cases drops on the unfamiliar photos, while the classifier stays confident on photos it cannot handle (AUROC 0.72) and fails silently. The figure `pad_ood_detection.png` shows the classifier's familiar and unfamiliar curves overlapping, while the retrieval model's separate cleanly.

Honest caveat: overall accuracy is low for both (~28%) — a dermatoscope-trained model on phone photos is a hard shift, and neither is deployable. The wins are the melanoma gap and the self-awareness, not raw accuracy.

Outputs: `results/pad_realworld_metrics.txt`, `results/pad_ood_detection.png`.

Run:

```
venv\Scripts\python.exe src\09_realworld_pad.py
```

## The result, in one paragraph

On clean, hospital-quality photos the standard classifier scores higher. But on the degraded photos people actually take, the retrieval "Lookup" model is both more robust — it keeps catching melanoma (about 40-50%) while the classifier collapses toward zero — and more honest — it knows when it is unsure (ECE 0.03 under heavy damage) while the classifier stays overconfident. Done right, the retrieval approach is the more trustworthy design for the real world. That is a concrete, honest answer to the question this project asked: can medical AI be honest about what it does not know?




