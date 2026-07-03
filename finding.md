# Findings

**Project:** Reliable skin-lesion screening when the photo isn't perfect.
**Question in one line:** on the low-quality photos people actually take, which model is more trustworthy — a standard classifier, or a case-retrieval model — measured by both accuracy and honesty about its own uncertainty?

---

## 1. What we tried to find out (the gap)

Almost every skin-cancer model is trained and judged on clean, hospital-grade dermatoscopic images. Very little work asks the practical question underneath: what happens on the blurry, dark, tilted photos a real person uploads from a phone, and does the *way* the model is built change how badly it breaks?

Two things are almost never reported together:
1. Does the model still catch the dangerous class (melanoma) when the photo degrades?
2. When it is wrong, does it *know* it is unsure, or does it stay confident?

For a screening tool, a confident wrong answer is worse than no answer, because it tells a sick person not to see a doctor. That trust question is the gap we set out to fill.

## 2. What we did

- **Data:** HAM10000 — 10,015 clean, doctor-diagnosed dermatoscopic images across 7 lesion types (details in `datasets.md`). Split by lesion into train 6,981 / val 1,532 / test 1,502, so the same lesion never leaks across piles.
- **Two designs, same data:**
  - **Memorizer** — a fine-tuned EfficientNet-B0 classifier (the standard approach).
  - **Lookup** — a retrieval model: DINOv2 image fingerprints plus a nearest-neighbour vote against the training library.
- **Crash test:** degraded the test photos on purpose — blur, JPEG compression, low light, noise, rotation — each at three increasing strengths, and re-ran both models.
- **Honesty test:** measured calibration (ECE and reliability) for both, on clean and degraded photos.
- **A correction:** the Lookup's first confidence signal (neighbour vote share) failed the honesty test, so we replaced it with a distance-based confidence, calibrated on a degraded validation set, and re-measured.
- Throughout, we used balanced accuracy (not plain accuracy) and tracked melanoma separately, because the data is two-thirds ordinary moles.

## 3. What we found

**Finding 1 — On clean photos, the classifier wins.**

| clean test set | accuracy | balanced accuracy | melanoma caught |
|----------------|----------|-------------------|-----------------|
| Memorizer | 76.2% | 74.5% | 56% |
| Lookup | 60.9% | 52.5% | 47% |

Expected: a model fine-tuned on skin beats a general-purpose retrieval model on easy images.

**Finding 2 — On degraded photos, the classifier's melanoma detection collapses; the Lookup holds. And overall accuracy hides it.**

Melanoma caught at the most severe damage level:

| damage (severe) | Memorizer | Lookup |
|-----------------|-----------|--------|
| low light | 0% | 50% |
| noise | 0% | 49% |
| blur | 4% | 40% |
| jpeg | 1% | 34% |
| rotation | 8% | 50% |

Even under blur and JPEG, where the Memorizer keeps a decent *overall* score, it is missing 96–99% of melanomas while the Lookup still catches about 40%. The danger is invisible in aggregate accuracy and only shows up when melanoma is looked at on its own.

**Finding 3 — Done right, the Lookup is also the most honest.**

Calibration, measured by ECE (lower = more honest):

| ECE | on clean photos | on heavily damaged photos |
|-----|-----------------|---------------------------|
| Memorizer | 0.08 | 0.14 (worse) |
| Lookup, vote-share confidence | 0.23 | 0.31 (worst) |
| Lookup, distance-based confidence | **0.05** | **0.03 (best)** |

As first built, the Lookup was badly overconfident, because its vote-share confidence sits near 0.84 no matter what and cannot express doubt. Switching to a distance-based confidence (how far the nearest cases are), calibrated on degraded validation data, made it the best-calibrated of all three. It stays honest exactly where the classifier gets worse.

**Finding 4 — It holds on real smartphone photos (the strongest evidence).**

We also ran both HAM-trained models, with no retraining, on PAD-UFES-20 — 2,106 real smartphone photos of skin lesions from a different dataset (a genuine real-world shift, not simulation). On these real photos:

| on real phone photos | Memorizer | Lookup |
|-----------------------|-----------|--------|
| melanoma caught | 11.5% | 40.4% |
| knows photos are unfamiliar (OOD AUROC) | 0.72 | 0.94 |

The retrieval model catches about 3.5x more melanomas on real phone photos, and reliably knows the photos are unfamiliar (out-of-distribution AUROC 0.94) while the classifier stays confident on data it cannot handle (0.72) and fails silently. Overall accuracy is low for both (~28%), as expected for a dermatoscope-trained model on phone photos; the wins are the melanoma gap and the self-awareness, not raw accuracy.

**The combined result:**

> On clean hospital photos the standard classifier scores higher. But on the degraded photos people actually take, the retrieval model is both more robust — it keeps catching melanoma while the classifier collapses toward zero — and more honest — it knows when it is unsure while the classifier stays overconfident.

## 4. How this differs from other research

None of the individual ingredients is new: skin classification, corruption robustness, calibration, and image retrieval have all been studied. The contribution is the **combination and two specific findings**:

- **Trust as the target, not accuracy.** We measure robustness and calibration together, on the same degraded images, for the dangerous class specifically.
- **Aggregate accuracy hides the melanoma collapse.** A model can look fine on the headline number while silently failing on the class that matters. We show this directly.
- **A retrieval model's honesty under distribution shift depends on its confidence signal.** Vote share is miscalibrated; a distance-based, shift-calibrated confidence stays well-calibrated (ECE ≈ 0.03) even under heavy corruption. This is the least-explored point and the sharpest candidate for genuine novelty.
- **An honest correction is part of the result.** The first confidence signal failed; we diagnosed why and fixed it, rather than reporting only the version that worked.

## 5. Honest limitations

- The controlled degradations are **simulated**; we partly address this by also testing on **real smartphone photos** (PAD-UFES-20), which are macroscopic — a different modality from the dermatoscopic training data, hence the large accuracy drop on them.
- Results are on a **single dataset** (HAM10000); other populations and cameras are untested.
- The retrieval fingerprint is a **general** model (DINOv2). A skin-specific one (for example Google's Derm Foundation) would likely raise accuracy and is a clear next step.
- This is a controlled study, **not a clinical trial**, and nothing here is a deployable product.
- Scope was one week; the calibration map was fit on a degraded validation subset.

## 6. Why it matters

Where dermatologists are scarce, a phone is often the only realistic way a person gets screened, and phone photos are exactly the degraded inputs studied here. A model that only works on clean clinical images is not much use to the people who need screening most. Treating trust — accuracy on the dangerous class *and* honesty about uncertainty — as a first-class goal speaks directly to whether these tools can be used safely outside a hospital.

**Bottom line:** the study gives a concrete, honest answer to the question it asked — *can medical AI be honest about what it does not know?* On degraded real-world photos, a retrieval design with the right uncertainty signal can be, and a standard classifier is not. This holds even on real smartphone photos from a separate dataset (PAD-UFES-20), where the retrieval model both catches more melanomas and reliably flags that the photos are unfamiliar.
