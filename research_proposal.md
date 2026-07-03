# Reliable skin-lesion screening when the photo isn't perfect

A short study comparing a standard image classifier with a case-retrieval approach, tested on the kind of low-quality photos people actually take.

Nishan Kharel
Draft — June 2026

---

## The short version

Almost every skin-cancer screening model is trained and tested on clean clinical images: good camera, steady lighting, the lesion centered in frame. Hand that same model a blurry, dark, or slightly tilted phone photo, which is what a real person uploads, and two things go wrong at once. Accuracy falls, and the model usually stays just as confident while giving the wrong answer.

I want to test a different design. Instead of a model that spits out a single label, I'll build one that pulls up the most similar past cases and decides from them, the way a clinician compares a new mole against ones they have seen before. Then I'll put both designs through the same set of degraded images and measure two things: how much accuracy each one loses as the photo gets worse, and whether it can tell when it is unsure.

## The gap

There is a lot of published work pushing accuracy higher on clean dermatoscopic datasets. There is much less that asks the practical question underneath it: what happens to these models on the photos people will actually send, and does the way we build the model change how badly it breaks?

That second part is the gap I care about. Most papers report one accuracy number on clean test images and stop there. They rarely report what the model does when the input is degraded, and they almost never check whether the model's confidence is honest under those conditions. For a screening tool, a wrong answer delivered with high confidence is worse than no answer, because it tells someone not to see a doctor when they should.

## Two ways to build the screener

There are two common ways to make a model decide "concerning" or "not concerning" from a skin photo. They behave very differently, and that difference is the whole point of this study.

**The standard classifier.** You train one network on thousands of labeled images until it learns to map a photo straight to a label. It is fast and accurate on clean data. The downside is that it gives you one answer and a confidence score with no explanation, and when it fails it tends to fail quietly, still reporting high confidence.

**Case retrieval.** Instead of memorizing a single decision boundary, this approach turns each image into a numerical fingerprint and stores all the known cases. For a new photo, it finds the closest stored cases and lets them vote. It can show the actual images behind its decision, so a person can sanity-check it, and when a photo is too far from anything it has seen, that distance is itself a useful warning sign.

I have built the retrieval pattern before, in a different domain. For a banknote recognition project I used image fingerprints plus a vector database to match folded and worn notes against clean references. The same machinery transfers directly to skin images, which is part of why this is doable quickly.

## The question

Put plainly:

> When skin photos degrade the way real phone photos do, which design keeps getting the right answer, and which one is more honest about being unsure?

## What I will actually do

1. Start from a public, labeled set of skin-lesion images (HAM10000, with the broader ISIC archive as backup). No clinical access or approvals are needed.
2. Build both versions: a fine-tuned classifier as the baseline, and a retrieval system using image embeddings from a pretrained model with nearest-neighbor voting.
3. Confirm both perform comparably on the clean test images, so the comparison is fair.
4. Degrade the test photos on purpose, in controlled steps: blur, JPEG compression, low light, sensor noise, and rotation. This copies common phone-photo problems at measurable severity levels.
5. Re-run both models across every level of degradation and record accuracy.
6. Measure not just accuracy but calibration, meaning whether a model's stated confidence matches how often it is actually right (Expected Calibration Error).
7. Write up the result with two clear figures: accuracy as image quality drops, and a confidence-honesty comparison.

Whatever the outcome, it is a real finding. If retrieval holds up better, that is a useful design recommendation. If it does not, then showing that the popular assumption is wrong is just as worth reporting.

## Why this is worth doing

Where I live, dermatologists are concentrated in a few cities, and for most people a phone is the only realistic way a skin check would ever happen. Those are exactly the conditions where image quality is worst, so a model that only works on clinical images is not much use to the people who need screening most. A study that takes degradation seriously, and that treats a model knowing its own limits as a first-class goal rather than an afterthought, speaks directly to whether these tools can be trusted outside a hospital.

Datasets Information:
10,015 images, ordinary JPGs, about 600×450 pixels each.
These are dermatoscopic images. A dermatoscope is a handheld magnifier with its own light that a dermatologist presses against the skin. So every photo is a clean, evenly-lit, zoomed-in close-up of one lesion.

That's important for your project: this data is the "clean hospital" condition. These are not the messy phone photos your study is about — your degradation steps (blur, low light, noise…) simulate phone photos on top of this clean data.


