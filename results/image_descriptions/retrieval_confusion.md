# retrieval_confusion.png

**File:** `results/retrieval_confusion.png`
**Step:** 4 — the retrieval model (the "Lookup")

**What it shows:** The confusion matrix of the retrieval model (DINOv2 fingerprints + nearest-neighbour vote, k=5) on the same clean test set of 1,502 photos. Same layout as the baseline matrix; green shading.

**How to read it:** Diagonal = correct, off-diagonal = mistakes, darker green = more photos.

**Takeaway:** On clean photos the Lookup is noisier than the classifier. It correctly identifies 690 of 1,004 harmless moles (69%), but mislabels 181 moles as melanoma — many false alarms. Melanoma recall is 79 of 167 (47%), with 30 melanomas missed as harmless moles. This confirms the expected result: a general-purpose retrieval model is weaker than a skin-specific fine-tuned classifier on clean images. Its strength shows up later, under image damage.
