# pad_ood_detection.png

**File:** `results/pad_ood_detection.png`
**Step:** 9 — the real-world test (PAD-UFES-20 smartphone photos)

**What it shows:** Two panels, each comparing a model's signal on its familiar HAM test photos (in-distribution, blue) against the real smartphone photos of PAD-UFES-20 (out-of-distribution, orange). Left: the classifier's confidence (max softmax probability). Right: the retrieval model's mean similarity to its nearest stored cases. Each title gives the out-of-distribution detection AUROC (0.5 = cannot tell familiar from unfamiliar, 1.0 = perfect).

**How to read it:** If the blue (familiar) and orange (unfamiliar) curves are separated, the model can tell that real-world photos are unlike what it was trained on. If they overlap, it cannot.

**Takeaway:** On the left, the classifier's two curves overlap and both pile up near 1.0 — it stays confident on real phone photos it cannot actually handle (AUROC 0.72). On the right, the retrieval model's curves separate cleanly — its similarity drops on the unfamiliar photos (AUROC 0.94). So on real smartphone photos, the retrieval model knows it is out of its depth while the classifier fails silently. On the same photos, the retrieval model also caught 40% of melanomas versus the classifier's 11%.
