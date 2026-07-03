# degradation_melanoma_recall.png

**File:** `results/degradation_melanoma_recall.png`
**Step:** 5 — the crash test (the clinically important view)

**What it shows:** The same five-panel layout as the balanced-accuracy figure, but the y-axis is melanoma recall — the share of real melanomas the model actually catches — as damage rises from 0 (clean) to 3 (severe). Two lines: the Memorizer (blue) and the Lookup (orange).

**How to read it:** Higher is safer, because it means fewer missed melanomas. This is the curve that matters clinically, since a missed melanoma is the dangerous error.

**Takeaway:** This is the core finding of the study. The Memorizer's melanoma detection collapses toward zero under every damage type — 0% under severe low light and noise, and just 1–8% under severe blur, JPEG, and rotation. The Lookup stays roughly flat at 40–50% throughout. Crucially, even under blur and JPEG, where the Memorizer looked fine on overall accuracy, it is missing almost every melanoma while the Lookup is not. Overall accuracy hides this; the melanoma view reveals it.
