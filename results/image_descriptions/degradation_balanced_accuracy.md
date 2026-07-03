# degradation_balanced_accuracy.png

**File:** `results/degradation_balanced_accuracy.png`
**Step:** 5 — the crash test

**What it shows:** Five panels, one per damage type (blur, jpeg, lowlight, noise, rotate). Each plots balanced accuracy (the fair overall score across all seven classes) as the damage strength rises from 0 (clean) to 3 (severe). Two lines: the Memorizer (blue) and the Lookup (orange).

**How to read it:** Higher is better. A line that drops steeply means the model is fragile to that kind of damage; a flat line means it is robust.

**Takeaway:** The result is honestly mixed on overall accuracy. Under low light, noise, and rotation, the Memorizer nosedives while the Lookup stays nearly flat and overtakes it. Under blur and JPEG, both fall but the Memorizer keeps its lead. So neither model wins outright on aggregate accuracy, which is why the melanoma-specific curve (the next figure) matters more.
