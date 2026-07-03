# baseline_confusion.png

**File:** `results/baseline_confusion.png`
**Step:** 3 — the baseline classifier (the "Memorizer")

**What it shows:** The confusion matrix of the baseline EfficientNet-B0 classifier on the clean test set of 1,502 photos. Rows are the true diagnosis, columns are what the model predicted, and the number in each cell is how many photos fell there. Darker blue means more photos.

**How to read it:** The diagonal (top-left to bottom-right) is correct predictions. Everything off the diagonal is a mistake. For example, the top-left cell (816) is harmless moles correctly called moles.

**Takeaway:** Even on clean photos the model confuses the dangerous class. 35 real melanomas were predicted as harmless moles (row `mel`, column `nv`) — the exact mistake that would send a sick person home — and 95 harmless moles were flagged as melanoma (false alarms). Overall melanoma recall is 56%. This is the reference point: the "standard" model is already shaky on melanoma before any photo damage.
