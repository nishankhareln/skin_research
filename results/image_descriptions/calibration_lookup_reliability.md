# calibration_lookup_reliability.png

**File:** `results/calibration_lookup_reliability.png`
**Step:** 7 — a real uncertainty signal makes the Lookup honest

**What it shows:** Two panels — clean photos (left) and heavily damaged photos (right). Reliability diagrams comparing three confidences: the Memorizer (blue), the Lookup's old vote-share confidence (orange), and the Lookup's new distance-based confidence (green). The dashed diagonal is perfect honesty, and each line's ECE is in the legend.

**How to read it:** The closer a line hugs the diagonal, the more honest that confidence is. Below the diagonal means overconfident.

**Takeaway:** This figure completes the study. The new distance-based Lookup confidence (green) hugs the diagonal in both panels and is the best calibrated of all three: ECE 0.05 on clean photos and 0.03 under heavy damage. It stays honest exactly where the Memorizer gets worse (0.08 → 0.14). The fix — using how far away the nearest cases are, instead of the vote share — turned the Lookup from the worst-calibrated model into the best. Combined with Step 5, the retrieval model is now both more robust and more honest on degraded real-world photos.
