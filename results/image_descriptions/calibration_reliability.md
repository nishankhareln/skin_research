# calibration_reliability.png

**File:** `results/calibration_reliability.png`
**Step:** 6 — the honesty test (first attempt)

**What it shows:** Two panels — clean photos on the left, heavily damaged photos on the right. Each is a reliability diagram: the x-axis is the confidence the model reports, the y-axis is how often it is actually right at that confidence. A dashed diagonal marks perfect honesty. Two lines: the Memorizer (blue) and the Lookup with vote-share confidence (orange), with each model's ECE in the legend (lower ECE = more honest).

**How to read it:** Points sitting ON the diagonal are perfectly calibrated. Points BELOW the diagonal mean the model is overconfident (it claims more confidence than its accuracy earns).

**Takeaway:** The surprise of Step 6. The Memorizer is reasonably calibrated (ECE 0.08 clean, 0.14 damaged), while the vote-share Lookup sits far below the diagonal (ECE 0.23 clean, 0.31 damaged) — badly overconfident. As first built, the classifier was the more honest model. This pointed to a flaw in the Lookup's confidence signal, addressed in the next figure.
