# calibration_overconfidence.png

**File:** `results/calibration_overconfidence.png`
**Step:** 6 — the honesty test (first attempt)

**What it shows:** Five panels, one per damage type. Each plots the "overconfidence gap" — the model's average confidence minus its actual accuracy — as damage rises from 0 to 3. Two lines: the Memorizer (blue) and the Lookup using its first confidence signal, the neighbour vote share (orange). A dashed line at 0 marks perfect honesty.

**How to read it:** Above the zero line means overconfident (the model sounds surer than it really is). The higher the line, the worse. Below zero would mean under-confident.

**Takeaway:** The vote-share Lookup sits well above zero almost everywhere (a gap of about 0.22–0.26), meaning it is persistently overconfident. The Memorizer's gap is smaller except where it collapses (severe noise and low light). This figure exposed that the vote-share was a poor way to measure the Lookup's confidence — it can barely drop below ~0.84 — which is what motivated the fix in Step 7.
