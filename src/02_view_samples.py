"""
Step 2: look at the data.

Reads the HAM10000 answer sheet (the metadata CSV), then saves a grid of
sample photos - a few examples of each of the 7 lesion types, labeled - so
we can see the real images with our own eyes before building any models.

Output: results/sample_photos.png
Run:    venv\Scripts\python.exe src\02_view_samples.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save to a file instead of opening a window
import matplotlib.pyplot as plt
from PIL import Image

# everything stays inside D:\skin_research
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# load the answer sheet
meta = pd.read_csv(DATA / "HAM10000_metadata")
print(f"loaded metadata: {len(meta)} rows")
print("columns:", list(meta.columns))
print("\nhow many photos of each type:")
print(meta["dx"].value_counts())

# plain-English name for each code
NAMES = {
    "nv": "harmless mole (nv)",
    "mel": "MELANOMA (mel)",
    "bkl": "benign keratosis (bkl)",
    "bcc": "basal cell carcinoma (bcc)",
    "akiec": "pre-cancer (akiec)",
    "vasc": "vascular (vasc)",
    "df": "dermatofibroma (df)",
}

classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
n_examples = 4   # photos shown per type

fig, axes = plt.subplots(
    len(classes), n_examples,
    figsize=(n_examples * 2.2, len(classes) * 2.2),
)

for r, cls in enumerate(classes):
    subset = meta[meta["dx"] == cls].head(n_examples)
    for c in range(n_examples):
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c < len(subset):
            img_id = subset.iloc[c]["image_id"]
            ax.imshow(Image.open(IMAGES / f"{img_id}.jpg"))
        else:
            ax.axis("off")
        if c == 0:
            ax.set_ylabel(NAMES[cls], rotation=0, ha="right",
                          va="center", fontsize=9)

fig.suptitle("HAM10000: sample photos by diagnosis", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = RESULTS / "sample_photos.png"
fig.savefig(out, dpi=110)
print(f"\nsaved sample grid to: {out}")
