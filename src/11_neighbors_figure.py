"""
Step 11 (error analysis / interpretability): show the retrieval model's evidence.

For several melanoma test images, display the query alongside the five nearest
stored cases the retrieval model consults, with each neighbour's true label and
similarity, and the model's resulting vote. This is the transparency a classifier
cannot offer, and it doubles as error analysis: rows where the neighbours are
mostly benign are exactly the melanomas the model (and a clinician) would find
hard.

Output: results/retrieval_neighbours.png
Run:  venv\Scripts\python.exe src\11_neighbors_figure.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from transformers import AutoModel
from PIL import Image, ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; IMAGES = DATA / "images"; RESULTS = ROOT / "results"; MODELS = ROOT / "models"
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]; CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
MEL = CLS2IDX["mel"]; K = 5; DINO_NAME = "facebook/dinov2-small"; N_QUERIES = 5
DINO_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    dino = AutoModel.from_pretrained(DINO_NAME).to(dev).eval()
    lib = np.load(MODELS / "dino_library.npz", allow_pickle=True)
    E, ylib, ids = lib["E"], lib["y"], lib["image_id"]
    counts = np.array([(ylib == c).sum() for c in range(len(CLASSES))], dtype=float); counts[counts == 0] = 1
    inv_freq = 1.0 / counts

    split = pd.read_csv(DATA / "split.csv")
    q_ids = split[(split.split == "test") & (split.dx == "mel")].image_id.tolist()[:N_QUERIES]

    fig, axes = plt.subplots(len(q_ids), K + 1, figsize=((K + 1) * 1.7, len(q_ids) * 1.9))
    for r, qid in enumerate(q_ids):
        qimg = Image.open(IMAGES / f"{qid}.jpg").convert("RGB")
        with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
            e = dino(pixel_values=DINO_TF(qimg).unsqueeze(0).to(dev)).pooler_output
        e = e.float().cpu().numpy()[0]; e = e / (np.linalg.norm(e) + 1e-8)
        sims = E @ e
        top = np.argpartition(-sims, K)[:K]; top = top[np.argsort(-sims[top])]
        votes = np.zeros(len(CLASSES))
        for n in top:
            votes[ylib[n]] += sims[n]
        votes *= inv_freq
        pred = CLASSES[int(votes.argmax())]
        ax = axes[r, 0]; ax.imshow(qimg.resize((150, 150))); ax.set_xticks([]); ax.set_yticks([])
        ok = "correct" if pred == "mel" else "MISSED"
        ax.set_ylabel(f"true: mel\nvote: {pred} ({ok})", fontsize=8)
        ax.set_title("query", fontsize=8)
        for c, n in enumerate(top):
            a = axes[r, c + 1]
            nb = Image.open(IMAGES / f"{ids[n]}.jpg").convert("RGB").resize((150, 150))
            a.imshow(nb); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{CLASSES[ylib[n]]}  {sims[n]:.2f}", fontsize=8)
    fig.suptitle("Retrieval evidence: each melanoma query and its 5 nearest stored cases", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RESULTS / "retrieval_neighbours.png", dpi=110)
    print("saved: results/retrieval_neighbours.png", flush=True)


if __name__ == "__main__":
    main()
