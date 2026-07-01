"""
Step 4: the case-retrieval "Lookup" model (DINOv2 fingerprints + kNN vote).

Uses the SAME split as Step 3 (data/split.csv) so the comparison is fair.

In order:
  1. Turn every TRAIN photo into a DINOv2 fingerprint -> the library.
  2. Turn every VAL and TEST photo into a fingerprint too.
  3. For a query photo, find the most similar library fingerprints (cosine)
     and vote, with inverse-frequency weighting so the common "mole" class
     does not dominate (mirrors the class weights used by the baseline).
  4. Pick k on the val set, then grade on the hidden test set with the SAME
     two scores as the baseline: accuracy and balanced accuracy.

Outputs:
  results/retrieval_metrics.txt
  results/retrieval_confusion.png
  models/dino_library.npz    (train fingerprints + labels, reused in Step 5)

Run:  venv\Scripts\python.exe src\04_retrieval_dinov2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel
from PIL import Image, ImageFile
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
LABELS = list(range(len(CLASSES)))
MODEL_NAME = "facebook/dinov2-small"      # swap to dinov2-base for stronger features
BATCH = 32
NUM_WORKERS = 4
K_GRID = [1, 5, 9, 15, 25]

# DINOv2 preprocessing: shortest edge 256, center crop 224, ImageNet normalize
_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Photos(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMAGES / f"{row['image_id']}.jpg").convert("RGB")
        return _tf(img), CLS2IDX[row["dx"]]


@torch.no_grad()
def embed(model, df, device):
    dl = DataLoader(Photos(df), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    embs, labs = [], []
    for x, y in dl:
        x = x.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = model(pixel_values=x)
        e = out.pooler_output
        if e is None:
            e = out.last_hidden_state[:, 0]      # CLS token fallback
        embs.append(e.float().cpu().numpy())
        labs.append(y.numpy())
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)   # unit length -> cosine = dot
    return E, np.concatenate(labs)


def knn_predict(train_E, train_y, query_E, k, inv_freq):
    sims = query_E @ train_E.T                       # cosine similarity
    topk = np.argpartition(-sims, k, axis=1)[:, :k]  # k most similar neighbours
    preds = np.empty(len(query_E), dtype=int)
    for i in range(len(query_E)):
        nbr = topk[i]
        votes = np.zeros(len(CLASSES))
        for j, n in enumerate(nbr):
            votes[train_y[n]] += sims[i, n]          # weight by similarity
        votes *= inv_freq                            # balance rare classes
        preds[i] = int(votes.argmax())
    return preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    split = pd.read_csv(DATA / "split.csv")
    tr = split[split.split == "train"]
    va = split[split.split == "val"]
    te = split[split.split == "test"]

    print(f"loading {MODEL_NAME} ...", flush=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("fingerprinting train / val / test ...", flush=True)
    train_E, train_y = embed(model, tr, device)
    val_E, val_y = embed(model, va, device)
    test_E, test_y = embed(model, te, device)
    np.savez(MODELS / "dino_library.npz", E=train_E, y=train_y,
             image_id=tr["image_id"].to_numpy())

    counts = np.array([(train_y == c).sum() for c in range(len(CLASSES))], dtype=float)
    counts[counts == 0] = 1
    inv_freq = 1.0 / counts

    # pick k on the val set (by balanced accuracy)
    best_k, best_bal = K_GRID[0], -1.0
    for k in K_GRID:
        pv = knn_predict(train_E, train_y, val_E, k, inv_freq)
        b = balanced_accuracy_score(val_y, pv)
        print(f"  k={k:>2}: val_balanced_acc={b:.3f}", flush=True)
        if b > best_bal:
            best_bal, best_k = b, k
    print("chosen k:", best_k, flush=True)

    # grade on the hidden test set
    pt = knn_predict(train_E, train_y, test_E, best_k, inv_freq)
    acc = accuracy_score(test_y, pt)
    bal = balanced_accuracy_score(test_y, pt)
    report = classification_report(test_y, pt, labels=LABELS,
                                   target_names=CLASSES, digits=3, zero_division=0)
    cm = confusion_matrix(test_y, pt, labels=LABELS)

    txt = "\n".join([
        f"RETRIEVAL (DINOv2 {MODEL_NAME} + kNN vote, k={best_k}) on the clean HAM10000 test set",
        f"test photos: {len(test_y)}",
        f"overall accuracy:   {acc:.3f}",
        f"balanced accuracy:  {bal:.3f}   (the fair score)",
        "",
        report,
    ])
    print("\n" + txt, flush=True)
    (RESULTS / "retrieval_metrics.txt").write_text(txt)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(LABELS); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (correct answer)")
    ax.set_title(f"retrieval confusion matrix (clean test set, k={best_k})")
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS / "retrieval_confusion.png", dpi=110)
    print("\nsaved: results/retrieval_metrics.txt, results/retrieval_confusion.png, "
          "models/dino_library.npz", flush=True)


if __name__ == "__main__":
    main()
