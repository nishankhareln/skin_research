"""
Step 3: train the baseline classifier (the "Memorizer") on clean photos.

In order, this script:
  1. Builds a train / val / test split GROUPED BY lesion_id, so the same
     lesion never lands in two piles (no cheating). The split is saved to
     data/split.csv so every later step uses the exact same hidden test set.
  2. Fine-tunes an EfficientNet-B0 (already trained on general images) on the
     7 skin-lesion classes, with class weights to fight the lopsided data.
  3. Grades it on the hidden test pile: overall accuracy AND balanced accuracy
     (the fair score that does not let "always guess mole" win), plus a
     per-class report and a confusion matrix.

Outputs:
  data/split.csv                      the train/val/test assignment
  models/baseline_efficientnet_b0.pt  trained weights (best val epoch)
  results/baseline_metrics.txt        the scores
  results/baseline_confusion.png      confusion-matrix figure

Run:  venv\Scripts\python.exe src\03_train_baseline.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True   # never crash on a half-read jpg

# ---------- paths (everything stays inside D:\skin_research) ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# ---------- fixed settings ----------
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
LABELS = list(range(len(CLASSES)))
SEED = 42
IMG_SIZE = 224
BATCH = 32
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 4 # photo-loading helpers. set to 0 if Windows complains.


# ---------- the photo loader ----------
# NOTE: this MUST sit at the top level of the file (not inside a function),
# otherwise Windows cannot hand it to the loading helpers. That was the bug
# in the first attempt.
class HAM(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMAGES / f"{row['image_id']}.jpg").convert("RGB")
        return self.tf(img), CLS2IDX[row["dx"]]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("device:", device, flush=True)

    meta = pd.read_csv(DATA / "HAM10000_metadata")

    # ---- split by lesion_id so the same lesion can't leak across piles ----
    lesion = meta.groupby("lesion_id")["dx"].first().reset_index()
    train_les, tmp_les = train_test_split(
        lesion, test_size=0.30, random_state=SEED, stratify=lesion["dx"])
    val_les, test_les = train_test_split(
        tmp_les, test_size=0.50, random_state=SEED, stratify=tmp_les["dx"])
    split_of = {}
    for lid in train_les["lesion_id"]:
        split_of[lid] = "train"
    for lid in val_les["lesion_id"]:
        split_of[lid] = "val"
    for lid in test_les["lesion_id"]:
        split_of[lid] = "test"
    meta["split"] = meta["lesion_id"].map(split_of)
    meta[["image_id", "lesion_id", "dx", "split"]].to_csv(DATA / "split.csv", index=False)
    print("split sizes (photos):", flush=True)
    print(meta["split"].value_counts(), flush=True)

    # ---- image pipelines ----
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), norm])

    tr = meta[meta.split == "train"]
    va = meta[meta.split == "val"]
    te = meta[meta.split == "test"]
    train_dl = DataLoader(HAM(tr, train_tf), batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=use_amp)
    val_dl = DataLoader(HAM(va, eval_tf), batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=use_amp)
    test_dl = DataLoader(HAM(te, eval_tf), batch_size=BATCH, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=use_amp)

    # ---- class weights so it can't win by always shouting "mole" ----
    counts = tr["dx"].value_counts().reindex(CLASSES).to_numpy(dtype=float)
    weights = counts.sum() / (len(CLASSES) * counts)
    class_w = torch.tensor(weights, dtype=torch.float32, device=device)
    print("class weights:", dict(zip(CLASSES, np.round(weights, 2))), flush=True)

    # ---- model: pretrained EfficientNet-B0 with a fresh 7-way head ----
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def predict(dl):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    out = model(x)
                ps.append(out.argmax(1).cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    # ---- train, keep the best epoch (by balanced accuracy on val) ----
    best_bal = -1.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        yv, pv = predict(val_dl)
        acc = accuracy_score(yv, pv)
        bal = balanced_accuracy_score(yv, pv)
        print(f"epoch {ep}/{EPOCHS}: train_loss={running/len(tr):.3f}  "
              f"val_acc={acc:.3f}  val_balanced_acc={bal:.3f}", flush=True)
        if bal > best_bal:
            best_bal = bal
            torch.save(model.state_dict(), MODELS / "baseline_efficientnet_b0.pt")

    # ---- final grade on the hidden test pile (best saved model) ----
    model.load_state_dict(torch.load(MODELS / "baseline_efficientnet_b0.pt",
                                     weights_only=True))
    yt, pt = predict(test_dl)
    acc = accuracy_score(yt, pt)
    bal = balanced_accuracy_score(yt, pt)
    report = classification_report(yt, pt, labels=LABELS, target_names=CLASSES,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(yt, pt, labels=LABELS)

    txt = "\n".join([
        "BASELINE EfficientNet-B0 on the clean HAM10000 test set",
        f"test photos: {len(yt)}",
        f"overall accuracy:   {acc:.3f}",
        f"balanced accuracy:  {bal:.3f}   (the fair score)",
        "",
        report,
    ])
    print("\n" + txt, flush=True)
    (RESULTS / "baseline_metrics.txt").write_text(txt)

    # ---- confusion-matrix figure ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(LABELS); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (correct answer)")
    ax.set_title("baseline confusion matrix (clean test set)")
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_confusion.png", dpi=110)
    print("\nsaved: models/baseline_efficientnet_b0.pt, "
          "results/baseline_metrics.txt, results/baseline_confusion.png", flush=True)


if __name__ == "__main__":
    main()
