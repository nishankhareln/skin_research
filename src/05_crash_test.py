"""
Step 5: the crash test.

Takes the SAME hidden test photos and damages them on purpose, harder and
harder, then re-runs BOTH models and records how each one holds up. This is
the core experiment.

Damage types (each at 3 increasing levels, plus level 0 = clean):
  blur, jpeg compression, low light, noise, rotation.

Both models see the SAME damaged photo; each then applies its own
preprocessing:
  - Memorizer: the trained EfficientNet-B0 classifier (models/baseline_...pt)
  - Lookup:    DINOv2 fingerprints of the damaged photo, voted against the
               CLEAN train library (models/dino_library.npz), k=5 (from Step 4)

Outputs:
  results/degradation_results.csv          every score at every level
  results/degradation_balanced_accuracy.png curves, one panel per damage type
  results/degradation_melanoma_recall.png   the clinically important curve

Run:  venv\Scripts\python.exe src\05_crash_test.py
Quick self-test:  set SKIN_SMOKE=1 first (uses 48 photos, 2 conditions).
"""
import os
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from transformers import AutoModel
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
SMOKE = os.environ.get("SKIN_SMOKE") == "1"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
MEL = CLS2IDX["mel"]
BATCH = 32
NUM_WORKERS = 4
K = 5                       # neighbours to consult (chosen on val in Step 4)
DINO_NAME = "facebook/dinov2-small"

# each damage type at 3 increasing strengths (index 1,2,3); 0 means clean
CORRUPTIONS = {
    "blur":     [0.8, 1.5, 3.0],    # gaussian blur radius (px)
    "jpeg":     [50, 25, 10],       # jpeg quality (lower = worse)
    "lowlight": [0.6, 0.4, 0.2],    # brightness factor (lower = darker)
    "noise":    [10, 25, 45],       # gaussian noise std (0-255)
    "rotate":   [15, 30, 45],       # degrees
}

_imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLF_TF = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(), _imagenet])
DINO_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), _imagenet])


def corrupt(img, ctype, level, idx):
    """Damage a PIL image. level 0 = no change."""
    if level == 0:
        return img
    p = CORRUPTIONS[ctype][level - 1]
    if ctype == "blur":
        return img.filter(ImageFilter.GaussianBlur(p))
    if ctype == "jpeg":
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(p))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if ctype == "lowlight":
        return ImageEnhance.Brightness(img).enhance(p)
    if ctype == "noise":
        rng = np.random.default_rng(1000 * level + idx)   # deterministic
        arr = np.asarray(img).astype(np.float32)
        arr = arr + rng.normal(0, p, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if ctype == "rotate":
        return img.rotate(p, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    raise ValueError(ctype)


class CorruptedTest(Dataset):
    def __init__(self, df, ctype, level):
        self.df = df.reset_index(drop=True)
        self.ctype = ctype
        self.level = level

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMAGES / f"{row['image_id']}.jpg").convert("RGB")
        img = corrupt(img, self.ctype, self.level, i)
        return CLF_TF(img), DINO_TF(img), CLS2IDX[row["dx"]]


def knn_predict(train_E, train_y, query_E, k, inv_freq):
    sims = query_E @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(query_E), dtype=int)
    for i in range(len(query_E)):
        nbr = topk[i]
        votes = np.zeros(len(CLASSES))
        for n in nbr:
            votes[train_y[n]] += sims[i, n]
        votes *= inv_freq
        preds[i] = int(votes.argmax())
    return preds


def scores(y, p):
    m = (y == MEL)
    mel_recall = float((p[m] == MEL).sum() / m.sum()) if m.sum() > 0 else float("nan")
    return accuracy_score(y, p), balanced_accuracy_score(y, p), mel_recall


@torch.no_grad()
def run_condition(df, ctype, level, clf, dino, train_E, train_y, inv_freq, device):
    dl = DataLoader(CorruptedTest(df, ctype, level), batch_size=BATCH,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    clf_preds, embs, ys = [], [], []
    for xc, xd, y in dl:
        xc, xd = xc.to(device), xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            clf_out = clf(xc)
            out = dino(pixel_values=xd)
        e = out.pooler_output
        if e is None:
            e = out.last_hidden_state[:, 0]
        clf_preds.append(clf_out.argmax(1).cpu().numpy())
        embs.append(e.float().cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    clf_p = np.concatenate(clf_preds)
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    ret_p = knn_predict(train_E, train_y, E, K, inv_freq)
    return scores(y, clf_p), scores(y, ret_p)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| smoke:", SMOKE, flush=True)
    RESULTS.mkdir(exist_ok=True)

    split = pd.read_csv(DATA / "split.csv")
    te = split[split.split == "test"]
    if SMOKE:
        te = te.head(48)

    # the Memorizer
    clf = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, len(CLASSES))
    clf.load_state_dict(torch.load(MODELS / "baseline_efficientnet_b0.pt", weights_only=True))
    clf = clf.to(device).eval()

    # the Lookup: DINOv2 + the clean train library
    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    lib = np.load(MODELS / "dino_library.npz", allow_pickle=True)
    train_E, train_y = lib["E"], lib["y"]
    counts = np.array([(train_y == c).sum() for c in range(len(CLASSES))], dtype=float)
    counts[counts == 0] = 1
    inv_freq = 1.0 / counts

    conditions = [("clean", 0)]
    if SMOKE:
        conditions += [("blur", 2)]
    else:
        conditions += [(c, s) for c in CORRUPTIONS for s in (1, 2, 3)]

    res, rows = {}, []
    for ctype, level in conditions:
        clf_s, ret_s = run_condition(te, ctype, level, clf, dino, train_E, train_y, inv_freq, device)
        res[(ctype, level)] = {"clf": clf_s, "ret": ret_s}
        rows.append({
            "corruption": ctype, "severity": level,
            "clf_acc": round(clf_s[0], 4), "clf_balanced": round(clf_s[1], 4),
            "clf_mel_recall": round(clf_s[2], 4),
            "ret_acc": round(ret_s[0], 4), "ret_balanced": round(ret_s[1], 4),
            "ret_mel_recall": round(ret_s[2], 4),
        })
        print(f"{ctype:9s} lvl {level}:  "
              f"Memorizer bal={clf_s[1]:.3f} mel={clf_s[2]:.3f}  |  "
              f"Lookup bal={ret_s[1]:.3f} mel={ret_s[2]:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "degradation_results.csv", index=False)
    print("\nsaved: results/degradation_results.csv", flush=True)

    if SMOKE:
        print("SMOKE STEP5 PASSED", flush=True)
        return

    # ---- plots: one panel per damage type ----
    def plot(metric_idx, title, fname):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()
        clean_clf = res[("clean", 0)]["clf"][metric_idx]
        clean_ret = res[("clean", 0)]["ret"][metric_idx]
        for ax_i, c in enumerate(CORRUPTIONS):
            ax = axes[ax_i]
            xs = [0, 1, 2, 3]
            clf_line = [clean_clf] + [res[(c, s)]["clf"][metric_idx] for s in (1, 2, 3)]
            ret_line = [clean_ret] + [res[(c, s)]["ret"][metric_idx] for s in (1, 2, 3)]
            ax.plot(xs, clf_line, marker="o", label="Memorizer")
            ax.plot(xs, ret_line, marker="s", label="Lookup")
            ax.set_title(c); ax.set_xlabel("damage level"); ax.set_ylabel(title)
            ax.set_ylim(0, 1); ax.set_xticks(xs); ax.legend(fontsize=8)
        axes[-1].axis("off")
        fig.suptitle(f"{title} as photos degrade  (level 0 = clean)")
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=110)

    plot(1, "balanced accuracy", "degradation_balanced_accuracy.png")
    plot(2, "melanoma recall", "degradation_melanoma_recall.png")
    print("saved: results/degradation_balanced_accuracy.png, "
          "results/degradation_melanoma_recall.png", flush=True)


if __name__ == "__main__":
    main()
