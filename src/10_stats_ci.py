"""
Step 10 (rigor): statistical significance via bootstrap, and a majority-class baseline.

Recomputes per-image predictions for both models on (a) the clean HAM test set
(in-distribution) and (b) the real PAD-UFES-20 photos, then reports 95% bootstrap
confidence intervals for the headline metrics:
  - melanoma recall (clean and PAD), both models
  - balanced accuracy (clean) and accuracy (PAD), both models
  - out-of-distribution AUROC (HAM vs PAD) for the classifier (max-softmax) and
    the retrieval model (mean neighbour similarity)
It also reports a trivial majority-class baseline (always predict 'nv').

No training. Inference only. Output: results/stats_ci.txt

Run:  venv\Scripts\python.exe src\10_stats_ci.py
Quick self-test: set SKIN_SMOKE=1 first.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from transformers import AutoModel
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
SMOKE = os.environ.get("SKIN_SMOKE") == "1"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HAM_IMAGES = DATA / "images"
PAD_IMAGES = DATA / "pad_ufes" / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
MEL = CLS2IDX["mel"]
K = 5
DINO_NAME = "facebook/dinov2-small"
N_BOOT = 200 if SMOKE else 2000
SEED = 42
PAD2HAM = {"BCC": "bcc", "MEL": "mel", "NEV": "nv", "ACK": "akiec", "SEK": "bkl"}

_imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLF_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), _imagenet])
DINO_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), _imagenet])


class ImgSet(Dataset):
    def __init__(self, paths, labels):
        self.paths = list(paths); self.labels = list(labels)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return CLF_TF(img), DINO_TF(img), self.labels[i]


def knn(train_E, train_y, q, k, inv_freq):
    sims = q @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(q), dtype=int); msim = np.empty(len(q))
    for i in range(len(q)):
        nbr = topk[i]; s = sims[i, nbr]
        votes = np.zeros(len(CLASSES))
        for j, n in enumerate(nbr):
            votes[train_y[n]] += s[j]
        votes *= inv_freq
        preds[i] = int(votes.argmax()); msim[i] = s.mean()
    return preds, msim


@torch.no_grad()
def run(paths, labels, clf, dino, train_E, train_y, inv_freq, device):
    dl = DataLoader(ImgSet(paths, labels), batch_size=32, shuffle=False,
                    num_workers=4, pin_memory=(device == "cuda"))
    embs, ys, cpred, cconf = [], [], [], []
    for xc, xd, y in dl:
        xc, xd = xc.to(device), xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = clf(xc); d = dino(pixel_values=xd)
        probs = out.float().softmax(1).cpu().numpy()
        cpred.append(probs.argmax(1)); cconf.append(probs.max(1))
        e = d.pooler_output
        if e is None:
            e = d.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy()); ys.append(np.asarray(y))
    E = np.concatenate(embs); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    y = np.concatenate(ys)
    rpred, msim = knn(train_E, train_y, E, K, inv_freq)
    return {"y": y, "cpred": np.concatenate(cpred), "cconf": np.concatenate(cconf),
            "rpred": rpred, "msim": msim}


RNG = np.random.default_rng(SEED)

def ci(fn, n):
    """Bootstrap 95% CI for a statistic fn(resampled_index) over n items."""
    vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        vals.append(fn(idx))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi

def recall_ci(y, pred, cls):
    pos = np.where(y == cls)[0]
    point = float((pred[pos] == cls).mean())
    lo, hi = ci(lambda idx: (pred[pos[idx]] == cls).mean(), len(pos))
    return point, lo, hi

def metric_ci(y, pred, fn):
    point = float(fn(y, pred))
    lo, hi = ci(lambda idx: fn(y[idx], pred[idx]), len(y))
    return point, lo, hi

def auroc_ci(labels, scores):
    point = float(roc_auc_score(labels, scores))
    n = len(labels)
    def stat(idx):
        yl = labels[idx]
        if yl.min() == yl.max():
            return np.nan
        return roc_auc_score(yl, scores[idx])
    vals = [stat(RNG.integers(0, n, n)) for _ in range(N_BOOT)]
    vals = [v for v in vals if not np.isnan(v)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi

def fmt(name, p, lo, hi):
    return f"{name:42s} {p:.3f}  (95% CI {lo:.3f}-{hi:.3f})"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| smoke:", SMOKE, flush=True)
    RESULTS.mkdir(exist_ok=True)

    split = pd.read_csv(DATA / "split.csv")
    ham = split[split.split == "test"]
    hpaths = [HAM_IMAGES / f"{i}.jpg" for i in ham.image_id]
    hlab = [CLS2IDX[d] for d in ham.dx]

    pad = pd.read_csv(DATA / "pad_ufes" / "metadata.csv")
    pad = pad[pad.diagnostic.isin(PAD2HAM)]
    ppaths = [PAD_IMAGES / f for f in pad.img_id]
    plab = [CLS2IDX[PAD2HAM[d]] for d in pad.diagnostic]
    if SMOKE:
        hpaths, hlab = hpaths[:80], hlab[:80]
        ppaths, plab = ppaths[:80], plab[:80]

    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    clf = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, len(CLASSES))
    clf.load_state_dict(torch.load(MODELS / "baseline_efficientnet_b0.pt", weights_only=True))
    clf = clf.to(device).eval()
    lib = np.load(MODELS / "dino_library.npz", allow_pickle=True)
    train_E, train_y = lib["E"], lib["y"]
    counts = np.array([(train_y == c).sum() for c in range(len(CLASSES))], dtype=float)
    counts[counts == 0] = 1
    inv_freq = 1.0 / counts

    print("running both models on clean HAM test ...", flush=True)
    h = run(hpaths, hlab, clf, dino, train_E, train_y, inv_freq, device)
    print("running both models on real PAD photos ...", flush=True)
    p = run(ppaths, plab, clf, dino, train_E, train_y, inv_freq, device)

    L = ["BOOTSTRAP 95% CONFIDENCE INTERVALS (" + str(N_BOOT) + " resamples)", ""]

    # majority baseline (clean)
    hy = h["y"]; maj = np.zeros_like(hy)
    L.append("Majority-class baseline (always 'nv'), clean HAM test:")
    L.append(f"  accuracy={accuracy_score(hy, maj):.3f}  balanced={balanced_accuracy_score(hy, maj):.3f}  melanoma recall=0.000")
    L.append("")

    L.append("Clean HAM test (in-distribution):")
    L.append("  " + fmt("Classifier melanoma recall", *recall_ci(h["y"], h["cpred"], MEL)))
    L.append("  " + fmt("Retrieval  melanoma recall", *recall_ci(h["y"], h["rpred"], MEL)))
    L.append("  " + fmt("Classifier balanced accuracy", *metric_ci(h["y"], h["cpred"], balanced_accuracy_score)))
    L.append("  " + fmt("Retrieval  balanced accuracy", *metric_ci(h["y"], h["rpred"], balanced_accuracy_score)))
    L.append("")

    L.append("Real PAD-UFES-20 smartphone photos:")
    L.append("  " + fmt("Classifier melanoma recall", *recall_ci(p["y"], p["cpred"], MEL)))
    L.append("  " + fmt("Retrieval  melanoma recall", *recall_ci(p["y"], p["rpred"], MEL)))
    L.append("  " + fmt("Classifier accuracy", *metric_ci(p["y"], p["cpred"], accuracy_score)))
    L.append("  " + fmt("Retrieval  accuracy", *metric_ci(p["y"], p["rpred"], accuracy_score)))
    L.append("")

    # OOD AUROC (HAM=in=0, PAD=out=1); higher score = more OOD
    ood_lab = np.concatenate([np.zeros(len(h["y"])), np.ones(len(p["y"]))])
    clf_ood = np.concatenate([1 - h["cconf"], 1 - p["cconf"]])
    ret_ood = np.concatenate([-h["msim"], -p["msim"]])
    L.append("Out-of-distribution detection AUROC (clean HAM vs real PAD):")
    L.append("  " + fmt("Classifier (max-softmax)", *auroc_ci(ood_lab, clf_ood)))
    L.append("  " + fmt("Retrieval (mean similarity)", *auroc_ci(ood_lab, ret_ood)))

    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    (RESULTS / "stats_ci.txt").write_text(txt)
    print("\nsaved: results/stats_ci.txt", flush=True)
    if SMOKE:
        print("SMOKE STEP10 PASSED", flush=True)


if __name__ == "__main__":
    main()
