"""
Step 7: give the Lookup a real uncertainty signal, then re-test its honesty.

Step 6 measured the Lookup's confidence as its neighbour VOTE SHARE, which is
almost always high (one class dominates 5 neighbours), so it could not express
"I'm unsure" and looked badly overconfident (ECE 0.23 clean / 0.31 damaged).

The natural uncertainty signal for a retrieval model is DISTANCE: if a photo's
nearest stored cases are far away (low similarity), the model is in unfamiliar
territory and should be unsure. Here we:
  1. Use the mean similarity to the 5 nearest cases as the raw signal.
  2. Calibrate it into a real probability with isotonic regression, fit on a
     held-out VAL set that we degrade the same way (so the map has seen low-
     similarity, hard cases).
  3. Re-measure calibration on clean + damaged test photos, and compare three
     confidences: the classifier's, the Lookup's old vote share, and the
     Lookup's new distance-based confidence.

The PREDICTIONS do not change (same weighted vote); only the confidence does,
so accuracy and melanoma recall are exactly as in Steps 4-5.

Outputs:
  results/calibration_lookup_compare.csv
  results/calibration_lookup_reliability.png

Run:  venv\Scripts\python.exe src\07_lookup_distance_confidence.py
Quick self-test:  set SKIN_SMOKE=1 first.
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
from sklearn.isotonic import IsotonicRegression
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
SEED = 42
BATCH = 32
NUM_WORKERS = 4
K = 5
DINO_NAME = "facebook/dinov2-small"
N_BINS = 15
VAL_CAL_N = 600                    # val photos used to fit the calibration map

CORRUPTIONS = {
    "blur":     [0.8, 1.5, 3.0],
    "jpeg":     [50, 25, 10],
    "lowlight": [0.6, 0.4, 0.2],
    "noise":    [10, 25, 45],
    "rotate":   [15, 30, 45],
}

_imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLF_TF = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(), _imagenet])
DINO_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), _imagenet])


def corrupt(img, ctype, level, idx):
    if level == 0:
        return img
    p = CORRUPTIONS[ctype][level - 1]
    if ctype == "blur":
        return img.filter(ImageFilter.GaussianBlur(p))
    if ctype == "jpeg":
        buf = BytesIO(); img.save(buf, format="JPEG", quality=int(p)); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if ctype == "lowlight":
        return ImageEnhance.Brightness(img).enhance(p)
    if ctype == "noise":
        rng = np.random.default_rng(1000 * level + idx)
        arr = np.asarray(img).astype(np.float32) + rng.normal(0, p, (img.size[1], img.size[0], 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if ctype == "rotate":
        return img.rotate(p, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    raise ValueError(ctype)


class TestImgs(Dataset):
    def __init__(self, df, ctype, level):
        self.df = df.reset_index(drop=True); self.ctype = ctype; self.level = level

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMAGES / f"{row['image_id']}.jpg").convert("RGB")
        img = corrupt(img, self.ctype, self.level, i)
        return CLF_TF(img), DINO_TF(img), CLS2IDX[row["dx"]]


def knn_all(train_E, train_y, query_E, k, inv_freq):
    """Return predictions, vote-share confidence, and mean top-k similarity."""
    sims = query_E @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(query_E), dtype=int)
    vote = np.empty(len(query_E))
    msim = np.empty(len(query_E))
    for i in range(len(query_E)):
        nbr = topk[i]
        s = sims[i, nbr]
        votes = np.zeros(len(CLASSES))
        for j, n in enumerate(nbr):
            votes[train_y[n]] += s[j]
        votes *= inv_freq
        tot = votes.sum()
        p = int(votes.argmax())
        preds[i] = p
        vote[i] = votes[p] / tot if tot > 0 else 0.0
        msim[i] = s.mean()
    return preds, vote, msim


def ece(conf, correct, n_bins=N_BINS):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(conf)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


def reliability(conf, correct, n_bins=N_BINS):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


@torch.no_grad()
def embed(df, ctype, level, dino, device):
    dl = DataLoader(TestImgs(df, ctype, level), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    embs, ys, need_clf = [], [], []
    for xc, xd, y in dl:
        xd = xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = dino(pixel_values=xd)
        e = out.pooler_output
        if e is None:
            e = out.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy()); ys.append(y.numpy())
        need_clf.append(xc)
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E, np.concatenate(ys), need_clf


@torch.no_grad()
def clf_confidence(need_clf, clf, device):
    preds, confs = [], []
    for xc in need_clf:
        xc = xc.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = clf(xc)
        probs = out.float().softmax(1).cpu().numpy()
        preds.append(probs.argmax(1)); confs.append(probs.max(1))
    return np.concatenate(preds), np.concatenate(confs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| smoke:", SMOKE, flush=True)
    RESULTS.mkdir(exist_ok=True)

    split = pd.read_csv(DATA / "split.csv")
    va = split[split.split == "val"]
    te = split[split.split == "test"]
    if SMOKE:
        va = va.head(48); te = te.head(48)
    else:
        va = va.sample(n=min(VAL_CAL_N, len(va)), random_state=SEED)

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

    conditions = [("clean", 0)]
    conditions += [("blur", 2)] if SMOKE else [(c, s) for c in CORRUPTIONS for s in (1, 2, 3)]

    # 1) fit the distance -> probability map on degraded VAL
    print("fitting calibration on val (this warms up the map) ...", flush=True)
    v_sim, v_correct = [], []
    for ctype, level in conditions:
        E, y, _ = embed(va, ctype, level, dino, device)
        pred, _, msim = knn_all(train_E, train_y, E, K, inv_freq)
        v_sim.append(msim); v_correct.append((pred == y).astype(float))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.concatenate(v_sim), np.concatenate(v_correct))

    # 2) evaluate on TEST: three confidences for the same runs
    per_img = {}   # (ctype, level) -> dict of arrays
    rows = []
    for ctype, level in conditions:
        E, y, need_clf = embed(te, ctype, level, dino, device)
        ret_pred, vote, msim = knn_all(train_E, train_y, E, K, inv_freq)
        dist_conf = iso.predict(msim)
        clf_pred, clf_conf = clf_confidence(need_clf, clf, device)
        methods = {
            "classifier":  (clf_conf, (clf_pred == y).astype(float)),
            "lookup_vote": (vote,      (ret_pred == y).astype(float)),
            "lookup_dist": (dist_conf, (ret_pred == y).astype(float)),
        }
        per_img[(ctype, level)] = methods
        for name, (conf, correct) in methods.items():
            rows.append({"corruption": ctype, "severity": level, "method": name,
                         "accuracy": round(float(correct.mean()), 4),
                         "mean_confidence": round(float(conf.mean()), 4),
                         "ece": round(float(ece(conf, correct)), 4)})
        print(f"{ctype:9s} lvl {level}:  Lookup mean-sim={msim.mean():.2f} "
              f"-> dist-conf={dist_conf.mean():.2f} (acc={methods['lookup_dist'][1].mean():.2f})",
              flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "calibration_lookup_compare.csv", index=False)
    print("\nsaved: results/calibration_lookup_compare.csv", flush=True)

    if SMOKE:
        print("SMOKE STEP7 PASSED", flush=True)
        return

    # pooled clean vs heavy damage
    def pool(level, name):
        cds = [per_img[(c, level)] for c in CORRUPTIONS] if level > 0 else [per_img[("clean", 0)]]
        conf = np.concatenate([d[name][0] for d in cds])
        cor = np.concatenate([d[name][1] for d in cds])
        return conf, cor

    print("\n--- honesty (ECE, lower = more honest) ---", flush=True)
    for level, tag in [(0, "clean"), (3, "heavy damage")]:
        line = f"{tag:12s}"
        for name in ("classifier", "lookup_vote", "lookup_dist"):
            line += f"  {name}={ece(*pool(level, name)):.2f}"
        print(line, flush=True)

    # reliability figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"classifier": "tab:blue", "lookup_vote": "tab:orange", "lookup_dist": "tab:green"}
    labels = {"classifier": "Memorizer", "lookup_vote": "Lookup (vote share)",
              "lookup_dist": "Lookup (distance, new)"}
    for ax, (level, name) in zip(axes, [(0, "clean photos"), (3, "heavily damaged photos")]):
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfectly honest")
        for m in ("classifier", "lookup_vote", "lookup_dist"):
            conf, cor = pool(level, m)
            xs, ys = reliability(conf, cor)
            ax.plot(xs, ys, marker="o", color=colors[m], label=f"{labels[m]} (ECE={ece(conf, cor):.2f})")
        ax.set_title(name); ax.set_xlabel("confidence it reports")
        ax.set_ylabel("how often it is actually right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Does a distance-based confidence make the Lookup honest?")
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration_lookup_reliability.png", dpi=110)
    print("saved: results/calibration_lookup_reliability.png", flush=True)


if __name__ == "__main__":
    main()
