"""
Step 6: the honesty test (calibration).

Step 5 showed the classifier FAILS on bad photos. This step asks the deeper
question: does it fail while still sounding CONFIDENT? A trustworthy screening
tool should get unsure when the photo is bad, not stay 90% sure while wrong.

For both models, on the clean photos and on every damaged version, we record
each prediction's CONFIDENCE and whether it was right, then measure:
  - ECE (Expected Calibration Error): one number for how far confidence is from
    reality. 0 = perfectly honest; higher = more overconfident/miscalibrated.
  - the overconfidence gap: mean confidence minus actual accuracy. Big positive
    gap = "sounds sure, is wrong".
  - reliability diagrams (confidence vs how often it is actually right).

Confidence:
  - Memorizer (classifier): the softmax probability of its chosen class.
  - Lookup (retrieval):      the share of the (weighted) neighbour vote that
                             went to the chosen class.

Outputs:
  results/calibration_results.csv
  results/calibration_overconfidence.png
  results/calibration_reliability.png

Run:  venv\Scripts\python.exe src\06_honesty_test.py
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
BATCH = 32
NUM_WORKERS = 4
K = 5
DINO_NAME = "facebook/dinov2-small"
N_BINS = 15

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
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(p))
        buf.seek(0)
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


def knn_conf(train_E, train_y, query_E, k, inv_freq):
    """Return (predictions, confidence) where confidence is the winning
    class's share of the weighted neighbour vote."""
    sims = query_E @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(query_E), dtype=int)
    confs = np.empty(len(query_E))
    for i in range(len(query_E)):
        votes = np.zeros(len(CLASSES))
        for n in topk[i]:
            votes[train_y[n]] += sims[i, n]
        votes *= inv_freq
        s = votes.sum()
        p = int(votes.argmax())
        preds[i] = p
        confs[i] = votes[p] / s if s > 0 else 0.0
    return preds, confs


def ece(conf, correct, n_bins=N_BINS):
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


def reliability(conf, correct, n_bins=N_BINS):
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


@torch.no_grad()
def run_condition(df, ctype, level, clf, dino, train_E, train_y, inv_freq, device):
    dl = DataLoader(CorruptedTest(df, ctype, level), batch_size=BATCH,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    ys, clf_pred, clf_conf, embs = [], [], [], []
    for xc, xd, y in dl:
        xc, xd = xc.to(device), xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            clf_out = clf(xc)
            out = dino(pixel_values=xd)
        probs = clf_out.float().softmax(1).cpu().numpy()
        clf_pred.append(probs.argmax(1))
        clf_conf.append(probs.max(1))
        e = out.pooler_output
        if e is None:
            e = out.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    clf_pred = np.concatenate(clf_pred)
    clf_conf = np.concatenate(clf_conf)
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    ret_pred, ret_conf = knn_conf(train_E, train_y, E, K, inv_freq)
    return {"y": y, "clf_pred": clf_pred, "clf_conf": clf_conf,
            "ret_pred": ret_pred, "ret_conf": ret_conf}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| smoke:", SMOKE, flush=True)
    RESULTS.mkdir(exist_ok=True)

    split = pd.read_csv(DATA / "split.csv")
    te = split[split.split == "test"]
    if SMOKE:
        te = te.head(48)

    clf = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, len(CLASSES))
    clf.load_state_dict(torch.load(MODELS / "baseline_efficientnet_b0.pt", weights_only=True))
    clf = clf.to(device).eval()

    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    lib = np.load(MODELS / "dino_library.npz", allow_pickle=True)
    train_E, train_y = lib["E"], lib["y"]
    counts = np.array([(train_y == c).sum() for c in range(len(CLASSES))], dtype=float)
    counts[counts == 0] = 1
    inv_freq = 1.0 / counts

    conditions = [("clean", 0)]
    conditions += [("blur", 2)] if SMOKE else [(c, s) for c in CORRUPTIONS for s in (1, 2, 3)]

    data, rows = {}, []
    for ctype, level in conditions:
        d = run_condition(te, ctype, level, clf, dino, train_E, train_y, inv_freq, device)
        data[(ctype, level)] = d
        for tag, pred, conf in [("clf", d["clf_pred"], d["clf_conf"]),
                                ("ret", d["ret_pred"], d["ret_conf"])]:
            correct = (pred == d["y"]).astype(float)
            acc = correct.mean()
            mconf = conf.mean()
            rows.append({"corruption": ctype, "severity": level, "model": tag,
                         "accuracy": round(float(acc), 4),
                         "mean_confidence": round(float(mconf), 4),
                         "overconfidence_gap": round(float(mconf - acc), 4),
                         "ece": round(float(ece(conf, correct)), 4)})
        print(f"{ctype:9s} lvl {level}:  "
              f"Memorizer conf={data[(ctype,level)]['clf_conf'].mean():.2f} "
              f"acc={(d['clf_pred']==d['y']).mean():.2f}  |  "
              f"Lookup conf={d['ret_conf'].mean():.2f} "
              f"acc={(d['ret_pred']==d['y']).mean():.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "calibration_results.csv", index=False)
    print("\nsaved: results/calibration_results.csv", flush=True)

    if SMOKE:
        print("SMOKE STEP6 PASSED", flush=True)
        return

    # ---- overconfidence gap curves (one panel per damage type) ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    def gap(ctype, level, tag):
        d = data[(ctype, level)]
        pred, conf = d[f"{tag}_pred"], d[f"{tag}_conf"]
        return float(conf.mean() - (pred == d["y"]).mean())
    for ax_i, c in enumerate(CORRUPTIONS):
        ax = axes[ax_i]
        xs = [0, 1, 2, 3]
        clf_line = [gap("clean", 0, "clf")] + [gap(c, s, "clf") for s in (1, 2, 3)]
        ret_line = [gap("clean", 0, "ret")] + [gap(c, s, "ret") for s in (1, 2, 3)]
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.plot(xs, clf_line, marker="o", label="Memorizer")
        ax.plot(xs, ret_line, marker="s", label="Lookup")
        ax.set_title(c); ax.set_xlabel("damage level")
        ax.set_ylabel("confidence - accuracy"); ax.set_xticks(xs); ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Overconfidence as photos degrade  (higher = sounds sure but is wrong)")
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration_overconfidence.png", dpi=110)

    # ---- reliability diagrams: clean vs pooled heavy damage ----
    def pool(level):
        cd = [data[(c, level)] for c in CORRUPTIONS] if level > 0 else [data[("clean", 0)]]
        out = {}
        for tag in ("clf", "ret"):
            conf = np.concatenate([d[f"{tag}_conf"] for d in cd])
            correct = np.concatenate([(d[f"{tag}_pred"] == d["y"]).astype(float) for d in cd])
            out[tag] = (conf, correct)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (level, name) in zip(axes, [(0, "clean photos"), (3, "heavily damaged photos")]):
        p = pool(level)
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfectly honest")
        for tag, label in [("clf", "Memorizer"), ("ret", "Lookup")]:
            xs, ys = reliability(*p[tag])
            ax.plot(xs, ys, marker="o", label=f"{label} (ECE={ece(*p[tag]):.2f})")
        ax.set_title(name); ax.set_xlabel("confidence it reports")
        ax.set_ylabel("how often it is actually right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Reliability: points below the dashed line mean overconfident")
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration_reliability.png", dpi=110)
    print("saved: results/calibration_overconfidence.png, "
          "results/calibration_reliability.png", flush=True)


if __name__ == "__main__":
    main()
