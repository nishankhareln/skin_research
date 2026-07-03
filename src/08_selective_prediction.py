"""
Step 8 (Plan 1): selective prediction — "refer to a doctor when unsure".

Both models can now be allowed to ABSTAIN on their least-confident photos and
send those to a human, instead of guessing. Because the retrieval model's
distance-based confidence is honest (Step 7), it should be able to hand off its
likely-wrong cases and keep high melanoma recall on the ones it answers. The
classifier, which stays confident even when wrong, should benefit far less.

Each model is triaged by its OWN confidence:
  - Memorizer: softmax probability of its predicted class.
  - Lookup:    distance-based confidence (mean similarity to nearest cases,
               isotonic-calibrated on a degraded val set, exactly as in Step 7).

We sweep the referral rate (how many least-confident photos go to a human) and
measure melanoma recall and accuracy on the photos each model KEEPS, on clean
photos and on degraded photos.

Outputs:
  results/selective_prediction.csv
  results/selective_prediction.png

Run:  venv\Scripts\python.exe src\08_selective_prediction.py
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
MEL = CLS2IDX["mel"]
SEED = 42
BATCH = 32
NUM_WORKERS = 4
K = 5
DINO_NAME = "facebook/dinov2-small"
VAL_CAL_N = 600

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
    sims = query_E @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(query_E), dtype=int)
    msim = np.empty(len(query_E))
    for i in range(len(query_E)):
        nbr = topk[i]
        s = sims[i, nbr]
        votes = np.zeros(len(CLASSES))
        for j, n in enumerate(nbr):
            votes[train_y[n]] += s[j]
        votes *= inv_freq
        preds[i] = int(votes.argmax())
        msim[i] = s.mean()
    return preds, msim


@torch.no_grad()
def embed(df, ctype, level, dino, device):
    dl = DataLoader(TestImgs(df, ctype, level), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    embs, ys, xcs = [], [], []
    for xc, xd, y in dl:
        xd = xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = dino(pixel_values=xd)
        e = out.pooler_output
        if e is None:
            e = out.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy()); ys.append(y.numpy()); xcs.append(xc)
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E, np.concatenate(ys), xcs


@torch.no_grad()
def clf_conf(xcs, clf, device):
    preds, confs = [], []
    for xc in xcs:
        xc = xc.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = clf(xc)
        probs = out.float().softmax(1).cpu().numpy()
        preds.append(probs.argmax(1)); confs.append(probs.max(1))
    return np.concatenate(preds), np.concatenate(confs)


def selective_curve(conf, pred, y, n_points=11):
    """Sort by confidence; keep the most confident fraction (coverage) and
    report melanoma recall and accuracy on the KEPT cases."""
    order = np.argsort(-conf)
    pred, y = pred[order], y[order]
    N = len(conf)
    rows = []
    for cov in np.linspace(1.0, 0.5, n_points):
        k = max(1, int(round(cov * N)))
        p, t = pred[:k], y[:k]
        m = (t == MEL)
        mel = float((p[m] == MEL).mean()) if m.sum() > 0 else np.nan
        acc = float((p == t).mean())
        rows.append((round(1 - cov, 3), round(cov, 3), mel, acc))   # referral, coverage, mel, acc
    return rows


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

    # fit the distance -> probability map on degraded val (same as Step 7)
    print("fitting confidence calibration on val ...", flush=True)
    v_sim, v_cor = [], []
    for ctype, level in conditions:
        E, y, _ = embed(va, ctype, level, dino, device)
        pred, msim = knn_all(train_E, train_y, E, K, inv_freq)
        v_sim.append(msim); v_cor.append((pred == y).astype(float))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.concatenate(v_sim), np.concatenate(v_cor))

    # collect per-image (conf, pred, y) for both models, clean vs degraded
    pools = {"clean": {"clf": [[], [], []], "ret": [[], [], []]},
             "degraded": {"clf": [[], [], []], "ret": [[], [], []]}}
    for ctype, level in conditions:
        E, y, xcs = embed(te, ctype, level, dino, device)
        ret_pred, msim = knn_all(train_E, train_y, E, K, inv_freq)
        ret_conf = iso.predict(msim)
        c_pred, c_conf = clf_conf(xcs, clf, device)
        bucket = "clean" if level == 0 else "degraded"
        pools[bucket]["clf"][0].append(c_conf); pools[bucket]["clf"][1].append(c_pred); pools[bucket]["clf"][2].append(y)
        pools[bucket]["ret"][0].append(ret_conf); pools[bucket]["ret"][1].append(ret_pred); pools[bucket]["ret"][2].append(y)
        print(f"  processed {ctype} lvl {level}", flush=True)

    rows = []
    curves = {}
    for bucket in ("clean", "degraded"):
        for model, name in [("clf", "Memorizer"), ("ret", "Lookup")]:
            conf = np.concatenate(pools[bucket][model][0])
            pred = np.concatenate(pools[bucket][model][1])
            y = np.concatenate(pools[bucket][model][2])
            curve = selective_curve(conf, pred, y)
            curves[(bucket, model)] = curve
            for referral, coverage, mel, acc in curve:
                rows.append({"photos": bucket, "model": name, "referral_rate": referral,
                             "coverage": coverage, "melanoma_recall_kept": round(mel, 4),
                             "accuracy_kept": round(acc, 4)})

    pd.DataFrame(rows).to_csv(RESULTS / "selective_prediction.csv", index=False)
    print("\nsaved: results/selective_prediction.csv", flush=True)

    # headline: at 20% referral on degraded photos
    def at(bucket, model, referral=0.2):
        for r, cov, mel, acc in curves[(bucket, model)]:
            if abs(r - referral) < 1e-6:
                return mel
        return np.nan
    print("\n--- melanoma recall on the photos KEPT (degraded) ---", flush=True)
    print(f"answer everything (0% referral):  Memorizer={at('degraded','clf',0.0):.2f}  Lookup={at('degraded','ret',0.0):.2f}", flush=True)
    print(f"refer least-confident 20%:         Memorizer={at('degraded','clf',0.2):.2f}  Lookup={at('degraded','ret',0.2):.2f}", flush=True)

    if SMOKE:
        print("SMOKE STEP8 PASSED", flush=True)
        return

    # plot: melanoma recall vs referral rate, clean and degraded
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, bucket in zip(axes, ["clean", "degraded"]):
        for model, name, mk in [("clf", "Memorizer", "o"), ("ret", "Lookup (distance)", "s")]:
            c = curves[(bucket, model)]
            xs = [r[0] for r in c]; ys = [r[2] for r in c]
            ax.plot(xs, ys, marker=mk, label=name)
        ax.set_title(f"{bucket} photos")
        ax.set_xlabel("fraction referred to a doctor")
        ax.set_ylabel("melanoma recall on the photos kept")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Refer the least-confident photos to a human: does melanoma recall improve on the rest?")
    fig.tight_layout()
    fig.savefig(RESULTS / "selective_prediction.png", dpi=110)
    print("saved: results/selective_prediction.png", flush=True)


if __name__ == "__main__":
    main()
