"""
Step 9 (the real-world test): run the HAM-trained models on REAL smartphone
photos (PAD-UFES-20), with no retraining.

HAM10000 is dermatoscopic (lens-magnified); PAD-UFES-20 is macroscopic
smartphone photos. That is a large, real distribution shift - exactly the
"photos people actually take" scenario. We expect BOTH models to lose accuracy.
The real question is whether each model KNOWS it has been handed unfamiliar
data, or fails silently.

We measure:
  1. Accuracy and melanoma recall on PAD (shared classes only), for both models.
  2. Out-of-distribution detection: can each model's own signal separate its
     familiar HAM test set from the unfamiliar PAD photos? (AUROC)
       - Classifier signal: max softmax probability (standard OOD baseline).
       - Retrieval signal: mean similarity to the nearest stored cases.
     A model that "knows" PAD is different will score high AUROC; one that stays
     blindly confident scores near 0.5.

Shared classes (PAD -> HAM): BCC->bcc, MEL->mel, NEV->nv, ACK->akiec, SEK->bkl.
SCC is dropped (no HAM equivalent).

Outputs:
  results/pad_realworld_metrics.txt
  results/pad_ood_detection.png

Run:  venv\Scripts\python.exe src\09_realworld_pad.py
Quick self-test:  set SKIN_SMOKE=1 first.
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BATCH = 32
NUM_WORKERS = 4
K = 5
DINO_NAME = "facebook/dinov2-small"
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


def knn(train_E, train_y, query_E, k, inv_freq):
    sims = query_E @ train_E.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = np.empty(len(query_E), dtype=int)
    msim = np.empty(len(query_E))
    for i in range(len(query_E)):
        nbr = topk[i]; s = sims[i, nbr]
        votes = np.zeros(len(CLASSES))
        for j, n in enumerate(nbr):
            votes[train_y[n]] += s[j]
        votes *= inv_freq
        preds[i] = int(votes.argmax()); msim[i] = s.mean()
    return preds, msim


@torch.no_grad()
def run(paths, labels, clf, dino, train_E, train_y, inv_freq, device):
    dl = DataLoader(ImgSet(paths, labels), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    embs, ys, clf_pred, clf_conf = [], [], [], []
    for xc, xd, y in dl:
        xc, xd = xc.to(device), xd.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = clf(xc); demb = dino(pixel_values=xd)
        probs = out.float().softmax(1).cpu().numpy()
        clf_pred.append(probs.argmax(1)); clf_conf.append(probs.max(1))
        e = demb.pooler_output
        if e is None:
            e = demb.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy()); ys.append(np.asarray(y))
    E = np.concatenate(embs); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    y = np.concatenate(ys)
    ret_pred, msim = knn(train_E, train_y, E, K, inv_freq)
    return {"y": y, "clf_pred": np.concatenate(clf_pred),
            "clf_conf": np.concatenate(clf_conf), "ret_pred": ret_pred, "msim": msim}


def recall_mel(y, pred):
    m = (y == MEL)
    return float((pred[m] == MEL).mean()) if m.sum() > 0 else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| smoke:", SMOKE, flush=True)
    RESULTS.mkdir(exist_ok=True)

    # HAM test = the familiar (in-distribution) set
    split = pd.read_csv(DATA / "split.csv")
    ham = split[split.split == "test"]
    ham_paths = [HAM_IMAGES / f"{i}.jpg" for i in ham.image_id]
    ham_labels = [CLS2IDX[d] for d in ham.dx]

    # PAD = the unfamiliar (out-of-distribution) real smartphone photos
    pad = pd.read_csv(DATA / "pad_ufes" / "metadata.csv")
    pad = pad[pad.diagnostic.isin(PAD2HAM)].copy()
    pad_paths = [PAD_IMAGES / f for f in pad.img_id]
    pad_labels = [CLS2IDX[PAD2HAM[d]] for d in pad.diagnostic]

    if SMOKE:
        ham_paths, ham_labels = ham_paths[:64], ham_labels[:64]
        pad_paths, pad_labels = pad_paths[:64], pad_labels[:64]

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

    print("running on HAM test (familiar) ...", flush=True)
    h = run(ham_paths, ham_labels, clf, dino, train_E, train_y, inv_freq, device)
    print("running on PAD-UFES-20 (real smartphone, unfamiliar) ...", flush=True)
    p = run(pad_paths, pad_labels, clf, dino, train_E, train_y, inv_freq, device)

    # accuracy on the real photos
    clf_acc = float((p["clf_pred"] == p["y"]).mean())
    ret_acc = float((p["ret_pred"] == p["y"]).mean())

    # OOD detection: HAM test = 0 (in), PAD = 1 (out). Higher score = more OOD.
    y_ood = np.concatenate([np.zeros(len(h["y"])), np.ones(len(p["y"]))])
    clf_ood_score = np.concatenate([1 - h["clf_conf"], 1 - p["clf_conf"]])   # low conf = OOD
    ret_ood_score = np.concatenate([-h["msim"], -p["msim"]])                 # low similarity = OOD
    auc_clf = roc_auc_score(y_ood, clf_ood_score)
    auc_ret = roc_auc_score(y_ood, ret_ood_score)

    lines = [
        "REAL-WORLD TEST: HAM-trained models on PAD-UFES-20 smartphone photos",
        f"HAM test photos (familiar): {len(h['y'])}   PAD photos (real, unfamiliar): {len(p['y'])}",
        "",
        "Accuracy on the real PAD photos (shared classes, no retraining):",
        f"  Memorizer (classifier):  acc={clf_acc:.3f}   melanoma recall={recall_mel(p['y'], p['clf_pred']):.3f}",
        f"  Lookup (retrieval):      acc={ret_acc:.3f}   melanoma recall={recall_mel(p['y'], p['ret_pred']):.3f}",
        "  (Low is expected: dermatoscopic-trained models on phone photos is a big shift.)",
        "",
        "Does the model KNOW PAD is unfamiliar?  (OOD-detection AUROC, 0.5=clueless, 1.0=perfect)",
        f"  Classifier max-softmax:   AUROC={auc_clf:.3f}",
        f"  Retrieval mean-similarity: AUROC={auc_ret:.3f}",
        "",
        "Average signal, familiar (HAM) vs unfamiliar (PAD):",
        f"  classifier confidence:  HAM={h['clf_conf'].mean():.3f}  PAD={p['clf_conf'].mean():.3f}",
        f"  retrieval similarity:   HAM={h['msim'].mean():.3f}  PAD={p['msim'].mean():.3f}",
    ]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (RESULTS / "pad_realworld_metrics.txt").write_text(txt)

    if SMOKE:
        print("SMOKE STEP9 PASSED", flush=True)
        return

    # figure: does each signal separate familiar (HAM) from unfamiliar (PAD)?
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(h["clf_conf"], bins=25, alpha=0.6, label="HAM test (familiar)", density=True)
    axes[0].hist(p["clf_conf"], bins=25, alpha=0.6, label="PAD real photos (unfamiliar)", density=True)
    axes[0].set_title(f"Classifier confidence (OOD AUROC={auc_clf:.2f})")
    axes[0].set_xlabel("max softmax probability")
    axes[1].hist(h["msim"], bins=25, alpha=0.6, label="HAM test (familiar)", density=True)
    axes[1].hist(p["msim"], bins=25, alpha=0.6, label="PAD real photos (unfamiliar)", density=True)
    axes[1].set_title(f"Retrieval similarity (OOD AUROC={auc_ret:.2f})")
    axes[1].set_xlabel("mean similarity to nearest cases")
    for ax in axes:
        ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("Can the model tell real phone photos are unfamiliar? (separated curves = yes)")
    fig.tight_layout()
    fig.savefig(RESULTS / "pad_ood_detection.png", dpi=110)
    print("\nsaved: results/pad_realworld_metrics.txt, results/pad_ood_detection.png", flush=True)


if __name__ == "__main__":
    main()
