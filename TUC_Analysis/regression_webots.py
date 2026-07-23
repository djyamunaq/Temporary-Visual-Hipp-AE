import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2

from utils import build_dataloader, set_seed
from main import load_feature_extractor

POOLS = ((1, 1), (2, 1), (1, 2), (2, 2), (2, 3), (3, 3), (4, 4), (5, 5), (6, 6))
ALPHA_GRID = np.logspace(-10, 3, 27)
INSPECT = (6, 6)  # pool whose worst-decoded frames get plotted
# transform normalisation
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


@torch.no_grad()
def extract(model, loader, pools, device):
    model.to(device).eval()
    X = {p: [] for p in pools}
    Y = []
    for batch in loader:
        fmap = model(batch[0].to(device))
        for p in pools:
            X[p].append(F.adaptive_avg_pool2d(fmap, p).flatten(1).cpu())
        Y.append(batch[1].float())
    return ({p: torch.cat(v).numpy().astype(np.float64) for p, v in X.items()},
            torch.cat(Y).numpy().astype(np.float64))


def err_cm(yt, yp):
    return np.linalg.norm(yt - yp, axis=1) * 100.0


def _fit(Xtr, Ytr, Xev, alpha):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(Xtr, Ytr)
    return m.predict(Xev)


def run_pool(X, Y, fit, val, te):
    alphas = len(fit) * ALPHA_GRID
    best = min(alphas, key=lambda a: np.median(err_cm(Y[val], _fit(X[fit], Y[fit], X[val], a))))
    trn = np.concatenate([fit, val])
    pred = _fit(X[trn], Y[trn], X[te], best)
    return err_cm(Y[te], pred), pred, best, best in (alphas[0], alphas[-1])


def plot_outliers(dataset, te, err, pred, Y, k, out):
    sel = np.argsort(err)[-k:][::-1]
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(2.6 * ((k + 1) // 2), 5.4))
    for ax, s in zip(axes.ravel(), sel):
        img = dataset[te[s]][0].permute(1, 2, 0).numpy() * STD + MEAN
        ax.imshow(img.clip(0, 1))
        ax.set_title(f"{err[s]:.0f} cm\ntrue {Y[te[s]][0]:.2f},{Y[te[s]][1]:.2f}\n"
                     f"pred {pred[s][0]:.2f},{pred[s][1]:.2f}", fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def split_indices(n, frac, mode, seed):
    cut = int(n * frac)
    if mode == "blocked":
        return np.arange(cut), np.arange(cut, n)
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:cut], perm[cut:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", default="../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv")
    p.add_argument("--feature-model-path", default="./attention_model/SAM_weights/")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["blocked", "random"], default="blocked")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out-outliers", default="outliers.png")
    p.add_argument("--out", default="pool_sweep.png")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((248, 328)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(args.data_csv, transform=tf, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, seed=args.seed)

    Xs, Y = extract(load_feature_extractor(args.feature_model_path), loader, POOLS, device)

    tr, te = split_indices(len(Y), args.train_frac, args.split, args.seed)
    n_val = max(1, int(0.2 * len(tr)))
    fit, val = tr[:-n_val], tr[-n_val:]

    chance = err_cm(Y[te], np.repeat(Y[fit].mean(0, keepdims=True), len(te), 0))
    errs, labels = [chance], ["chance"]
    print(f"chance      median {np.median(chance):6.1f} cm")
    for p in POOLS:
        e, pred, a, edge = run_pool(Xs[p], Y, fit, val, te)
        errs.append(e)
        labels.append(f"{p[0]}x{p[1]}\nd={Xs[p].shape[1]}")
        print(f"{p[0]}x{p[1]:<10} median {np.median(e):6.1f} cm   alpha={a:.3g}"
              f"{'  EDGE' if edge else ''}")
        if p == INSPECT:
            plot_outliers(loader.dataset, te, e, pred, Y, args.topk, args.out_outliers)

    fig, ax = plt.subplots(figsize=(2 * len(errs), 6))
    ax.boxplot(errs, tick_labels=labels, showfliers=True,
               flierprops=dict(marker=".", markersize=2, markerfacecolor="k",
                               markeredgecolor="none", alpha=0.25))
    ax.axhline(np.median(chance), ls="--", lw=0.8, c="grey")
    ax.set_ylabel("localisation error (cm)")
    ax.set_xlabel("spatial pool size")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
