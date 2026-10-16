import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ae_model.plotting import ratemaps

EPS = 1e-8
CONDITIONS = [("1x1", False), ("1x1", True), ("2x2", False), ("2x2", True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="./eval_results_n128/cache")
    p.add_argument("--subset", default="1.0")
    p.add_argument("--n-units", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_ratemaps.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(len(CONDITIONS), args.n_units, figsize=(1.6 * args.n_units, 1.7 * len(CONDITIONS)),
                             squeeze=False)
    occupancy, visited = None, None
    for row, (pool, att) in zip(axes, CONDITIONS):
        name = f"features_only_pool{pool}{'_att' if att else ''}_subset{args.subset}"
        cache = Path(args.cache_dir) / f"{name}.npz"
        for ax in row:
            ax.axis("off")
        if not cache.exists():
            row[0].set_title(f"{name}: not found", fontsize=8, loc="left")
            continue

        data = np.load(cache)
        emb, pos = data["emb"], data["pos"]
        if occupancy is None:
            occupancy = ratemaps(np.ones((len(pos), 1)), pos)[0]
            visited = occupancy > 0.01

        units = np.sort(rng.choice(emb.shape[1], args.n_units, replace=False))
        maps = ratemaps(emb[:, units], pos)
        silent = (emb.max(axis=0) <= EPS).mean()
        for ax, unit, m in zip(row, units, maps):
            rate = np.where(visited, m / np.where(visited, occupancy, 1), np.nan)
            if np.nanmax(rate) > 0:
                rate = rate / np.nanmax(rate)
            ax.imshow(rate, cmap=cmap, origin="lower", vmin=0, vmax=1)
            ax.set_title("silent" if emb[:, unit].max() <= EPS else f"unit {unit}", fontsize=8)
        row[0].text(-0.15, 0.5, f"{pool}{' + att' if att else ''}\n{100 * silent:.0f}% silent",
                    transform=row[0].transAxes, ha="right", va="center", fontsize=10)

    fig.suptitle(f"Example latent ratemaps (random units, training subset {args.subset})", fontsize=11)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    fig.savefig(args.out, dpi=300)
    fig.savefig(Path(args.out).with_suffix(".pdf"))


if __name__ == "__main__":
    main()
