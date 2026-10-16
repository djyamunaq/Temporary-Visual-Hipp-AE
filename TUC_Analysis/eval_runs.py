import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

N_BINS = 50
FILTER_WIDTH = 2
MIN_OCC_FRAC = 0.01
EPS = 1e-8

POOL_STYLE = {"1x1": dict(ls="--", marker="s"), "2x2": dict(ls="-", marker="o")}


PANELS = [
    ("r2", "Reconstruction R² of the input"),
    ("latent_active_per_frame", "Latent: units active per frame"),
    ("latent_mean_cosine", "Latent: mean cosine similarity between units"),
    ("latent_spatial_sparsity", "Latent: spatial sparsity of active units"),
]
CACHE_KEYS = {"emb", "pos", "r2", "latent_active_per_frame", "latent_mean_cosine", "latent_spatial_sparsity"}


def build_ratemaps(act, pos):
    p = pos - pos.min(axis=0)
    idx = np.rint(p / p.max() * (N_BINS - 1)).astype(int)
    flat = idx[:, 1] * N_BINS + idx[:, 0]

    occupancy = gaussian_filter(np.bincount(flat, minlength=N_BINS ** 2).reshape(N_BINS, N_BINS).astype(float), FILTER_WIDTH)
    visited = occupancy >= MIN_OCC_FRAC * occupancy.max()

    maps = np.zeros((act.shape[1], N_BINS, N_BINS))
    for u in range(act.shape[1]):
        total = np.bincount(flat, weights=act[:, u], minlength=N_BINS ** 2).reshape(N_BINS, N_BINS)
        rate = np.where(visited, gaussian_filter(total, FILTER_WIDTH) / np.where(visited, occupancy, 1), 0)
        maps[u] = rate / max(rate.max(), 1e-12)
    return maps, visited


def spatial_sparsity(act, pos):
    alive = act.max(axis=0) > EPS
    if not alive.any():
        return np.nan
    maps, visited = build_ratemaps(act[:, alive], pos)
    v = maps[:, visited]
    return np.mean(1 - v.mean(axis=1) ** 2 / (v ** 2).mean(axis=1))


def mean_cosine(act):
    h = act[:, act.max(axis=0) > EPS].astype(np.float64)
    if h.shape[1] < 2:
        return np.nan
    sscp = h.T @ h
    d = np.sqrt(np.diag(sscp))
    cos = sscp / np.outer(d, d)
    return cos[~np.eye(len(cos), dtype=bool)].mean()


def compute_outputs(run_dir, state):
    import torch
    from torchvision.transforms import v2
    from main import build_model_from_config, load_feature_extractor
    from utils import build_dataloader
    from training_functions import get_eval_metrics

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, cfg = build_model_from_config(str(run_dir), device)
    if "feature_extractor" not in state:
        state["feature_extractor"] = load_feature_extractor(cfg.feature_model_path)

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((248, 328)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(cfg.data_csv, transform=transform, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, seed=cfg.seed)
    emb, pos, r2, inputs = get_eval_metrics(dataloader=loader, ae_model=model,
                                            feature_extractor=state["feature_extractor"],
                                            device=device, return_inputs=True)
    pos = np.asarray(pos, dtype=float)[:, :2]
    return dict(
        emb=np.asarray(emb, dtype=np.float32),
        pos=pos,
        r2=float(r2),
        latent_active_per_frame=(emb > EPS).mean(),
        latent_mean_cosine=mean_cosine(emb),
        latent_spatial_sparsity=spatial_sparsity(emb, pos),
    )


def get_outputs(run_dir, cache_dir, recompute, state):
    cache = cache_dir / f"{run_dir.name}.npz"
    mtime = (run_dir / "best_model.pt").stat().st_mtime
    if cache.exists() and not recompute:
        cached = dict(np.load(cache))
        if cached["mtime"] == mtime and CACHE_KEYS <= cached.keys():
            return cached
    out = compute_outputs(run_dir, state)
    np.savez_compressed(cache, mtime=mtime, **out)
    return out


def plot_summary(df, path):
    sizes = sorted(df.subset_size.unique())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for condition, g in df.groupby("condition"):
            ax.plot(g.subset_size, g[metric],
                    color="C1" if g.attention.iloc[0] else "C0",
                    label=condition,
                    **POOL_STYLE[g.pool.iloc[0]])
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:g}" for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("training subset fraction")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(Path(path).with_suffix(".pdf"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-base", default="./ae_model/feature_extractor_ae_checkpoint_n128/")
    p.add_argument("--out-dir", default="./eval_results_n128/")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    state, rows = {}, []
    for run_dir in sorted(Path(args.checkpoint_base).iterdir()):
        if not (run_dir / "config.json").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("grid_cells") or cfg.get("subset_size") is None:
            continue
        if not (run_dir / "metrics.json").exists():
            print(f"[skip] {run_dir.name}: no metrics.json")
            continue

        print(run_dir.name, flush=True)
        out = get_outputs(run_dir, cache_dir, args.recompute, state)
        emb = out["emb"]
        rows.append(dict(
            pool="x".join(map(str, cfg["pool_output_size"])),
            attention=bool(cfg["attention"]),
            subset_size=float(cfg["subset_size"]),
            r2=float(out["r2"]),
            latent_active_per_frame=(emb > EPS).mean(),
            latent_never_active=(emb.max(axis=0) <= EPS).mean(),
            latent_mean_cosine=mean_cosine(emb),
            latent_spatial_sparsity=spatial_sparsity(emb, out["pos"]),
        ))

    df = pd.DataFrame(rows).sort_values(["pool", "attention", "subset_size"]).reset_index(drop=True)
    df["condition"] = df.pool + df.attention.map({True: " + att", False: ""})
    df.to_csv(out_dir / "eval_results.csv", index=False)

    metrics = [m for m, _ in PANELS]
    parts = [f"{title}\n{df.pivot_table(index='subset_size', columns='condition', values=m, dropna=False).round(4).to_string()}"
             for m, title in PANELS]
    with_att = df[df.attention].set_index(["pool", "subset_size"])[metrics]
    without_att = df[~df.attention].set_index(["pool", "subset_size"])[metrics]
    parts.append(f"attention minus no attention\n{(with_att - without_att).dropna(how='all').round(4).to_string()}")
    text = "\n\n".join(parts)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)

    plot_summary(df, out_dir / "eval_summary.png")


if __name__ == "__main__":
    main()
