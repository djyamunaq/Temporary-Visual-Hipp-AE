import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ae_model.plotting import stats_place_fields

N_BINS = 50
FILTER_WIDTH = 2
MIN_OCC_FRAC = 0.01
EPS = 1e-8

PANELS = [
    ("r2_clean", "Reconstruction R2 (clean)"),
    ("r2_noisy", "Reconstruction R2 (noisy)"),
    ("frac_units_with_field", "Fraction of units with a place field"),
    ("mean_fields_per_field_unit", "Fields per unit (units with >=1)"),
    ("spatial_sparsity", "Spatial sparsity (1 - Treves-Rolls a)"),
    ("mean_field_frac_area", "Mean field size / visited area"),
    ("frac_dead_units", "Fraction of dead units"),
    ("frac_active_per_frame", "Fraction of units active per frame"),
]


def compute_outputs(run_dir, state):
    import torch
    from torchvision.transforms import v2
    from main import build_model_from_config, load_feature_extractor
    from utils import build_dataloader, set_seed
    from training_functions import get_eval_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, cfg = build_model_from_config(str(run_dir), device)
    model.eval()
    if "feature_extractor" not in state:
        state["feature_extractor"] = load_feature_extractor(cfg.feature_model_path)

    base = [v2.ToDtype(torch.float32, scale=True), v2.Resize((248, 328))]
    norm = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    noise = v2.GaussianNoise(mean=0.0, sigma=cfg.noise_sigma, clip=True)
    transforms = {"clean": v2.Compose(base + [norm]), "noisy": v2.Compose(base + [noise, norm])}

    set_seed(cfg.seed)
    out = {}
    for name, transform in transforms.items():
        loader = build_dataloader(cfg.data_csv, transform=transform, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, seed=cfg.seed)
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            emb, pos, r2 = get_eval_metrics(dataloader=loader, ae_model=model,
                                            feature_extractor=state["feature_extractor"], grid_cell_encoder=None,
                                            save_path=os.path.join(tmp, f"features_{name}"), device=device)
        out[f"r2_{name}"] = np.asarray(r2, dtype=float)
        if name == "clean":
            out["emb"] = np.asarray(emb, dtype=np.float32)
            out["pos"] = np.asarray(pos, dtype=float)[:, :2]
    return out


def get_outputs(run_dir, cache_dir, recompute, state):
    cache = cache_dir / f"{run_dir.name}.npz"
    mtime = (run_dir / "best_model.pt").stat().st_mtime
    if cache.exists() and not recompute:
        cached = dict(np.load(cache))
        if cached["mtime"] == mtime:
            return cached
    out = compute_outputs(run_dir, state)
    np.savez_compressed(cache, mtime=mtime, **out)
    return out


def build_ratemaps(emb, pos):
    p = pos - pos.min(axis=0)
    idx = np.rint(p / p.max() * (N_BINS - 1)).astype(int)
    flat = idx[:, 1] * N_BINS + idx[:, 0]

    occupancy = gaussian_filter(np.bincount(flat, minlength=N_BINS ** 2).reshape(N_BINS, N_BINS).astype(float), FILTER_WIDTH)
    visited = occupancy >= MIN_OCC_FRAC * occupancy.max()

    maps = np.zeros((emb.shape[1], N_BINS, N_BINS))
    for u in range(emb.shape[1]):
        total = np.bincount(flat, weights=emb[:, u], minlength=N_BINS ** 2).reshape(N_BINS, N_BINS)
        rate = np.where(visited, gaussian_filter(total, FILTER_WIDTH) / np.where(visited, occupancy, 1), 0)
        maps[u] = rate / max(rate.max(), 1e-12)
    return maps, visited


def place_field_metrics(emb, pos):
    maps, visited = build_ratemaps(emb, pos)
    num_fields, _, sizes = stats_place_fields(maps)
    has_field = num_fields > 0

    v = maps[:, visited]
    with np.errstate(invalid="ignore"):
        sparsity = 1 - v.mean(axis=1) ** 2 / (v ** 2).mean(axis=1)

    return dict(
        frac_units_with_field=has_field.mean(),
        mean_fields_per_field_unit=num_fields[has_field].mean() if has_field.any() else np.nan,
        mean_field_frac_area=sizes.mean() / visited.sum() if sizes.size else np.nan,
        spatial_sparsity=np.nanmean(sparsity),
    )


def plot_summary(df, path):
    sizes = sorted(df.subset_size.unique())
    colors = {pool: f"C{i}" for i, pool in enumerate(sorted(df.pool.unique()))}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for condition, g in df.groupby("condition"):
            att = g.attention.iloc[0]
            ax.plot(g.subset_size, g[metric], color=colors[g.pool.iloc[0]], ls="--" if att else "-",
                    marker="s" if att else "o", label=condition)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:g}" for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("training subset fraction")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-base", default="./ae_model/feature_extractor_ae_checkpoint/")
    p.add_argument("--out-dir", default="./eval_results")
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
        row = dict(
            pool="x".join(map(str, cfg["pool_output_size"])),
            attention=bool(cfg["attention"]),
            subset_size=float(cfg["subset_size"]),
            r2_clean=out["r2_clean"].mean(),
            r2_noisy=out["r2_noisy"].mean(),
            frac_dead_units=(out["emb"].max(axis=0) <= EPS).mean(),
            frac_active_per_frame=(out["emb"] > EPS).mean(),
            **place_field_metrics(out["emb"], out["pos"]),
        )
        reported = json.loads((run_dir / "metrics.json").read_text())["clean"]
        if abs(row["r2_clean"] - reported) > 1e-3:
            print(f"  WARNING: clean R2 {row['r2_clean']:.4f} differs from metrics.json ({reported:.4f})")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["pool", "attention", "subset_size"]).reset_index(drop=True)
    df["condition"] = df.pool + df.attention.map({True: " + att", False: ""})
    df.to_csv(out_dir / "eval_results.csv", index=False)

    metrics = [m for m, _ in PANELS]
    parts = [f"{m}\n{df.pivot_table(index='subset_size', columns='condition', values=m, dropna=False).round(4).to_string()}"
             for m in metrics]
    with_att = df[df.attention].set_index(["pool", "subset_size"])[metrics]
    without_att = df[~df.attention].set_index(["pool", "subset_size"])[metrics]
    parts.append(f"attention minus no attention\n{(with_att - without_att).dropna(how='all').round(4).to_string()}")
    text = "\n\n".join(parts)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)

    plot_summary(df, out_dir / "eval_summary.png")


if __name__ == "__main__":
    main()
