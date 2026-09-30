import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(base):
    rows, curves = [], {}
    for run_dir in sorted(Path(base).iterdir()):
        cfg_path, loss_path = run_dir / "config.json", run_dir / "loss_history.npy"
        if not cfg_path.exists() or not loss_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("grid_cells") or cfg.get("subset_size") is None:
            continue

        loss = np.load(loss_path)
        loss = loss.reshape(len(loss), -1)
        n_train = len(np.load(run_dir / "subset_indices.npy"))
        key = ("x".join(map(str, cfg["pool_output_size"])), float(cfg["subset_size"]), bool(cfg["attention"]))
        curves[key] = loss

        for c in range(loss.shape[1]):
            rows.append(dict(
                pool=key[0], subset_size=key[1], attention="att" if key[2] else "none", component=c,
                epochs=len(loss),
                steps=len(loss) * -(-n_train // cfg["batch_size"]),
                best_epoch=int(np.nanargmin(loss[:, c])) + 1,
                min_loss=np.nanmin(loss[:, c]),
                final_loss=loss[-1, c],
            ))
    return pd.DataFrame(rows), curves


def plot_curves(curves, component, path):
    pools = sorted({k[0] for k in curves})
    sizes = sorted({k[1] for k in curves})
    fig, axes = plt.subplots(len(pools), len(sizes), figsize=(3.5 * len(sizes), 3 * len(pools)),
                             sharey=True, squeeze=False)
    for i, pool in enumerate(pools):
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            for att in (False, True):
                loss = curves.get((pool, size, att))
                if loss is not None and component < loss.shape[1]:
                    ax.plot(np.arange(1, len(loss) + 1), loss[:, component], color="C1" if att else "C0",
                            ls="--" if att else "-", label="attention" if att else "no attention")
            ax.set_yscale("log")
            ax.set_title(f"pool {pool}, subset {size:g}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(pools) - 1:
                ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel("training loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)


def summary_text(df):
    fmt = lambda v: f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"
    parts = []
    for c, g in df.groupby("component"):
        piv = g.pivot_table(index=["pool", "subset_size"], columns="attention",
                            values=["min_loss", "final_loss", "best_epoch", "steps"])
        if ("min_loss", "att") in piv and ("min_loss", "none") in piv:
            piv[("min_loss", "att / none")] = piv[("min_loss", "att")] / piv[("min_loss", "none")]
        parts.append(f"loss component {c}\n{piv.to_string(float_format=fmt)}")
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-base", default="./ae_model/feature_extractor_ae_checkpoint/")
    p.add_argument("--out-dir", default="./eval_results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, curves = load_runs(args.checkpoint_base)
    df.to_csv(out_dir / "loss_summary.csv", index=False)
    text = summary_text(df)
    (out_dir / "loss_summary.txt").write_text(text)
    print(text)

    n_components = df.component.max() + 1
    for c in range(n_components):
        name = "loss_curves.png" if n_components == 1 else f"loss_curves_component{c}.png"
        plot_curves(curves, c, out_dir / name)


if __name__ == "__main__":
    main()
