import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2

from utils import build_dataloader, set_seed
from main import load_feature_extractor, build_model_from_config

POOLS = ((1, 1), (2, 2))
ALPHA_GRID = np.logspace(-16, 3, 39)
COLORS = {"no attention": "#9fb7d6", "attention": "#f2b880"}


def load_attention_models(checkpoint_base, subset, device):
    models = {}
    for p in POOLS:
        run_dir = Path(checkpoint_base) / f"features_only_pool{p[0]}x{p[1]}_att_subset{subset}"
        model, _, cfg = build_model_from_config(str(run_dir), device)
        assert cfg.attention and tuple(cfg.pool_output_size) == p, f"{run_dir} is not an attention run with pool {p}"
        model.eval()
        if hasattr(model.pool, "A_raw"):
            A = torch.sigmoid(model.pool.A_raw)
            print(f"{p[0]}x{p[1]} attention A: shape {tuple(A.shape)}, "
                  f"min {A.min():.3f}, mean {A.mean():.3f}, max {A.max():.3f}")
        models[p] = model
    return models


@torch.no_grad()
def extract(feature_extractor, att_models, loader, device):
    feature_extractor.to(device).eval()
    X = {(p, att): [] for p in POOLS for att in (False, True)}
    Y = []
    for batch in loader:
        fmap = feature_extractor(batch[0].to(device))
        for p in POOLS:
            X[(p, False)].append(F.adaptive_avg_pool2d(fmap, p).flatten(1).cpu())
            X[(p, True)].append(att_models[p].pool_flatten(fmap).flatten(1).cpu())
        Y.append(batch[1][:, :2].float())
    return ({k: torch.cat(v).numpy().astype(np.float64) for k, v in X.items()},
            torch.cat(Y).numpy().astype(np.float64))


def err_cm(yt, yp):
    return np.linalg.norm(yt - yp, axis=1) * 100.0


def _fit(Xtr, Ytr, Xev, alpha):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(Xtr, Ytr)
    return m.predict(Xev)


def run_pool(X, Y, fit, val, te, alpha=None):
    """Fit ridge on z-scored features and return test errors.

    alpha=None: select alpha on the validation set from len(fit) * ALPHA_GRID.
    alpha=float: use that sklearn alpha as given for every condition (no selection).
    """
    trn = np.concatenate([fit, val])
    if alpha is not None:
        return err_cm(Y[te], _fit(X[trn], Y[trn], X[te], alpha)), alpha, False
    alphas = len(fit) * ALPHA_GRID
    best = min(alphas, key=lambda a: np.median(err_cm(Y[val], _fit(X[fit], Y[fit], X[val], a))))
    pred = _fit(X[trn], Y[trn], X[te], best)
    return err_cm(Y[te], pred), best, best in (alphas[0], alphas[-1])


def split_indices(n, frac, mode, seed):
    cut = int(n * frac)
    if mode == "blocked":
        return np.arange(cut), np.arange(cut, n)
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:cut], perm[cut:]


def sig_label(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def paired_ttest(e_no, e_att, te, block, alternative):
    """Paired t-test on per-frame differences d = err_att - err_noatt (negative = attention more accurate).

    Both conditions are evaluated on the same test frames, so the test is paired
    (equivalent to scipy.stats.ttest_rel). With block > 1, differences are first averaged over
    contiguous blocks of `block` temporally ordered test frames to reduce autocorrelation.
    """
    order = np.argsort(te)  # temporal order of test frames
    d = (e_att - e_no)[order]
    lag1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    r = max(lag1, 0.0)
    n_eff = len(d) * (1 - r) / (1 + r)  # AR(1) approximation
    d_z = d.mean() / d.std(ddof=1)      # Cohen's d_z on frame-level differences
    if block > 1:
        nb = len(d) // block
        d = d[:nb * block].reshape(nb, block).mean(1)
    n = len(d)
    df = n - 1
    res = stats.ttest_1samp(d, 0.0, alternative=alternative)
    half = stats.t.ppf(0.975, df) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, df=df, t=float(res.statistic), p=float(res.pvalue), mean_diff=float(d.mean()),
                ci_lo=float(d.mean() - half), ci_hi=float(d.mean() + half), d_z=float(d_z),
                lag1=lag1, n_eff=float(n_eff), n_frames=len(order))


def plot(errs, chance, dims, tests, out, show_sig=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    positions, data, colors = [], [], []
    for i, p in enumerate(POOLS):
        for j, att in enumerate((False, True)):
            positions.append(1.6 * i + (j - 0.5) * 0.7)
            data.append(errs[(p, att)])
            colors.append(COLORS["attention" if att else "no attention"])

    box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True,
                     medianprops=dict(color="k", lw=1.2),
                     flierprops=dict(marker=".", markersize=2, markerfacecolor="k",
                                     markeredgecolor="none", alpha=0.25))
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("k")

    chance_med = np.median(chance)
    top = max(max(np.max(d) for d in data), chance_med)
    h = 0.03 * top
    if show_sig:
        for i, p in enumerate(POOLS):
            x1, x2 = positions[2 * i], positions[2 * i + 1]
            y = max(np.max(data[2 * i]), np.max(data[2 * i + 1])) + h
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="k", lw=1)
            lab = sig_label(tests[p]["p"])
            ax.text((x1 + x2) / 2, y + h, lab, ha="center", va="bottom", fontsize=9 if lab == "n.s." else 12)

    ax.axhline(chance_med, ls="--", lw=0.8, c="grey")
    ax.set_xticks([1.6 * i for i in range(len(POOLS))],
                  [f"{p[0]}x{p[1]}\nd={dims[p]}" for p in POOLS])
    ax.set_xlim(-0.8, 1.6 * (len(POOLS) - 1) + 0.8)
    ax.set_ylim(0, top + (5 if show_sig else 1) * h)
    ax.set_ylabel("localisation error (cm)")
    ax.set_xlabel("spatial pool size")
    handles = [Patch(facecolor=c, edgecolor="k", label=l) for l, c in COLORS.items()]
    handles.append(Line2D([], [], ls="--", lw=0.8, c="grey", label="chance (median)"))
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(Path(out).with_suffix(".pdf"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-base", default="./ae_model/feature_extractor_ae_checkpoint_n128/",
                   help="Folder containing features_only_pool{1x1,2x2}_att_subset<subset>.")
    p.add_argument("--subset", default="1.0", help="Subset tag of the attention runs to load.")
    p.add_argument("--data-csv", default="../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv")
    p.add_argument("--feature-model-path", default="./attention_model/SAM_weights/")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["blocked", "random"], default="blocked")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=None,
                   help="Fixed ridge alpha (sklearn scale, on z-scored features) shared by all conditions. "
                        "Default None: select alpha per condition on the validation set.")
    p.add_argument("--plot-significance", action="store_true",
                   help="Draw significance brackets (n.s./*/**/***) between paired boxes.")
    p.add_argument("--alternative", choices=["two-sided", "less", "greater"], default="two-sided",
                   help="H1 for mean(err_att - err_noatt). 'less' = attention more accurate (one-sided).")
    p.add_argument("--block-size", type=int, default=1,
                   help="Average paired differences over contiguous blocks of this many test frames "
                        "before testing (1 = frame-level test).")
    p.add_argument("--out", default="xy_regression_attention.png")
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

    att_models = load_attention_models(args.checkpoint_base, args.subset, device)
    Xs, Y = extract(load_feature_extractor(args.feature_model_path), att_models, loader, device)

    tr, te = split_indices(len(Y), args.train_frac, args.split, args.seed)
    n_val = max(1, int(0.2 * len(tr)))
    fit, val = tr[:-n_val], tr[-n_val:]

    chance = err_cm(Y[te], np.repeat(Y[fit].mean(0, keepdims=True), len(te), 0))
    print(f"chance             median {np.median(chance):6.1f} cm")
    errs = {}
    for p in POOLS:
        for att in (False, True):
            e, a, edge = run_pool(Xs[(p, att)], Y, fit, val, te, args.alpha)
            errs[(p, att)] = e
            name = f"{p[0]}x{p[1]}{' + att' if att else ''}"
            print(f"{name:<18} median {np.median(e):6.1f} cm   mean {e.mean():6.1f} cm   "
                  f"alpha={a:.3g}{'  EDGE' if edge else ''}")

    unit = "frames" if args.block_size == 1 else f"blocks of {args.block_size} frames"
    print(f"\nPaired t-test, d = err_att - err_noatt (negative = attention better), "
          f"H1: {args.alternative}, unit: {unit}")
    tests = {}
    for p in POOLS:
        r = paired_ttest(errs[(p, False)], errs[(p, True)], te, args.block_size, args.alternative)
        tests[p] = r
        print(f"  {p[0]}x{p[1]}: n = {r['n']}, mean d = {r['mean_diff']:+.2f} cm "
              f"[95% CI {r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], "
              f"t({r['df']}) = {r['t']:.3f}, p = {r['p']:.3g}, d_z = {r['d_z']:+.3f}  {sig_label(r['p'])}")
        print(f"        frame-level lag-1 autocorr of d = {r['lag1']:.3f} -> "
              f"AR(1) effective n ~ {r['n_eff']:.0f} of {r['n_frames']} frames")

    dims = {p: Xs[(p, False)].shape[1] for p in POOLS}
    plot(errs, chance, dims, tests, args.out, show_sig=args.plot_significance)
    np.savez(Path(args.out).with_suffix(".npz"), chance=chance,
             **{f"pool{p[0]}x{p[1]}{'_att' if att else ''}": e for (p, att), e in errs.items()},
             **{f"ttest_pool{p[0]}x{p[1]}": np.array([r["t"], r["df"], r["p"], r["mean_diff"], r["d_z"]])
                for p, r in tests.items()})


if __name__ == "__main__":
    main()
