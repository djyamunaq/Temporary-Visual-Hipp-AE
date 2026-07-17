"""
Batch analysis exporter for the hippocampal autoencoder experiments.

Given a root folder containing several training runs (each produced by
``main.py`` and identifiable by a ``config.json``), this script reproduces
every figure and quantitative result of ``attention_notebook.ipynb`` and writes
them into a dedicated analysis-outputs folder.

The ``--data-root`` is expected to hold one subfolder per experiment run, and
each run contains an experiment-type subfolder holding the checkpoint::

    <data-root>/<experiment_id>/<experiment_type>/{config.json, best_model.pt, loss_history.npy}

with ``<experiment_type>`` one of ``features_only`` / ``features_only_pool1x1``
(selected via ``--experiment-type``). This nesting is mirrored under
``--out-root``.

Layout of the outputs (``--out-root``)::

    <out-root>/
      <experiment_id>/
        <experiment_type>/                    # e.g. features_only, features_only_pool1x1
            feature_maps/                     # feature extractor is frozen
                image_{index}.jpg             # 512-channel feature-map montage
                clusterings/image_{index}/
                    kmeans-bisec_{k}.jpg
                    kmeans_{k}.jpg
                    spectral_{k}.jpg
                    optics.jpg
            config.json
            loss_curve.png
            reconstructions/features/... , grids/...
            ratemaps_100.png
            ratemap_sum.png
            sscp_matrix.png
            example_place_field.png
            centroid_density.png
            centroids_scatter.png
            place_field_percentages.png
            metrics.json
            arrays.npz

Run from inside ``TUC_Analysis`` so the relative ``data_csv`` /
``feature_model_path`` values stored in each ``config.json`` resolve correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive: never pop windows, safe for batch jobs
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import OPTICS, KMeans, BisectingKMeans, SpectralClustering
from torchvision.transforms import v2

from ae_model.cnn_hippocampal_ae import Conv_AE, load_ae_model
from ae_model.plotting import ratemaps, stats_place_fields, format_centroids
from attention_model.conv_encoder import ConvEncoder
from grid_cells.encoder import load_grid_encoder
from training_functions import get_eval_metrics
from utils import build_dataloader, set_seed, timer


# Image resize used everywhere in the project (main.py / notebook).
RESIZE = (248, 328)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLUSTER_KS = [4, 6, 8, 12, 16, 32, 64]

# Experiment types expected as subfolders directly under --data-root.
EXPERIMENT_TYPES = ["features_only", "features_only_pool1x1"]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Export all attention_notebook figures and metrics for a "
        "root of autoencoder experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        required=True,
        help="Folder with one subfolder per run: "
        "<data-root>/<experiment_id>/<experiment_type>/config.json.",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Destination folder for analysis outputs (kept separate from data-root). "
        "The <experiment_id>/<experiment_type> nesting is mirrored here.",
    )
    p.add_argument(
        "--experiment-type",
        choices=EXPERIMENT_TYPES + ["both"],
        default="both",
        help="Which experiment type (under each experiment_id) to analyze.",
    )
    p.add_argument(
        "--n-featuremap-images",
        type=int,
        default=8,
        help="Number of dataset images for the shared feature-map / clustering section.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available else cpu.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest cross-platform).",
    )
    p.add_argument(
        "--recon-prob",
        type=float,
        default=0.1,
        help="Per-batch probability of saving a reconstruction figure during eval.",
    )
    p.add_argument(
        "--data-csv", default=None, help="Override the data.csv path from config.json."
    )
    p.add_argument(
        "--feature-model-path",
        default=None,
        help="Override the feature extractor path from config.json.",
    )
    p.add_argument(
        "--skip-feature-maps",
        action="store_true",
        help="Skip the (expensive) shared feature-map / clustering section.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Discovery + shared helpers                                                  #
# --------------------------------------------------------------------------- #
def discover_experiments(data_root: Path, exp_type: str):
    """Find runs at <data_root>/<experiment_id>/<exp_type>/config.json.

    ``experiment_id`` is the run subfolder name, preserved so the input nesting
    can be mirrored in the outputs.
    """
    experiments = []
    for exp_id_dir in sorted(
        (p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()
    ):
        exp_dir = exp_id_dir / exp_type
        cfg_path = exp_dir / "config.json"
        if not cfg_path.is_file():
            continue
        with open(cfg_path, "r") as f:
            config = json.load(f)
        experiments.append(
            {
                "experiment_id": exp_id_dir.name,
                "exp_dir": exp_dir,
                "config": config,
            }
        )
    return experiments


def build_transform():
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(RESIZE),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_feature_extractor(
    feature_model_path: str, device: torch.device
) -> ConvEncoder:
    """Load the frozen ConvEncoder feature extractor (same remap as main.py)."""
    ckpt = torch.load(
        os.path.join(feature_model_path, "best.ckpt"),
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    encoder_state_dict = ckpt["state_dict"]
    for k in list(encoder_state_dict.keys()):
        new_k = k.replace("encoder.encoder", "encoder").replace(
            "encoder.concept_proj", "concept_proj"
        )
        if new_k != k:
            encoder_state_dict[new_k] = encoder_state_dict.pop(k)

    feature_extractor = ConvEncoder()
    feature_extractor.load_state_dict(ckpt["state_dict"])
    feature_extractor.eval().to(device)
    return feature_extractor


def resolve_paths(config: dict, args) -> tuple[str, str]:
    """Resolve data_csv and feature_model_path, honouring CLI overrides."""
    data_csv = args.data_csv or config["data_csv"]
    feature_model_path = args.feature_model_path or config["feature_model_path"]
    return data_csv, feature_model_path


# --------------------------------------------------------------------------- #
# Shared feature-map / clustering section (notebook cells 3-4)                #
# --------------------------------------------------------------------------- #
def _save_feature_montage(features_hwc: np.ndarray, path: Path):
    """Montage of all feature-map channels for one image (cell 3)."""
    h, w, n_channels = features_hwc.shape
    ncols = 32
    nrows = int(np.ceil(n_channels / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 0.8, nrows * 0.8))
    axs = axs.flat
    for i in range(n_channels):
        axs[i].imshow(features_hwc[:, :, i], cmap="inferno")
        axs[i].axis("off")
    for j in range(n_channels, nrows * ncols):
        axs[j].set_visible(False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_label_image(label_img: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(label_img, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def export_feature_maps(
    feature_extractor: ConvEncoder,
    loader,
    n_images: int,
    device: torch.device,
    out_dir: Path,
):
    """Feature-map montages + per-image clusterings, computed once (frozen extractor)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clusterings_dir = out_dir / "clusterings"

    images, _ = next(iter(loader))
    n_images = min(n_images, images.shape[0])

    with torch.inference_mode():
        feats = (
            feature_extractor(images[:n_images].float().to(device)).cpu().numpy()
        )  # (N, C, H, W)

    for index in range(n_images):
        features = feats[index]  # (C, H, W)
        features_hwc = np.moveaxis(features, 0, -1)  # (H, W, C)
        _save_feature_montage(features_hwc, out_dir / f"image_{index}.jpg")

        h, w, c = features_hwc.shape
        feature_reshaped = np.reshape(features_hwc, (h * w, c))

        img_cluster_dir = clusterings_dir / f"image_{index}"
        img_cluster_dir.mkdir(parents=True, exist_ok=True)

        # OPTICS is density-based (k has no meaning): computed once per image.
        optics = OPTICS(min_samples=20).fit(feature_reshaped)
        _save_label_image(
            np.reshape(optics.labels_, (h, w)), "Optics", img_cluster_dir / "optics.jpg"
        )

        for k in CLUSTER_KS:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(
                feature_reshaped
            )
            bisect = BisectingKMeans(n_clusters=k, random_state=0, n_init=5).fit(
                feature_reshaped
            )
            spectral = SpectralClustering(
                n_clusters=k, assign_labels="cluster_qr", random_state=0
            ).fit(feature_reshaped)

            _save_label_image(
                np.reshape(bisect.labels_, (h, w)),
                f"KMeans-Bisec k={k}",
                img_cluster_dir / f"kmeans-bisec_{k}.jpg",
            )
            _save_label_image(
                np.reshape(kmeans.labels_, (h, w)),
                f"KMeans k={k}",
                img_cluster_dir / f"kmeans_{k}.jpg",
            )
            _save_label_image(
                np.reshape(spectral.labels_, (h, w)),
                f"Spectral k={k}",
                img_cluster_dir / f"spectral_{k}.jpg",
            )

        print(f"  feature maps + clusterings done for image {index}")


# --------------------------------------------------------------------------- #
# Per-experiment AE figures (notebook cells 6-17)                             #
# --------------------------------------------------------------------------- #
def _save_loss_curve(history: np.ndarray, path: Path):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_ratemaps_grid(all_ratemaps: np.ndarray, path: Path):
    """Grid of up to 100 unit ratemaps (cell 10 / plot_ratemaps, made robust to n<100)."""
    n = min(100, all_ratemaps.shape[0])
    fig = plt.figure(figsize=(20, 20), dpi=150)
    for i in range(n):
        plt.subplot(10, 10, i + 1)
        plt.imshow(all_ratemaps[i], cmap="hot", origin="lower")
        plt.axis("off")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_ratemap_sum(all_ratemaps: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(all_ratemaps.sum(axis=0), cmap="hot", origin="lower")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_sscp(embeddings: np.ndarray, path: Path):
    sscp = np.dot(embeddings.T, embeddings)
    fig = plt.figure(figsize=(7, 6))
    plt.title("Sums of Squares and Cross Products Matrix", fontsize=15)
    plt.imshow(sscp, origin="lower")
    plt.xlabel("Units")
    plt.ylabel("Units")
    plt.colorbar()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_example_place_field(
    all_ratemaps, unit, all_num_fields, sizes_per_field, centroids_per_field, path: Path
):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(all_ratemaps[unit], cmap="hot", origin="lower")
    if all_num_fields[unit] != 0:
        for c in centroids_per_field[unit]:
            plt.scatter(c[1], c[0], color="green", marker="x", s=30)
    plt.title(f"Unit {unit} - {all_num_fields[unit]} place field(s)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_centroid_density(centroids: np.ndarray, path: Path):
    fig = plt.figure(figsize=(7, 6))
    if len(centroids) > 0:
        heatmap, xedges, yedges = np.histogram2d(
            centroids[:, 1], centroids[:, 0], bins=(20, 20)
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="viridis")
        plt.colorbar(label="Density")
    plt.title("Centroids density", fontsize=15)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_centroids_scatter(centroids: np.ndarray, path: Path):
    fig = plt.figure(figsize=(7, 7))
    if len(centroids) > 0:
        plt.plot(centroids[:, 1], centroids[:, 0], "bx", markersize=3)
    plt.title(f"Centroids (n = {len(centroids)})", fontsize=15)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_place_field_percentages(
    all_num_fields: np.ndarray, n_hidden: int, path: Path
) -> dict:
    """Bar chart + dict of the % of units with each number of place fields (cell 17)."""
    counts = np.bincount(all_num_fields.astype(int))
    percentages = {
        str(i): float(counts[i] / n_hidden * 100) for i in range(len(counts))
    }

    fig = plt.figure(figsize=(7, 6))
    plt.bar(list(percentages.keys()), list(percentages.values()))
    plt.title("% of place fields", fontsize=15)
    plt.xlabel("Place fields")
    plt.ylabel("% of units")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return percentages


def build_ae_model(config: dict, exp_dir: Path, device: torch.device):
    """Rebuild Conv_AE (+ grid encoder for grid runs) and load trained weights."""
    n_hidden = config["n_hidden"]
    grid_cells = bool(config.get("grid_cells", False))

    if grid_cells:
        grid_encoder = load_grid_encoder(
            str(exp_dir / "grid_encoder.pt"), map_location=device
        )
        grid_encoder.eval().to(device)
        ae_model = Conv_AE(
            n_hidden=n_hidden,
            last_layer_activation=torch.nn.Sigmoid(),
            d_aux=grid_encoder.n_cells,
        )
        weight_path = exp_dir / "best_model_aux_input.pt"
    else:
        grid_encoder = None
        ae_model = Conv_AE(
            n_hidden=n_hidden, last_layer_activation=torch.nn.Sigmoid(), d_aux=None
        )
        weight_path = exp_dir / "best_model.pt"

    ae_model = load_ae_model(ae_model, str(weight_path), device=device)
    ae_model.to(device)
    return ae_model, grid_encoder


def process_experiment(
    exp: dict, feature_extractor, args, device: torch.device, out_dir: Path
):
    config = exp["config"]
    exp_dir = exp["exp_dir"]
    n_hidden = config["n_hidden"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Echo the config used for this run.
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    data_csv, _ = resolve_paths(config, args)

    ae_model, grid_encoder = build_ae_model(config, exp_dir, device)

    # Loss curve from the saved training history.
    loss_history_path = exp_dir / "loss_history.npy"
    if loss_history_path.exists():
        _save_loss_curve(np.load(loss_history_path), out_dir / "loss_curve.png")

    # Eval: embeddings, positions, per-batch R2 (+ reconstruction figures).
    loader = build_dataloader(
        data_csv,
        transform=build_transform(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    embeddings, positions, r2_per_batch = get_eval_metrics(
        dataloader=loader,
        ae_model=ae_model,
        feature_extractor=feature_extractor,
        device=device,
        grid_cell_encoder=grid_encoder,
        feature_reconstruction_path=str(out_dir / "reconstructions"),
        prob_plot=args.recon_prob,
    )
    accuracy = float(np.mean(r2_per_batch))

    # Ratemaps and derived figures.
    all_ratemaps = ratemaps(embeddings, positions, n_bins=50, filter_width=3)
    _save_ratemaps_grid(all_ratemaps, out_dir / "ratemaps_100.png")
    _save_ratemap_sum(all_ratemaps, out_dir / "ratemap_sum.png")
    _save_sscp(embeddings, out_dir / "sscp_matrix.png")

    all_num_fields, centroids, sizes = stats_place_fields(
        all_ratemaps,
        peak_as_centroid=True,
        min_pix_cluster=0.02,
        max_pix_cluster=0.5,
        active_threshold=0.2,
    )
    centroids_per_field, sizes_per_field = format_centroids(
        all_num_fields, centroids, sizes
    )

    # Example single-unit place field: prefer a unit that actually has fields.
    units_with_fields = np.where(all_num_fields > 0)[0]
    rng = np.random.default_rng(args.seed)
    unit = (
        int(rng.choice(units_with_fields))
        if len(units_with_fields)
        else int(rng.integers(len(all_num_fields)))
    )
    _save_example_place_field(
        all_ratemaps,
        unit,
        all_num_fields,
        sizes_per_field,
        centroids_per_field,
        out_dir / "example_place_field.png",
    )

    _save_centroid_density(centroids, out_dir / "centroid_density.png")
    _save_centroids_scatter(centroids, out_dir / "centroids_scatter.png")
    percentages = _save_place_field_percentages(
        all_num_fields, n_hidden, out_dir / "place_field_percentages.png"
    )

    # Quantitative results.
    metrics = {
        "experiment_id": exp["experiment_id"],
        "exp_dir": str(exp_dir),
        "n_hidden": n_hidden,
        "grid_cells": bool(config.get("grid_cells", False)),
        "config": config,
        "r2_accuracy_mean": accuracy,
        "n_units": int(all_ratemaps.shape[0]),
        "total_place_fields": int(np.sum(all_num_fields)),
        "mean_num_fields_per_unit": float(np.mean(all_num_fields)),
        "mean_field_size": float(np.mean(sizes)) if len(sizes) else 0.0,
        "place_field_percentages": percentages,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    np.savez_compressed(
        out_dir / "arrays.npz",
        embeddings=embeddings,
        positions=positions,
        r2_per_batch=r2_per_batch,
        ratemaps=all_ratemaps,
        num_fields=all_num_fields,
        centroids=centroids,
        sizes=sizes,
    )

    print(
        f"  R2={accuracy:.4f}  total_place_fields={metrics['total_place_fields']}  -> {out_dir}"
    )
    return out_dir


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run_experiment_type(
    exp_type: str,
    data_root: Path,
    out_root: Path,
    args,
    device: torch.device,
    get_feature_extractor,
    ensure_feature_maps,
):
    """Analyze one experiment type across all experiment_ids, mirroring the nesting."""
    experiments = discover_experiments(data_root, exp_type)
    if not experiments:
        print(
            f"\n[{exp_type}] skipped: no '{exp_type}/config.json' found under any "
            f"experiment_id in '{data_root}'."
        )
        return [], []

    print(f"\n{'=' * 60}\nExperiment type: {exp_type}")
    print(f"Discovered {len(experiments)} experiment(s):")
    for exp in experiments:
        print(
            f"  [{exp['experiment_id']}] n_hidden={exp['config']['n_hidden']} "
            f"grid_cells={exp['config'].get('grid_cells', False)}  ({exp['exp_dir']})"
        )

    succeeded, failed = [], []
    for exp in experiments:
        tag = f"{exp['experiment_id']}/{exp_type}"
        # Mirror <experiment_id>/<experiment_type> under out_root.
        out_dir = out_root / exp["experiment_id"] / exp_type
        print(f"\n[{tag}] processing n_hidden={exp['config']['n_hidden']} ...")
        try:
            data_csv, feature_model_path = resolve_paths(exp["config"], args)
            feature_extractor = get_feature_extractor(feature_model_path)
            with timer(f"Experiment {tag}"):
                process_experiment(exp, feature_extractor, args, device, out_dir)
                if not args.skip_feature_maps:
                    ensure_feature_maps(
                        feature_model_path, data_csv, out_dir / "feature_maps"
                    )
            succeeded.append(tag)
        except Exception:
            failed.append(tag)
            print(f"  FAILED experiment {tag} ({exp['exp_dir']}):")
            traceback.print_exc()

    return succeeded, failed


def main():
    args = parse_args()
    set_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_types = (
        EXPERIMENT_TYPES if args.experiment_type == "both" else [args.experiment_type]
    )

    print(f"Device: {device}")
    print(f"Experiment type(s): {', '.join(selected_types)}")

    # Shared feature extractor (frozen, identical across runs); cached by path.
    feature_extractor_cache: dict[str, ConvEncoder] = {}

    def get_feature_extractor(path: str) -> ConvEncoder:
        if path not in feature_extractor_cache:
            feature_extractor_cache[path] = load_feature_extractor(path, device)
        return feature_extractor_cache[path]

    # Feature maps depend only on the (frozen) feature extractor, so compute them
    # once per feature-model path and copy into each mirrored experiment folder.
    feature_map_cache: dict[str, Path] = {}

    def ensure_feature_maps(feature_model_path: str, data_csv: str, fm_dir: Path):
        cached = feature_map_cache.get(feature_model_path)
        if cached is not None and cached.exists():
            if fm_dir.resolve() != cached.resolve():
                shutil.copytree(cached, fm_dir, dirs_exist_ok=True)
            return
        print("  exporting feature-map / clustering section...")
        with timer("  feature-map section"):
            loader = build_dataloader(
                data_csv,
                transform=build_transform(),
                batch_size=max(args.n_featuremap_images, 1),
                shuffle=True,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            export_feature_maps(
                get_feature_extractor(feature_model_path),
                loader,
                args.n_featuremap_images,
                device,
                fm_dir,
            )
        feature_map_cache[feature_model_path] = fm_dir

    succeeded, failed = [], []
    for exp_type in selected_types:
        s, f = run_experiment_type(
            exp_type,
            data_root,
            out_root,
            args,
            device,
            get_feature_extractor,
            ensure_feature_maps,
        )
        succeeded += s
        failed += f

    if not succeeded and not failed:
        raise SystemExit(
            f"No experiments found for type(s) {selected_types} under {data_root}"
        )

    print("\n" + "=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed experiments: {failed}")
    print(f"Outputs written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
