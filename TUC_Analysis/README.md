# TUC Contribution

## TODOs & Questions

- Erik will implement the following things within this pipeline: Instead of learning a convolution of the feature maps from the CERTHs SAM-distilled ConvEncoder, we will pool the feature maps and use them as input for the AE. Every "landmark" will be represented by a single feature vector instead of a feature map. Questions would be: Is that what you want? A SimCLR pipeline on top of the feature maps would be overkill for such a simple environment. Pooling is also more biologically plausible. We can use a simple average pooling (first) or a more complex pooling method (e.g., attention-based pooling) to extract the most relevant features from the feature maps.
- Everything is really dependent on what to do with the bottleneck embeddings. Sparse vs. overlapping representations? To make this pipeline as good as possible, we need to know what the downstream task is. If the downstream task is to learn a spatial map of the environment, then we should aim for sparse representations. If the downstream task is to learn a more general representation of the environment, then we should aim for overlapping representations.  
- What should be done with the grid-cell auxiliary? Should we keep it or remove it? 
- Why the change from a classical Deep Net to Spiking Neural Networks? 

---

## Repository layout

```
.
├── main.py                        # training + eval of the AE feature extractor with optional grid-cell auxiliary
├── training_functions.py          # train / train_aux / get_eval_metrics
├── utils.py                       # dataloader, dataset, seed, timing helpers
│
├── ae_model/
│   ├── dense_hippocampal_ae.py    # Dense autoencoder that pools the feature maps with optional aux head
│   ├── plotting.py                # Ratemap computation and place-field analysis from Denis
│   └── feature_extractor_ae_checkpoint/   # Saved checkpoints of the AE training
│
├── attention_model/
│   ├── conv_encoder.py            # Frozen ConvEncoder feature extractor (CERTH distilled SAM-based CNN)
│   ├── Visualization.py           # Attention / feature visualisation utilities (not used in the pipeline)
│   └── SAM_weights/               # Pre-trained ConvEncoder weights (best.ckpt)
│
├── grid_cells/
│   ├── encoder.py                 # GridCellEncoder deterministic Fourier grid codes
│   └── utils.py                   # Grid-cell helper functions
│
├── attention_notebook.ipynb       # Feature-extractor AE only analysis (no grid-cell auxiliary)
├── gridcell_notebook.ipynb        # Grid-cell auxiliary AE analysis (with grid-cell auxiliary)
└── vit_notebook.ipynb             # ViT feature analysis (SOTA comparison)
```

---

## Modes

### Features-only
The autoencoder compresses ConvEncoder features into a low-dimensional bottleneck and reconstructs them. No positional signal is used during training.

```bash
python main.py --data-csv path/to/data.csv
```

### Grid-cell auxiliary
A frozen, analytically defined grid-cell code (based on agent XY position) is appended as an auxiliary reconstruction target. The bottleneck is jointly optimised to reconstruct both visual features and the grid code, encouraging spatially structured representations.

```bash
python main.py --data-csv path/to/data.csv --grid-cells
```

Key auxiliary hyperparameters: `--beta` (grid-loss weight), `--scales`, `--orientations`, `--cells-per-module`.

---

## Notebooks

Each notebook (`attention_notebook.ipynb`, `gridcell_notebook.ipynb`) can run the full training loop (equivalent to `main.py`) **or** load a pre-trained checkpoint and go straight to evaluation. Post-training analysis uses `ae_model/plotting.py` to compute spatial ratemaps, detect place fields, and visualise their statistics. `vit_notebook.ipynb` is just a sanity check on how DiNO trained ViT features look like in the Webots environment (SOTA comparison).

---


## Analysis

`analysis.py` is a batch exporter that reproduces every figure and quantitative result of `attention_notebook.ipynb` for a whole root of training runs, without opening a notebook. Each run (produced by `main.py`) is discovered by its `config.json`, re-loaded from its checkpoint, evaluated, and written to a mirrored output tree.

### Expected input layout

Point `--data-root` at a folder holding one subfolder per run. Each run contains an experiment-type subfolder with the checkpoint and config:

```
<data-root>/
└── <experiment_id>/
    └── <experiment_type>/            # features_only | features_only_pool1x1
        ├── config.json
        ├── best_model.pt             # best_model_aux_input.pt for grid-cell runs
        ├── loss_history.npy
        └── grid_encoder.pt           # grid-cell runs only
```

The same `<experiment_id>/<experiment_type>` nesting is mirrored under `--out-root`. For each run the exporter writes feature-map montages and per-image clusterings, the loss curve, reconstruction figures, spatial ratemaps, an SSCP matrix, place-field statistics and example figures, plus `metrics.json` and `arrays.npz` with the raw arrays.

> Run from inside `TUC_Analysis` so the relative `data_csv` / `feature_model_path` values stored in each `config.json` resolve correctly.

### Usage

```bash
python analysis.py --data-root path/to/runs --out-root path/to/analysis_outputs
```

### Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--data-root` | str | **required** | Root folder with one subfolder per run (`<data-root>/<experiment_id>/<experiment_type>/config.json`). |
| `--out-root` | str | **required** | Destination for the analysis outputs; the `<experiment_id>/<experiment_type>` nesting is mirrored here. |
| `--experiment-type` | `features_only` \| `features_only_pool1x1` \| `both` | `both` | Which experiment type (under each `experiment_id`) to analyze. |
| `--n-featuremap-images` | int | `8` | Number of dataset images used in the shared feature-map / clustering section. |
| `--device` | str | auto | Torch device (e.g. `cuda`, `cpu`). Defaults to CUDA if available, else CPU. |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--batch-size` | int | `512` | Batch size used during evaluation. |
| `--num-workers` | int | `0` | DataLoader workers (`0` is safest cross-platform). |
| `--recon-prob` | float | `0.1` | Per-batch probability of saving a reconstruction figure during eval. |
| `--data-csv` | str | from config | Override the `data.csv` path stored in each `config.json`. |
| `--feature-model-path` | str | from config | Override the frozen feature-extractor path stored in each `config.json`. |
| `--skip-feature-maps` | flag | off | Skip the (expensive) shared feature-map / clustering section. |

### Examples

Analyze only the `features_only` runs on CPU and skip the expensive clustering section:

```bash
python analysis.py \
    --data-root path/to/runs \
    --out-root path/to/analysis_outputs \
    --experiment-type features_only \
    --device cpu \
    --skip-feature-maps
```

Override the dataset and feature-extractor paths for every run:

```bash
python analysis.py \
    --data-root path/to/runs \
    --out-root path/to/analysis_outputs \
    --data-csv path/to/data.csv \
    --feature-model-path ./attention_model/SAM_weights
```

Each run is processed independently: a failure in one run is logged (with traceback) and does not stop the rest of the batch. The frozen feature extractor and its feature-map section are cached per feature-model path, so shared work is computed once and copied into each mirrored output folder.
