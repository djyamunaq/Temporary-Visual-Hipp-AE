# TUC Contribution

## TODOs & Questions

- DONE: Erik will implement the following things within this pipeline: Instead of learning a convolution of the feature maps from the CERTHs SAM-distilled ConvEncoder, we will pool the feature maps and use them as input for the AE. Every "landmark" will be represented by a single feature vector instead of a feature map. Questions would be: Is that what you want? A SimCLR pipeline on top of the feature maps would be overkill for such a simple environment. Pooling is also more biologically plausible. We can use a simple average pooling (first) or a more complex pooling method (e.g., attention-based pooling) to extract the most relevant features from the feature maps.
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
├── vit_notebook.ipynb             # ViT feature analysis (SOTA comparison)
└── requirements.txt               # Python dependencies (not all are strictly necessary for the pipeline)
```

---

## Modes

### Features-only
The autoencoder compresses ConvEncoder features into a low-dimensional bottleneck and reconstructs them. No positional signal is used during training. Now with a Pooling layer that pools the feature maps into a single feature vector per landmark.

```bash
python main.py --data-csv path/to/data.csv
```

### Grid-cell auxiliary (with Pooling experimental can cause bugs)
A frozen, analytically defined grid-cell code (based on agent XY position) is appended as an auxiliary reconstruction target. The bottleneck is jointly optimised to reconstruct both visual features and the grid code, encouraging spatially structured representations.

```bash
python main.py --data-csv path/to/data.csv --grid-cells
```

Key auxiliary hyperparameters: `--beta` (grid-loss weight), `--scales`, `--orientations`, `--cells-per-module`.

---

## Notebooks

Each notebook (`attention_notebook.ipynb`, `gridcell_notebook.ipynb`) can run the full training loop (equivalent to `main.py`) **or** load a pre-trained checkpoint and go straight to evaluation. Post-training analysis uses `ae_model/plotting.py` to compute spatial ratemaps, detect place fields, and visualise their statistics. `vit_notebook.ipynb` is just a sanity check on how DiNO trained ViT features look like in the Webots environment (SOTA comparison).

---

