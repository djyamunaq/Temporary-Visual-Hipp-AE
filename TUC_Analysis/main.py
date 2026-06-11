"""
Training script for the hippocampal Conv autoencoder on Webots frames.

With --grid-cells the autoencoder receives a grid-cell code of the agent's
position as an auxiliary target and reconstructs both the image features and
the grid code.
"""

import argparse
import os

import torch
from torchvision.transforms import v2
import numpy as np

from grid_cells.encoder import GridCellEncoder, GridModule, save_grid_encoder, load_grid_encoder
from ae_model.cnn_hippocampal_ae import Conv_AE
from utils import build_dataloader, WebotsFrameDataset, set_seed, get_parameters, timer
from attention_model.conv_encoder import ConvEncoder
from training_functions import train, train_aux, get_eval_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train a hippocampal autoencoder on Webots frames, with optional grid-cell auxiliary target.")

    p.add_argument("--grid-cells", action="store_true",
                   help="Use the grid-cell auxiliary target (sets d_aux and beta).")

    p.add_argument("--data-csv", default="../Denis/HIP_AE_VISUAL/Datasets/Tmaze_2/data.csv")
    p.add_argument("--feature-model-path", default="./attention_model/SAM_weights/")
    p.add_argument("--checkpoint-base", default="./ae_model/feature_extractor_ae_checkpoint/",
                   help="Base dir; a mode subdir is appended.")

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=None, help="Random seed; None for nondeterministic.")

    p.add_argument("--n_hidden", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_learning_rate", type=float, default=1e-7)
    p.add_argument("--alpha", type=float, default=1e5)
    p.add_argument("--c_factor", type=float, default=1000.0)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight of the grid-cell reconstruction loss; "
                        "forced to 0 without --grid-cells.")
    p.add_argument("--arena-size", type=float, default=1.2, help="[m]")
    p.add_argument("--grid-res", type=int, default=200)
    p.add_argument("--cells-per-module", type=int, default=4)
    p.add_argument("--scales", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.60, 0.80, 1.0])
    p.add_argument("--orientations", type=float, nargs="+", default=[0.00, 0.10, 0.20, 0.30, 0.40, 0.50])

    args = p.parse_args()
    if len(args.scales) != len(args.orientations):
        p.error(f"--scales ({len(args.scales)}) and --orientations "
                f"({len(args.orientations)}) must have equal length.")
    return args


def load_feature_extractor(feature_model_path):
    """Load the frozen ConvEncoder feature extractor."""
    ckpt = torch.load(feature_model_path + "best.ckpt",
                      weights_only=False, map_location=torch.device("cpu"))

    encoder_state_dict = ckpt["state_dict"]
    for k in list(encoder_state_dict.keys()):
        new_k = k.replace("encoder.encoder", "encoder").replace("encoder.concept_proj", "concept_proj")
        if new_k != k:
            encoder_state_dict[new_k] = encoder_state_dict.pop(k)

    feature_extractor = ConvEncoder()
    feature_extractor.load_state_dict(ckpt["state_dict"])
    feature_extractor.eval()
    return feature_extractor


def build_grid_encoder(args):
    modules = [
        GridModule(scale=s, orientation=o, n_cells=args.cells_per_module)
        for s, o in zip(args.scales, args.orientations)
    ]
    return GridCellEncoder(modules)


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_tag = "grid" if args.grid_cells else "features_only"
    checkpoint_dir = os.path.join(args.checkpoint_base, mode_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Mode: {mode_tag} | device: {device}")
    print(f"Checkpoints -> {checkpoint_dir}")
    
    # generate config json for this run
    config = vars(args)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        import json
        json.dump(config, f, indent=4)

    # data
    tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((248, 328)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = build_dataloader(
        args.data_csv,
        transform=tf,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # frozen feature extractor
    feature_extractor = load_feature_extractor(args.feature_model_path)

    # autoencoder + grid-cell encoder (None when features only)
    if args.grid_cells:
        grid_cell_encoder = build_grid_encoder(args)
        ae_model = Conv_AE(
            n_hidden=args.n_hidden,
            last_layer_activation=torch.nn.Sigmoid(),
            d_aux=grid_cell_encoder.n_cells,
        )
        beta = args.beta
        save_grid_encoder(grid_cell_encoder, os.path.join(checkpoint_dir, "grid_encoder.pt"))
    else:
        grid_cell_encoder = None
        ae_model = Conv_AE(n_hidden=args.n_hidden, last_layer_activation=torch.nn.Sigmoid(), d_aux=None)
        beta = 0.0
    
    print(f"Number of parameters: {get_parameters(ae_model)} M")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.min_learning_rate
    )

    # train (grid-cell encoder and beta omitted when features only)
    with timer("Training complete in: "):
        if grid_cell_encoder is not None:
            history = train_aux(
                ae_model=ae_model,
                feature_extractor=feature_extractor,
                grid_cell_encoder=grid_cell_encoder,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                alpha=args.alpha,
                C_factor=args.c_factor,
                beta=beta,
                device=device,
                num_epochs=args.num_epochs,
                scheduler=scheduler,
                checkpoint_path=checkpoint_dir,
                patience=args.patience,
            )
        else:
            history = train(
                feature_extractor=feature_extractor,
                ae_model=ae_model,
                num_epochs=args.num_epochs,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                alpha=args.alpha,
                C_factor=args.c_factor,
                device=device,
                checkpoint_path=checkpoint_dir,
                patience=args.patience,
            )

    np.save(os.path.join(checkpoint_dir, "loss_history.npy"), np.asarray(history))

    # eval
    with timer("Extracting embeddings in: "):
        embeddings, positions, r2_features = get_eval_metrics(
            dataloader=loader,
            ae_model=ae_model,
            feature_extractor=feature_extractor,
            grid_cell_encoder=grid_cell_encoder,
            device=device,
        )

    accuracy = np.mean(r2_features)
    print(f"Embeddings shape: {np.asarray(embeddings).shape}")
    print(f"Accuracy (feature reconstruction): {accuracy}")


if __name__ == "__main__":
    main()
    