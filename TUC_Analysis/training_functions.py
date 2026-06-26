import torch
import torch.nn as nn
import os
from tqdm import tqdm
from typing import Optional
from pathlib import Path
import numpy as np


def train(
    ae_model: nn.Module, 
    feature_extractor: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    alpha: float, 
    C_factor: float,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: Optional[str] = None,
    patience: Optional[int] = 10,
):
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    best_loss = float('inf')
    patience_counter = 0
    history = []    

    ae_model.train()
    ae_model.to(device)
    feature_extractor.eval()
    feature_extractor.to(device)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        running_loss = 0.
        for i, data in enumerate(loader):            
            inputs, _ = data 
            inputs = inputs.to(device)
            with torch.no_grad():
                features = feature_extractor(inputs)
            reconstruction_loss, _ = ae_model.training_step(optimizer=optimizer, criterion=criterion, x=features, aux=None, alpha=alpha, C_factor=C_factor)
            running_loss += reconstruction_loss

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(loader)
        history.append(epoch_loss)
        pbar.set_postfix(loss=f'{epoch_loss:.4f}', patience=f'{patience_counter}/{patience}')

        # Checkpoint best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(ae_model.state_dict(), os.path.join(checkpoint_path, 'best_model.pt'))
        else:
            patience_counter += 1

        # Early stopping
        if patience is not None and patience_counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} — best loss: {best_loss:.4f}")
            break

    return history


# grid cell encoder training function
def train_aux(
    ae_model: nn.Module, 
    feature_extractor: nn.Module,
    grid_cell_encoder: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    alpha: float, 
    C_factor: float,
    beta: float,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: Optional[str] = None,
    patience: Optional[int] = 10,
):
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    best_loss = float('inf')
    patience_counter = 0
    history = []    

    ae_model.train()
    ae_model.to(device)

    feature_extractor.eval()
    feature_extractor.to(device)

    grid_cell_encoder.to(device)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        running_loss = 0.
        for i, data in enumerate(loader):            
            inputs, xy = data 
            inputs = inputs.to(device); xy = xy.to(device)
            
            with torch.no_grad():
                features = feature_extractor(inputs)
                grid_cell_input = grid_cell_encoder(xy)

            reconstruction_loss, _ = ae_model.training_step(optimizer=optimizer, criterion=criterion, x=features, aux=grid_cell_input, alpha=alpha, C_factor=C_factor, beta=beta)
            running_loss += reconstruction_loss

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(loader)
        history.append(epoch_loss)
        pbar.set_postfix(loss=f'{epoch_loss:.4f}', patience=f'{patience_counter}/{patience}')

        # Checkpoint best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(ae_model.state_dict(), os.path.join(checkpoint_path, 'best_model_aux_input.pt'))
        else:
            patience_counter += 1

        # Early stopping
        if patience is not None and patience_counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} — best loss: {best_loss:.4f}")
            break

    return history


def get_eval_metrics(
    feature_extractor: nn.Module,
    ae_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    grid_cell_encoder: Optional[nn.Module] = None,
    feature_reconstruction_path: Optional[str] = None,
    prob_plot: float = 0.1
):
    from sklearn.metrics import r2_score

    ae_model.eval().to(device)
    feature_extractor.eval().to(device)
    if grid_cell_encoder is not None:
        grid_cell_encoder.eval().to(device)

    if feature_reconstruction_path is not None:
        feat_save_dir = Path(feature_reconstruction_path) / "features"
        feat_save_dir.mkdir(parents=True, exist_ok=True)
        if grid_cell_encoder is not None:
            grid_save_dir = Path(feature_reconstruction_path) / "grids"
            grid_save_dir.mkdir(parents=True, exist_ok=True)

    latent_vectors = []
    positions = []
    r2_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image, xy = batch
            image = image.to(device); xy = xy.to(device)

            grid_cell_input = grid_cell_encoder(xy) if grid_cell_encoder is not None else None
            features = feature_extractor(image)
            rec_features, rec_grid, h = ae_model(features, aux=grid_cell_input)

            features_np     = ae_model.pool_flatten(features).cpu().numpy()
            rec_features_np = rec_features.cpu().numpy()

            # R2 per sample: flatten (C, H, W) - (D,) per sample, then average over batch
            r2_batch = r2_score(
                features_np.reshape(len(features_np), -1),
                rec_features_np.reshape(len(rec_features_np), -1),
                multioutput='uniform_average'
            )
            r2_scores.append(r2_batch)

            latent_vectors.append(h.cpu().numpy())
            positions.append(xy.cpu().numpy())

            if feature_reconstruction_path is not None and np.random.rand() < prob_plot:
                _save_comparison_figure(
                    original=features_np,
                    reconstructed=rec_features_np,
                    save_path=feat_save_dir / f"batch_{batch_idx:05d}.png",
                    title="Feature reconstruction",
                    image=image.cpu().numpy(),
                )
                if grid_cell_encoder is not None:
                    _save_comparison_figure(
                        original=grid_cell_input.cpu().numpy(),
                        reconstructed=rec_grid.cpu().numpy(),
                        save_path=grid_save_dir / f"batch_{batch_idx:05d}.png",
                        title="Grid reconstruction"
                    )

    return (
        np.concatenate(latent_vectors, axis=0),
        np.concatenate(positions, axis=0),
        np.array(r2_scores)
    )
    

def _save_comparison_figure(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
    title: str = "",
    image: Optional[np.ndarray] = None,
    n_samples: int = 4,
):  
    """
    Handles two input shapes:
      - (B, C, H, W): mean over C → imshow (H, W) heatmap
      - (B, D):        plot as 1D line
    If image (B, C, H, W) is passed and input is spatial, adds a third row with the raw image (mean over C).
    Randomly draws min(n_samples, B) samples from the batch.
    """
    import matplotlib.pyplot as plt

    B = original.shape[0]
    n = min(n_samples, B)
    idx = np.random.choice(B, size=n, replace=False)

    original      = original[idx]
    reconstructed = reconstructed[idx]
    if image is not None:
        image = image[idx]

    is_spatial = original.ndim == 4
    n_rows = 3 if (is_spatial and image is not None) else 2

    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        orig_i = original[i]
        rec_i  = reconstructed[i]

        if is_spatial:
            orig_plot = orig_i.mean(axis=0)
            rec_plot  = rec_i.mean(axis=0)
            
            axes[0, i].imshow(orig_plot, cmap='viridis', aspect='auto', vmin=orig_plot.min(), vmax=orig_plot.max())
            axes[1, i].imshow(rec_plot,  cmap='viridis', aspect='auto', vmin=rec_plot.min(), vmax=rec_plot.max())
            for ax in axes[:2, i]:
                ax.axis('off')
            if image is not None:
                img_i = image[i].mean(axis=0)
                axes[2, i].imshow(img_i, cmap='gray', aspect='auto')
                axes[2, i].axis('off')
                axes[2, i].set_title(f"Image {i}", fontsize=8)
        else:
            axes[0, i].plot(orig_i)
            axes[1, i].plot(rec_i)
            axes[0, i].set_ylim(
                min(orig_i.min(), rec_i.min()),
                max(orig_i.max(), rec_i.max())
            )
            axes[1, i].set_ylim(axes[0, i].get_ylim())

        axes[0, i].set_title(f"Original {i}: min={orig_i.min():.2f}, max={orig_i.max():.2f}", fontsize=8)
        axes[1, i].set_title(f"Recon {i}: min={rec_i.min():.2f}, max={rec_i.max():.2f}",    fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
