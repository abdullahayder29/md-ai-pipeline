#!/usr/bin/env python3
"""
================================================================================
  CONFORMATIONAL ANALYSIS MODULE
  PyTorch Autoencoder for MD Trajectory Dimensionality Reduction
================================================================================

  PURPOSE:
  This script is the "AI brain" of the pipeline. It takes a cleaned MD
  trajectory as input and uses a Deep Learning Autoencoder to project the
  high-dimensional protein conformational space into a compact 2D "latent space"
  that can be visualized and interpreted.

  SCIENTIFIC BACKGROUND:
  ---------------------
  Each frame of an MD trajectory can be described by the 3D coordinates of
  every Cα atom. If a protein has 300 residues, each frame is a point in a
  300×3 = 900-dimensional space. It's impossible to visualize this directly.

  Traditional methods like PCA (Principal Component Analysis) find the axes
  of MAXIMUM VARIANCE using a LINEAR transformation. While fast and interpretable,
  PCA cannot capture non-linear relationships (e.g., a protein that folds into a
  U-shape — PCA would "unfold" it linearly).

  An Autoencoder learns a NON-LINEAR "compression" and "decompression" function.
  The bottleneck layer (the "latent space") captures the most important structural
  variations in a low-dimensional representation that PCA would miss.

  ARCHITECTURE:
  Input (900D) → Encoder → Latent Space (2D) → Decoder → Reconstructed Input (900D)
  The model trains by minimizing the RECONSTRUCTION ERROR: the difference between
  the original conformation and the Decoder's attempt to reconstruct it from
  just 2 numbers.

  USAGE (called by Nextflow):
  ---------------------------
  conformational_analysis.py \
    --trajectory protein_ca.xtc \
    --topology   protein_ca.pdb \
    --latent-dim 2 \
    --epochs     150 \
    --batch-size 64 \
    --output-prefix .

================================================================================
"""

import argparse
import sys
import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import MDAnalysis as mda
from MDAnalysis.analysis import align
import matplotlib
matplotlib.use('Agg')    # Use non-interactive backend (no display needed on HPC)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Always use proper logging in production scripts — never raw `print()`.
# This allows log level filtering and timestamps.
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_trajectory(trajectory_path: str, topology_path: str) -> np.ndarray:
    """
    Load an MD trajectory and extract Cα atom positions for every frame.

    Args:
        trajectory_path : Path to the .xtc trajectory file
        topology_path   : Path to the .pdb reference structure (needed by MDAnalysis)

    Returns:
        coordinates : numpy array of shape (n_frames, n_atoms * 3)
                      Each row is one flattened protein conformation.

    WHY FLATTEN?
    We reshape (n_frames, n_atoms, 3) → (n_frames, n_atoms*3) to give the
    neural network a 1D feature vector per frame. The network learns which
    dimensions encode important structural variation.
    """
    log.info(f"Loading trajectory: {trajectory_path}")
    log.info(f"Using topology:     {topology_path}")

    # MDAnalysis Universe is the central data structure: it holds the trajectory
    # and provides a rich selection language to filter atoms
    universe = mda.Universe(topology_path, trajectory_path)

    # Select only Cα atoms — these define the protein backbone shape
    ca_atoms = universe.select_atoms("name CA")

    n_frames = len(universe.trajectory)
    n_atoms  = len(ca_atoms)

    log.info(f"Trajectory loaded: {n_frames} frames, {n_atoms} Cα atoms")
    log.info(f"Feature vector size per frame: {n_atoms * 3} dimensions")

    # Align all frames to the first frame to remove global rotation/translation
    # (This should already be done by GROMACS, but we add it here as a safety net)
    ref = mda.Universe(topology_path)  # First frame as reference
    aligner = align.AlignTraj(universe, ref, select="name CA", in_memory=True)
    aligner.run()

    # Extract coordinates from every frame into a numpy array
    coordinates = np.zeros((n_frames, n_atoms * 3), dtype=np.float32)
    for i, ts in enumerate(universe.trajectory):
        # Flatten (n_atoms, 3) → (n_atoms*3,) for neural network input
        coordinates[i] = ca_atoms.positions.flatten()

    log.info(f"Coordinate matrix shape: {coordinates.shape}")
    return coordinates


def preprocess_features(coordinates: np.ndarray):
    """
    Normalize the feature matrix using z-score standardization.

    WHY SCALE?
    Neural networks train much faster and more stably when input features have
    zero mean and unit variance. Without scaling, features with large absolute
    values (e.g., x-coordinates in Angstroms) dominate the loss function.

    Returns:
        scaled_coords : normalized coordinate array
        scaler        : fitted StandardScaler (needed to invert transform later)
    """
    log.info("Standardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coordinates)
    return scaled_coords.astype(np.float32), scaler


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class MDAutoencoder(nn.Module):
    """
    A Symmetric Deep Autoencoder for MD Conformational Analysis.

    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────┐
    │  INPUT LAYER    (input_dim)  ← e.g., 900 features        │
    │                                                          │
    │  ENCODER:                                                │
    │    Linear(input_dim → 512)  + BatchNorm + ELU + Dropout  │
    │    Linear(512 → 256)        + BatchNorm + ELU + Dropout  │
    │    Linear(256 → 128)        + BatchNorm + ELU            │
    │                                                          │
    │  BOTTLENECK (LATENT SPACE):                              │
    │    Linear(128 → latent_dim)  ← e.g., 2 dimensions        │
    │                                                          │
    │  DECODER (mirror of encoder):                           │
    │    Linear(latent_dim → 128) + BatchNorm + ELU + Dropout  │
    │    Linear(128 → 256)        + BatchNorm + ELU + Dropout  │
    │    Linear(256 → 512)        + BatchNorm + ELU            │
    │    Linear(512 → input_dim)                               │
    │                                                          │
    │  OUTPUT LAYER   (input_dim) ← reconstructed conformation │
    └──────────────────────────────────────────────────────────┘

    Key design choices:
    - ELU activation: smoother than ReLU, avoids "dying neuron" problem
    - BatchNorm: stabilizes training, acts as regularization
    - Dropout(0.2): prevents overfitting, improves generalization
    - Symmetric: encoder and decoder have mirror architectures (common for AEs)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Args:
            input_dim  : Number of input features (n_atoms * 3)
            latent_dim : Size of the bottleneck (usually 2 or 3 for visualization)
        """
        super(MDAutoencoder, self).__init__()

        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── ENCODER ──────────────────────────────────────────────────────────
        # The encoder progressively compresses the input into a lower-dimensional
        # representation. Think of it as "learning what's important."
        self.encoder = nn.Sequential(
            # Block 1: input_dim → 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),           # Normalize activations within the batch
            nn.ELU(),                      # Exponential Linear Unit activation
            nn.Dropout(0.2),               # Randomly zero 20% of neurons during training

            # Block 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            # Block 3: 256 → 128 (no dropout near the bottleneck)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )

        # ── BOTTLENECK (LATENT SPACE) ─────────────────────────────────────────
        # This single layer is the core of the Autoencoder.
        # It forces the model to represent each conformation in just `latent_dim`
        # numbers — a severe information bottleneck.
        # No activation function here: we want an unconstrained latent space.
        self.bottleneck = nn.Linear(128, latent_dim)

        # ── DECODER ──────────────────────────────────────────────────────────
        # The decoder tries to RECONSTRUCT the original conformation from the
        # compressed latent representation. It's the mirror image of the encoder.
        self.decoder = nn.Sequential(
            # Block 1: latent_dim → 128
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2),

            # Block 2: 128 → 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            # Block 3: 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),

            # Output layer: 512 → input_dim
            # No activation: we want raw values to match the (scaled) input
            nn.Linear(512, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input conformations into the latent space.
        Used during inference to extract latent coordinates.
        """
        h = self.encoder(x)
        return self.bottleneck(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct conformations from latent coordinates.
        Used during training to compute the reconstruction loss.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode then decode.
        Returns:
            z_latent     : the bottleneck representation (latent coordinates)
            x_reconstructed : the decoder's attempt to recreate the input
        """
        z_latent       = self.encode(x)
        x_reconstructed = self.decode(z_latent)
        return z_latent, x_reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_autoencoder(
    model        : MDAutoencoder,
    data_tensor  : torch.Tensor,
    n_epochs     : int,
    batch_size   : int,
    device       : torch.device,
    learning_rate: float = 1e-3
) -> list:
    """
    Train the Autoencoder using mini-batch gradient descent.

    TRAINING CONCEPT:
    1. Feed a mini-batch of conformations through the network (forward pass)
    2. Calculate the RECONSTRUCTION LOSS (MSE between input and output)
    3. Backpropagate the loss gradient through the network
    4. Update weights using the Adam optimizer to reduce the loss
    5. Repeat until convergence

    Args:
        model       : The MDAutoencoder instance
        data_tensor : All trajectory frames as a PyTorch tensor
        n_epochs    : Number of complete passes over the training data
        batch_size  : Number of frames per mini-batch
        device      : 'cuda' or 'cpu'
        learning_rate : Adam optimizer learning rate

    Returns:
        loss_history : List of average loss per epoch (for plotting)
    """
    log.info(f"Starting training: {n_epochs} epochs, batch_size={batch_size}, device={device}")

    # ── DataLoader ──────────────────────────────────────────────────────────
    # PyTorch's DataLoader handles batching and shuffling automatically.
    # shuffle=True randomizes frame order each epoch — important for MD data
    # which is temporally autocorrelated.
    dataset    = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Adam (Adaptive Moment Estimation) is the de facto standard optimizer.
    # It adapts the learning rate per parameter, converging faster than SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # ReduceLROnPlateau halves the learning rate when the loss plateaus.
    # This "fine-tunes" the model when it's close to convergence.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # ── Loss Function ─────────────────────────────────────────────────────────
    # Mean Squared Error (MSE): penalizes the average squared distance between
    # original and reconstructed coordinates. Perfect reconstruction = loss of 0.
    criterion = nn.MSELoss()

    # Move model to GPU if available
    model = model.to(device)
    model.train()    # Set to training mode (enables dropout, batchnorm tracking)

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for (batch,) in dataloader:
            batch = batch.to(device)      # Move data to the same device as model

            # ── Forward Pass ──────────────────────────────────────────────────
            optimizer.zero_grad()         # Clear gradients from previous step
            _, x_reconstructed = model(batch)

            # ── Loss Calculation ──────────────────────────────────────────────
            loss = criterion(x_reconstructed, batch)

            # ── Backward Pass ─────────────────────────────────────────────────
            loss.backward()               # Compute gradients via backpropagation
            optimizer.step()             # Update weights using Adam

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        # Progress logging every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch [{epoch:4d}/{n_epochs}] | Loss: {avg_loss:.6f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    log.info("Training complete.")
    return loss_history


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: INFERENCE — EXTRACT LATENT COORDINATES
# ─────────────────────────────────────────────────────────────────────────────
def extract_latent_coordinates(
    model       : MDAutoencoder,
    data_tensor : torch.Tensor,
    device      : torch.device
) -> np.ndarray:
    """
    After training, pass all trajectory frames through the ENCODER ONLY
    to extract their 2D latent representations.

    This is the actual "dimensionality reduction" result: each frame is
    now represented as a single point in 2D space.

    Args:
        model       : Trained MDAutoencoder
        data_tensor : All trajectory frames
        device      : 'cuda' or 'cpu'

    Returns:
        latent_coords : numpy array of shape (n_frames, latent_dim)
    """
    log.info("Extracting latent space coordinates...")

    model.eval()    # Disable dropout and batchnorm tracking during inference
    latent_coords = []

    # torch.no_grad() disables gradient computation — faster, uses less memory
    with torch.no_grad():
        # Process in batches to avoid GPU memory overflow on large trajectories
        batch_size = 512
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size].to(device)
            z     = model.encode(batch)
            latent_coords.append(z.cpu().numpy())    # Move back to CPU for numpy

    return np.concatenate(latent_coords, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_latent_space(
    latent_coords : np.ndarray,
    output_path   : str,
    n_frames      : int
):
    """
    Create a publication-quality visualization of the conformational landscape.

    We color-code each point by SIMULATION TIME to reveal temporal patterns:
    - Are conformations sampling one region early and another later? (slow dynamics)
    - Are they rapidly interconverting? (fast dynamics)
    - Are there distinct clusters? (metastable states)

    Args:
        latent_coords : (n_frames, latent_dim) array of encoder outputs
        output_path   : File path to save the figure
        n_frames      : Total number of trajectory frames (for color axis)
    """
    log.info(f"Generating latent space visualization → {output_path}")

    # Color each frame by its time index (0 = early, n_frames = late)
    time_array = np.arange(n_frames)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.35)

    # ── Left panel: Scatter plot ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    sc  = ax1.scatter(
        latent_coords[:, 0],
        latent_coords[:, 1],
        c         = time_array,
        cmap      = 'plasma',
        alpha     = 0.6,
        s         = 8,          # Small points to show density
        linewidths = 0,
        rasterized = True       # Rasterize scatter for faster PDF rendering
    )

    cbar = plt.colorbar(sc, ax=ax1, shrink=0.8)
    cbar.set_label('Simulation Frame (time →)', fontsize=11)

    ax1.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12, fontweight='bold')
    ax1.set_title('Conformational Landscape\n(Autoencoder Latent Space)', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ── Right panel: Kernel Density Estimate (2D histogram) ──────────────────
    ax2 = fig.add_subplot(gs[1])
    sns.kdeplot(
        x      = latent_coords[:, 0],
        y      = latent_coords[:, 1],
        ax     = ax2,
        cmap   = 'Blues',
        fill   = True,
        thresh = 0.05,
        levels = 10,
        alpha  = 0.85
    )

    ax2.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Density\n(Free Energy Landscape Proxy)', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Annotation: high-density regions = low free energy = stable conformations
    ax2.text(0.02, 0.02, "High density = Low free energy\n(Stable states)",
             transform=ax2.transAxes, fontsize=8, alpha=0.6,
             verticalalignment='bottom')

    fig.suptitle('MD Trajectory — PyTorch Autoencoder Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Latent space plot saved → {output_path}")


def plot_training_loss(loss_history: list, output_path: str):
    """
    Plot the training loss curve to verify the model converged properly.

    WHAT TO LOOK FOR:
    - Smooth decrease: the model is learning
    - Plateau: the model has converged (good!)
    - Oscillating or increasing: learning rate may be too high
    - Not decreasing: possible bug, or learning rate too low

    Args:
        loss_history : List of per-epoch average loss values
        output_path  : File path to save the figure
    """
    log.info(f"Generating training loss plot → {output_path}")

    epochs = range(1, len(loss_history) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, loss_history, linewidth=2, color='steelblue', label='Training Loss (MSE)')
    ax.fill_between(epochs, loss_history, alpha=0.15, color='steelblue')

    # Mark the final loss value
    ax.axhline(y=loss_history[-1], color='crimson', linestyle='--', alpha=0.7,
               label=f'Final loss: {loss_history[-1]:.5f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('Autoencoder Training Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')    # Log scale makes the convergence pattern clearer

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Training loss plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6: MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_arguments():
    """
    Parse command-line arguments. All parameters are passed in by Nextflow
    from the process block in main.nf, making this script self-contained
    and testable outside of Nextflow too.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Autoencoder for MD Conformational Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trajectory',    required=True, help='Path to .xtc trajectory file')
    parser.add_argument('--topology',      required=True, help='Path to .pdb reference topology')
    parser.add_argument('--latent-dim',    type=int,   default=2,   help='Autoencoder latent space dimensions')
    parser.add_argument('--epochs',        type=int,   default=150, help='Number of training epochs')
    parser.add_argument('--batch-size',    type=int,   default=64,  help='Mini-batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Adam optimizer learning rate')
    parser.add_argument('--output-prefix', type=str,   default='.',  help='Directory for output files')
    return parser.parse_args()


def main():
    args   = parse_arguments()
    outdir = Path(args.output_prefix)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Device Selection ──────────────────────────────────────────────────────
    # Automatically use GPU (CUDA) if available, otherwise fall back to CPU.
    # On a GROMACS GPU node, CUDA should be available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Computing device: {device}")
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load and preprocess trajectory ────────────────────────────────
    raw_coords = load_trajectory(args.trajectory, args.topology)
    scaled_coords, scaler = preprocess_features(raw_coords)

    input_dim = scaled_coords.shape[1]
    n_frames  = scaled_coords.shape[0]
    log.info(f"Input dimensionality: {input_dim} | Latent dimensionality: {args.latent_dim}")

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(scaled_coords, dtype=torch.float32)

    # ── Step 2: Build the Autoencoder ─────────────────────────────────────────
    model = MDAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model architecture:\n{model}")
    log.info(f"Total trainable parameters: {n_params:,}")

    # ── Step 3: Train the model ───────────────────────────────────────────────
    loss_history = train_autoencoder(
        model        = model,
        data_tensor  = data_tensor,
        n_epochs     = args.epochs,
        batch_size   = args.batch_size,
        device       = device,
        learning_rate = args.learning_rate
    )

    # ── Step 4: Extract latent coordinates ───────────────────────────────────
    latent_coords = extract_latent_coordinates(model, data_tensor, device)
    log.info(f"Latent coordinates shape: {latent_coords.shape}")

    # ── Step 5: Save outputs ──────────────────────────────────────────────────

    # 5a. Save latent coordinates as CSV (for downstream analysis, clustering, etc.)
    latent_df = pd.DataFrame(
        latent_coords,
        columns=[f'z{i+1}' for i in range(args.latent_dim)]
    )
    latent_df.insert(0, 'frame', range(n_frames))
    csv_path = outdir / 'latent_coordinates.csv'
    latent_df.to_csv(csv_path, index=False)
    log.info(f"Latent coordinates saved → {csv_path}")

    # 5b. Save the trained model weights (for later fine-tuning or inference)
    model_path = outdir / 'autoencoder_model.pt'
    torch.save({
        'model_state_dict' : model.state_dict(),
        'input_dim'        : input_dim,
        'latent_dim'       : args.latent_dim,
        'loss_history'     : loss_history,
        'scaler_mean'      : scaler.mean_,
        'scaler_scale'     : scaler.scale_,
    }, model_path)
    log.info(f"Model weights saved → {model_path}")

    # 5c. Generate visualizations
    plot_latent_space(
        latent_coords = latent_coords,
        output_path   = str(outdir / 'latent_space.png'),
        n_frames      = n_frames
    )
    plot_training_loss(
        loss_history = loss_history,
        output_path  = str(outdir / 'reconstruction_loss.png')
    )

    log.info("=" * 60)
    log.info("  AI Analysis complete! Summary:")
    log.info(f"  Frames analyzed     : {n_frames}")
    log.info(f"  Input dimensions    : {input_dim}")
    log.info(f"  Latent dimensions   : {args.latent_dim}")
    log.info(f"  Final training loss : {loss_history[-1]:.6f}")
    log.info(f"  Outputs in          : {outdir.resolve()}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
