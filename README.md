# MD-AI Pipeline
### A Nextflow-Orchestrated Molecular Dynamics + PyTorch Conformational Analysis Workflow

[![Nextflow](https://img.shields.io/badge/Nextflow-≥23.04-brightgreen)](https://www.nextflow.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2023.3-blue)](https://www.gromacs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This repository contains a production-ready, modular bioinformatics pipeline that automates the full lifecycle of protein conformational analysis — from a raw PDB structure file to a publication-quality visualization of the protein's conformational landscape.

The core innovation is the integration of a PyTorch Deep Learning Autoencoder as the analytical engine, replacing traditional linear methods (PCA) with non-linear dimensionality reduction that can capture complex conformational transitions hidden in MD trajectory data.

```
Raw PDB File
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NEXTFLOW ORCHESTRATION LAYER                      │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│  │   PREPARE    │──▶│  PRODUCTION  │──▶│ TRAJECTORY PROCESSING │   │
│  │   SYSTEM     │   │     MD       │   │   (PBC Correction)    │   │
│  │  (GROMACS)   │   │  (GROMACS)   │   │      (GROMACS)        │   │
│  └──────────────┘   └──────────────┘   └───────────┬───────────┘   │
│                                                     │               │
│                                         ┌───────────▼───────────┐   │
│                                         │     AI ANALYSIS       │   │
│                                         │  (PyTorch Autoencoder)│   │
│                                         └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
Latent Space Map + Free Energy Landscape Proxy
```

---

Scientific Background

### Why Molecular Dynamics?
MD simulations propagate Newton's equations of motion for every atom in a biological system, generating a **trajectory** — a time-series of protein conformations. This allows us to observe processes like folding, ligand binding, and allosteric communication that are inaccessible to static crystallography.

### Why Dimensionality Reduction?
A protein with 300 residues has **~900 degrees of freedom** (x, y, z per Cα atom). Each frame of the trajectory is a point in this 900-dimensional space. Direct analysis is impossible. We must project this space onto 2–3 dimensions while preserving the most important structural differences.

### Why an Autoencoder Instead of PCA?
| Method | Type | Captures non-linear relationships? | Interpretable |
|--------|------|-------------------------------------|---------------|
| PCA    | Linear | ❌ No | ✅ Yes |
| t-SNE  | Non-linear | ✅ Yes | ❌ No (no encoder function) |
| **Autoencoder** | **Non-linear** | **✅ Yes** | **⚠️ Partial** |

An Autoencoder is uniquely suited here because:
1. It learns a **continuous, invertible mapping** (unlike t-SNE)
2. The decoder can **generate new conformations** by sampling the latent space
3. The latent space is a **learnable, differentiable function** of the input
4. It scales to **very large systems** where PCA becomes memory-limited

---

## Repository Structure

```
md-ai-pipeline/
├── main.nf                    # Pipeline orchestration logic (Nextflow DSL2)
├── nextflow.config            # Infrastructure configuration (CPU/GPU/HPC profiles)
│
├── bin/
│   └── conformational_analysis.py  # PyTorch Autoencoder (AI analysis module)
│
├── data/
│   ├── input/
│   │   └── protein.pdb        # Your input structure (you provide this)
│   ├── mdp/
│   │   ├── em.mdp             # Energy minimization parameters
│   │   ├── nvt.mdp            # NVT equilibration parameters
│   │   ├── npt.mdp            # NPT equilibration parameters
│   │   └── md.mdp             # Production MD parameters
│   └── test/
│       └── small_peptide.pdb  # Minimal test structure for CI
│
├── results/                   # Pipeline outputs (generated at runtime)
│   ├── 01_system_prep/
│   ├── 02_production_md/
│   ├── 03_processed_trajectory/
│   ├── 04_ai_analysis/
│   │   ├── latent_space.png          ← Main result: conformational landscape
│   │   ├── reconstruction_loss.png   ← Training convergence
│   │   ├── latent_coordinates.csv    ← Raw data for custom analysis
│   │   └── autoencoder_model.pt      ← Saved PyTorch model
│   └── reports/
│       ├── pipeline_report.html      ← Execution report
│       └── pipeline_timeline.html    ← Gantt chart of process execution
│
└── README.md
```

---

## Architecture Deep-Dive

### Layer 1: Nextflow Orchestration (`main.nf`)

Nextflow uses a **dataflow programming model**: you define processes and connect them with channels. The framework automatically resolves execution order, handles parallelism, and manages failures.

```
CHANNEL: pdb_file → PREPARE_SYSTEM → [minimized.gro, topol.top]
                                              │
                                              ▼
                              PRODUCTION_MD (GPU) → [md.xtc, md.tpr]
                                              │
                                              ▼
                              PROCESS_TRAJECTORY → [protein_ca.xtc]
                                              │
                                              ▼
                              AI_ANALYSIS → [latent_space.png, model.pt]
```

**Key Nextflow concepts used in this pipeline:**

| Concept | Where Used | Why |
|---------|------------|-----|
| `process` | All 4 stages | Define atomic units of work |
| `channel` | `workflow {}` block | Connect processes without explicit ordering |
| `publishDir` | Every process | Copy results to permanent location |
| `label` | `PRODUCTION_MD` | Map process to GPU resources in config |
| `conda` | Every process | Auto-manage software environments |
| `-profile` | CLI flag | Switch between laptop/HPC/cloud environments |
| `workflow.onComplete` | End of file | Run post-pipeline cleanup/reporting |

### Layer 2: GROMACS System Preparation

The `PREPARE_SYSTEM` process performs the standard GROMACS setup protocol:

```
PDB → pdb2gmx → editconf → solvate → genion → grompp → mdrun (EM)
       ↓            ↓          ↓          ↓
   topology     periodic    solvated   neutralized
    (top)        box         system      system
```

Each step is essential:
- **`pdb2gmx`**: Assigns force field parameters and adds hydrogen atoms
- **`editconf`**: Creates a periodic boundary box (dodecahedron is most space-efficient)
- **`solvate`**: Fills the box with TIP3P water molecules
- **`genion`**: Adds Na⁺/Cl⁻ ions to achieve electrical neutrality
- **Energy minimization**: Relaxes steric clashes before heating

### Layer 3: PyTorch Autoencoder (`bin/conformational_analysis.py`)

The Python module is deliberately separated from Nextflow. It is a fully independent, testable Python script:

```
MDAnalysis → numpy array      PyTorch DataLoader
(trajectory)  (n_frames,       (mini-batches)
              n_atoms×3)            │
                  │                 ▼
                  └──────→  Encoder → Latent Space (2D)
                                           │
                                     Decoder ←┘
                                           │
                                    MSE Loss
                                    (backprop)
```

**Autoencoder architecture:**
```
Input (900) → Linear(512)+BN+ELU+Drop → Linear(256)+BN+ELU+Drop →
Linear(128)+BN+ELU → Bottleneck(2) → Linear(128)+BN+ELU+Drop →
Linear(256)+BN+ELU+Drop → Linear(512)+BN+ELU → Output(900)
```

---

## Quick Start

### Prerequisites

- [Nextflow](https://www.nextflow.io/docs/latest/install.html) ≥ 23.04
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) (recommended)
- GROMACS (installed via conda by the pipeline)
- An NVIDIA GPU (optional, CPU fallback available)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/md-ai-pipeline.git
cd md-ai-pipeline

# 2. Make the analysis script executable
chmod +x bin/conformational_analysis.py

# 3. Verify Nextflow installation
nextflow -version
```

### Running the Pipeline

**Option A: Standard local run (no GPU)**
```bash
nextflow run main.nf -profile standard
```

**Option B: Local run with GPU acceleration**
```bash
nextflow run main.nf -profile gpu_local
```

**Option C: HPC cluster (SLURM)**
```bash
nextflow run main.nf -profile slurm
```

**Option D: Quick test (fast validation)**
```bash
nextflow run main.nf -profile test
```

**Customizing parameters at runtime:**
```bash
nextflow run main.nf \
  --pdb_file   /path/to/your/protein.pdb \
  --sim_time   50 \           # 50 ns simulation
  --latent_dim 3 \            # 3D latent space
  --n_epochs   300 \
  -profile     slurm \
  -resume                     # Resume from last successful stage
```

> **💡 The `-resume` flag** is one of Nextflow's most powerful features. If your pipeline fails at Stage 3, `-resume` will skip Stages 1 and 2 (already completed) and restart from Stage 3. This saves enormous computational time on HPC clusters.

### Input Requirements

| File | Format | Notes |
|------|--------|-------|
| `protein.pdb` | PDB | Standard crystallographic format; download from [RCSB PDB](https://www.rcsb.org/) |
| `em.mdp`, `nvt.mdp`, `npt.mdp`, `md.mdp` | GROMACS MDP | Pre-configured in `data/mdp/` |

---

## Output Interpretation

### `latent_space.png` — The Main Result

This is your **conformational landscape**. Each point represents one trajectory frame, colored by simulation time.

| Pattern | Interpretation |
|---------|---------------|
| One tight cluster | Protein stays in one stable conformation |
| Two distinct clusters | Two metastable states (conformational switching) |
| Diffuse cloud | Highly flexible/disordered protein |
| Points moving left → right over time | Slow conformational drift during simulation |

### `reconstruction_loss.png` — Model Quality Check

- **Decreasing, then plateau**: ✅ Model converged correctly
- **Loss still decreasing at end**: Increase `--n_epochs`
- **Loss not decreasing**: Learning rate may need tuning (`--learning-rate 5e-4`)

### `latent_coordinates.csv` — Raw Data for Custom Analysis

Use this CSV for further analysis in Python/R:
```python
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('results/04_ai_analysis/latent_coordinates.csv')

# K-Means clustering to identify conformational states
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['z1', 'z2']])
```

---

## Adapting to Your System

This pipeline is **molecule-agnostic**. To analyze a different protein:

1. Replace `data/input/protein.pdb` with your structure
2. Adjust `params.forcefield` in `nextflow.config` if needed (e.g., `charmm36m-ildn` for membrane proteins)
3. Tune `params.sim_time` based on the timescale of your process of interest
4. Run: `nextflow run main.nf --pdb_file data/input/my_protein.pdb`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `conda` environment creation slow | Install `mamba` and set `conda.useMamba = true` in config |
| GPU not detected in PyTorch | Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
| GROMACS `maxwarn` errors | Increase `maxwarn` in the relevant `grompp` call in `main.nf` |
| Autoencoder loss not decreasing | Try lower learning rate: `--learning-rate 5e-4` or more epochs |
| Pipeline fails mid-run | Use `nextflow run main.nf -resume` to restart from last checkpoint |

---

## Understanding Nextflow: A Beginner's Guide

### Core Concepts Illustrated by This Pipeline

**1. Processes = Functions**
A process is like a function: it takes inputs, does work, and produces outputs. Unlike a regular function, it can run on any compute resource (local, cloud, HPC).

**2. Channels = Pipes**
Channels connect processes. When Process A emits output on a channel, Process B automatically receives it as input. You don't write `if/then` logic — Nextflow infers the order.

**3. Work Directory**
Every process run gets its own isolated directory under `work/`. This is how Nextflow enables `-resume`: if the inputs haven't changed, the cached output is reused.

**4. The `bin/` Directory**
Any executable in `bin/` is automatically on the PATH for all processes. This is the canonical way to ship analysis scripts with your pipeline.

**5. Config vs. Logic**
`nextflow.config` handles infrastructure (CPUs, memory, environment, scheduler).
`main.nf` handles science (what runs, in what order, on what data).
Never mix them.

---

## License

MIT License — see [LICENSE](LICENSE) for details.




}
```
