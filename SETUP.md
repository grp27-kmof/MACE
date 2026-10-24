# 🔋 MACE QMOF K⁺ Battery Screening Pipeline — Setup Guide

A step-by-step guide to installing dependencies, configuring the environment, and running the pipeline.

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **OS** | Linux / Windows (WSL recommended) | Native Windows supported |
| **Python** | 3.11 | Via Conda |
| **CUDA** | 12.1 | Required for GPU acceleration |
| **Conda** | ≥ 23.x | Miniconda or Anaconda |
| **Git** | Any | For cloning the repo |

> [!IMPORTANT]
> A CUDA 12.1-compatible GPU is **strongly recommended** for CI-NEB and MACE inference. CPU mode is supported but significantly slower.

---

## ⚙️ Environment Setup

### Option A — Conda (Recommended)

Use the provided `environment.yml` to create a fully pinned environment in one step:

```bash
conda env create -f environment.yml
conda activate mace_clean_env
```

This installs:
- PyTorch 2.5.1 + CUDA 12.1
- MACE-Torch 0.3.16
- ASE, PyMatGen, e3nn, bvlain, ions, matscipy, and all scientific dependencies.

---

### Option B — pip (Virtual Environment)

If you prefer `pip`, create a virtual environment and install from `requirements.txt`:

```bash
python -m venv mace_env
# On Linux/macOS:
source mace_env/bin/activate
# On Windows:
mace_env\Scripts\activate

pip install -r requirements.txt
```

> [!NOTE]
> The `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cu121` for CUDA wheels. Make sure your pip version supports extra index URLs (pip ≥ 21.x).

---

## 🤖 MACE Model Weights

The pipeline loads one model, set in `pipeline_config.yaml`:

| Model file | Size | Use |
|---|---|---|
| `2023-12-03-mace-128-L1_epoch-199.model` | ~42 MB | The model every run uses |

It already sits in `MACE/`, next to `qmof_screening_pipeline.py`.

```yaml
model:
  model_path: 2023-12-03-mace-128-L1_epoch-199.model
  device: cuda   # or cpu
  default_dtype: float32
```

---

## 📁 Directory Structure

```
mace/
├── SETUP.md                   ← You are here
├── environment.yml            ← Conda environment spec
├── requirements.txt           ← pip requirements spec
└── MACE/
    ├── qmof_screening_pipeline.py     ← Main pipeline script
    ├── pipeline_config.yaml           ← All tunable parameters
    ├── bvse_ions_path.py              ← BVSE pathfinding helper
    ├── 2023-12-03-mace-128-L1_epoch-199.model  ← MACE weights
    ├── input/sample_176.cif           ← Example CIF
    └── output/sample_176_diffusion_barrier.xyz
```

---

## ▶️ Running the Pipeline

### 1. Single CIF File (Quick Test)

Verify your installation on a single structure:

```bash
cd MACE
python qmof_screening_pipeline.py \
    --cif_file input/sample_176.cif \
    --config pipeline_config.yaml \
    --device cpu
```

### 2. Batch Mode — Entire Directory

Screen all `.cif` files in a directory using a single GPU:

```bash
python qmof_screening_pipeline.py \
    --cif_dir ./input/ \
    --config pipeline_config.yaml \
    --device cuda \
    --workers 1
```

### 3. Multi-GPU Parallel Mode

Distribute workload over multiple GPUs (e.g., 4 GPUs):

```bash
python qmof_screening_pipeline.py \
    --cif_dir ./input/ \
    --config pipeline_config.yaml \
    --device cuda \
    --workers 4 \
    --output_dir ./output/
```

---

## 🔧 Key Configuration Parameters

All parameters live in `pipeline_config.yaml`. Key settings:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `geometry` | `k_ion_diameter` | `2.76` Å | Minimum pore size for K⁺ |
| `bvse` | `mobile_ion` | `"K+"` | Ion species for BVSE |
| `bvse` | `resolution` | `0.1` Å | BVSE grid spacing |
| `pathfinding` | `max_hop_distance` | `8.0` Å | Max K⁺ hop distance |
| `neb` | `n_images` | `7` | NEB image count |
| `neb` | `max_steps` | `2000` | Total NEB budget |
| `physics` | `T` | `298.0` K | Temperature for Arrhenius |
| `physics` | `conductivity_threshold` | `1e-4` S/cm | Flag threshold for DFT queue |
| `model` | `device` | `cpu` | `cpu` or `cuda` |

---

## 📤 Outputs

| File | Description |
|---|---|
| `screening_results.csv` | PLD, POAV, barrier, D, σ for all processed MOFs |
| `screening_failures.csv` | Error log for failed structures with tracebacks |
| `dft_validation_queue.csv` | Promising candidates (σ ≥ 10⁻⁴ S/cm) for DFT |
| `neb_trajectories/` | NEB trajectory files (if `save_neb_trajectories: true`) |

---

## 🔌 Optional Dependencies

These are **not** required but improve accuracy and performance:

| Package | Purpose | Fallback |
|---|---|---|
| `zeoplusplus` | Accurate PLD / POAV via Voronoi tessellation | Grid-based fallback in pipeline |
| `pymatgen-analysis-diffusion` | `DistinctPathFinder` for pathway enumeration | Distance-based fallback in pipeline |

Install them if available:

```bash
pip install zeoplusplus pymatgen-analysis-diffusion
```

---

## 🐛 Troubleshooting

**CUDA not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, verify your CUDA toolkit version matches `cu121` wheels.

**`mace-torch` import error:**
```bash
pip install mace-torch==0.3.16 --extra-index-url https://download.pytorch.org/whl/cu121
```

**`bvlain` / `ions` not found:**
```bash
pip install bvlain==0.25.1 ions==0.5.1
```

**Volume collapse / NaN forces during NEB:**
Reduce `maxstep` and `spring_constant` in `pipeline_config.yaml`:
```yaml
neb:
  maxstep: 0.02
  spring_constant: 0.05
```

---

## 📚 References

- [MACE](https://github.com/ACEsuit/mace) — Fast & accurate equivariant ML interatomic potentials
- [QMOF Database](https://github.com/arosen93/QMOF) — Quantum-chemical MOF dataset
- [ASE](https://wiki.fysik.dtu.dk/ase/) — Atomic Simulation Environment
- [PyMatGen](https://pymatgen.org/) — Python Materials Genomics
- [bvlain / ions](https://github.com/dembart/ions) — Bond Valence Site Energy pathfinding
