# MACE QMOF K⁺ screening pipeline

Screens metal–organic frameworks as K⁺ battery anode candidates. For each CIF it estimates the K⁺ migration barrier with a MACE climbing-image nudged elastic band (CI-NEB), then converts that barrier into a room-temperature diffusion coefficient and ionic conductivity.

The calculation is a dilute-limit hop: one K⁺ per supercell, inserted into the empty framework.

## What the pipeline does

`MACE/qmof_screening_pipeline.py` runs four stages.

1. **Clean.** Loads the CIF, drops oxidation states, keeps the majority species on disordered sites, and removes isolated H/O solvent fragments of four atoms or fewer.
2. **Geometry filter.** Estimates the pore-limiting diameter (PLD) and probe-occupiable accessible volume (POAV). Zeo++ is used when it is installed; otherwise a real-space grid is used. Structures with PLD below 2.76 Å (the K⁺ diameter) are marked sterically blocked and skipped.
3. **Pathfinding.** Places K⁺ in the largest voids, snaps each ion to a local minimum with MACE, then picks a hop. Bond-valence site energy (BVSE) via the `ions` library is tried first. If that fails, `DistinctPathFinder` is used, then a distance-based pair search. Hops longer than 8 Å are rejected. BVSE also rejects hops shorter than 1.5 Å.
4. **CI-NEB and transport.** Relaxes the two endpoints, builds a 7-image band (IDPP interpolation, FIRE optimizer, flat-band warm-up, then climbing image), and reports

   `D = D0 exp(−Ea / kT)`

   `σ = n z² e² D / kT`

   with `D0 = 10⁻³ cm²/s`, `T = 298 K`, `z = 1`, and `n = 1 / V` (one K⁺ per supercell). Structures with `σ ≥ 10⁻⁴ S/cm` are written to the DFT queue. Barriers above 3 eV are logged as likely unphysical.

The migration barrier is `max(E) − min(E_endpoints)` along the relaxed band.

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11 | Conda environment in `environment.yml` |
| PyTorch 2.5.1 + CUDA 12.1 | CPU works; a CI-NEB on one MOF is much slower |
| MACE-Torch 0.3.16, ASE, pymatgen, PyYAML | Pinned in `environment.yml` and `requirements.txt` |
| A MACE `.model` file | Both shipped weights live in `MACE/` |

Optional, with automatic fallbacks:

| Package | Used for | If missing |
|---|---|---|
| `zeoplusplus` | PLD and POAV | Grid estimate inside the pipeline |
| `pymatgen-analysis-diffusion` | `DistinctPathFinder` | Pair search on snapped K⁺ sites |
| `ions` and `bvlain` | BVSE pathfinding | Void-based pathfinding |

## Setup

Model weights (`*.model`) are stored with Git LFS. Install LFS once, then clone:

```bash
git lfs install
git clone <repository-url>
```

From the repository root:

```bash
conda env create -f environment.yml
conda activate mace_clean_env
```

Or, with pip:

```bash
python -m venv mace_env
# Windows: mace_env\Scripts\activate
# Linux/macOS: source mace_env/bin/activate
pip install -r requirements.txt
```

The default model in `MACE/pipeline_config.yaml` is `2023-12-03-mace-128-L1_epoch-199.model` (fast screening, ~42 MB). `mace-mh-1.model` (~56 MB) is the higher-accuracy alternative. Set `model.model_path` to switch. The 2023 model has a single `Default` head; a `--head` name that is not in the file is ignored and that head is used.

GPU check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Layout

```
mace/
├── README.md
├── SETUP.md                         # longer install and troubleshooting notes
├── environment.yml
├── requirements.txt
└── MACE/
    ├── qmof_screening_pipeline.py   # main entry point
    ├── pipeline_config.yaml
    ├── bvse_ions_path.py            # standalone BVSE site finder
    ├── 2023-12-03-mace-128-L1_epoch-199.model
    ├── mace-mh-1.model
    ├── input/
    │   └── sample_176.cif           # example structure
    └── output/
        └── sample_176_diffusion_barrier.xyz
```

Run commands from `MACE/` so the relative model path in the config resolves.

## Run

One structure:

```bash
cd MACE
python qmof_screening_pipeline.py --cif_file input/sample_176.cif --config pipeline_config.yaml --device cuda --output_dir output
```

Every CIF in a directory, results under `output/`:

```bash
python qmof_screening_pipeline.py --cif_dir input --config pipeline_config.yaml --device cuda --workers 1 --output_dir output
```

Several GPUs (one worker per GPU):

```bash
python qmof_screening_pipeline.py --cif_dir input --config pipeline_config.yaml --device cuda --workers 4 --output_dir output
```

Fixed hop, given as fractional coordinates. Both ends are required. Each site is snapped to a MACE minimum before the NEB.

```bash
python qmof_screening_pipeline.py --cif_file input/sample_176.cif --config pipeline_config.yaml --device cuda --output_dir output --initial_site 0.25 0.50 0.30 --final_site 0.70 0.50 0.30
```

Useful overrides: `--model_path`, `--device cpu|cuda`, `--n_images`, `--fmax`, `--T`, `--supercell A B C`.

To print only the BVSE endpoints for a later manual hop:

```bash
python bvse_ions_path.py input/sample_176.cif --mobile_ion K+
```

### Guest potassium already in the CIF

The pipeline appends one migrating K⁺ and treats every atom already in the file as framework. A CIF that is already packed with K will be screened as a crowded pore, and the pathfinder often returns a very short hop between occupied sites. For the dilute-limit barrier, remove those guest K ions first and pass the bare framework.

## Configuration

All knobs are in `MACE/pipeline_config.yaml`.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `geometry` | `k_ion_diameter` | 2.76 Å | PLD cutoff |
| `bvse` | `min_hop_distance` | 1.5 Å | Shortest hop BVSE will accept |
| `bvse` | `energy_asymmetry_threshold` | 1.0 eV | Max \|E(start) − E(end)\| after the MACE snap |
| `pathfinding` | `max_hop_distance` | 8.0 Å | Longest hop |
| `neb` | `n_images` | 7 | Images in the band, including both endpoints |
| `neb` | `fmax` | 0.10 eV/Å | FIRE convergence |
| `neb` | `warmup_steps` / `max_steps` | 200 / 2000 | Flat-band warm-up, then CI-NEB |
| `physics` | `conductivity_threshold` | 1e-4 S/cm | DFT-queue cutoff |
| `physics` | `barrier_unphysical_threshold` | 3.0 eV | Logged as unreliable |
| `model` | `device` | `cpu` | `cpu` or `cuda` |
| `output` | `save_neb_trajectories` | `true` | Write the band to disk |

## Outputs

Written to the working directory, or to `--output_dir` when that flag is set.

| File | Contents |
|---|---|
| `screening_results.csv` | PLD, POAV, hop length, barrier, D, σ, convergence, DFT flag |
| `screening_failures.csv` | Structures that raised an error, with the traceback |
| `dft_validation_queue.csv` | Rows with σ at or above the conductivity threshold |
| `neb_trajectories/<name>_neb.extxyz` | Full NEB band in one file |
| `neb_trajectories/<name>_cp2k/image00.xyz` … | One extended XYZ per image. For the default 7-image band these are `image00` through `image06` |

`image00.xyz` is the relaxed start and the last image is the relaxed end. `rel_energy_eV` on a frame, when present, is the energy relative to the lower endpoint.

## When a run fails

- **Sterically blocked.** PLD is under 2.76 Å. The pore cannot fit K⁺.
- **No path.** Fewer than two distinct K⁺ sites were found, or every pair is outside the hop window.
- **Endpoint asymmetry.** BVSE found an edge whose two ends differ by more than 1 eV after the MACE snap. With `fallback_to_void: true` the void search still runs.
- **Volume collapse or NaN forces.** The NEB is aborted. Lower `neb.maxstep` (try `0.02`) and `neb.spring_constant` (try `0.05`) and rerun.
- **Barrier above 3 eV.** The row is still written. Treat Ea, D, and σ as unreliable until the band is checked.

`SETUP.md` has the install-error notes (CUDA wheel mismatch, missing `mace-torch`, missing `ions`).

## References

- [MACE](https://github.com/ACEsuit/mace) — equivariant machine-learning interatomic potentials
- [QMOF](https://github.com/arosen93/QMOF) — quantum-chemical MOF database
- [ASE](https://wiki.fysik.dtu.dk/ase/) — optimizers and NEB
- [pymatgen](https://pymatgen.org/) — structure I/O and diffusion path finding
- [ions](https://github.com/dembart/ions) — bond-valence site energies
