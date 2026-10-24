#!/usr/bin/env python3
"""
qmof_screening_pipeline.py
==========================
Unified, object-oriented pipeline for high-throughput screening of the QMOF
database to discover optimal K+ battery anode MOF materials.

Stages
------
1. GeometryFilter   -- Zeo++ (or grid fallback) PLD/void screening
2. PathFinder       -- Automated K+ site discovery + MACE snap relaxation
3. MACEEvaluator    -- CI-NEB with FIRE optimizer (7 images, IDPP, k=0.05)
4. PostProcessor    -- Arrhenius D, Nernst-Einstein sigma, DFT AL queue

Usage (single CIF)
------------------
    python qmof_screening_pipeline.py --cif_file input/sample_176.cif \
        --model_path ./2023-12-03-mace-128-L1_epoch-199.model --device cpu

Usage (batch, all CIFs in a directory)
---------------------------------------
    python qmof_screening_pipeline.py --cif_dir ./qmof_cifs/ \
        --config pipeline_config.yaml --workers 4 --device cuda

Multiprocessing / multi-GPU
----------------------------
    The `process_single_cif()` function is designed to be called from a
    multiprocessing.Pool.  Each worker receives a GPU index and pins itself
    to that device.  Example::

        from qmof_screening_pipeline import process_single_cif
        from multiprocessing import Pool
        with Pool(4) as p:
            p.starmap(process_single_cif,
                      [(cif, cfg, i % 4) for i, cif in enumerate(cifs)])
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import csv
import logging
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party -- core scientific stack
# ---------------------------------------------------------------------------
import numpy as np

# Pymatgen
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

# ASE
from ase import Atoms
from ase.optimize import FIRE

try:
    from ase.mep import NEB  # ASE >= 3.23
except ImportError:
    from ase.neb import NEB  # ASE < 3.23

# MACE
try:
    import torch
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([slice])
    from mace.calculators import MACECalculator, mace_mp
    _MACE_AVAILABLE = True
except ImportError:
    _MACE_AVAILABLE = False
    warnings.warn("mace-torch not found.  Install with:  pip install mace-torch")

# ---------------------------------------------------------------------------
# Optional soft-import dependencies (fallbacks activate automatically)
# ---------------------------------------------------------------------------
try:
    # Zeo++ Python wrapper
    import zeoplusplus as _zpp_mod  # noqa: F401
    from pymatgen.io.zeopp import ZeoPPAnalyzer
    _ZPP_AVAILABLE = True
except ImportError:
    _ZPP_AVAILABLE = False

try:
    from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder
    _DPF_AVAILABLE = True
except ImportError:
    _DPF_AVAILABLE = False

try:
    from ase.constraints import FixAtoms
    _FIXATOMS_AVAILABLE = True
except ImportError:
    _FIXATOMS_AVAILABLE = False

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from ions import Percolator, Edge
    from ions.calculators.bvse_calculator import BVSECalculator
    from ions.decorate import assign_bvse_params
    _BVSE_AVAILABLE = True
except ImportError as e:
    import traceback
    traceback.print_exc()
    _BVSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qmof_pipeline")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K_B_EV   = 8.617333262145e-5   # eV / K
K_B_SI   = 1.380649e-23        # J / K
E_CHARGE = 1.602176634e-19     # C
ANG3_TO_CM3 = 1e-24            # 1 Ang^3 = 1e-24 cm^3

# ---------------------------------------------------------------------------
# Default configuration (mirrors pipeline_config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "geometry": {
        "k_ion_diameter": 2.76,
        "min_void_radius": 1.38,
        "grid_spacing": 0.20,
        "n_candidate_voids": 10,
        "use_zeopp": True,
    },
    "bvse": {
        "enabled": True,
        "mobile_ion": "K+",
        "resolution": 0.2,
        "r_cut": 10.0,
        "max_candidates": 5,
        "energy_asymmetry_threshold": 1.0,
        "fallback_to_void": True,
    },
    "pathfinding": {
        "max_hop_distance": 4.5,
        "snap_fmax": 0.10,
        "snap_steps": 300,
        "snap_maxstep": 0.05,
        "use_distinct_path_finder": True,
    },
    "neb": {
        "n_images": 7,
        "spring_constant": 0.05,
        "fmax": 0.05,
        "maxstep": 0.05,
        "max_steps": 500,
        "interpolation": "idpp",
        "endpoint_relax_fmax": 0.05,
        "endpoint_relax_steps": 1000,
        "volume_collapse_threshold": 0.15,
    },
    "physics": {
        "D0": 1e-3,
        "T": 298.0,
        "z": 1,
        "conductivity_threshold": 1e-4,
        "barrier_unphysical_threshold": 3.0,
    },
    "supercell": [1, 1, 2],
    "model": {
        "device": "cuda",
        "model_path": None,
        "head": None,
        "default_dtype": "float32",
        "mace_mp_size": "medium",
    },
    "output": {
        "results_csv": "screening_results.csv",
        "failures_csv": "screening_failures.csv",
        "dft_queue_csv": "dft_validation_queue.csv",
        "neb_trajectories_dir": "neb_trajectories",
        "save_neb_trajectories": False,
        "log_level": "INFO",
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config and deep-merge with built-in defaults."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and _YAML_AVAILABLE:
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh)
        if user_cfg:
            _deep_merge(cfg, user_cfg)
    elif config_path and not _YAML_AVAILABLE:
        logger.warning("PyYAML not installed; using built-in defaults.  "
                       "pip install pyyaml")
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ===========================================================================
# Dataclasses -- inter-stage data transfer objects
# ===========================================================================

@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    pld: float = 0.0
    poav: float = 0.0
    void_centers: List[np.ndarray] = field(default_factory=list)


@dataclass
class PathResult:
    found: bool
    paths: List[Tuple] = field(default_factory=list)
    # each tuple: (initial_frac: ndarray, final_frac: ndarray, hop_Ang: float)
    reason: str = ""


@dataclass
class NEBResult:
    converged: bool
    barrier_eV: float = float("nan")
    energies: List[float] = field(default_factory=list)
    reason: str = ""
    volume_initial: float = 0.0
    volume_final: float = 0.0


@dataclass
class ScreeningResult:
    cif_name: str
    status: str
    pld: float = float("nan")
    poav: float = float("nan")
    hop_distance_ang: float = float("nan")
    barrier_eV: float = float("nan")
    D_cm2s: float = float("nan")
    sigma_Scm: float = float("nan")
    n_carriers_cm3: float = float("nan")
    converged: bool = False
    al_flagged: bool = False
    runtime_s: float = 0.0
    notes: str = ""


# ===========================================================================
# Stage 0: CIF Cleaner
# ===========================================================================

class CIFCleaner:
    """
    Load and sanitise a raw QMOF CIF file.

    Steps
    -----
    1. Remove oxidation states (MACE uses neutral species labels).
    2. Resolve fractional occupancies deterministically (keep majority species).
    3. Remove isolated solvent molecules (H/O fragments <= 4 atoms) using
       PBC-aware connectivity analysis with element-specific bond cutoffs.
    4. Validate that the cleaned structure is non-empty.
    """

    SOLVENT_SPECIES = {"H", "O"}
    SOLVENT_MAX_ATOMS = 4

    # Generous bond cutoffs (Angstrom) used for connectivity analysis
    _METAL_SYMBOLS = {
        "Zr", "Cu", "Zn", "Fe", "Co", "Ni", "Al", "Cr", "Ti", "V",
        "Mn", "Mo", "W", "Hf", "Ce", "La", "In", "Bi", "Mg", "Ca",
        "Ba", "Sr", "Cd", "Ag", "Au", "Pt", "Pd", "Ru", "Rh", "Ir",
        "Sc", "Y", "Nd", "Sm", "Eu", "Tb", "Dy", "Er", "Yb", "Lu",
    }

    def clean(self, cif_path: str) -> Structure:
        struct = Structure.from_file(cif_path)
        struct = self._remove_oxidation_states(struct)
        struct = self._resolve_occupancies(struct)
        struct = self._remove_solvents(struct)
        self._validate(struct, cif_path)
        return struct

    # ------------------------------------------------------------------
    def _remove_oxidation_states(self, struct: Structure) -> Structure:
        try:
            struct.remove_oxidation_states()
        except Exception:
            pass
        return struct

    # ------------------------------------------------------------------
    def _resolve_occupancies(self, struct: Structure) -> Structure:
        """Keep only the dominant species at each disordered site."""
        disordered = [
            (i, max(s.species, key=lambda sp: s.species[sp]))
            for i, s in enumerate(struct)
            if len(s.species) > 1
        ]
        if not disordered:
            return struct

        idx_map = {i: sp for i, sp in disordered}
        new_sp, new_fc = [], []
        for i, site in enumerate(struct):
            new_sp.append(idx_map.get(i, site.specie))
            new_fc.append(site.frac_coords)

        logger.debug(f"Resolved {len(disordered)} disordered sites.")
        return Structure(struct.lattice, new_sp, new_fc)

    # ------------------------------------------------------------------
    def _remove_solvents(self, struct: Structure) -> Structure:
        """
        Build a PBC-aware connectivity graph and remove isolated H/O fragments.
        Uses pymatgen's get_distance() which handles periodic images correctly.
        """
        n = len(struct)
        adj = [[] for _ in range(n)]

        for i in range(n):
            ei = struct[i].specie.symbol
            for j in range(i + 1, n):
                ej = struct[j].specie.symbol
                cutoff = self._bond_cutoff(ei, ej)
                d = struct.get_distance(i, j)
                if d < cutoff:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        components = []
        for start in range(n):
            if not visited[start]:
                stack, comp = [start], []
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    comp.append(node)
                    stack.extend(adj[node])
                components.append(comp)

        to_remove = []
        for comp in components:
            syms = {struct[i].specie.symbol for i in comp}
            if syms.issubset(self.SOLVENT_SPECIES) and len(comp) <= self.SOLVENT_MAX_ATOMS:
                to_remove.extend(comp)

        if to_remove:
            logger.debug(f"Removing {len(to_remove)} solvent atoms "
                         f"({len([c for c in components if set(struct[c[0]].specie.symbol for c in [c]) <= self.SOLVENT_SPECIES])} fragments).")
            struct.remove_sites(sorted(to_remove, reverse=True))

        return struct

    def _bond_cutoff(self, ei: str, ej: str) -> float:
        pair = {ei, ej}
        # Metal -- O/N/S bonds in MOFs can reach 2.8 Ang
        if pair & self._METAL_SYMBOLS and pair & {"O", "N", "S", "Cl", "F"}:
            return 2.8
        if pair == {"C", "C"}:
            return 1.75
        if pair & {"H"}:
            return 1.30
        if pair == {"C", "N"} or pair == {"C", "O"} or pair == {"C", "S"}:
            return 1.60
        return 2.0  # conservative fallback

    def _validate(self, struct: Structure, cif_path: str) -> None:
        if len(struct) == 0:
            raise ValueError(f"Structure is empty after cleaning: {cif_path}")
        heavy = [s for s in struct if s.specie.symbol not in {"H", "O"}]
        if not heavy:
            raise ValueError(f"No heavy framework atoms remain: {cif_path}")


# ===========================================================================
# Stage 1: Geometric Pre-Filter
# ===========================================================================

class GeometryFilter:
    """
    Screen a cleaned Structure for K+ steric accessibility.

    Primary:  Zeo++ via pymatgen.io.zeopp.ZeoPPAnalyzer
    Fallback: PBC-aware grid search (always available, no extra deps)

    Decision rule
    -------------
    PLD < k_ion_diameter (2.76 Ang)  -->  "Sterically Blocked"
    """

    def __init__(self, cfg: dict):
        gc = cfg["geometry"]
        self.k_diameter    = gc["k_ion_diameter"]
        self.k_radius      = self.k_diameter / 2.0
        self.min_void_r    = gc["min_void_radius"]
        self.grid_spacing  = gc["grid_spacing"]
        self.n_candidates  = gc["n_candidate_voids"]
        self.use_zeopp     = gc["use_zeopp"] and _ZPP_AVAILABLE

    def filter(self, struct: Structure) -> FilterResult:
        if self.use_zeopp:
            try:
                return self._filter_zeopp(struct)
            except Exception as exc:
                logger.warning(f"Zeo++ failed ({exc}); using grid fallback.")
        return self._filter_grid(struct)

    # ------------------------------------------------------------------
    def _filter_zeopp(self, struct: Structure) -> FilterResult:
        analyzer = ZeoPPAnalyzer(struct)

        # Pore Limiting Diameter
        try:
            pld = float(analyzer.get_pore_diameter())
        except Exception:
            pld = float(analyzer.get_largest_included_sphere_along_free_sphere_path())

        # Probe-Occupiable Accessible Volume
        try:
            poav = float(analyzer.get_accessible_volume(probe_radius=self.k_radius))
        except Exception:
            poav = 0.0

        if pld < self.k_diameter:
            return FilterResult(passed=False, reason="Sterically Blocked",
                                pld=pld, poav=poav)

        # Void centres from Voronoi nodes
        void_centers = []
        try:
            voronoi_nodes = analyzer.get_voronoi_nodes()
            accessible = sorted(
                [(np.array(fc), r) for fc, r in voronoi_nodes if r >= self.min_void_r],
                key=lambda x: x[1], reverse=True,
            )
            void_centers = [fc for fc, _ in accessible[: self.n_candidates]]
        except Exception:
            void_centers = []

        # Fallback: use grid if Zeo++ gives no void centres
        if not void_centers:
            grid_result = self._filter_grid(struct)
            void_centers = grid_result.void_centers

        return FilterResult(passed=True, reason="OK",
                            pld=pld, poav=poav, void_centers=void_centers)

    # ------------------------------------------------------------------
    def _filter_grid(self, struct: Structure) -> FilterResult:
        """
        Brute-force grid: sample unit cell at `grid_spacing` intervals (fractional),
        compute PBC-correct distance to nearest atom at each point, find voids.
        """
        lattice = struct.lattice
        step = self.grid_spacing
        fa = np.arange(0.0, 1.0, step)
        grid = np.stack(np.meshgrid(fa, fa, fa, indexing="ij"), axis=-1).reshape(-1, 3)
        n_grid = len(grid)

        atom_frac = np.array([s.frac_coords for s in struct])  # (N, 3)

        min_dists = np.empty(n_grid)
        for k in range(n_grid):
            df = atom_frac - grid[k]            # (N, 3)
            df -= np.round(df)                   # minimum image in fractional
            # Convert to cartesian properly (matrix multiply)
            cart = df @ lattice.matrix           # (N, 3)
            min_dists[k] = np.linalg.norm(cart, axis=1).min()

        pld_estimate = float(min_dists.max() * 2.0)

        if pld_estimate < self.k_diameter:
            return FilterResult(passed=False, reason="Sterically Blocked",
                                pld=pld_estimate, poav=0.0)

        # POAV estimate: fraction of cell accessible to K+
        n_accessible = int((min_dists >= self.k_radius).sum())
        poav_estimate = (n_accessible / n_grid) * lattice.volume

        # Pick void centres greedily
        order = np.argsort(min_dists)[::-1]
        selected_fracs: List[np.ndarray] = []
        for idx in order:
            if min_dists[idx] < self.min_void_r:
                break
            frac = grid[idx]
            too_close = False
            for prev in selected_fracs:
                df = np.abs(frac - prev)
                df = np.where(df > 0.5, 1.0 - df, df)
                if np.linalg.norm(df) < 2.0 * step:
                    too_close = True
                    break
            if not too_close:
                selected_fracs.append(frac.copy())
            if len(selected_fracs) >= self.n_candidates:
                break

        return FilterResult(
            passed=True, reason="OK",
            pld=pld_estimate, poav=poav_estimate,
            void_centers=selected_fracs,
        )


# ===========================================================================
# Stage 2: Automated PathFinder
# ===========================================================================

class PathFinder:
    """
    Discover K+ hopping sites and paths autonomously.

    Algorithm
    ---------
    1. For each void centre:
       a. Insert K+ at that fractional position in the supercell.
       b. Fix all framework atoms; relax only K+ with MACE-FIRE (fast snap).
       c. Record the relaxed K+ fractional position.
    2. Deduplicate sites that converged to the same pocket.
    3. Primary:  DistinctPathFinder (pymatgen-analysis-diffusion).
       Fallback: Enumerate all pairs within max_hop_distance; pick shortest.

    This approach is chemistry-agnostic -- works for carboxylate MOFs,
    zeolitic imidazolate frameworks, paddlewheels, MIL, UiO, etc.
    """

    def __init__(self, cfg: dict, calc):
        pc = cfg["pathfinding"]
        self.max_hop     = pc["max_hop_distance"]
        self.snap_fmax   = pc["snap_fmax"]
        self.snap_steps  = pc["snap_steps"]
        self.snap_maxstep = pc["snap_maxstep"]
        self.use_dpf     = pc["use_distinct_path_finder"] and _DPF_AVAILABLE
        self.max_paths   = pc.get("max_paths", 5)
        self.calc        = calc

    # ------------------------------------------------------------------
    def find_paths(self, struct: Structure,
                   void_centers: List[np.ndarray],
                   supercell: List[int]) -> PathResult:
        if not void_centers:
            return PathResult(found=False,
                              reason="No void centres supplied by GeometryFilter.")

        super_struct = struct.copy()
        super_struct.make_supercell(np.diag(supercell))

        # --- Snap each void to a stable K+ site ---
        relaxed: List[np.ndarray] = []
        sc_arr = np.array(supercell, dtype=float)
        for i, vf in enumerate(void_centers):
            try:
                vf_super = (vf / sc_arr) % 1.0
                site = self._snap_k(super_struct, vf_super)
                if site is not None:
                    relaxed.append(site)
            except Exception as exc:
                logger.debug(f"Snap failed for void {i}: {exc}")

        if len(relaxed) < 2:
            return PathResult(found=False,
                              reason=f"Fewer than 2 stable K+ sites found "
                                     f"({len(relaxed)} after snap relaxation).")

        relaxed = self._deduplicate(relaxed, super_struct.lattice)
        if len(relaxed) < 2:
            return PathResult(found=False,
                              reason="All void centres collapsed to a single K+ site.")

        logger.info(f"  PathFinder: {len(relaxed)} distinct K+ sites identified.")

        # --- Pathfinding ---
        if self.use_dpf:
            try:
                return self._dpf_paths(struct, relaxed, supercell)
            except Exception as exc:
                logger.warning(f"DistinctPathFinder failed ({exc}); using pair fallback.")

        return self._pair_paths(super_struct, relaxed)

    # ------------------------------------------------------------------
    def _snap_k(self, super_struct: Structure,
                void_frac: np.ndarray) -> Optional[np.ndarray]:
        """Insert K+ at void_frac, freeze framework, relax K+ with FIRE."""
        working = super_struct.copy()
        working.append(Element("K"), void_frac % 1.0)
        atoms = AseAtomsAdaptor.get_atoms(working)
        atoms.wrap()
        
        # Add a tiny random perturbation (0.05 Angstroms) to K+ to break 
        # perfect symmetry (prevents getting stuck on saddle points where F=0)
        k_idx = len(atoms) - 1
        pos = atoms.positions
        perturbation = np.random.normal(0, 0.05, 3)
        pos[k_idx] += perturbation
        atoms.set_positions(pos)

        if _FIXATOMS_AVAILABLE:
            mask = [i != k_idx for i in range(len(atoms))]
            atoms.set_constraint(FixAtoms(mask=mask))

        atoms.calc = self.calc
        opt = FIRE(atoms, maxstep=self.snap_maxstep)
        opt.run(fmax=self.snap_fmax, steps=self.snap_steps)

        forces = atoms.get_forces()
        if not np.isfinite(forces).all():
            return None

        k_cart = atoms.get_positions()[k_idx]
        lat = Lattice(atoms.get_cell())
        return lat.get_fractional_coords(k_cart) % 1.0

    # ------------------------------------------------------------------
    def _deduplicate(self, sites: List[np.ndarray],
                     lattice: Lattice, tol: float = 0.30) -> List[np.ndarray]:
        """Remove sites within `tol` Angstrom of an already-kept site."""
        unique = [sites[0]]
        for s in sites[1:]:
            is_dup = False
            for u in unique:
                df = s - u
                df -= np.round(df)
                cart = df @ lattice.matrix
                if np.linalg.norm(cart) < tol:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(s)
        return unique

    # ------------------------------------------------------------------
    def _dpf_paths(self, prim_struct: Structure,
                   relaxed_super: List[np.ndarray],
                   supercell: List[int]) -> PathResult:
        """Use DistinctPathFinder on primitive cell."""
        sc_arr = np.array(supercell, dtype=float)
        # Map first supercell site back to primitive cell
        prim_frac = (relaxed_super[0] * sc_arr) % 1.0

        k_struct = prim_struct.copy()
        k_struct.append(Element("K"), prim_frac)

        dpf = DistinctPathFinder(
            k_struct,
            migrating_specie="K",
            max_path_length=self.max_hop,
            symprec=0.1,
            perc_mode=">1d",
        )
        raw = dpf.get_paths()
        if not raw:
            return PathResult(found=False,
                              reason="DistinctPathFinder found no paths within "
                                     f"{self.max_hop} Ang.")

        paths = []
        for p in raw[:self.max_paths]:
            try:
                init_super = (np.array(p.isite.frac_coords) / sc_arr) % 1.0
                fin_super  = (np.array(p.esite.frac_coords) / sc_arr) % 1.0
                paths.append((
                    init_super,
                    fin_super,
                    float(p.length),
                ))
            except Exception:
                continue

        if not paths:
            return PathResult(found=False,
                              reason="DPF paths could not be parsed.")
        return PathResult(found=True, paths=paths)

    # ------------------------------------------------------------------
    def _pair_paths(self, super_struct: Structure,
                    relaxed: List[np.ndarray]) -> PathResult:
        """Fallback: all unique pairs within max_hop_distance."""
        lat = super_struct.lattice
        pairs = []
        for i in range(len(relaxed)):
            for j in range(i + 1, len(relaxed)):
                df = relaxed[j] - relaxed[i]
                df -= np.round(df)
                cart = df @ lat.matrix
                d = float(np.linalg.norm(cart))
                if d <= self.max_hop:
                    pairs.append((relaxed[i].copy(), relaxed[j].copy(), d))

        if not pairs:
            return PathResult(
                found=False,
                reason=f"No site pairs within {self.max_hop} Ang "
                       f"({len(relaxed)} unique sites found).",
            )

        pairs.sort(key=lambda x: x[2])
        return PathResult(found=True, paths=pairs[:self.max_paths])


# ===========================================================================
# BVSE PathFinder Integration
# ===========================================================================

def _assign_oxidation_states_for_bvse(struct: Structure) -> Structure:
    """Remove oxidation states and assign oxidation states for BVSE."""
    try:
        struct.remove_oxidation_states()
    except Exception:
        pass
    # BVSE needs oxidation states; use pymatgen's BVAnalyzer to add them
    from pymatgen.analysis.bond_valence import BVAnalyzer
    try:
        analyzer = BVAnalyzer()
        struct = analyzer.get_oxi_state_decorated_structure(struct)
    except Exception as e:
        logger.warning(f"  Could not assign oxidation states via BVAnalyzer: {e}. Falling back to default guesser.")
        # Fallback manual guesser
        guesses = {}
        for el in struct.composition.elements:
            symbol = el.symbol
            if symbol == 'O': guesses[symbol] = -2
            elif symbol == 'H': guesses[symbol] = 1
            elif symbol in ['F', 'Cl', 'Br', 'I']: guesses[symbol] = -1
            elif symbol == 'N': guesses[symbol] = -3
            elif symbol == 'S': guesses[symbol] = -2
            elif symbol in ['Li', 'Na', 'K', 'Rb', 'Cs']: guesses[symbol] = 1
            elif symbol in ['Mg', 'Ca', 'Sr', 'Ba', 'Zn', 'Cd']: guesses[symbol] = 2
            elif symbol in ['Al', 'Ga', 'In', 'Sc', 'Y', 'La']: guesses[symbol] = 3
            elif symbol in ['Ti', 'Zr', 'Hf', 'Ce', 'C']: guesses[symbol] = 4
            else:
                guesses[symbol] = 3 if el.is_metal else -2
        try:
            struct.add_oxidation_state_by_element(guesses)
        except Exception as exc:
            logger.error(f"  Failed to manually assign oxidation states: {exc}")
    return struct

def find_paths_bvse(struct: Structure, cfg: dict, pf: PathFinder, void_centers: List[np.ndarray]) -> PathResult:
    """
    Computes BVSE minimum energy path using the ions library.
    Extracts candidate K+ sites from void_centers, snaps them using MACE,
    builds a percolating network, and evaluates the BVSE barrier for each edge
    to find the optimal migration path.
    """
    if not _BVSE_AVAILABLE:
        return PathResult(found=False, reason="ions library not available.")
        
    bc = cfg["bvse"]
    sc_arr = np.array(cfg["supercell"], dtype=float)
    
    if not void_centers:
        return PathResult(found=False, reason="void_centers required for ions BVSE pathfinding.")
        
    sc_bare = struct.copy()
    sc_bare.make_supercell(np.diag(cfg["supercell"]))
    
    logger.info("  Snapping void centers to MACE minima for ions Percolator...")
    relaxed = []
    for vf in void_centers:
        try:
            vf_super = (vf / sc_arr) % 1.0
            site = pf._snap_k(sc_bare, vf_super)
            if site is not None:
                relaxed.append(site)
        except Exception as exc:
            logger.debug(f"Snap failed for void: {exc}")
            
    relaxed = pf._deduplicate(relaxed, sc_bare.lattice)
    if len(relaxed) < 2:
        return PathResult(found=False, reason="Fewer than 2 stable K+ sites found after MACE snap.")

    # Build Atoms object with all K+ sites
    from pymatgen.io.ase import AseAtomsAdaptor
    working_sc = _assign_oxidation_states_for_bvse(sc_bare)
    atoms_mof = AseAtomsAdaptor.get_atoms(working_sc)
    
    oxi = [s.specie.oxi_state for s in working_sc]
    atoms_mof.set_array('oxi_states', np.array(oxi, dtype=float))
    
    for frac in relaxed:
        atoms_mof.append('K')
        cart = frac @ sc_bare.lattice.matrix
        atoms_mof.positions[-1] = cart
        oxi_arr = atoms_mof.get_array('oxi_states')
        oxi_arr[-1] = 1.0
        atoms_mof.set_array('oxi_states', oxi_arr)
        
    # Assign BVSE params with no K-K interaction
    try:
        from ions.decorate import assign_bvse_params
        from ions.data import bvse_data
        
        # Inject missing generic parameters for framework anions (N, C, P, etc.)
        # This prevents failure for MOFs like ZIF-8 that contain elements not originally in BVSE batteries
        if 'K' in bvse_data and 1 in bvse_data['K']:
            for sym, q in zip(atoms_mof.symbols, atoms_mof.get_array('oxi_states')):
                if q < 0:
                    if sym not in bvse_data['K'][1]:
                        bvse_data['K'][1][sym] = {}
                    if q not in bvse_data['K'][1][sym]:
                        # Default mock parameters for Morse/Coulomb interactions
                        bvse_data['K'][1][sym][q] = {'r0': 2.0, 'r_min': 2.8, 'alpha': 2.3, 'd0': 0.35}

        assign_bvse_params(atoms_mof, 'K', 1, self_interaction=False)
    except Exception as e:
        return PathResult(found=False, reason=f"assign_bvse_params failed: {e}")
    
    logger.info("  Running ions Percolator...")
    try:
        from ions import Percolator, Edge
        perc = Percolator(atoms_mof, mobile_specie="K", upper_bound=bc.get("r_cut", 10.0))
        cutoff = bc.get("r_cut", 10.0)
        bottleneck = 0.5
        edges = perc.unique_edges(cutoff, bottleneck, method="naive")
    except Exception as e:
        return PathResult(found=False, reason=f"Percolator failed: {e}")
        
    if len(edges) == 0:
        return PathResult(found=False, reason="No percolation edges found by ions Percolator.")
        
    logger.info(f"  Evaluating endpoints via MACE for {len(edges)} unique edges...")
    candidates = []
    
    min_hop = bc.get("min_hop_distance", 1.5)
    # Use config max_hop, but also allow PLD-based adaptive upper bound:
    # Large-pore MOFs (ZIF-8, IRMOF) can have hops up to ~2x PLD.
    cfg_max_hop = cfg.get("pathfinding", {}).get("max_hop_distance", 8.0)
    pld_guess = getattr(pf, '_last_pld', None)  # optionally set by caller
    if pld_guess and pld_guess > 0:
        adaptive_max = max(cfg_max_hop, 2.0 * pld_guess)
    else:
        adaptive_max = cfg_max_hop
    max_hop = adaptive_max
    logger.info(f"  Hop distance window: [{min_hop:.2f}, {max_hop:.2f}] Ang")
    
    def _mk_atoms(sc, frac):
        from pymatgen.core import Element
        from pymatgen.io.ase import AseAtomsAdaptor
        s = sc.copy()
        s.append(Element("K"), frac % 1.0)
        at = AseAtomsAdaptor.get_atoms(s)
        at.wrap()
        at.calc = pf.calc
        return at
    
    all_hop_distances = []
    for edge_data in edges:
        source, target = int(edge_data[0]), int(edge_data[1])
        offset = edge_data[2:]
        
        try:
            init_frac = (atoms_mof.positions[source] @ np.linalg.inv(sc_bare.lattice.matrix)) % 1.0
            fin_cart = atoms_mof.positions[target] + offset @ sc_bare.lattice.matrix
            fin_frac = (fin_cart @ np.linalg.inv(sc_bare.lattice.matrix)) % 1.0
            
            hop_d = float(np.linalg.norm(fin_cart - atoms_mof.positions[source]))
            all_hop_distances.append(hop_d)
            
            if min_hop <= hop_d <= max_hop:
                atoms_init = _mk_atoms(sc_bare, init_frac)
                e_init = float(atoms_init.get_potential_energy())
                
                atoms_fin = _mk_atoms(sc_bare, fin_frac)
                e_fin = float(atoms_fin.get_potential_energy())
                
                delta_e = abs(e_init - e_fin)
                
                candidates.append({
                    'init_frac': init_frac,
                    'fin_frac': fin_frac,
                    'hop_d': hop_d,
                    'asymmetry': delta_e,
                    'barrier': hop_d  # rank by hop distance when BVSE not used
                })
        except Exception as e:
            logger.debug(f"Failed to evaluate edge {source}->{target}: {e}")
        
    if not candidates:
        # Diagnose: report actual hop distances so user can tune max_hop_distance
        if all_hop_distances:
            hops_str = ", ".join(f"{d:.2f}" for d in sorted(all_hop_distances))
            logger.warning(
                f"  No edges in [{min_hop:.2f}, {max_hop:.2f}] Ang. "
                f"All hop distances: [{hops_str}] Ang. "
                f"Consider raising pathfinding.max_hop_distance in config."
            )
        return PathResult(found=False, reason="Failed to evaluate any valid candidates matching hop distance.")
        
    # Sort by hop distance (since we dropped BVSE barrier)
    candidates.sort(key=lambda x: x['barrier'])
    
    # Apply max_candidates
    candidates = candidates[:bc.get("max_candidates", 10)]
    
    # Filter by asymmetry threshold
    valid_candidates = [c for c in candidates if c['asymmetry'] <= bc["energy_asymmetry_threshold"]]
    
    if not valid_candidates:
        best_asym = candidates[0]['asymmetry']
        return PathResult(
            found=False, 
            reason=f"BVSE endpoints failed MACE validation: best asymmetry {best_asym:.2f} eV > {bc['energy_asymmetry_threshold']} eV"
        )
        
    best = valid_candidates[0]
    logger.info(f"  Best BVSE edge asymmetry: {best['asymmetry']:.3f} eV, distance: {best['hop_d']:.3f} A")
    
    return PathResult(found=True, paths=[(best['init_frac'], best['fin_frac'], best['hop_d'])])


# ===========================================================================
# Stage 3: MACE CI-NEB Evaluator
# ===========================================================================

class MACEEvaluator:
    """
    Run a Climbing-Image NEB using the MACE universal potential and FIRE.

    Safety guards
    -------------
    - NaN force detection: aborts immediately if any NaN force is detected.
    - Volume collapse: |DeltaV/V| > threshold after endpoint relaxation -> abort.
    - Hard step cap: exits after max_steps regardless of convergence.
    - KeyboardInterrupt: extracts partial barrier gracefully.
    - IDPP -> linear fallback: if IDPP interpolation fails, retries with linear.
    """

    def __init__(self, cfg: dict, calc, supercell: List[int]):
        nc = cfg["neb"]
        self.n_images    = nc["n_images"]
        self.k           = nc["spring_constant"]
        self.fmax        = nc["fmax"]
        self.maxstep     = nc["maxstep"]
        self.max_steps   = nc["max_steps"]
        self.warmup_steps = nc.get("warmup_steps", 150)   # flat-band pre-relax steps
        self.interp      = nc["interpolation"]
        self.ep_fmax     = nc["endpoint_relax_fmax"]
        self.ep_steps    = nc["endpoint_relax_steps"]
        self.vol_thr     = nc["volume_collapse_threshold"]
        self.calc        = calc
        self.supercell   = supercell
        self.save_traj   = cfg["output"]["save_neb_trajectories"]
        self.traj_dir    = cfg["output"]["neb_trajectories_dir"]

    # ------------------------------------------------------------------
    def evaluate(self, struct: Structure,
                 init_frac: np.ndarray,
                 fin_frac: np.ndarray,
                 mof_name: str = "MOF") -> NEBResult:

        # Build supercell
        sc = struct.copy()
        sc.make_supercell(np.diag(self.supercell))

        # Insert mobile K+
        init_atoms = self._mk_atoms(sc, init_frac)
        fin_atoms  = self._mk_atoms(sc, fin_frac)

        # Relax endpoints
        logger.info("  Relaxing initial endpoint ...")
        init_rel, v_init = self._relax_ep(init_atoms)
        if init_rel is None:
            return NEBResult(converged=False,
                             reason="NaN forces in initial endpoint relaxation.")

        logger.info("  Relaxing final endpoint ...")
        fin_rel, v_fin = self._relax_ep(fin_atoms)
        if fin_rel is None:
            return NEBResult(converged=False,
                             reason="NaN forces in final endpoint relaxation.")

        # Log endpoint energies and warn on large asymmetry
        e_init = float(init_rel.get_potential_energy())
        e_fin  = float(fin_rel.get_potential_energy())
        delta_e = abs(e_init - e_fin)
        logger.info(f"  Endpoint energies: init={e_init:.4f} eV, "
                    f"fin={e_fin:.4f} eV, |ΔE|={delta_e:.4f} eV")
        if delta_e > 1.0:
            logger.warning(
                f"  Large endpoint energy asymmetry ({delta_e:.2f} eV). "
                "The 'final' site may not be a true energy minimum. "
                "Consider re-running bvse_ions_path.py or trying a different site."
            )

        # Volume collapse check
        v_max = max(v_init, v_fin, 1.0)
        dv = abs(v_init - v_fin) / v_max
        if dv > self.vol_thr:
            return NEBResult(
                converged=False,
                reason=f"Structural Collapse (|DeltaV/V| = {dv:.2%})",
                volume_initial=v_init,
                volume_final=v_fin,
            )

        # Build NEB images (start flat, no climbing)
        logger.info(f"  Building {self.n_images}-image NEB ...")
        images = [init_rel.copy() for _ in range(self.n_images)]
        
        # FIX: PBC wrap the final K+ position to the nearest image of the initial K+
        # This prevents 12+ Angstrom linear interpolation paths across unit cell boundaries.
        fin_wrapped = fin_rel.copy()
        cell = init_rel.get_cell().array
        pos_i = init_rel.positions[-1]
        pos_f = fin_rel.positions[-1]
        diff = pos_f - pos_i
        diff_wrapped = diff - np.round(diff @ np.linalg.inv(cell)) @ cell
        fin_wrapped.positions[-1] = pos_i + diff_wrapped
        
        images[0]  = init_rel.copy()
        images[-1] = fin_wrapped

        # Interpolate
        neb_flat = NEB(images, climb=False, k=self.k, allow_shared_calculator=True)
        try:
            neb_flat.interpolate(method=self.interp)
        except Exception as exc:
            logger.warning(f"  {self.interp.upper()} failed ({exc}); trying linear.")
            try:
                neb_flat.interpolate(method="linear")
            except Exception as exc2:
                return NEBResult(converged=False,
                                 reason=f"Interpolation failed: {exc2}",
                                 volume_initial=v_init, volume_final=v_fin)

        for img in images:
            img.calc = self.calc

        # Setup trajectory if requested
        traj_path = None
        if self.save_traj:
            Path(self.traj_dir).mkdir(exist_ok=True)
            traj_path = str(Path(self.traj_dir) / f"{mof_name}_neb.extxyz")

        converged   = False
        interrupted = False

        try:
            # ---- Phase 1: flat-band warm-up (climb=False) ---------------
            if self.warmup_steps > 0:
                logger.info(
                    f"  Phase 1 – flat-band warm-up: {self.warmup_steps} steps "
                    f"(climb=False, k={self.k}, maxstep={self.maxstep}) ..."
                )
                opt_flat = FIRE(neb_flat, maxstep=self.maxstep)
                opt_flat.run(fmax=self.fmax, steps=self.warmup_steps)

            # ---- Phase 2: Climbing-Image CI-NEB -------------------------
            remaining = max(1, self.max_steps - self.warmup_steps)
            # Rebuild NEB object with climb=True (reuses already-moved images)
            neb_ci = NEB(images, climb=True, k=self.k,
                         allow_shared_calculator=True)
            logger.info(
                f"  Phase 2 – CI-NEB: up to {remaining} steps "
                f"(climb=True, fmax={self.fmax}, maxstep={self.maxstep}) ..."
            )
            opt_ci = FIRE(neb_ci, maxstep=self.maxstep)
            converged = opt_ci.run(fmax=self.fmax, steps=remaining)

            # NaN check after run
            for img in images[1:-1]:
                f = img.get_forces()
                if not np.isfinite(f).all():
                    return NEBResult(
                        converged=False,
                        reason="NaN/Inf forces in NEB images after FIRE.",
                        volume_initial=v_init, volume_final=v_fin,
                    )
        except KeyboardInterrupt:
            logger.warning("  KeyboardInterrupt: extracting partial barrier.")
            interrupted = True
        except Exception as exc:
            msg = str(exc).lower()
            if "nan" in msg or "inf" in msg or "overflow" in msg:
                return NEBResult(converged=False,
                                 reason=f"Numerical explosion in FIRE: {exc}",
                                 volume_initial=v_init, volume_final=v_fin)
            return NEBResult(converged=False,
                             reason=f"Unexpected NEB error: {exc}",
                             volume_initial=v_init, volume_final=v_fin)
        finally:
            if traj_path is not None:
                try:
                    from ase.io import write
                    import os
                    write(traj_path, images)
                    
                    cp2k_dir = Path(self.traj_dir) / f"{mof_name}_cp2k"
                    cp2k_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(images):
                        write(str(cp2k_dir / f"image{i:02d}.xyz"), img)
                except Exception as e:
                    logger.warning(f"  Failed to save NEB trajectory: {e}")

        # Extract energies
        try:
            energies = [float(img.get_potential_energy()) for img in images]
        except Exception as exc:
            return NEBResult(converged=False,
                             reason=f"Energy extraction failed: {exc}",
                             volume_initial=v_init, volume_final=v_fin)

        if not np.isfinite(energies).all():
            return NEBResult(converged=False,
                             reason="Non-finite energies in NEB band.",
                             volume_initial=v_init, volume_final=v_fin)

        barrier = float(max(energies) - min(energies[0], energies[-1]))
        reason = ("OK" if (converged and not interrupted)
                  else "Partial (KeyboardInterrupt)" if interrupted
                  else "Partial (max_steps reached)")

        return NEBResult(
            converged=(converged and not interrupted),
            barrier_eV=barrier,
            energies=energies,
            reason=reason,
            volume_initial=v_init,
            volume_final=v_fin,
        )

    # ------------------------------------------------------------------
    def _mk_atoms(self, sc: Structure, frac: np.ndarray) -> Atoms:
        s = sc.copy()
        s.append(Element("K"), frac % 1.0)
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.wrap()
        return atoms

    def _relax_ep(self, atoms: Atoms) -> Tuple[Optional[Atoms], float]:
        """Relax one endpoint in two stages for accuracy.

        Stage A – K+-only relax with framework fixed.
            Pins K+ to the correct local basin identified by BVSE.
            Uses FixAtoms to freeze all non-K atoms.
        Stage B – Full unconstrained fine-relaxation.
            Lets the framework breathe slightly around the inserted K+.
        Returns (atoms, volume) or (None, 0) on failure.
        """
        from ase.constraints import FixAtoms

        atoms.calc = self.calc
        vol_before = abs(float(np.linalg.det(atoms.get_cell())))

        # Identify K+ index (last atom)
        k_idx = len(atoms) - 1
        k_pos_before = atoms.positions[k_idx].copy()

        # ---- Stage A: K+ only, framework fixed -------------------------
        framework_mask = [i for i in range(len(atoms)) if i != k_idx]
        fix = FixAtoms(indices=framework_mask)
        atoms.set_constraint(fix)
        try:
            opt_a = FIRE(atoms, maxstep=self.maxstep)
            opt_a.run(fmax=self.ep_fmax, steps=self.ep_steps)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.warning(f"  Endpoint K+-only relax exception: {exc}")
            return None, 0.0

        k_drift_A = float(np.linalg.norm(atoms.positions[k_idx] - k_pos_before))
        logger.info(f"    K+ drift in Stage A (K-only): {k_drift_A:.3f} Ang")
        if k_drift_A > 1.5:
            logger.warning(
                f"    K+ moved {k_drift_A:.2f} Ang during K+-only relax -- "
                "this site may not be a stable basin. Barrier may be inaccurate."
            )

        # ---- Stage B: Full unconstrained fine-relaxation ---------------
        atoms.set_constraint([])   # remove FixAtoms
        try:
            opt_b = FIRE(atoms, maxstep=self.maxstep)
            opt_b.run(fmax=self.ep_fmax, steps=self.ep_steps)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.warning(f"  Endpoint full-relax exception: {exc}")
            return None, 0.0

        forces = atoms.get_forces()
        if not np.isfinite(forces).all():
            return None, 0.0

        vol_after = abs(float(np.linalg.det(atoms.get_cell())))
        return atoms, vol_after


# ===========================================================================
# Stage 4: Post-Processor
# ===========================================================================

class PostProcessor:
    """
    Convert NEB barrier to diffusivity and ionic conductivity.

    Carrier concentration
    ---------------------
    Assumes one mobile K+ ion per supercell (standard dilute-limit approximation).
    n = 1 / V_supercell

    Arrhenius diffusivity
    ---------------------
    D = D0 * exp(-Ea / (kB * T))

    Nernst-Einstein conductivity
    ----------------------------
    sigma = (n * z^2 * e^2 * D) / (kB * T)
    """

    def __init__(self, cfg: dict):
        p = cfg["physics"]
        self.D0                  = p["D0"]
        self.T                   = p["T"]
        self.z                   = p["z"]
        self.sigma_threshold     = p["conductivity_threshold"]
        self.barrier_unphysical  = p["barrier_unphysical_threshold"]

    def compute(self, neb_result: NEBResult,
                super_volume_ang3: float,
                ) -> Tuple[float, float, float, bool]:
        """
        Returns
        -------
        (D_cm2s, sigma_Scm, n_cm3, al_flagged)
        """
        Ea = neb_result.barrier_eV

        if Ea > self.barrier_unphysical:
            logger.warning(
                f"  Barrier {Ea:.3f} eV > {self.barrier_unphysical} eV threshold "
                f"-- result may be unreliable (NEB not fully converged?)."
            )

        vol_cm3 = super_volume_ang3 * ANG3_TO_CM3
        n_cm3   = 1.0 / vol_cm3          # one K+ per supercell

        D       = self.D0 * math.exp(-Ea / (K_B_EV * self.T))
        D_m2s   = D * 1e-4               # cm^2/s -> m^2/s
        n_m3    = n_cm3 * 1e6            # cm^-3 -> m^-3

        sigma_Sm  = (n_m3 * self.z**2 * E_CHARGE**2 * D_m2s) / (K_B_SI * self.T)
        sigma_Scm = sigma_Sm * 1e-2      # S/m -> S/cm

        al_flagged = sigma_Scm >= self.sigma_threshold
        return D, sigma_Scm, n_cm3, al_flagged


# ===========================================================================
# MACE Calculator Factory
# ===========================================================================

def build_mace_calculator(cfg: dict, gpu_id: Optional[int] = None):
    """
    Build a MACE ASE calculator.

    Priority
    --------
    1. User-specified model_path (local file)
    2. Local cache / working directory candidates
    3. Auto-download MACE-MP-0 from HuggingFace (mace_mp helper)

    Parameters
    ----------
    gpu_id : int, optional
        If set and device=="cuda", pins to "cuda:<gpu_id>" for multi-GPU workers.
    """
    if not _MACE_AVAILABLE:
        raise ImportError(
            "mace-torch is not installed.  "
            "Install with:  pip install mace-torch"
        )

    mc     = cfg["model"]
    device = mc["device"]
    if gpu_id is not None and device == "cuda":
        device = f"cuda:{gpu_id}"

    dtype   = mc.get("default_dtype", "float32")
    head    = mc.get("head")
    mp_size = mc.get("mace_mp_size", "medium")

    def _make(model_path_str: str):
        kwargs = dict(
            model_paths=model_path_str,
            device=device,
            default_dtype=dtype,
        )
        if head:
            kwargs["head"] = head
        return MACECalculator(**kwargs)

    # 1. Explicit model_path
    model_path = mc.get("model_path")
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"MACE model not found: {model_path}")
        logger.info(f"Loading MACE from {p}")
        return _make(str(p))

    # 2. Local cache search
    home       = Path.home()
    cache_dir  = home / ".mace_cache"
    cwd        = Path(".")
    candidates = [
        cache_dir / "2024-01-07-mace-128-L2_epoch-199.model",
        cache_dir / "2023-12-03-mace-128-L1_epoch-199.model",
        cwd       / "2023-12-03-mace-128-L1_epoch-199.model",
        cwd       / "mace-mh-1.model",
        cwd       / "models" / "mace-mp-0-medium.model",
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"Using cached model: {c}")
            return _make(str(c))

    # 3. Auto-download
    logger.info(f"Auto-downloading MACE-MP-0 ({mp_size}) from HuggingFace ...")
    return mace_mp(model=mp_size, device=device, default_dtype=dtype)


# ===========================================================================
# CSV helpers
# ===========================================================================

_RESULT_FIELDS = [
    "cif_name", "status", "pld", "poav", "hop_distance_ang",
    "barrier_eV", "D_cm2s", "sigma_Scm", "n_carriers_cm3",
    "converged", "al_flagged", "runtime_s", "notes",
]
_FAILURE_FIELDS = ["cif_name", "status", "notes", "traceback"]
_DFT_FIELDS     = [
    "cif_name", "barrier_eV", "sigma_Scm", "D_cm2s", "n_carriers_cm3", "notes",
]


def _ensure_csv(path: str, fields: List[str]) -> None:
    p = Path(path)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()


def _append_csv(path: str, fields: List[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore").writerow(row)


# ===========================================================================
# Standalone per-CIF function  (multiprocessing-safe entry point)
# ===========================================================================

def process_single_cif(
    cif_path: str,
    cfg: Dict[str, Any],
    gpu_id: Optional[int] = None,
    calc=None,
) -> ScreeningResult:
    """
    Run the full 4-stage pipeline on a single CIF file.

    Parameters
    ----------
    cif_path : str
        Path to the .cif file.
    cfg : dict
        Pipeline configuration (from load_config()).
    gpu_id : int, optional
        CUDA device index for multiprocessing workers.
    calc : ASE Calculator, optional
        Pre-built MACE calculator (pass this in serial mode to avoid
        re-loading model weights for every MOF).

    Returns
    -------
    ScreeningResult
    """
    t0       = time.time()
    cif_name = Path(cif_path).stem
    out_cfg  = cfg["output"]

    logger.info(f"[{cif_name}] ---- starting pipeline ----")

    try:
        # ----------------------------------------------------------------
        # Stage 0: Clean
        # ----------------------------------------------------------------
        cleaner = CIFCleaner()
        struct  = cleaner.clean(cif_path)
        logger.info(f"[{cif_name}] Cleaned: {len(struct)} atoms, "
                    f"vol={struct.volume:.1f} Ang^3")

        # ----------------------------------------------------------------
        # Stage 1: Geometric pre-filter
        # ----------------------------------------------------------------
        gf   = GeometryFilter(cfg)
        filt = gf.filter(struct)
        logger.info(f"[{cif_name}] PLD={filt.pld:.3f} Ang, "
                    f"POAV={filt.poav:.1f} Ang^3  -->  {filt.reason}")

        if not filt.passed:
            result = ScreeningResult(
                cif_name=cif_name,
                status="Sterically Blocked",
                pld=filt.pld,
                poav=filt.poav,
                runtime_s=time.time() - t0,
                notes=filt.reason,
            )
            _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))
            return result

        # ----------------------------------------------------------------
        # Build MACE calculator (once per CIF in serial, once per worker
        # in parallel mode)
        # ----------------------------------------------------------------
        if calc is None:
            calc = build_mace_calculator(cfg, gpu_id=gpu_id)

        # ----------------------------------------------------------------
        # Stage 2: Pathfinding (or manual site override)
        # ----------------------------------------------------------------
        manual_init = cfg["pathfinding"].get("manual_initial_site")
        manual_fin  = cfg["pathfinding"].get("manual_final_site")
        pf = PathFinder(cfg, calc)

        if manual_init is not None and manual_fin is not None:
            # User provided explicit fractional coordinates -- snap them first
            raw_init = np.array(manual_init, dtype=float) % 1.0
            raw_fin  = np.array(manual_fin,  dtype=float) % 1.0
            
            sc_bare = struct.copy()
            sc_bare.make_supercell(np.diag(cfg["supercell"]))

            logger.info(f"[{cif_name}] Snapping manual K+ sites to MACE minima...")
            init_frac = pf._snap_k(sc_bare, raw_init)
            fin_frac  = pf._snap_k(sc_bare, raw_fin)

            if init_frac is None or fin_frac is None:
                logger.error(f"[{cif_name}] Failed to snap manual sites (forces NaN).")
                result = ScreeningResult(
                    cif_name=cif_name,
                    status="Manual Snap Failed",
                    pld=filt.pld, poav=filt.poav, runtime_s=time.time() - t0,
                    notes="NaN forces during K+ snapping."
                )
                _append_csv(out_cfg["failures_csv"], _FAILURE_FIELDS, asdict(result)) # type: ignore
                return result

            df = fin_frac - init_frac
            df -= np.round(df)
            hop_d = float(np.linalg.norm(df @ sc_bare.lattice.matrix))
            logger.info(
                f"[{cif_name}] Manual K+ sites snapped & ready.\n"
                f"  Initial: {raw_init} -> {init_frac}\n"
                f"  Final:   {raw_fin} -> {fin_frac}\n"
                f"  Hop distance: {hop_d:.3f} Ang"
            )
        else:
            bvse_success = False
            if cfg.get("bvse", {}).get("enabled", False) and _BVSE_AVAILABLE:
                logger.info(f"[{cif_name}] Attempting BVSE pathfinding using ions...")
                pr_bvse = find_paths_bvse(struct, cfg, pf, filt.void_centers)
                if pr_bvse.found:
                    init_frac, fin_frac, hop_d = pr_bvse.paths[0]
                    logger.info(f"[{cif_name}] BVSE PathFinder success: Hop distance {hop_d:.3f} Ang")
                    bvse_success = True
                else:
                    logger.warning(f"[{cif_name}] BVSE PathFinder failed: {pr_bvse.reason}")
            elif cfg.get("bvse", {}).get("enabled", False) and not _BVSE_AVAILABLE:
                logger.warning(f"[{cif_name}] BVSE enabled but ions library not available. Skipping BVSE.")

            if not bvse_success:
                if cfg.get("bvse", {}).get("enabled", False) and not cfg.get("bvse", {}).get("fallback_to_void", True):
                    logger.info(f"[{cif_name}] BVSE failed and fallback_to_void=False. Aborting.")
                    result = ScreeningResult(
                        cif_name=cif_name,
                        status="BVSE Failed",
                        pld=filt.pld, poav=filt.poav, runtime_s=time.time() - t0,
                        notes="BVSE failed and fallback_to_void is False."
                    )
                    _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))
                    return result

                logger.info(f"[{cif_name}] Running void-based PathFinder...")
                pr  = pf.find_paths(struct, filt.void_centers, cfg["supercell"])

                if not pr.found:
                    logger.info(f"[{cif_name}] No K+ path: {pr.reason}")
                    result = ScreeningResult(
                        cif_name=cif_name,
                        status="No Path Found",
                        pld=filt.pld,
                        poav=filt.poav,
                        runtime_s=time.time() - t0,
                        notes=pr.reason,
                    )
                    _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))
                    return result

                init_frac, fin_frac, hop_d = pr.paths[0]
                logger.info(f"[{cif_name}] Shortest hop: {hop_d:.3f} Ang  "
                            f"({len(pr.paths)} paths found)")

        # ----------------------------------------------------------------
        # Stage 3: CI-NEB
        # ----------------------------------------------------------------
        ev  = MACEEvaluator(cfg, calc, cfg["supercell"])
        nr  = ev.evaluate(struct, init_frac, fin_frac, mof_name=cif_name)

        if math.isnan(nr.barrier_eV) or not nr.reason.startswith(("OK", "Partial")):
            logger.warning(f"[{cif_name}] NEB failed: {nr.reason}")
            _append_csv(out_cfg["failures_csv"], _FAILURE_FIELDS, {
                "cif_name": cif_name,
                "status": "NEB Failure",
                "notes": nr.reason,
                "traceback": "",
            })
            result = ScreeningResult(
                cif_name=cif_name,
                status="NEB Failure",
                pld=filt.pld,
                poav=filt.poav,
                hop_distance_ang=hop_d,
                runtime_s=time.time() - t0,
                notes=nr.reason,
            )
            _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))
            return result

        logger.info(f"[{cif_name}] Barrier Ea = {nr.barrier_eV:.4f} eV  "
                    f"[{'converged' if nr.converged else 'partial'}]")

        # ----------------------------------------------------------------
        # Stage 4: Post-processing
        # ----------------------------------------------------------------
        sc_bare = struct.copy()
        sc_bare.make_supercell(np.diag(cfg["supercell"]))
        super_vol = sc_bare.volume

        pp = PostProcessor(cfg)
        D, sigma, n_cm3, al_flagged = pp.compute(nr, super_vol)

        logger.info(
            f"[{cif_name}] D={D:.3e} cm^2/s  sigma={sigma:.3e} S/cm  "
            f"n={n_cm3:.3e} cm^-3"
            + ("  [-> DFT queue]" if al_flagged else "")
        )

        status = "OK" if nr.converged else "Partial"
        result = ScreeningResult(
            cif_name=cif_name,
            status=status,
            pld=filt.pld,
            poav=filt.poav,
            hop_distance_ang=hop_d,
            barrier_eV=nr.barrier_eV,
            D_cm2s=D,
            sigma_Scm=sigma,
            n_carriers_cm3=n_cm3,
            converged=nr.converged,
            al_flagged=al_flagged,
            runtime_s=time.time() - t0,
            notes=nr.reason,
        )
        _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))

        if al_flagged:
            _append_csv(out_cfg["dft_queue_csv"], _DFT_FIELDS, {
                "cif_name":        cif_name,
                "barrier_eV":      nr.barrier_eV,
                "sigma_Scm":       sigma,
                "D_cm2s":          D,
                "n_carriers_cm3":  n_cm3,
                "notes":           nr.reason,
            })

        return result

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        tb_str = traceback.format_exc()
        logger.error(f"[{cif_name}] Unhandled error: {exc}\n{tb_str}")
        _append_csv(out_cfg["failures_csv"], _FAILURE_FIELDS, {
            "cif_name":  cif_name,
            "status":    "Error",
            "notes":     str(exc),
            "traceback": tb_str[:600],
        })
        result = ScreeningResult(
            cif_name=cif_name,
            status="Error",
            runtime_s=time.time() - t0,
            notes=str(exc),
        )
        _append_csv(out_cfg["results_csv"], _RESULT_FIELDS, asdict(result))
        return result


# ===========================================================================
# Batch orchestrator
# ===========================================================================

class QMOFScreener:
    """
    Iterate over a collection of CIF files and run the full pipeline on each.

    Serial mode   (workers=1):
        Builds MACE calculator once; reuses across all MOFs.
        Fastest for single-GPU workstations.

    Parallel mode (workers>1):
        Uses multiprocessing.Pool.  Each worker process builds its own
        MACE calculator independently.  GPU indices assigned round-robin.
        Suitable for multi-GPU servers.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def run(self, cif_paths: List[str], workers: int = 1) -> List[ScreeningResult]:
        out = self.cfg["output"]
        for path, fields in [
            (out["results_csv"],  _RESULT_FIELDS),
            (out["failures_csv"], _FAILURE_FIELDS),
            (out["dft_queue_csv"], _DFT_FIELDS),
        ]:
            _ensure_csv(path, fields)

        mode = "serial" if workers <= 1 else f"{workers} workers"
        logger.info(f"Screening {len(cif_paths)} CIFs [{mode}] ...")

        if workers <= 1:
            return self._serial(cif_paths)
        return self._parallel(cif_paths, workers)

    # ------------------------------------------------------------------
    def _serial(self, cif_paths: List[str]) -> List[ScreeningResult]:
        logger.info("Building MACE calculator (serial) ...")
        calc    = build_mace_calculator(self.cfg)
        results = []
        n       = len(cif_paths)
        for i, cif in enumerate(cif_paths, 1):
            logger.info(f"=== [{i}/{n}] {Path(cif).name} ===")
            res = process_single_cif(cif, self.cfg, calc=calc)
            results.append(res)
            _log_result(res, i, n)
        _print_summary(results, self.cfg)
        return results

    # ------------------------------------------------------------------
    def _parallel(self, cif_paths: List[str], workers: int) -> List[ScreeningResult]:
        import multiprocessing as mp

        device = self.cfg["model"]["device"]
        args = [
            (cif, self.cfg, i % workers if device == "cuda" else None, None)
            for i, cif in enumerate(cif_paths)
        ]

        results = []
        with mp.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.starmap(process_single_cif, args), 1):
                results.append(res)
                _log_result(res, i, len(cif_paths))

        _print_summary(results, self.cfg)
        return results


# ===========================================================================
# Reporting helpers
# ===========================================================================

_STATUS_ICON = {
    "OK":                "OK",
    "Partial":           "~~",
    "Sterically Blocked":"XX",
    "No Path Found":     "??",
    "NEB Failure":       "!!",
    "Error":             "EE",
}


def _log_result(res: ScreeningResult, i: int, n: int) -> None:
    icon   = _STATUS_ICON.get(res.status, "  ")
    ea_str = f"Ea={res.barrier_eV:.3f} eV" if math.isfinite(res.barrier_eV) else "Ea=N/A"
    sg_str = f"sigma={res.sigma_Scm:.2e} S/cm" if math.isfinite(res.sigma_Scm) else "sigma=N/A"
    al_str = " [DFT-queue]" if res.al_flagged else ""
    logger.info(
        f"  [{icon}] ({i}/{n}) {res.cif_name}: {res.status} | "
        f"{ea_str} | {sg_str} | {res.runtime_s:.0f}s{al_str}"
    )


def _print_summary(results: List[ScreeningResult], cfg: dict) -> None:
    ok      = [r for r in results if r.status in ("OK", "Partial")]
    blocked = [r for r in results if r.status == "Sterically Blocked"]
    failed  = [r for r in results if r.status not in ("OK", "Partial",
                                                        "Sterically Blocked")]
    al_q    = [r for r in ok if r.al_flagged]

    logger.info("=" * 65)
    logger.info(f"QMOF SCREENING COMPLETE  ({len(results)} structures)")
    logger.info(f"  Sterically blocked  : {len(blocked)}")
    logger.info(f"  NEB succeeded       : {len(ok)}")
    logger.info(f"  Failures / errors   : {len(failed)}")
    logger.info(f"  DFT validation queue: {len(al_q)}")

    if ok:
        barriers = [r.barrier_eV for r in ok if math.isfinite(r.barrier_eV)]
        sigmas   = [r.sigma_Scm  for r in ok if math.isfinite(r.sigma_Scm)]
        if barriers:
            best = ok[int(np.argmin(barriers))]
            logger.info(
                f"  Best barrier        : {min(barriers):.4f} eV  ({best.cif_name})"
            )
        if sigmas:
            best_s = ok[int(np.argmax(sigmas))]
            logger.info(
                f"  Best conductivity   : {max(sigmas):.4e} S/cm  ({best_s.cif_name})"
            )

    out = cfg["output"]
    logger.info(f"  Results CSV   : {out['results_csv']}")
    logger.info(f"  Failures CSV  : {out['failures_csv']}")
    logger.info(f"  DFT queue CSV : {out['dft_queue_csv']}")
    logger.info("=" * 65)


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QMOF K+ Battery Anode Screening Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--cif_file", metavar="FILE",
                     help="Single CIF to process.")
    inp.add_argument("--cif_dir", metavar="DIR",
                     help="Directory of CIF files (batch mode).")

    p.add_argument("--config", metavar="YAML", default=None,
                   help="Path to pipeline_config.yaml.")

    # Model overrides
    p.add_argument("--model_path", default=None,
                   help="Override: path to local MACE .model file.")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Override: compute device.")
    p.add_argument("--head", default=None,
                   help="Override: multi-head name (e.g. mp_pbe_refit_add).")

    # NEB overrides
    p.add_argument("--n_images", type=int, default=None,
                   help="Override: number of NEB images.")
    p.add_argument("--fmax", type=float, default=None,
                   help="Override: NEB FIRE convergence threshold (eV/Ang).")
    p.add_argument("--T", type=float, default=None,
                   help="Override: temperature in K.")
    p.add_argument("--supercell", nargs=3, type=int, default=None,
                   metavar=("A", "B", "C"),
                   help="Override: supercell dimensions.")

    # Manual K+ site override (skips automated PathFinder)
    p.add_argument("--initial_site", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Manual initial K+ fractional coordinates. "
                        "Skips automated PathFinder (Stage 2).")
    p.add_argument("--final_site", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Manual final K+ fractional coordinates. "
                        "Must be used together with --initial_site.")

    # Batch
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes.")
    p.add_argument("--output_dir", default=None,
                   help="Directory for output CSVs.")
    return p


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model_path:  cfg["model"]["model_path"]  = args.model_path
    if args.device:      cfg["model"]["device"]       = args.device
    if args.head:        cfg["model"]["head"]          = args.head
    if args.n_images:    cfg["neb"]["n_images"]        = args.n_images
    if args.fmax:        cfg["neb"]["fmax"]            = args.fmax
    if args.T:           cfg["physics"]["T"]           = args.T
    if args.supercell:   cfg["supercell"]              = args.supercell

    # Manual K+ site override
    if args.initial_site is not None and args.final_site is not None:
        cfg["pathfinding"]["manual_initial_site"] = args.initial_site
        cfg["pathfinding"]["manual_final_site"]   = args.final_site
    elif args.initial_site is not None or args.final_site is not None:
        raise ValueError(
            "--initial_site and --final_site must both be provided together."
        )

    if args.output_dir:
        d = Path(args.output_dir)
        d.mkdir(parents=True, exist_ok=True)
        for key in ("results_csv", "failures_csv", "dft_queue_csv", "neb_trajectories_dir"):
            cfg["output"][key] = str(d / Path(cfg["output"][key]).name)
    return cfg


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    cfg    = _apply_overrides(cfg, args)

    # Apply log level from config
    log_level = getattr(logging, cfg["output"]["log_level"].upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Collect CIF paths
    if args.cif_file:
        cif_paths = [args.cif_file]
    else:
        cif_dir   = Path(args.cif_dir)
        cif_paths = sorted(
            list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.CIF"))
        )
        cif_paths = [str(p) for p in cif_paths]
        if not cif_paths:
            logger.error(f"No .cif files found in {args.cif_dir}")
            sys.exit(1)
        logger.info(f"Found {len(cif_paths)} CIF files in {args.cif_dir}")

    screener = QMOFScreener(cfg)
    screener.run(cif_paths, workers=args.workers)
    sys.exit(0)


if __name__ == "__main__":
    main()
