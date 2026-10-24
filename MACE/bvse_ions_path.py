#!/usr/bin/env python3
"""
bvse_ions_path.py — Find K+ migration path using the `ions` library.

Uses Bond Valence Site Energy (BVSE) to automatically find the Minimum Energy Path
(MEP) for K+ in any MOF. Outputs the initial and final fractional coordinates
for use with MACE-NEB.

Dependencies:
    pip install ions pymatgen ase numpy scipy

Usage:
    python bvse_ions_path.py MOF.cif --mobile_ion K+ [--resolution 0.2] [--r_cut 10.0]

Output:
    Initial site: 0.123456 0.789012 0.345678
    Final site:   0.654321 0.210988 0.567890
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------------------------
try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    raise ImportError("pymatgen is required. Install with: pip install pymatgen")

try:
    import bvlain
except ImportError:
    raise ImportError("bvlain is required. Install with: pip install bvlain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BVSE:
    """
    Bond Valence Site Energy (BVSE) pathway analysis class.
    Wraps the bvlain library and implements minimum energy path (MEP) search.
    """
    def __init__(self, structure, mobile_ion="K+", resolution=0.2, r_cut=10.0):
        self.structure = structure
        self.mobile_ion = mobile_ion
        self.resolution = resolution
        self.r_cut = r_cut
        self.grid_data = None
        self.calc = None

    def compute_grid(self):
        from bvlain import Lain
        self.calc = Lain()
        # Read structure (disabling internal oxi_check since we clean and decorate it first)
        self.calc.read_structure(self.structure, oxi_check=False)
        
        # Standardise mobile ion notation for bvse_distribution (e.g. K+ or K1+)
        ion = self.mobile_ion
        if not (ion.endswith("+") or ion.endswith("-")):
            ion += "+"
            
        logger.info(f"Computing BVSE grid using bvlain for mobile ion {ion}...")
        self.grid_data = self.calc.bvse_distribution(
            mobile_ion=ion,
            r_cut=self.r_cut,
            resolution=self.resolution,
        )

    def get_barrier(self):
        try:
            barriers = self.calc.percolation_barriers()
            valid = [v for v in barriers.values() if v is not None and np.isfinite(v)]
            if valid:
                return float(min(valid))
        except Exception:
            pass
        return 0.0

    def get_mep(self):
        if self.grid_data is None:
            raise RuntimeError("Grid data not computed. Run compute_grid() first.")

        grid = self.grid_data
        nx, ny, nz = grid.shape
        
        # 1. Find local minima in the 3D grid
        local_minima = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    val = grid[i, j, k]
                    is_min = True
                    # Check 6-neighbors with PBC
                    for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        ni, nj, nk = (i + di) % nx, (j + dj) % ny, (k + dk) % nz
                        if grid[ni, nj, nk] < val:
                            is_min = False
                            break
                    if is_min:
                        local_minima.append((val, (i, j, k)))

        if len(local_minima) < 2:
            logger.warning("Fewer than 2 local minima found in BVSE grid; using lowest energy grid points.")
            flat_indices = np.argsort(grid.flatten())
            for idx in flat_indices[:10]:
                i, j, k = np.unravel_index(idx, grid.shape)
                local_minima.append((grid[i, j, k], (int(i), int(j), int(k))))

        # 2. Find a pair of local minima representing a physically reasonable migration path (1.5 to 5.5 Angstroms)
        selected_pair = None
        min_energy_sum = float("inf")
        lat = self.structure.lattice
        
        for idx1 in range(len(local_minima)):
            val1, (i1, j1, k1) = local_minima[idx1]
            f1 = np.array([i1 / nx, j1 / ny, k1 / nz])
            for idx2 in range(idx1 + 1, len(local_minima)):
                val2, (i2, j2, k2) = local_minima[idx2]
                f2 = np.array([i2 / nx, j2 / ny, k2 / nz])
                
                df = f2 - f1
                df -= np.round(df)
                cart = df @ lat.matrix
                d = np.linalg.norm(cart)
                if 1.5 <= d <= 5.5:
                    if val1 + val2 < min_energy_sum:
                        min_energy_sum = val1 + val2
                        selected_pair = ((i1, j1, k1), (i2, j2, k2))

        # Fallback to closest non-identical pair if no pair matches the distance window
        if selected_pair is None:
            min_dist = float("inf")
            for idx1 in range(len(local_minima)):
                val1, (i1, j1, k1) = local_minima[idx1]
                f1 = np.array([i1 / nx, j1 / ny, k1 / nz])
                for idx2 in range(idx1 + 1, len(local_minima)):
                    val2, (i2, j2, k2) = local_minima[idx2]
                    f2 = np.array([i2 / nx, j2 / ny, k2 / nz])
                    
                    df = f2 - f1
                    df -= np.round(df)
                    cart = df @ lat.matrix
                    d = np.linalg.norm(cart)
                    if d > 0.5:  # Not the same site
                        if d < min_dist:
                            min_dist = d
                            selected_pair = ((i1, j1, k1), (i2, j2, k2))

        if selected_pair is None:
            raise RuntimeError("Could not identify distinct start and end sites in the BVSE grid.")

        # 3. Pathfinding (minimax/Dijkstra bottleneck search on the grid)
        import heapq
        start, end = selected_pair
        
        min_max_energy = {start: grid[start]}
        parent = {}
        pq = [(grid[start], start)]
        
        while pq:
            max_e, curr = heapq.heappop(pq)
            if curr == end:
                break
            if max_e > min_max_energy.get(curr, float("inf")):
                continue
                
            i, j, k = curr
            for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                neighbor = ((i + di) % nx, (j + dj) % ny, (k + dk) % nz)
                alt = max(max_e, grid[neighbor])
                if alt < min_max_energy.get(neighbor, float("inf")):
                    min_max_energy[neighbor] = alt
                    parent[neighbor] = curr
                    heapq.heappush(pq, (alt, neighbor))

        # Reconstruct path
        path = []
        curr = end
        while curr in parent:
            path.append(curr)
            curr = parent[curr]
        path.append(start)
        path.reverse()

        # Convert to fractional coordinates
        mep = []
        for i, j, k in path:
            mep.append([i / nx, j / ny, k / nz])
            
        return mep


def clean_structure(struct: Structure) -> Structure:
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
        logger.warning(f"Could not assign oxidation states via BVAnalyzer: {e}. Falling back to default guesser.")
        # Fallback manual guesser
        guesses = {}
        for el in struct.composition.elements:
            symbol = el.symbol
            if symbol == 'O':
                guesses[symbol] = -2
            elif symbol == 'H':
                guesses[symbol] = 1
            elif symbol == 'F':
                guesses[symbol] = -1
            elif symbol == 'Cl':
                guesses[symbol] = -1
            elif symbol == 'N':
                guesses[symbol] = -3
            elif symbol == 'S':
                guesses[symbol] = -2
            elif symbol in ['Li', 'Na', 'K', 'Rb', 'Cs']:
                guesses[symbol] = 1
            elif symbol in ['Mg', 'Ca', 'Sr', 'Ba', 'Zn', 'Cd']:
                guesses[symbol] = 2
            elif symbol in ['Al', 'Ga', 'In', 'Sc', 'Y', 'La']:
                guesses[symbol] = 3
            elif symbol in ['Ti', 'Zr', 'Hf', 'Ce']:
                guesses[symbol] = 4
            elif symbol == 'C':
                guesses[symbol] = 4
            else:
                if el.is_metal:
                    guesses[symbol] = 3
                else:
                    guesses[symbol] = -2
        try:
            struct.add_oxidation_state_by_element(guesses)
        except Exception as exc:
            logger.error(f"Failed to manually assign oxidation states: {exc}")
    return struct


def get_bvse_path_ions(
    cif_path: str,
    mobile_ion: str = "K+",
    resolution: float = 0.2,
    r_cut: float = 10.0,
) -> tuple:
    """
    Use ions.BVSE to compute the BVSE landscape and extract the Minimum Energy Path (MEP).

    Returns:
        (initial_frac, final_frac) as numpy arrays of fractional coordinates.
    """
    # 1. Load and clean structure
    struct = Structure.from_file(cif_path)
    struct = clean_structure(struct)
    logger.info(f"Structure: {len(struct)} atoms, volume={struct.volume:.1f} Å³")

    # 2. Initialize BVSE calculator
    logger.info(f"Running BVSE for {mobile_ion} (resolution={resolution} Å, r_cut={r_cut} Å)...")
    bvse = BVSE(
        structure=struct,
        mobile_ion=mobile_ion,
        resolution=resolution,
        r_cut=r_cut,
    )

    # 3. Compute the BVSE grid
    bvse.compute_grid()
    logger.info("BVSE grid computed.")

    # 4. Get the Minimum Energy Path (MEP)
    # The MEP is a list of fractional coordinates along the path.
    mep = bvse.get_mep()
    if mep is None or len(mep) < 2:
        raise RuntimeError("Could not extract a valid MEP. Try adjusting resolution or r_cut.")

    logger.info(f"MEP contains {len(mep)} points.")

    # 5. Extract initial and final points
    initial_frac = np.array(mep[0]) % 1.0
    final_frac = np.array(mep[-1]) % 1.0

    # Optionally, also get the BVSE barrier estimate (for reference)
    try:
        barrier = bvse.get_barrier()
        logger.info(f"BVSE barrier estimate: {barrier:.4f} eV (approximate)")
    except Exception:
        pass

    return initial_frac, final_frac


def main():
    parser = argparse.ArgumentParser(
        description="Find K+ migration path using ions library."
    )
    parser.add_argument("cif", type=str, help="MOF CIF file")
    parser.add_argument("--mobile_ion", type=str, default="K+",
                        help="Mobile ion species (e.g., K+, Li+)")
    parser.add_argument("--resolution", type=float, default=0.2,
                        help="Grid resolution (Å)")
    parser.add_argument("--r_cut", type=float, default=10.0,
                        help="Interaction cutoff (Å)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove oxidation states and solvents")
    args = parser.parse_args()

    logger.info("=== BVSE + ions Path Finder ===")
    logger.info(f"CIF: {args.cif}")
    logger.info(f"Mobile ion: {args.mobile_ion}")

    try:
        init_frac, final_frac = get_bvse_path_ions(
            args.cif,
            mobile_ion=args.mobile_ion,
            resolution=args.resolution,
            r_cut=args.r_cut,
        )
    except Exception as e:
        logger.error(f"Pathfinding failed: {e}")
        sys.exit(1)

    print(f"\nInitial site: {init_frac[0]:.6f} {init_frac[1]:.6f} {init_frac[2]:.6f}")
    print(f"Final site:   {final_frac[0]:.6f} {final_frac[1]:.6f} {final_frac[2]:.6f}")
    print("\n# Use these with --initial_site and --final_site in mace_mof_diffusion.py")
    print("# For the supercell, apply the same fractional coordinates (they are independent of supercell size).")


if __name__ == "__main__":
    main()