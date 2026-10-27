"""Build a cosine-modulated umbrella-sampling cell.

The supercell is created with a single ASE replication.  Using the new
ASE-natural umbrella convention, the flat-index decode (iz fastest, ix
slowest) means `m = (0, 0, mz)` produces a real-space wave along z
without any NSUP two-stage trick or axis transposition.
"""
from ase import Atoms

from fmdlib.supercell import write_groupid_supercell

unit_cell = Atoms(
    symbols=["Ba", "Ti", "O", "O", "O"],
    scaled_positions=[
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ],
    cell=[4.0, 4.0, 4.0],
    pbc=True,
)

# Inhomogeneous (cosine) z-axis bias: 12 unit cells along z, 3 along x and y
# → wavelength along z = 12 unit cells = 48 Å with m = (0, 0, 1).
sys = write_groupid_supercell("cell.xyz", unit_cell, (3, 3, 12))
print(
    f"wrote cell.xyz: {len(sys)} atoms, "
    f"{int(sys.arrays['groupid'].max())} groups"
)
