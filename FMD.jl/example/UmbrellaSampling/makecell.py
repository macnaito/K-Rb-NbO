"""Build a uniform-bias umbrella-sampling cell with per-unit-cell groupid.

The supercell is created with a single ASE replication; the umbrella's
flat-index decode is ASE-natural (iz fastest, ix slowest), so no NSUP
two-stage trick is needed.
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
    cell=[4.0, 4.0, 4.044],
    pbc=True,
)

# 9³ unit cells = 729 groups, 3645 atoms.
sys = write_groupid_supercell("cell.xyz", unit_cell, (9, 9, 9))
print(
    f"wrote cell.xyz: {len(sys)} atoms, "
    f"{int(sys.arrays['groupid'].max())} groups"
)
