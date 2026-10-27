"""Build a 180-degree domain-wall cell and the target polarization map for it.

Writes three files:

  prototype.xyz  ideal cubic perovskite, no displacements.  This is the
                 structure declared to have P = 0 (the polarization gauge).
  cell.xyz       the same supercell with the up/down domain pattern applied,
                 i.e. the configuration the run starts from.
  domain.json    the domain sign of each group and the per-group weight mask,
                 in the flat group order FMD expects
                 (ix*Ny*Nz + iy*Nz + iz, iz fastest).

The target polarization itself is NOT written here: input.jl measures it from
this cell with `group_p_gauged`, which is the same quantity the bias restrains.
Hard-coding a number invites the two to disagree.

The seed pattern is geometric: each unit cell gets the soft-mode displacement
of a relaxed single-domain cell, with the sign flipped between the two halves
of the box along x.  Two 180-degree walls result, since the cell is periodic.

Shell-model parameters and the soft-mode displacement amplitude for BaTiO3 are
from H. Azuma et al., Acta Mater. 296, 121216 (2025),
doi:10.1016/j.actamat.2025.121216.
"""

import json

import numpy as np
from ase import Atoms

from fmdlib.supercell import write_groupid_supercell

# Ideal cubic perovskite.  Order matters: the gauge is stored per atom index,
# so prototype.xyz and cell.xyz must list atoms identically.
UNIT = Atoms(
    symbols=["Ba", "Ti", "O", "O", "O"],
    scaled_positions=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
    cell=[4.0, 4.0, 4.0],
    pbc=True,
)

NX, NY, NZ = 12, 4, 4          # wall normal along x
NG = NX * NY * NZ

# Soft-mode displacement of the relaxed tetragonal cell, in Angstrom, as the
# B-site and oxygen shifts relative to the A-site sublattice.  Polarization
# points along +z for the "up" domain.
U_TI = 0.076
U_O = -0.093

# Half-width, in unit cells, of the band released around each wall.  The
# domains are restrained; the wall is left free to find its own width and
# internal structure, which is usually the thing being measured.
RELEASE = 2


def build():
    proto = write_groupid_supercell("prototype.xyz", UNIT, (NX, NY, NZ))

    # Domain pattern: +z polarization for the first half of x, -z for the
    # second.  `sign` is indexed by the unit cell's ix.
    domain = np.where(np.arange(NX) < NX // 2, 1.0, -1.0)

    cell = proto.copy()
    groupid = proto.arrays["groupid"]           # 1-based, flat, iz fastest
    ix_of_group = (groupid - 1) // (NY * NZ)
    symbols = np.array(cell.get_chemical_symbols())
    shift = np.zeros((len(cell), 3))
    s = domain[ix_of_group]
    shift[symbols == "Ti", 2] = U_TI * s[symbols == "Ti"]
    shift[symbols == "O", 2] = U_O * s[symbols == "O"]
    cell.positions += shift
    cell.arrays["groupid"] = groupid
    cell.write("cell.xyz", format="extxyz")

    # Domain sign per group, and the released band around each wall.
    #
    # The sign flips BETWEEN cells, so the walls sit at half-integer ix: with
    # NX = 12 the flips are between ix = 5|6 and (periodically) ix = 11|0, i.e.
    # at ix = 5.5 and 11.5.  Measuring the distance from a cell's centre
    # (ix + 0.5) to those positions is what makes the released band symmetric
    # about each wall; using integer ix would release one more cell on one
    # side than the other.
    ix = np.arange(NG) // (NY * NZ)
    sign = domain[ix]
    centre = ix + 0.5
    walls = np.array([NX / 2.0, float(NX)])       # 6.0 and 12.0 ≡ 0.0
    dist = np.min(
        [np.abs(((centre - w) + NX / 2) % NX - NX / 2) for w in walls], axis=0
    )
    weight = np.where(dist < RELEASE, 0.0, 1.0)

    with open("domain.json", "w") as fh:
        json.dump(
            {
                "gridsize": [NX, NY, NZ],
                "domain_sign": sign.tolist(),
                "weight": weight.tolist(),
            },
            fh,
        )

    print(
        f"wrote prototype.xyz / cell.xyz: {len(cell)} atoms, {NG} groups; "
        f"domain.json: {int((weight == 0).sum())} of {NG} groups released "
        f"around the walls at ix = {walls[0] - 0.5} and {walls[1] - 0.5}"
    )


if __name__ == "__main__":
    build()
