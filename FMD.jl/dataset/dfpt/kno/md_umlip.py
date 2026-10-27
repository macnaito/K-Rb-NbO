"""NPT-MD sampling with MACE-MP foundation model for KNbO3.

Usage (single seed, for HPC array job):
    python md_umlip.py --seed 0

Output: mdtraj_seed<N>.xyz  (ASE extxyz, properties: energy forces stress)

Settings mirror example/python_workflow/kno_production/workflow.toml
[stage.sample_umlip].
"""

import argparse
from pathlib import Path

import numpy as np
import ase.io
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs
from mace.calculators import mace_mp

# ── MD settings ──────────────────────────────────────────────────────────────
MODEL       = "medium-omat-0"
DEVICE      = "cuda"
DTYPE       = "float32"

INPUT_XYZ   = Path(__file__).parent / "init.xyz"

T_K         = 1500.0
DT_FS       = 1.0
NSTEPS      = 10000
LOGINTERVAL = 20
TTIME_FS    = 25.0
PTIME_FS    = 75.0
B_EVA3      = 0.6          # bulk modulus estimate [eV/Å³]
# ─────────────────────────────────────────────────────────────────────────────


def run(seed: int) -> None:
    atoms = ase.io.read(str(INPUT_XYZ))
    atoms.pbc = True
    atoms.calc = mace_mp(model=MODEL, device=DEVICE, default_dtype=DTYPE)

    np.random.seed(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    pfactor = (PTIME_FS * fs) ** 2 * B_EVA3
    dyn = NPT(
        atoms,
        timestep=DT_FS * fs,
        temperature_K=T_K,
        ttime=TTIME_FS * fs,
        pfactor=pfactor,
        mask=None,
        externalstress=0.0,
        logfile="-",
        loginterval=LOGINTERVAL,
    )

    trajectory: list = []

    def capture() -> None:
        snap = atoms.copy()
        snap.info["energy"] = float(atoms.get_potential_energy())
        snap.arrays["forces"] = atoms.get_forces().astype(np.float64)
        snap.info["stress"] = atoms.get_stress().astype(np.float64)
        trajectory.append(snap)

    dyn.attach(capture, interval=LOGINTERVAL)
    dyn.run(NSTEPS)

    out = Path(f"mdtraj_seed{seed}.xyz")
    ase.io.write(str(out), trajectory)
    print(f"wrote {len(trajectory)} frames → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.seed)
