"""Sample structures from MD trajectories and relax with MACE.

Reads all mdtraj_seed*.xyz, randomly samples NSAMPLES structures,
runs variable-cell relaxation (FIRE + FrechetCellFilter) with MACE,
and writes the full relaxation trajectory (every step of every sample)
to relax.xyz.

Usage:
    python relax_umlip.py [--nsamples 100] [--max-steps 200] [--seed 0]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import ase.io
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from mace.calculators import mace_mp

# ── settings ─────────────────────────────────────────────────────────────────
MODEL     = "medium-omat-0"
DEVICE    = "cuda"
DTYPE     = "float32"

NSAMPLES  = 100
MAX_STEPS = 20
FMAX      = 0.05   # eV/Å  (loose; DFT will refine)
# ─────────────────────────────────────────────────────────────────────────────


def load_trajectories() -> list:
    files = sorted(Path(".").glob("mdtraj_seed*.xyz"))
    if not files:
        raise FileNotFoundError("no mdtraj_seed*.xyz files found in current directory")
    frames = []
    for f in files:
        frames.extend(ase.io.read(str(f), index=":"))
    print(f"loaded {len(frames)} frames from {len(files)} trajectory file(s)")
    return frames


def run(nsamples: int, max_steps: int, seed: int) -> None:
    frames = load_trajectories()

    random.seed(seed)
    selected = random.sample(frames, k=min(nsamples, len(frames)))
    print(f"sampled {len(selected)} structures (seed={seed})")

    calc = mace_mp(model=MODEL, device=DEVICE, default_dtype=DTYPE)

    relaxed = []
    n_conv_traj = 0
    for i, atoms in enumerate(selected):
        atoms = atoms.copy()
        atoms.pbc = True
        atoms.calc = calc

        ecf = FrechetCellFilter(atoms)
        opt = FIRE(ecf, logfile=None)

        traj = []
        def capture(atoms=atoms, traj=traj):
            snap = atoms.copy()
            snap.info["mace_energy"]   = float(atoms.get_potential_energy())
            snap.arrays["mace_forces"] = atoms.get_forces().astype(np.float64)
            snap.info["mace_stress"]   = atoms.get_stress().astype(np.float64)
            traj.append(snap)
        opt.attach(capture, interval=1)

        converged = opt.run(fmax=FMAX, steps=max_steps)

        if traj:
            traj[-1].info["relax_steps"] = opt.get_number_of_steps()
            traj[-1].info["converged"]   = bool(converged)

        relaxed.extend(traj)
        n_conv_traj += int(bool(converged))
        status = "✓" if converged else f">{max_steps}"
        n_steps = opt.get_number_of_steps()
        print(f"  [{i+1:>{len(str(len(selected)))}}/{len(selected)}] steps={n_steps:3d} frames={len(traj)} {status}")

    out = Path("relax.xyz")
    ase.io.write(str(out), relaxed)
    print(f"wrote {len(relaxed)} frames from {len(selected)} trajectories → {out}  ({n_conv_traj} converged)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples",  type=int, default=NSAMPLES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()
    run(args.nsamples, args.max_steps, args.seed)
