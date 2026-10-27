"""VASP single-point input generation and output collection for KNbO3.

Structures are pre-relaxed by MACE (relax_umlip.py), so VASP only computes
energy / forces / stress at the given geometry (NSW=0, IBRION=-1).
ISIF=2 requests stress without cell optimisation.

Usage:
    # 1. write VASP inputs
    python gen_vasp.py [--nsamples 100] [--seed 0]

    # 2. after VASP finishes, collect outputs into dft.xyz
    python gen_vasp.py --collect [--output dft.xyz]
"""

import argparse
import os
import random
from pathlib import Path
from natsort import natsorted

import numpy as np
import ase.io
from ase.calculators.vasp import Vasp

# ── VASP settings ─────────────────────────────────────────────────────────────
VASP_PARAMS = dict(
    xc       = "PBEsol",   # GGA = PS
    prec     = "Accurate",
    kspacing = 0.3,
    encut    = 600,
    ediff    = 1e-6,
    ismear   = 0,
    sigma    = 0.05,
    nsw      = 0,          # static: MACE already relaxed the structure
    ibrion   = -1,         # no ionic relaxation
    isif     = 2,          # compute stress but do not relax cell
    isym     = 0,
    kpar     = 2,
    ncore    = 4,
    nwrite   = 1,
    lwave    = False,
)

NSAMPLES  = 100
INPUT_XYZ = Path("relax.xyz")
WORKDIR   = Path("workdir")
# ─────────────────────────────────────────────────────────────────────────────


def write_inputs(nsamples: int, seed: int) -> None:
    frames = ase.io.read(str(INPUT_XYZ), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    print(f"loaded {len(frames)} structures from {INPUT_XYZ}")

    random.seed(seed)
    selected = random.sample(frames, k=min(nsamples, len(frames)))
    print(f"sampled {len(selected)} structures (seed={seed})")

    has_pp = bool(os.environ.get("VASP_PP_PATH") or os.environ.get("PMG_VASP_PSP_DIR"))
    if not has_pp:
        print("warning: VASP_PP_PATH not set — POTCAR will not be written")

    for i, atoms in enumerate(selected):
        order = sorted(range(len(atoms)), key=lambda j: atoms[j].symbol)
        atoms = atoms[order]

        wdir = WORKDIR / f"{i:04d}"
        wdir.mkdir(parents=True, exist_ok=True)

        calc = Vasp(directory=str(wdir), **VASP_PARAMS)
        calc.write_input(atoms)

        if not has_pp:
            (wdir / "POTCAR").unlink(missing_ok=True)

        if (i + 1) % 10 == 0 or i == len(selected) - 1:
            print(f"  {i+1}/{len(selected)} → {wdir}")

    print(f"done: {len(selected)} directories under {WORKDIR}/")


def collect(output: Path) -> None:
    vaspruns = natsorted(WORKDIR.glob("*/vasprun.xml"))
    if not vaspruns:
        raise FileNotFoundError(f"no vasprun.xml found under {WORKDIR}/")
    print(f"found {len(vaspruns)} vasprun.xml file(s)")

    frames = []
    n_failed = 0
    for vasprun in vaspruns:
        try:
            atoms = ase.io.read(str(vasprun))
        except Exception as e:
            print(f"  skip {vasprun.parent.name}: {e}")
            n_failed += 1
            continue

        atoms.info["energy"] = float(atoms.get_potential_energy())
        atoms.arrays["forces"] = atoms.get_forces().astype(np.float64)
        atoms.info["stress"] = atoms.get_stress().astype(np.float64)
        frames.append(atoms)

    ase.io.write(str(output), frames)
    print(f"wrote {len(frames)} frames → {output}  ({n_failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect",  action="store_true",
                        help="collect vasprun.xml outputs instead of writing inputs")
    parser.add_argument("--output",   type=Path, default=Path("dft.xyz"),
                        help="output xyz for --collect (default: dft.xyz)")
    parser.add_argument("--nsamples", type=int,  default=NSAMPLES)
    parser.add_argument("--seed",     type=int,  default=0)
    args = parser.parse_args()

    if args.collect:
        collect(args.output)
    else:
        write_inputs(args.nsamples, args.seed)
