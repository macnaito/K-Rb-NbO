"""VASP DFPT (LEPSILON+LPEAD) input generation and output collection for KNbO3.

Samples from MACE-relaxed structures after filtering by max force/stress
(poorly relaxed cells cause DFPT convergence failure).

Usage:
    # write VASP inputs
    python gen_vasp.py [--nsamples 30] [--seed 0]

    # after VASP finishes, collect BEC + dielectric tensor into dfpt.xyz
    python gen_vasp.py --collect [--output dfpt.xyz]

Output of --collect: dfpt.xyz in FMD extxyz format
  per-atom: species, pos, becex (3), becey (3), becez (3)
    becex[i, α] = ∂P_x/∂u_{i,α}  (= ∂F_α/∂E_x by Maxwell, matching FMD's
    `linear_response!`).  This is the transpose of VASP/ASE's native
    bec[i, α=pol, β=disp] storage; the slicing in `collect()` does the swap.
  per-frame: dielectric_tensor (Voigt-6, eV/Å³)
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
    xc        = "PBEsol",   # GGA = PS
    prec      = "Accurate",
    kspacing  = 0.25,       # denser than singlepoint for DFPT accuracy
    encut     = 600,
    ediff     = 1e-8,       # tight SCF for response functions
    ismear    = 0,
    sigma     = 0.01,
    nsw       = 0,
    ibrion    = -1,
    isym      = 0,
    lepsilon  = True,
    lpead     = True,
    kpar      = 1,
    ncore     = 1,
    nwrite    = 1,
    lwave     = False,
)

# ── sampling settings ─────────────────────────────────────────────────────────
NSAMPLES      = 30
MAX_FORCE_EVA = 5.0    # eV/Å  — drop poorly relaxed frames before DFPT
MAX_STRESS    = 0.05   # eV/Å³
INPUT_XYZ     = Path("relax.xyz")
WORKDIR       = Path("workdir")
# ─────────────────────────────────────────────────────────────────────────────


def write_inputs(nsamples: int, seed: int) -> None:
    frames = ase.io.read(str(INPUT_XYZ), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    print(f"loaded {len(frames)} structures from {INPUT_XYZ}")

    # Filter by stored MACE forces/stress before sampling
    ok = []
    for atoms in frames:
        f = atoms.arrays.get("mace_forces", atoms.get_positions() * 0)
        s = np.asarray(atoms.info.get("mace_stress", np.zeros(6)))
        if np.max(np.abs(f)) < MAX_FORCE_EVA and np.max(np.abs(s)) < MAX_STRESS:
            ok.append(atoms)
    print(f"  {len(ok)}/{len(frames)} passed force/stress filter")

    random.seed(seed)
    selected = random.sample(ok, k=min(nsamples, len(ok)))
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

        results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
        bec = results.get("born_effective_charges")   # (N, 3, 3)
        eps = results.get("dielectric_tensor")         # (3, 3)

        if bec is None or eps is None:
            print(f"  skip {vasprun.parent.name}: missing BEC or dielectric tensor")
            n_failed += 1
            continue

        bec = np.asarray(bec, dtype=np.float64)
        eps = np.asarray(eps, dtype=np.float64)

        # Build a clean atoms with only the FMD-required data
        out = ase.Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.cell,
            pbc=True,
        )
        # VASP/ASE bec[i, α, β] = ∂P_α/∂u_β (α=row=polarization, β=col=displacement).
        # FMD becex[i, α] = ∂F_α/∂E_x = ∂P_x/∂u_α (Maxwell), so slice the row α=x.
        out.arrays["becex"] = bec[:, 0, :]   # ∂P_x/∂u_β  (N, 3)
        out.arrays["becey"] = bec[:, 1, :]   # ∂P_y/∂u_β  (N, 3)
        out.arrays["becez"] = bec[:, 2, :]   # ∂P_z/∂u_β  (N, 3)
        # Dielectric tensor as Voigt-6 [xx, yy, zz, yz, xz, xy]
        out.info["dielectric_tensor"] = np.array([
            eps[0, 0], eps[1, 1], eps[2, 2],
            eps[1, 2], eps[0, 2], eps[0, 1],
        ], dtype=np.float64)

        frames.append(out)

    ase.io.write(str(output), frames, format="extxyz")
    print(f"wrote {len(frames)} frames → {output}  ({n_failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect",  action="store_true",
                        help="collect vasprun.xml outputs instead of writing inputs")
    parser.add_argument("--output",   type=Path, default=Path("dfpt.xyz"),
                        help="output xyz for --collect (default: dfpt.xyz)")
    parser.add_argument("--nsamples", type=int,  default=NSAMPLES)
    parser.add_argument("--seed",     type=int,  default=0)
    args = parser.parse_args()

    if args.collect:
        collect(args.output)
    else:
        write_inputs(args.nsamples, args.seed)
