"""Fine-tune MACE-MP on DFT single-point data and build a combined dataset.

Usage:
    # 1. fine-tune
    python finetune_mace.py [--train dft.xyz] [--device cuda]

    # 2. re-evaluate MD/relax frames with the fine-tuned model and merge with DFT
    python finetune_mace.py --collect [--output synth.xyz] [--device cuda]

Output of --collect: synth.xyz in FMD extxyz format
  - DFT frames:  method=PBEsol
  - MACE frames: method=MACE
  Both carry potential_energy=, virial= (9-float column-major), forces.

Settings mirror workflow.toml [stage.finetune_mace].
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import ase.io

# ── settings ─────────────────────────────────────────────────────────────────
MODEL          = "medium-omat-0"
NAME           = "ft"
VALID_FRACTION = 0.05
MAX_EPOCHS     = 50
BATCH_SIZE     = 2
LR             = 0.01
DTYPE          = "float32"
DEVICE         = "cuda"
SEED           = 42

# LoRA: scaling = alpha/rank (= 0.125 here — conservative, preserves base
# model behavior and reduces overfitting risk on small fine-tuning datasets)
LORA_RANK  = 8
LORA_ALPHA = 1.0

ENERGY_KEY = "energy"
FORCES_KEY = "forces"
STRESS_KEY = "stress"
# ─────────────────────────────────────────────────────────────────────────────


# ── FMD-compatible labelling ──────────────────────────────────────────────────

def _label(atoms, *, potential_energy, forces, stress_voigt6, method):
    """Return a copy of atoms with FMD-compatible info/arrays set.

    FMD extxyz reader accepts:
      potential_energy=  (scalar, eV)
      virial=            (Voigt-6 array, eV; written by ASE as "v1 v2 ... v6")
      method=            (string tag for per-method loss weighting)
    ASE's extxyz writer emits all atoms.info keys automatically.
    """
    atoms = atoms.copy()
    volume = atoms.get_volume()
    atoms.info["potential_energy"] = float(potential_energy)
    atoms.info["virial"] = (-np.asarray(stress_voigt6) * volume).astype(np.float64)
    atoms.info["method"] = method
    atoms.arrays["forces"] = np.asarray(forces, dtype=np.float64)
    for key in ("energy", "stress", "mace_energy", "mace_stress", "relax_steps", "converged"):
        atoms.info.pop(key, None)
        atoms.arrays.pop(key, None)
    return atoms

# ─────────────────────────────────────────────────────────────────────────────


def build_argv(train_file: Path, device: str) -> list[str]:
    return [
        "mace_run_train",
        "--name",               NAME,
        "--train_file",         str(train_file),
        "--foundation_model",   MODEL,
        "--energy_key",         ENERGY_KEY,
        "--forces_key",         FORCES_KEY,
        "--stress_key",         STRESS_KEY,
        "--E0s",                "foundation",
        "--valid_fraction",     str(VALID_FRACTION),
        "--max_num_epochs",     str(MAX_EPOCHS),
        "--batch_size",         str(BATCH_SIZE),
        "--lr",                 str(LR),
        "--default_dtype",      DTYPE,
        "--device",             device,
        "--seed",               str(SEED),
        # LoRA: replaces --multiheads_finetuning / --pt_train_file mp
        "--lora",               "True",
        "--lora_rank",          str(LORA_RANK),
        "--lora_alpha",         str(LORA_ALPHA),
        # explicit stress-aware loss (vs default "weighted" which is E+F only)
        "--loss",               "stress",
        "--ema",
        "--ema_decay",          "0.99",
    ]


def finetune(train_file: Path, device: str) -> None:
    if not train_file.exists():
        raise FileNotFoundError(f"train file not found: {train_file}")

    argv = build_argv(train_file, device)
    print("command:")
    print(" ", " ".join(argv), "\n")

    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    compiled = Path(f"{NAME}_compiled.model")
    if compiled.exists():
        print(f"\nfine-tune complete → {compiled}")
    else:
        print(f"\nwarning: {compiled} not found — check mace_run_train output")


def collect(model_path: Path, output: Path, device: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"compiled model not found: {model_path}")

    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype=DTYPE,
    )

    all_frames = []

    # ── 1. re-evaluate MD trajectories + relaxed structures ──────────────────
    for src in sorted(Path(".").glob("mdtraj_seed*.xyz")) + [Path("relax.xyz")]:
        if not src.exists():
            continue
        frames = ase.io.read(str(src), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        for atoms in frames:
            atoms.pbc = True
            atoms.calc = calc
            all_frames.append(_label(
                atoms,
                potential_energy = atoms.get_potential_energy(),
                forces           = atoms.get_forces(),
                stress_voigt6    = atoms.get_stress(),
                method           = "MACE",
            ))
        print(f"  {src.name}: {len(frames)} frames re-evaluated")

    n_mace = len(all_frames)

    # ── 2. load and convert DFT frames ───────────────────────────────────────
    dft_path = Path("dft.xyz")
    if dft_path.exists():
        frames = ase.io.read(str(dft_path), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        for atoms in frames:
            # Read via calculator interface: ASE 3.28+ attaches extxyz energy/
            # forces/stress to a SinglePointCalculator, not to info/arrays.
            all_frames.append(_label(
                atoms,
                potential_energy = atoms.get_potential_energy(),
                forces           = atoms.get_forces(),
                stress_voigt6    = atoms.get_stress(),
                method           = "PBEsol",
            ))
        n_dft = len(frames)
        print(f"  {dft_path.name}: {n_dft} DFT frames loaded")
    else:
        n_dft = 0
        print(f"warning: {dft_path} not found — DFT frames skipped")

    ase.io.write(str(output), all_frames, format="extxyz")
    print(f"\nwrote {n_dft} DFT (PBEsol) + {n_mace} MACE = {len(all_frames)} frames → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="re-evaluate MACE frames and merge with DFT into synth.xyz")
    parser.add_argument("--output",  type=Path,  default=Path("synth.xyz"),
                        help="output for --collect (default: synth.xyz)")
    parser.add_argument("--model",   type=Path,  default=Path(f"{NAME}_compiled.model"),
                        help="compiled model for --collect")
    parser.add_argument("--train",   type=Path,  default=Path("dft.xyz"))
    parser.add_argument("--device",  type=str,   default=DEVICE)
    args = parser.parse_args()

    if args.collect:
        collect(args.model, args.output, args.device)
    else:
        finetune(args.train, args.device)
