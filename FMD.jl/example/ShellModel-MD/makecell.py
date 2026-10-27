#!/usr/bin/env python3
"""Cubic BaTiO3 supercell generator (pre-processing, stdlib only).

Writes an extxyz with the minimal columns FMD.jl needs: `species` and
`pos`. Everything else is auto-populated on read (velocities are redrawn
by npt! from params[:seed] whenever initial_temperature > 0, shell
displacements start at zero).

Usage:
    python3 makecell.py                 # 16^3 cells -> input.xyz (20,480 atoms)
    python3 makecell.py -n 4 -o small.xyz
"""
import argparse

# 5-atom cubic perovskite basis (fractional coordinates).
BASIS = [
    ("Ba", 0.0, 0.0, 0.0),
    ("Ti", 0.5, 0.5, 0.5),
    ("O",  0.5, 0.5, 0.0),
    ("O",  0.5, 0.0, 0.5),
    ("O",  0.0, 0.5, 0.5),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=16,
                   help="supercell size N (N^3 unit cells, default 16)")
    p.add_argument("-a", type=float, default=4.00,
                   help="cubic lattice constant in Å (default 4.00; the NPT "
                        "equilibration refines it, so precision is not needed)")
    p.add_argument("-o", default="input.xyz", help="output file (default input.xyz)")
    args = p.parse_args()
    if args.n < 1:
        p.error("-n must be >= 1")

    n, a0 = args.n, args.a
    box = n * a0
    lines = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                for sym, fx, fy, fz in BASIS:
                    x, y, z = (ix + fx) * a0, (iy + fy) * a0, (iz + fz) * a0
                    lines.append(f"{sym:2s} {x:20.12e} {y:20.12e} {z:20.12e}")

    with open(args.o, "w") as f:
        f.write(f"{len(lines)}\n")
        f.write(f'Lattice="{box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0 {box}" '
                f'Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.o}: {len(lines)} atoms, cubic {box:.2f} Å box")


if __name__ == "__main__":
    main()
