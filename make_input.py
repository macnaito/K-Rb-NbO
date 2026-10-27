#!/usr/bin/env python3
"""(K,Rb)NbO3 cubic supercell generator for md.jl (stdlib only).

Writes an extxyz in the same layout as the bundled input.xyz:

- atoms ordered in species blocks: A site (K/Rb), Nb, O(0.5,0.5,0),
  O(0.5,0,0.5), O(0,0.5,0.5); inside each block the unit cells run
  ix -> iy -> iz with iz fastest
- groupid = unit-cell index (1 .. n^3, same ordering), so all 5 atoms of
  one perovskite cell share a groupid
- the A sites to be Rb are drawn without replacement from
  random.Random(seed); the same (n, rb, seed) always gives the same file

Usage:
    python3 make_input.py                         # 8^3, x=0.10 (51 Rb), seed 0 -> input.xyz
    python3 make_input.py --rb 0.2 --seed 3 -o x20.xyz
    python3 make_input.py --nrb 51                # exact Rb count instead of a fraction
"""
import argparse
import os
import random

# A-site species is decided per cell; the other four sites are fixed.
BASIS = [
    ("A",  0.0, 0.0, 0.0),
    ("Nb", 0.5, 0.5, 0.5),
    ("O",  0.5, 0.5, 0.0),
    ("O",  0.5, 0.0, 0.5),
    ("O",  0.0, 0.5, 0.5),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=8,
                   help="supercell size N (N^3 unit cells, default 8)")
    p.add_argument("-a", type=float, default=4.02297265625,
                   help="cubic lattice constant in Å (default 4.02297265625, "
                        "the value of the bundled input.xyz)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rb", type=float, default=0.10,
                   help="Rb fraction x on the A site (default 0.10); the count "
                        "is round(x * N^3)")
    g.add_argument("--nrb", type=int, default=None,
                   help="exact number of Rb atoms (overrides --rb)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument("-o", default="input.xyz", help="output file (default input.xyz)")
    p.add_argument("-f", "--force", action="store_true",
                   help="overwrite the output file if it exists")
    args = p.parse_args()

    if args.n < 1:
        p.error("-n must be >= 1")
    if args.a <= 0:
        p.error("-a must be > 0")
    ncell = args.n ** 3
    if args.nrb is not None:
        nrb = args.nrb
    else:
        if not 0.0 <= args.rb <= 1.0:
            p.error("--rb must lie in [0, 1]")
        nrb = round(args.rb * ncell)
    if not 0 <= nrb <= ncell:
        p.error(f"Rb count {nrb} is outside 0..{ncell}")
    if os.path.exists(args.o) and not args.force:
        p.error(f"{args.o} already exists (use -f to overwrite)")

    rb_cells = set(random.Random(args.seed).sample(range(ncell), nrb))

    n, a0 = args.n, args.a
    box = n * a0
    lines = []
    for sym, fx, fy, fz in BASIS:
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    icell = (ix * n + iy) * n + iz
                    s = ("Rb" if icell in rb_cells else "K") if sym == "A" else sym
                    x, y, z = (ix + fx) * a0, (iy + fy) * a0, (iz + fz) * a0
                    lines.append(f"{s:<2s}{x:17.8f}{y:17.8f}{z:17.8f}{icell + 1:9d}")

    with open(args.o, "w") as f:
        f.write(f"{len(lines)}\n")
        f.write(f'Lattice="{box:.8f} 0.0 0.0 0.0 {box:.8f} 0.0 0.0 0.0 {box:.8f}" '
                f'Properties=species:S:1:pos:R:3:groupid:I:1 pbc="T T T"\n')
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.o}: {len(lines)} atoms ({nrb} Rb / {ncell} A sites, "
          f"x = {nrb / ncell:.4f}), cubic {box:.4f} Å box, seed {args.seed}")


if __name__ == "__main__":
    main()
