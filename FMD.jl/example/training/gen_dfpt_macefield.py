#!/usr/bin/env python3
"""[optional] MACE-Field で BEC + 誘電テンソルを生成 → dfpt.xyz

shell モデルのフィット（input_fitsm.jl）に使う DFPT 相当データを，VASP なしで
MACE-Field（誘電応答を予測する MACE 拡張, github.com/mdi-group/mace-field）
から作る．

精度の注意: BaTiO₃ では MACE-Field の Z*/ε は DFPT 比 ±10% 程度．手元で
ワークフロー全体を試すには十分だが，論文品質の shell フィットには VASP DFPT
（dataset/dfpt/kno が生成パイプラインの雛形）を使うこと．

前提: pip install ase, mace-field（上記リポジトリ）,
      モデルファイル（README の入手先参照）

使い方:
    python3 gen_dfpt_macefield.py 'out_T400_P0/step_*.xyz' \
        --model MACEField-MH-0-omat-dielectric.model --nsamples 20 -o dfpt.xyz
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import ase.io


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="入力 extxyz（glob 可）")
    p.add_argument("--model", required=True, help="MACE-Field モデルファイル (.model)")
    p.add_argument("--head", default="mp-dielectric",
                   help="マルチヘッドモデルの head 名 (default: mp-dielectric)")
    p.add_argument("--nsamples", type=int, default=20,
                   help="等間隔サンプルするフレーム数 (default: 20)")
    p.add_argument("-o", "--output", default="dfpt.xyz")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    paths = sorted(set(q for pat in args.inputs for q in glob.glob(pat)))
    if not paths:
        raise SystemExit(f"gen_dfpt_macefield.py: no files match {args.inputs!r}")

    # step_<n>.xyz 群は番号順（時間順）に並べ替える．辞書順のままだと
    # step_1000 < step_50 になり，等間隔サンプルが時間の等間隔にならない．
    def stepnum(p):
        m = re.fullmatch(r"step_(\d+)\.xyz", Path(p).name)
        return int(m.group(1)) if m else None
    nums = [stepnum(p) for p in paths]
    if all(n is not None for n in nums):
        paths = [p for _, p in sorted(zip(nums, paths))]

    idx = np.unique(np.linspace(0, len(paths) - 1, args.nsamples).astype(int))
    paths = [paths[i] for i in idx]

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype="float64", model_type="MACEField",
                          head=args.head, electric_field=[0.0, 0.0, 0.0])

    frames = []
    for path in paths:
        src = ase.io.read(path)
        atoms = ase.Atoms(src.get_chemical_symbols(), positions=src.get_positions(),
                          cell=src.cell, pbc=True)
        atoms.calc = calc
        atoms.get_potential_energy()
        n = len(atoms)
        # bec[i, α, β] = ∂P_α/∂u_β（α = 分極成分の行）．
        bec = np.asarray(calc.results["becs"], dtype=np.float64).reshape(n, 3, 3)
        eps = np.eye(3) + np.asarray(calc.results["polarizability"],
                                     dtype=np.float64).reshape(3, 3)

        out = ase.Atoms(atoms.symbols, positions=atoms.positions, cell=atoms.cell, pbc=True)
        # FMD becex[i, α] = ∂F_α/∂E_x = ∂P_x/∂u_α（Maxwell 関係）なので
        # 分極成分の行 α = x を切り出す（VASP 収集側 dataset/dfpt と同じ転置規約）．
        out.arrays["becex"] = bec[:, 0, :]
        out.arrays["becey"] = bec[:, 1, :]
        out.arrays["becez"] = bec[:, 2, :]
        # 誘電テンソルは Voigt-6 [xx, yy, zz, yz, xz, xy]
        out.info["dielectric_tensor"] = np.array([
            eps[0, 0], eps[1, 1], eps[2, 2],
            eps[1, 2], eps[0, 2], eps[0, 1],
        ], dtype=np.float64)
        frames.append(out)

    ase.io.write(args.output, frames, format="extxyz")
    print(f"wrote {args.output}: {len(frames)} frames")

    # 妥当性の目視チェック．BaTiO₃ の DFPT 目安（立方晶）:
    #   Z*(Ba) ≈ +2.7, Z*(Ti) ≈ +7.2, Z*(O∥) ≈ −5.7 / Z*(O⊥) ≈ −2.1（3 サイト平均 −3.3）
    #   音響サムルール Σ_i Z*_i ≈ 0，ε∞ ≈ 5–6
    # ASR と ε 対角は転置・単位規約（4π / 体積）の取り違えをその場で検出する．
    syms = np.array(frames[0].get_chemical_symbols())
    Z = np.stack([np.stack([f.arrays["becex"], f.arrays["becey"], f.arrays["becez"]],
                           axis=1) for f in frames])        # (nframe, N, 3, 3)
    ztr = np.einsum("fnaa->fn", Z).mean(axis=0) / 3.0        # 対角平均 (N,)
    for s in dict.fromkeys(syms):
        print(f"  <Z*({s})> = {ztr[syms == s].mean():+.2f}")
    asr = np.abs(Z.sum(axis=1)).max()
    print(f"  acoustic sum rule: max|Σ_i Z*_i| = {asr:.3f} (≈ 0 expected)")
    eps_d = np.mean([f.info['dielectric_tensor'][:3].mean() for f in frames])
    print(f"  <ε∞ diag> = {eps_d:.2f} (BaTiO3 DFPT ≈ 5–6 expected)")


if __name__ == "__main__":
    main()
