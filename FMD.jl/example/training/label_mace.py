#!/usr/bin/env python3
"""Stage 2 — MACE foundation model による単点ラベリング → FMD 学習データ

input_sample.jl が書いた out_T*/step_<n>.xyz を集め，MACE でエネルギー・力・
応力を評価し，FMD extxyz（potential_energy= / virial= / forces / method=）
1 ファイルにまとめる．glob は out_T*（サンプリング出力）に限定すること —
out_* だと input_md.jl の出力 out_md/ まで拾ってしまう．

既定モデルは r²SCAN 系の mace-matpes-r2scan-0．shell モデルのパラメータ
セット（r²SCAN-D4 フィット）と参照汎関数が揃うため，残差（= ACE が学習する
Δ）が小さく保たれる．

前提: pip install ase mace-torch  （GPU 不要．初回実行時にモデルを自動取得）

使い方:
    python3 label_mace.py 'out_T*/step_*.xyz' -o train.xyz --burnin 20
    # 別フレーバーで再ラベル（差分学習のベース/ターゲット生成にも使える）:
    python3 label_mace.py some.xyz -o newref.xyz --model medium-mpa-0 --method-tag MACE_MPA0
"""
import argparse
import glob
import re
import time
from pathlib import Path

import numpy as np
import ase.io


def collect_inputs(patterns, burnin):
    """glob 展開し，ディレクトリごとに step 番号順へ整列して先頭 burnin 個を捨てる．"""
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"label_mace.py: no files match {pat!r}")
        paths.extend(Path(p) for p in hits)
    paths = list(dict.fromkeys(paths))              # パターン重複による二重ラベルを防ぐ

    def stepnum(p):
        m = re.fullmatch(r"step_(\d+)\.xyz", p.name)
        return int(m.group(1)) if m else None

    bydir = {}
    for p in paths:
        bydir.setdefault(p.parent, []).append(p)

    kept = []
    for d, ps in sorted(bydir.items()):
        nums = [stepnum(p) for p in ps]
        if all(n is not None for n in nums):        # step_<n>.xyz 群 → 番号順 + burnin
            ps = [p for _, p in sorted(zip(nums, ps))][burnin:]
            if not ps:
                raise SystemExit(f"label_mace.py: --burnin {burnin} drops every frame in {d}")
        elif burnin > 0:
            # 番号付けできないファイルに burn-in は適用できない．黙って全フレームを
            # 通すと非平衡フレームが学習データに混ざるので，明示的に止める．
            bad = [p.name for p, n in zip(ps, nums) if n is None][:3]
            raise SystemExit(f"label_mace.py: --burnin needs step_<n>.xyz naming, but {d} "
                             f"has {bad} — label these files separately without --burnin")
        kept.extend(ps)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="入力 extxyz（glob 可，例 'out_*/step_*.xyz'）")
    p.add_argument("-o", "--output", default="train.xyz")
    p.add_argument("--model", default="mace-matpes-r2scan-0",
                   help="mace_mp のモデル名（default: mace-matpes-r2scan-0 = r²SCAN 系）")
    p.add_argument("--method-tag", default="MACE_R2SCAN",
                   help="method= タグ．fit 入力の method_weight のキーと一致させる")
    p.add_argument("--burnin", type=int, default=0,
                   help="ディレクトリごとに捨てる熱平衡化前スナップショット数")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    files = collect_inputs(args.inputs, args.burnin)
    print(f"labeling {len(files)} frames with {args.model} on {args.device} ...", flush=True)

    from mace.calculators import mace_mp
    # mace_mp は未知のモデル名を黙って既定モデルにフォールバックするので，
    # タイプミスが別フレーバーのラベルにならないよう先に名前を検証する．
    try:
        from mace.calculators.foundations_models import mace_mp_urls
        if args.model in mace_mp_urls or Path(args.model).is_file():
            pass
        else:
            raise SystemExit(f"label_mace.py: unknown model {args.model!r} — "
                             f"valid names: {', '.join(sorted(mace_mp_urls))}")
    except ImportError:
        print("warning: cannot validate model name against this mace version "
              "(mace_mp_urls not found) — a typo silently falls back to the default model")

    # float64: 学習ラベルはソルバノイズを避けるため倍精度で評価する．
    calc = mace_mp(model=args.model, device=args.device, default_dtype="float64")

    frames, t0 = [], time.time()
    for i, path in enumerate(files):
        src = ase.io.read(path)
        # species / pos / cell だけの素の Atoms に落とす（FMD の shell 列や速度は
        # 参照ラベルに不要．残すと readxyz が shell 変位を初期値として拾ってしまう）．
        atoms = ase.Atoms(src.get_chemical_symbols(), positions=src.get_positions(),
                          cell=src.cell, pbc=True)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=True)          # eV/Å³, 引張が正

        out = ase.Atoms(atoms.symbols, positions=atoms.positions, cell=atoms.cell, pbc=True)
        out.info["potential_energy"] = float(energy)
        # FMD の virial= 規約（eV, Voigt-6）: virial = −stress · V
        out.info["virial"] = (-np.asarray(stress) * atoms.get_volume()).astype(np.float64)
        out.info["method"] = args.method_tag
        out.arrays["forces"] = np.asarray(forces, dtype=np.float64)
        frames.append(out)

        if (i + 1) % 20 == 0 or i + 1 == len(files):
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(files)}  ({rate:.2f} frames/s)", flush=True)

    ase.io.write(args.output, frames, format="extxyz")
    print(f"wrote {args.output}: {len(frames)} frames, method={args.method_tag}")


if __name__ == "__main__":
    main()
