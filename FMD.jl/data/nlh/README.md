# NLH 遮蔽核斥力ポテンシャル 係数テーブル

`nlh_coeffs.dat` は Nordlund–Lehtola–Hobler (NLH) 反発ポテンシャルの pair-specific
係数表。`src/Calculators/NLH.jl` の `NLHParam` がロードする。

## 出典

K. Nordlund, S. Lehtola, G. Hobler,
*Repulsive interatomic potentials calculated at three levels of theory*,
Phys. Rev. A **111**, 032818 (2025).

データ: doi:10.5281/zenodo.14172633 —
"Data sets for the publication *Repulsive interatomic potentials calculated at
three levels of theory*"、アーカイブ `nlh_potentials_opendata.tar.gz` 内の
`nlh/nlh_coeffs.dat` を**改変なし**で同梱。

- sha256: `621e7a44168107c67baba87079986a851006f82a36198727c1850c81bc11d516`
- 320,935 bytes / 4,279 行（ヘッダ 1 + 4,278 ペア）

ライセンス: **CC-BY 4.0**（原データの条件をそのまま継承）。

## 形式

```
# Z1 Z2     a1        b1       a2       b2       a3       b3 error(V>30) error(V>10)
 1  1 -8.99999   9.55658  9.99999  8.89086  0.00000  0.00000  1.32%  3.14%
```

- 1 行 1 ペア、`Z1 ≤ Z2`（対称なので片側のみ収録）。`Z ≤ 92` の全 92·93/2 = 4278 組
- ポテンシャル: `V(r) = (1/4πε₀)·Z₁Z₂e²/r · φ(r)`,  `φ(r) = Σᵢ aᵢ exp(−bᵢ r)`
- 単位: `r` は Å、`bᵢ` は Å⁻¹、`aᵢ` は無次元、`V` は eV。
  原子単位へは `b_au = b_Å · UNITS[:Å]`（`V` は `Z₁Z₂/r_Bohr · φ` で Hartree）
- `Σᵢ aᵢ = 1`（`r → 0` で遮蔽なしの Coulomb 極限 `Z₁Z₂/r` に漸近）。
  全 4278 ペアで厳密に成立する — `NLHParam` はこれを整合性チェックに使う
- 最後の 2 列は原典の DFT データに対する fit 誤差（`V > 30 eV` / `V > 10 eV` の領域）。
  平均 4.8%、最大 10.0%。**fit は反発の強い短距離領域に対してのみ検証されている**

## 注意: 減衰しない遮蔽関数を持つ 24 ペア

`aᵢ ≠ 0` かつ `bᵢ = 0` の項を持つペアが 24 件ある（H–He, O–Na, Mg–Mg など）。
この項は `r → ∞` で消えないので `V(r) → aᵢ·Z₁Z₂/r`、すなわち**遮蔽されない Coulomb
裾**になる（O–Na では 5 Å で 1.27 eV、10 Å でも 0.63 eV）。原典の fit が上記のとおり
短距離のみを対象としている以上これは外挿の産物であり、MD のベースライン項としては
使えない。

`NLHParam` のローダはこれらのペアを **error** で弾く（黙って使わない）。
