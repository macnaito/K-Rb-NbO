export UNITS, ATOMIC_NUMBERS

# 1 a.u. of time (ℏ/Hartree) in seconds — single source for :second and :fs.
const _AU_TIME_S = 1.0545718176462e-34/4.3597447222071e-18

"""
    UNITS :: NamedTuple

Conversion factors from one atomic unit (Hartree, Bohr, electron mass, …)
into the named SI / chemistry unit. Every numeric quantity inside FMD lives
in atomic units; divide by a `UNITS[...]` factor to convert *into* atomic
units when assembling input, multiply when reporting back.

# Keys

| Key | Meaning |
|---|---|
| `eV`     | Hartree / eV (= 27.211…)            |
| `Å`      | Bohr / Å (= 0.5292…)                |
| `dalton` | unified atomic mass unit / electron mass |
| `Kelvin` | Hartree / kB (≈ 3.158e5)            |
| `Dirac`  | reduced Planck constant ℏ in J·s    |
| `Hartree`| Hartree in J                        |
| `second` | atomic time unit (ℏ/Hartree) in s   |
| `fs`     | atomic time unit (ℏ/Hartree) in fs (≈ 0.02419) |

Examples:

```julia
rc       = 9.0 / UNITS[:Å]      # 9 Å in Bohr
T_au     = 300.0 / UNITS[:Kelvin]  # 300 K in Hartree
E_in_eV  = E_au * UNITS[:eV]
timestep = 1.0 / UNITS[:fs]     # 1 fs in a.u. (≈ 41.341)
```
"""
const UNITS = (;
    eV = 27.21138625,
    Å = 0.529177210903,
    dalton = 1.66053906660e-27/9.10938356e-31,
    Kelvin = 1/(8.617333262145e-5/27.21138625),
    Dirac = 1.0545718176462e-34,  # Js
    Hartree = 4.3597447222071e-18, # J
    second = _AU_TIME_S,
    fs = _AU_TIME_S*1e15,
    # One a.u. of pressure (Hartree/Bohr³) in GPa. Stress params are raw
    # a.u.; a compressibility from a bulk modulus B in GPa reads
    # κ_T = UNITS[:GPa] / B_GPa (Issue #140).
    GPa = 4.3597447222071e-18 / (0.529177210903e-10)^3 * 1e-9
)

"""
    ATOMIC_MASSES :: Dict{String,Float64}

Element-symbol → atomic mass in atomic units (mass-of-electron). Internally
`amu × UNITS[:dalton]`. Used by [`readxyz`](@ref) to populate
`atoms.arrays[:masses]` from the file's species column.
"""
const ATOMIC_MASSES = Dict{String, Float64}(
    # --- 第1周期 ---
    "H"  => 1.008*UNITS[:dalton],    # 水素
    "He" => 4.0026*UNITS[:dalton],   # ヘリウム

    # --- 第2周期 ---
    "Li" => 6.94*UNITS[:dalton],     # リチウム
    "Be" => 9.0122*UNITS[:dalton],   # ベリリウム
    "B"  => 10.81*UNITS[:dalton],    # ホウ素
    "C"  => 12.011*UNITS[:dalton],   # 炭素
    "N"  => 14.007*UNITS[:dalton],   # 窒素
    "O"  => 15.999*UNITS[:dalton],   # 酸素
    "F"  => 18.998*UNITS[:dalton],   # フッ素
    "Ne" => 20.180*UNITS[:dalton],   # ネオン

    # --- 第3周期 ---
    "Na" => 22.990*UNITS[:dalton],   # ナトリウム
    "Mg" => 24.305*UNITS[:dalton],   # マグネシウム
    "Al" => 26.982*UNITS[:dalton],   # アルミニウム
    "Si" => 28.085*UNITS[:dalton],   # ケイ素
    "P"  => 30.974*UNITS[:dalton],   # リン
    "S"  => 32.06*UNITS[:dalton],    # 硫黄
    "Cl" => 35.45*UNITS[:dalton],    # 塩素
    "Ar" => 39.948*UNITS[:dalton],   # アルゴン

    # --- 第4周期 ---
    "K"  => 39.098*UNITS[:dalton],   # カリウム
    "Ca" => 40.078*UNITS[:dalton],   # カルシウム
    "Sc" => 44.956*UNITS[:dalton],   # スカンジウム
    "Ti" => 47.867*UNITS[:dalton],   # チタン
    "V"  => 50.942*UNITS[:dalton],   # バナジウム
    "Cr" => 51.996*UNITS[:dalton],   # クロム
    "Mn" => 54.938*UNITS[:dalton],   # マンガン
    "Fe" => 55.845*UNITS[:dalton],   # 鉄
    "Co" => 58.933*UNITS[:dalton],   # コバルト
    "Ni" => 58.693*UNITS[:dalton],   # ニッケル
    "Cu" => 63.546*UNITS[:dalton],   # 銅
    "Zn" => 65.38*UNITS[:dalton],    # 亜鉛
    "Ga" => 69.723*UNITS[:dalton],   # ガリウム
    "Ge" => 72.630*UNITS[:dalton],   # ゲルマニウム
    "As" => 74.922*UNITS[:dalton],   # ヒ素
    "Se" => 78.971*UNITS[:dalton],   # セレン
    "Br" => 79.904*UNITS[:dalton],   # 臭素
    "Kr" => 83.798*UNITS[:dalton],   # クリプトン

    # --- 第5周期 ---
    "Rb" => 85.468*UNITS[:dalton],   # ルビジウム
    "Sr" => 87.62*UNITS[:dalton],    # ストロンチウム
    "Y"  => 88.906*UNITS[:dalton],   # イットリウム
    "Zr" => 91.224*UNITS[:dalton],   # ジルコニウム
    "Nb" => 92.906*UNITS[:dalton],   # ニオブ
    "Mo" => 95.95*UNITS[:dalton],    # モリブデン
    "Tc" => 98.0*UNITS[:dalton],     # テクネチウム (放射性: 98Tc)
    "Ru" => 101.07*UNITS[:dalton],   # ルテニウム
    "Rh" => 102.91*UNITS[:dalton],   # ロジウム
    "Pd" => 106.42*UNITS[:dalton],   # パラジウム
    "Ag" => 107.87*UNITS[:dalton],   # 銀
    "Cd" => 112.41*UNITS[:dalton],   # カドミウム
    "In" => 114.82*UNITS[:dalton],   # インジウム
    "Sn" => 118.71*UNITS[:dalton],   # スズ
    "Sb" => 121.76*UNITS[:dalton],   # アンチモン
    "Te" => 127.60*UNITS[:dalton],   # テルル
    "I"  => 126.904*UNITS[:dalton],  # ヨウ素
    "Xe" => 131.29*UNITS[:dalton],   # キセノン

    # --- 第6周期 (ランタノイド含む) ---
    "Cs" => 132.905*UNITS[:dalton],  # セシウム
    "Ba" => 137.327*UNITS[:dalton],  # バリウム
    "La" => 138.905*UNITS[:dalton],  # ランタン
    "Ce" => 140.116*UNITS[:dalton],  # セリウム
    "Pr" => 140.908*UNITS[:dalton],  # プラセオジム
    "Nd" => 144.242*UNITS[:dalton],  # ネオジム
    "Pm" => 145.0*UNITS[:dalton],    # プロメチウム (放射性: 145Pm)
    "Sm" => 150.36*UNITS[:dalton],   # サマリウム
    "Eu" => 151.964*UNITS[:dalton],  # ユウロピウム
    "Gd" => 157.25*UNITS[:dalton],   # ガドリニウム
    "Tb" => 158.925*UNITS[:dalton],  # テルビウム
    "Dy" => 162.500*UNITS[:dalton],  # ジスプロシウム
    "Ho" => 164.930*UNITS[:dalton],  # ホルミウム
    "Er" => 167.259*UNITS[:dalton],  # エルビウム
    "Tm" => 168.934*UNITS[:dalton],  # ツリウム
    "Yb" => 173.045*UNITS[:dalton],  # イッテルビウム
    "Lu" => 174.967*UNITS[:dalton],  # ルテチウム
    "Hf" => 178.49*UNITS[:dalton],   # ハフニウム
    "Ta" => 180.948*UNITS[:dalton],  # タンタル
    "W"  => 183.84*UNITS[:dalton],   # タングステン
    "Re" => 186.207*UNITS[:dalton],  # レニウム
    "Os" => 190.23*UNITS[:dalton],   # オスミウム
    "Ir" => 192.217*UNITS[:dalton],  # イリジウム
    "Pt" => 195.084*UNITS[:dalton],  # 白金
    "Au" => 196.967*UNITS[:dalton],  # 金
    "Hg" => 200.592*UNITS[:dalton],  # 水銀
    "Tl" => 204.38*UNITS[:dalton],   # タリウム
    "Pb" => 207.2*UNITS[:dalton],    # 鉛
    "Bi" => 208.980*UNITS[:dalton],  # ビスマス
    "Po" => 209.0*UNITS[:dalton],    # ポロニウム (放射性)
    "At" => 210.0*UNITS[:dalton],    # アスタチン (放射性)
    "Rn" => 222.0*UNITS[:dalton],    # ラドン (放射性)

    # --- 第7周期 (アクチノイド含む) ---
    "Fr" => 223.0*UNITS[:dalton],    # フランシウム (放射性)
    "Ra" => 226.0*UNITS[:dalton],    # ラジウム (放射性)
    "Ac" => 227.0*UNITS[:dalton],    # アクチニウム (放射性)
    "Th" => 232.038*UNITS[:dalton],  # トリウム
    "Pa" => 231.036*UNITS[:dalton],  # プロトアクチニウム
    "U"  => 238.029*UNITS[:dalton],  # ウラン
    "Np" => 237.0*UNITS[:dalton],    # ネプツニウム (放射性)
    "Pu" => 244.0*UNITS[:dalton],    # プルトニウム (放射性)
    "Am" => 243.0*UNITS[:dalton],    # アメリシウム (放射性)
    "Cm" => 247.0*UNITS[:dalton],    # キュリウム (放射性)
    "Bk" => 247.0*UNITS[:dalton],    # バークリウム (放射性)
    "Cf" => 251.0*UNITS[:dalton],    # カリホルニウム (放射性)
    "Es" => 252.0*UNITS[:dalton],    # アインスタイニウム (放射性)
    "Fm" => 257.0*UNITS[:dalton],    # フェルミウム (放射性)
    "Md" => 258.0*UNITS[:dalton],    # メンデレビウム (放射性)
    "No" => 259.0*UNITS[:dalton],    # ノーベリウム (放射性)
    "Lr" => 262.0*UNITS[:dalton],    # ローレンシウム (放射性)
    "Rf" => 267.0*UNITS[:dalton],    # ラザホージウム (放射性)
    "Db" => 268.0*UNITS[:dalton],    # ドブニウム (放射性)
    "Sg" => 269.0*UNITS[:dalton],    # シーボーギウム (放射性)
    "Bh" => 270.0*UNITS[:dalton],    # ボーリウム (放射性)
    "Hs" => 269.0*UNITS[:dalton],    # ハッシウム (放射性)
    "Mt" => 278.0*UNITS[:dalton],    # マイトネリウム (放射性)
    "Ds" => 281.0*UNITS[:dalton],    # ダームスタチウム (放射性)
    "Rg" => 282.0*UNITS[:dalton],    # レントゲニウム (放射性)
    "Cn" => 285.0*UNITS[:dalton],    # コペルニシウム (放射性)
    "Nh" => 286.0*UNITS[:dalton],    # ニホニウム (放射性)
    "Fl" => 289.0*UNITS[:dalton],    # フレロビウム (放射性)
    "Mc" => 290.0*UNITS[:dalton],    # モスコビウム (放射性)
    "Lv" => 293.0*UNITS[:dalton],    # リバモリウム (放射性)
    "Ts" => 294.0*UNITS[:dalton],    # テネシン (放射性)
    "Og" => 294.0*UNITS[:dalton],    # オガネソン (放射性)
)

"""
    ATOMIC_NUMBERS :: Dict{String,Int}

Element-symbol → atomic number Z. Same key set as [`ATOMIC_MASSES`](@ref).

Screened-nuclear repulsion terms are parametrised by the nuclear charge of
the pair rather than by species id, so they need this table. There is no
in-tree consumer at the moment — the ZBL short-range wall pin that used to
be one was removed in Issue #160, and the NLH baseline of Issue #124 will
be the next.
"""
const ATOMIC_NUMBERS = Dict{String, Int}(
    # --- 第1周期 ---
    "H"  => 1,   "He" => 2,
    # --- 第2周期 ---
    "Li" => 3,   "Be" => 4,   "B"  => 5,   "C"  => 6,   "N"  => 7,
    "O"  => 8,   "F"  => 9,   "Ne" => 10,
    # --- 第3周期 ---
    "Na" => 11,  "Mg" => 12,  "Al" => 13,  "Si" => 14,  "P"  => 15,
    "S"  => 16,  "Cl" => 17,  "Ar" => 18,
    # --- 第4周期 ---
    "K"  => 19,  "Ca" => 20,  "Sc" => 21,  "Ti" => 22,  "V"  => 23,
    "Cr" => 24,  "Mn" => 25,  "Fe" => 26,  "Co" => 27,  "Ni" => 28,
    "Cu" => 29,  "Zn" => 30,  "Ga" => 31,  "Ge" => 32,  "As" => 33,
    "Se" => 34,  "Br" => 35,  "Kr" => 36,
    # --- 第5周期 ---
    "Rb" => 37,  "Sr" => 38,  "Y"  => 39,  "Zr" => 40,  "Nb" => 41,
    "Mo" => 42,  "Tc" => 43,  "Ru" => 44,  "Rh" => 45,  "Pd" => 46,
    "Ag" => 47,  "Cd" => 48,  "In" => 49,  "Sn" => 50,  "Sb" => 51,
    "Te" => 52,  "I"  => 53,  "Xe" => 54,
    # --- 第6周期 (ランタノイド含む) ---
    "Cs" => 55,  "Ba" => 56,  "La" => 57,  "Ce" => 58,  "Pr" => 59,
    "Nd" => 60,  "Pm" => 61,  "Sm" => 62,  "Eu" => 63,  "Gd" => 64,
    "Tb" => 65,  "Dy" => 66,  "Ho" => 67,  "Er" => 68,  "Tm" => 69,
    "Yb" => 70,  "Lu" => 71,  "Hf" => 72,  "Ta" => 73,  "W"  => 74,
    "Re" => 75,  "Os" => 76,  "Ir" => 77,  "Pt" => 78,  "Au" => 79,
    "Hg" => 80,  "Tl" => 81,  "Pb" => 82,  "Bi" => 83,  "Po" => 84,
    "At" => 85,  "Rn" => 86,
    # --- 第7周期 (アクチノイド含む) ---
    "Fr" => 87,  "Ra" => 88,  "Ac" => 89,  "Th" => 90,  "Pa" => 91,
    "U"  => 92,  "Np" => 93,  "Pu" => 94,  "Am" => 95,  "Cm" => 96,
    "Bk" => 97,  "Cf" => 98,  "Es" => 99,  "Fm" => 100, "Md" => 101,
    "No" => 102, "Lr" => 103, "Rf" => 104, "Db" => 105, "Sg" => 106,
    "Bh" => 107, "Hs" => 108, "Mt" => 109, "Ds" => 110, "Rg" => 111,
    "Cn" => 112, "Nh" => 113, "Fl" => 114, "Mc" => 115, "Lv" => 116,
    "Ts" => 117, "Og" => 118,
)
