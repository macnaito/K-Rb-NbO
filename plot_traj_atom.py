#!/usr/bin/env python3
"""毎ステップ出力した xyz から，元素ごとに 1 原子を選んでコア変位と shelldisp の時間変化を図にする．

    python3 plot_traj_atom.py [outdir] [K_id Nb_id O_id]
        outdir 既定: output_result/out_nve_0.5fs_traj
    → <outdir>/traj_atom.html

t=0 は入力 (step_39000.xyz)．原子は id 列で追跡する (CellList の並べ替えに影響されない)．
単位: xyz の pos・shelldisp は Å．
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output_result/out_nve_0.5fs_traj")
INPUT = Path("output_result/out-150K_0Gpa_relax/step_39000.xyz")
DT_FS = 0.5


def read_frame(path, want):
    """want: 集合 of id → {id: (species, pos[3], shelldisp[3])}, lattice 対角."""
    with open(path) as f:
        n = int(f.readline())
        hdr = f.readline()
        lat = [float(x) for x in re.search(r'Lattice="([^"]+)"', hdr).group(1).split()]
        props = re.search(r"Properties=(\S+)", hdr).group(1).split(":")
        # 列の位置を Properties から決める
        cols, c = {}, 0
        for name, _typ, ncol in zip(props[0::3], props[1::3], props[2::3]):
            cols[name] = (c, int(ncol))
            c += int(ncol)
        res = {}
        for _ in range(n):
            v = f.readline().split()
            i = int(v[cols["id"][0]])
            if want is None or i in want:
                p0, s0 = cols["pos"][0], cols["shelldisp"][0]
                res[i] = (v[cols["species"][0]], [float(x) for x in v[p0:p0 + 3]],
                          [float(x) for x in v[s0:s0 + 3]])
    return res, [lat[0], lat[4], lat[8]]


# 追跡する原子: 引数指定がなければ元素ごとに最小 id
first, L = read_frame(INPUT, None)
if len(sys.argv) > 4:
    pick = [int(x) for x in sys.argv[2:5]]
else:
    pick = [min(i for i, (sp, _, _) in first.items() if sp == el) for el in ("K", "Nb", "O")]
want = set(pick)

frames = sorted(OUT.glob("step_*.xyz"), key=lambda p: int(p.stem.split("_")[1]))
traj = {i: {"t": [0.0], "pos": [first[i][1]], "sd": [first[i][2]]} for i in pick}
for p in frames:
    step = int(p.stem.split("_")[1])
    fr, _ = read_frame(p, want)
    for i in pick:
        traj[i]["t"].append(step * DT_FS)
        traj[i]["pos"].append(fr[i][1])
        traj[i]["sd"].append(fr[i][2])

atoms = []
for i in pick:
    tr = traj[i]
    p0, d0 = tr["pos"][0], tr["sd"][0]
    u = [[(p[a] - p0[a]) - round((p[a] - p0[a]) / L[a]) * L[a] for a in range(3)] for p in tr["pos"]]
    dd = [[d[a] - d0[a] for a in range(3)] for d in tr["sd"]]
    # 平均まわりの揺らぎ振幅 (rms) と 1 ステップの平均移動量
    def rms(x):
        m = sum(x) / len(x)
        return (sum((y - m) ** 2 for y in x) / len(x)) ** 0.5
    def corr(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        c = sum((p - mx) * (q - my) for p, q in zip(x, y))
        return c / (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    def stepmean(seq):
        s = [sum((seq[k + 1][a] - seq[k][a]) ** 2 for a in range(3)) ** 0.5 for k in range(len(seq) - 1)]
        return sum(s) / len(s)
    info = {"id": i, "sp": first[i][0], "d0": d0,
            "u_rms": [rms([x[a] for x in u]) for a in range(3)],
            "dd_rms": [rms([x[a] for x in dd]) for a in range(3)],
            "u_step": stepmean(u), "dd_step": stepmean(dd),
            "corr": [corr([x[a] for x in u], [x[a] for x in dd]) for a in range(3)]}
    atoms.append({"info": info, "t": tr["t"],
                  "u": [[round(x[a], 6) for x in u] for a in range(3)],
                  "dd": [[round(x[a], 7) for x in dd] for a in range(3)],
                  "d": [[round(x[a], 6) for x in tr["sd"]] for a in range(3)]})
    print(f"{first[i][0]} id={i}: frames={len(tr['t'])} |Δr|/step={info['u_step']:.2e} Å  "
          f"|Δd|/step={info['dd_step']:.2e} Å  u_rms={['%.3f' % x for x in info['u_rms']]}  "
          f"Δd_rms={['%.4f' % x for x in info['dd_rms']]}")

html = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>1 原子の追跡</title>
<style>
:root{--bg:#fbfbfa;--card:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--grid:rgba(0,0,0,.06);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--grid:rgba(255,255,255,.08);
--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
:root[data-theme="dark"]{--bg:#141413;--card:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--grid:rgba(255,255,255,.08);
--s1:#3987e5;--s2:#d95926;--s3:#199e70}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 4px;font-weight:600}
.sub{color:var(--muted);margin:0 0 14px;font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.bar button{font:inherit;font-size:13px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.bar button[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.card h2{font-size:15px;margin:0 0 2px;font-weight:600}
.card p{margin:0 0 10px;color:var(--muted);font-size:13px}
.legend{display:flex;gap:16px;font-size:13px;margin:0 0 8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.wrap{position:relative;height:300px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:12px}
</style>
</head>
<body>
<main>
<h1>1 原子のコア変位とシェル変位の追跡</h1>
<p class="sub">NVE・dt=0.5 fs・cgtol=1e-6・外挿次数 2・0.3 ps（毎ステップ出力）．t=0 は step_39000（150 K 平衡）．</p>
<div class="bar" id="pick"></div>
<div class="stats" id="stats"></div>

<div class="card">
<h2>コア変位 u(t) = r(t) − r(0)</h2>
<p>コア（原子核側）の位置の時間変化．</p>
<div class="legend" id="lg1"></div>
<div class="wrap"><canvas id="c1"></canvas></div>
</div>

<div class="card">
<h2>シェル変位の変化 Δd(t) = d(t) − d(0)</h2>
<p>コアから見たシェルのずれ（shelldisp）の変化．大きさは元素で大きく違う（K はコア変位の約 1/50，O はコア変位と同程度）．</p>
<div class="legend" id="lg2"></div>
<div class="wrap"><canvas id="c2"></canvas></div>
</div>

<div class="card">
<h2>成分ごとの重ね描き（各系列を最大振幅で規格化）</h2>
<p>u と Δd の位相関係を見るため，同じ成分の 2 本を ±1 に規格化して重ねる（単位が違うのではなく大きさが違うため）．</p>
<div class="bar" id="comp"></div>
<div class="legend" id="lg3"></div>
<div class="wrap"><canvas id="c3"></canvas></div>
</div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const ATOMS = __ATOMS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const AX = ['x', 'y', 'z'];
let cur = ATOMS.length - 1, comp = 0, charts = [];

function axis(t){return {title:{display:true,text:t,color:css('--muted')},ticks:{color:css('--muted')},grid:{color:css('--grid')},border:{color:css('--line')}};}
function opts(ylab, fmt){
  return {responsive:true,maintainAspectRatio:false,animation:false,parsing:false,
    interaction:{mode:'index',axis:'x',intersect:false},
    plugins:{legend:{display:false},tooltip:{callbacks:{title:it=>`t = ${it[0].parsed.x.toFixed(1)} fs`,
      label:c=>`${c.dataset.label}: ${fmt(c.parsed.y)}`}}},
    scales:{x:{...axis('時間 (fs)'),type:'linear',min:0,max:300},y:axis(ylab)}};
}
const ds = (lab, t, y, color, dash) => ({label:lab, data:t.map((x,i)=>({x, y:y[i]})), borderColor:color,
  backgroundColor:color, borderWidth:2, pointRadius:0, pointHoverRadius:4, borderDash:dash||[]});
function legend(el, items){ el.innerHTML = items.map(([l,c,d])=>`<span><i style="background:${d?'none':c};${d?`border-top:2px dashed ${c};height:0`:''}"></i>${l}</span>`).join(''); }

function draw(){
  charts.forEach(c=>c.destroy()); charts=[];
  const a = ATOMS[cur], s = a.info, cols=[css('--s1'),css('--s2'),css('--s3')];
  const f3 = v => v.toExponential(2) + ' Å';
  document.getElementById('stats').innerHTML = [
    [`${s.sp} (id ${s.id})`, '追跡している原子'],
    [s.u_step.toExponential(2)+' Å', 'コアの 1 ステップ移動 |Δr|'],
    [s.dd_step.toExponential(2)+' Å', 'シェル変位の 1 ステップ変化 |Δd|'],
    [Math.max(...s.u_rms).toFixed(3)+' Å', 'コア変位の揺らぎ rms（最大成分）'],
    [Math.hypot(...s.d0).toFixed(4)+' Å', 'シェル変位 |d(0)|'],
    [s.corr.map(c=>(c>=0?'+':'−')+Math.abs(c).toFixed(2)).join(' / '), 'u と Δd の相関係数 (x / y / z)'],
  ].map(([b,t])=>`<div class="stat"><b>${b}</b><span>${t}</span></div>`).join('');
  legend(document.getElementById('lg1'), AX.map((x,i)=>[`u_${x}`, cols[i]]));
  legend(document.getElementById('lg2'), AX.map((x,i)=>[`Δd_${x}`, cols[i]]));
  charts.push(new Chart(document.getElementById('c1'), {type:'line', options:opts('u (Å)', f3),
    data:{datasets:AX.map((x,i)=>ds(`u_${x}`, a.t, a.u[i], cols[i]))}}));
  charts.push(new Chart(document.getElementById('c2'), {type:'line', options:opts('Δd (Å)', f3),
    data:{datasets:AX.map((x,i)=>ds(`Δd_${x}`, a.t, a.dd[i], cols[i]))}}));
  const norm = y => { const m = Math.max(...y.map(Math.abs)) || 1; return y.map(v=>v/m); };
  const c = cols[comp];
  legend(document.getElementById('lg3'), [[`u_${AX[comp]}（規格化）`, c], [`Δd_${AX[comp]}（規格化）`, css('--muted'), true]]);
  charts.push(new Chart(document.getElementById('c3'), {type:'line', options:opts('規格化した値', v=>v.toFixed(3)),
    data:{datasets:[ds(`u_${AX[comp]}`, a.t, norm(a.u[comp]), c), ds(`Δd_${AX[comp]}`, a.t, norm(a.dd[comp]), css('--muted'), [5,4])]}}));
}
function buttons(el, labels, get, set){
  el.innerHTML=''; labels.forEach((l,i)=>{const b=document.createElement('button'); b.textContent=l;
    b.setAttribute('aria-pressed', get()===i); b.onclick=()=>{set(i); [...el.children].forEach((x,j)=>x.setAttribute('aria-pressed', j===i)); draw();}; el.appendChild(b);});
}
buttons(document.getElementById('pick'), ATOMS.map(a=>`${a.info.sp}（id ${a.info.id}）`), ()=>cur, i=>cur=i);
buttons(document.getElementById('comp'), ['x 成分','y 成分','z 成分'], ()=>comp, i=>comp=i);
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""
dst = OUT / "traj_atom.html"
dst.write_text(html.replace("__ATOMS__", json.dumps(atoms, separators=(",", ":"))), encoding="utf-8")
print("wrote", dst)
