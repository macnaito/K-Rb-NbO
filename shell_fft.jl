# shelldisp の周波数解析: shell 予測子の sin 近似 (誤差 ≈ (iθ)^{k+1}·d, θ = ωΔt) を検証する．
#
#     julia shell_fft.jl [outdir]
#         outdir 既定: output_result/out_nve_0.5fs_traj (毎ステップ xyz, dt = 0.5 fs, k = 2 で走らせたもの)
#     → <outdir>/shell_fft.html, <outdir>/shell_fft.txt
#
# 1. 元素ごとの shelldisp パワースペクトル (平均・一次トレンド除去, Hann 窓) とコア速度スペクトル (VDOS)
# 2. 予測子誤差 e_k(n+1) = d(n+1) − pred_k = ∇^{k+1} d(n+1) (恒等式) を k = 0..4 で軌跡から直接計算し，
#    |e_k| / |∇d| を sin 模型 (スペクトル < fcut から得た θ_eff) の予言と比べる
# 3. e_k の位相: 速度型 v = (d(n+2) − d(n))/2 と変位型 a = d(n+2) − 2d(n+1) + d(n) への相関
#    sin 模型の予言: k=0 +v, k=1 +a, k=2 −v, k=3 −a, k=4 +v
using Printf

const OUT = length(ARGS) >= 1 ? ARGS[1] : "output_result/out_nve_0.5fs_traj"
const INPUT = "output_result/out-150K_0Gpa_relax/step_39000.xyz"
const DT_FS = 0.5
const FMAX_THZ = 60.0      # スペクトル表示範囲
const DF_THZ = 0.1
const FCUT_THZ = 40.0      # これより上は CG 残差ノイズとみなす (スペクトルで確認)

function read_frame(path)
    open(path) do f
        n = parse(Int, readline(f))
        hdr = readline(f)
        props = split(match(r"Properties=(\S+)", hdr).captures[1], ":")
        cols = Dict{String,Int}(); c = 1
        for j in 1:3:length(props)
            cols[props[j]] = c; c += parse(Int, props[j+2])
        end
        ip, ii, is, iv = cols["pos"], cols["id"], cols["shelldisp"], cols["velo"]
        sp = Vector{String}(undef, n); id = zeros(Int, n)
        sd = zeros(3, n); ve = zeros(3, n)
        for a in 1:n
            v = split(readline(f))
            sp[a] = v[1]; id[a] = parse(Int, v[ii])
            for c in 1:3
                sd[c, a] = parse(Float64, v[is+c-1]); ve[c, a] = parse(Float64, v[iv+c-1])
            end
        end
        p = sortperm(id)
        return sp[p], sd[:, p], ve[:, p]
    end
end

files = [INPUT; sort(filter(f -> occursin(r"^step_\d+\.xyz$", f), readdir(OUT)),
                     by = f -> parse(Int, match(r"\d+", f).match)) .|> f -> joinpath(OUT, f)]
sp, sd0, ve0 = read_frame(files[1])
na, N = length(sp), length(files)
D = zeros(3, na, N); V = zeros(3, na, N)
D[:, :, 1] = sd0; V[:, :, 1] = ve0
for (n, f) in enumerate(files)
    n == 1 && continue
    _, s, v = read_frame(f); D[:, :, n] = s; V[:, :, n] = v
end
println("frames = $N ($(N-1) steps), atoms = $na")
species = ["K", "Nb", "O"]

# ---- 1. スペクトル (直接 DFT, 振幅は Hann 窓付き) --------------------------------
freqs = collect(0:DF_THZ:FMAX_THZ)
tps = (0:N-1) .* (DT_FS * 1e-3)                    # ps
win = [0.5 - 0.5cos(2π * (n - 1) / (N - 1)) for n in 1:N]
E = [cis(-2π * f * t) for t in tps, f in freqs]   # N × nf
function detrend(x)
    t = collect(1.0:length(x)); tm = sum(t) / length(t); xm = sum(x) / length(x)
    b = sum((t .- tm) .* (x .- xm)) / sum((t .- tm) .^ 2)
    return x .- xm .- b .* (t .- tm)
end
function spectrum(A, sel)
    P = zeros(length(freqs))
    for a in sel, c in 1:3
        x = detrend(A[c, a, :]) .* win
        P .+= abs2.(transpose(E) * x)
    end
    return P ./ length(sel)
end
Psd = Dict(s => spectrum(D, findall(==(s), sp)) for s in species)
Pv = Dict(s => spectrum(V, findall(==(s), sp)) for s in species)
Psd_all = spectrum(D, 1:na); Pv_all = spectrum(V, 1:na)

# ---- 2. 予測子誤差 ------------------------------------------------------------
nabla(X) = X[:, :, 2:end] .- X[:, :, 1:end-1]
diffs = Any[D]
for m in 1:5; push!(diffs, nabla(diffs[end])); end
# ∇^m d は index n (= 元の n+m) に対応．共通の時刻範囲 n+1 ∈ 6..N-1 (v, a に n+2 が要る) で揃える
tr = 6:N-1
at(m, X, rng) = X[:, :, rng .- m]                  # ∇^m d(t) を元の時刻 t で取り出す
normmean(X) = sum(sqrt.(sum(abs2, X; dims = 1))) / (size(X, 2) * size(X, 3))
dd = at(1, diffs[2], tr)                            # ∇d(t) = 1 ステップの変位
vv = (D[:, :, tr .+ 1] .- D[:, :, tr .- 1]) ./ 2
aa = D[:, :, tr .+ 1] .- 2 .* D[:, :, tr] .+ D[:, :, tr .- 1]
corr(X, Y) = sum(X .* Y) / sqrt(sum(abs2, X) * sum(abs2, Y))

# sin 模型の予言: |e_k|_rms / |∇d|_rms = sqrt(∫P|2sin(θ/2)|^{2(k+1)} / ∫P|2sin(θ/2)|^2), f < FCUT
θf = 2π .* freqs .* 1e12 .* (DT_FS * 1e-15)
low = freqs .<= FCUT_THZ
mom(m) = sum(Psd_all[low] .* (2 .* sin.(θf[low] ./ 2)) .^ (2m))
rms(X) = sqrt(sum(abs2, X) / (size(X, 2) * size(X, 3)))

io = IOBuffer()
@printf(io, "frames=%d dt=%.2f fs  window=%.3f ps  resolution=%.2f THz\n", N, DT_FS, (N-1)*DT_FS*1e-3, 1/((N-1)*DT_FS*1e-3))
@printf(io, "mean |∇d| per atom-step = %.3e Å\n\n", normmean(dd))
@printf(io, "%2s %12s %12s %12s %12s %10s %8s %8s\n", "k", "mean|e_k|Å", "|e|/|∇d|", "rms ratio", "sin-model", "θ_eff", "C_v", "C_a")
rows = []
for k in 0:4
    e = at(k + 1, diffs[k+2], tr)
    r_mean = normmean(e) / normmean(dd)
    r_rms = rms(e) / rms(dd)
    r_sin = sqrt(mom(k + 1) / mom(1))
    θeff = r_rms^(1 / k)                            # r ≈ θ^k (k ≥ 1)
    cv, ca = corr(e, vv), corr(e, aa)
    push!(rows, (k, normmean(e), r_mean, r_rms, r_sin, cv, ca))
    @printf(io, "%2d %12.3e %12.3e %12.3e %12.3e %10s %8.3f %8.3f\n", k, normmean(e), r_mean, r_rms, r_sin,
            k == 0 ? "-" : @sprintf("%.4f", θeff), cv, ca)
end
# 加重平均周波数
for (lab, P) in (("shelldisp", Psd_all), ("core velo", Pv_all))
    fm = sum(P[low] .* freqs[low]) / sum(P[low])
    @printf(io, "\n<f>_%s (f<%.0f THz) = %.2f THz", lab, FCUT_THZ, fm)
end
fe = sqrt(mom(3) / mom(1)) |> r -> sqrt(r) / (2π * DT_FS * 1e-3)   # θ_eff for k=2 → THz
@printf(io, "\nθ_eff(k=2) from spectrum → f_eff = %.2f THz\n", fe)
txt = String(take!(io)); print(txt)
write(joinpath(OUT, "shell_fft.txt"), txt)

# ---- HTML ---------------------------------------------------------------------
js(v) = "[" * join((@sprintf("%.6g", x) for x in v), ",") * "]"
norm1(P) = P ./ maximum(P)
cols = Dict("K" => "#7c5cbf", "Nb" => "#2a9d8f", "O" => "#e76f51")
ds_sd = join(["{label:'$s shelldisp',data:$(js(norm1(Psd[s]))),borderColor:'$(cols[s])',borderWidth:1.5,pointRadius:0}" for s in species], ",")
ds_v = join(["{label:'$s core velo',data:$(js(norm1(Pv[s]))),borderColor:'$(cols[s])',borderDash:[4,3],borderWidth:1,pointRadius:0}" for s in species], ",")
html = """
<!doctype html><html><head><meta charset="utf-8"><title>shelldisp spectrum</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>body{font-family:sans-serif;margin:16px;max-width:1000px}pre{background:#f4f4f4;padding:8px;overflow-x:auto}
.box{height:380px;margin-bottom:24px}</style></head><body>
<h3>shelldisp パワースペクトル ($(OUT), $(N) frames)</h3>
<div class="box"><canvas id="c1"></canvas></div>
<div class="box"><canvas id="c2"></canvas></div>
<pre>$(txt)</pre>
<script>
const f=$(js(freqs));
const opt=(t,log)=>({responsive:true,maintainAspectRatio:false,animation:false,
 plugins:{title:{display:true,text:t}},
 scales:{x:{type:'linear',title:{display:true,text:'f (THz)'}},y:{type:log?'logarithmic':'linear',title:{display:true,text:'power (max=1)'}}}});
const mk=(ds)=>ds.map(d=>({...d,data:d.data.map((y,i)=>({x:f[i],y:Math.max(y,1e-12)}))}));
new Chart(c1,{type:'line',data:{datasets:mk([$(ds_sd),$(ds_v)])},options:opt('linear',false)});
new Chart(c2,{type:'line',data:{datasets:mk([$(ds_sd)])},options:opt('log (noise floor)',true)});
</script></body></html>
"""
write(joinpath(OUT, "shell_fft.html"), html)
println("→ ", joinpath(OUT, "shell_fft.html"))
