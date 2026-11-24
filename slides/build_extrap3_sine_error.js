// 3 点外挿の誤差をサイン波で解析するスライド (4 枚) を作る
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const { applyTheme } = require("/Users/hirokinaito/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/fd514745-9d97-453f-802a-e06098d85af6/841f75c4-660b-49b7-9b23-dc16a01cc4db/skills/pptx/scripts/apply_theme.js");

const OUT = "/Users/hirokinaito/Desktop/KNbO/slides/extrap3_sine_error.pptx";

const THEME = {
  name: "Shell Extrapolation",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A41", lt1: "FFFFFF", dk2: "4A5768", lt2: "EEF2F6",
    accent1: "D9822B", accent2: "2A7F9E", accent3: "7A8899",
    accent4: "B5453A", accent5: "5B8C5A", accent6: "8E6CB0",
    hlink: "2A7F9E", folHlink: "7A8899",
  },
};
const HEX = THEME.colors;

// ---------- 図: サイン波と 3 点外挿 (θ を誇張) ----------
function diagramSVG() {
  const W = 900, H = 600, th = 0.7, psi = 0.25;
  const t0 = psi - 2 * th - 0.45, t1 = psi + th + 0.12;
  const X = t => 70 + (t - t0) / (t1 - t0) * (W - 200);
  const Y = y => H / 2 - y * 230;
  const f = t => Math.sin(t);
  const ts = [psi - 2 * th, psi - th, psi, psi + th];
  const [dm2, dm1, d0, dp1] = ts.map(f);
  const pred = 3 * d0 - 3 * dm1 + dm2;
  // 3 点を通る 2 次式 (外挿に使う多項式)
  const lag = t => {
    const [a, b, c] = ts.slice(0, 3);
    return dm2 * (t - b) * (t - c) / ((a - b) * (a - c)) + dm1 * (t - a) * (t - c) / ((b - a) * (b - c)) + d0 * (t - a) * (t - b) / ((c - a) * (c - b));
  };
  let curve = "", par = "";
  for (let i = 0; i <= 200; i++) {
    const t = t0 + (t1 - t0) * i / 200;
    curve += `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(f(t)).toFixed(1)}`;
  }
  for (let i = 0; i <= 100; i++) {
    const t = ts[0] - 0.1 + (ts[3] + 0.08 - ts[0] + 0.1) * i / 100;
    par += `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(lag(t)).toFixed(1)}`;
  }
  const teal = "#" + HEX.accent2, amber = "#" + HEX.accent1, ink = "#" + HEX.dk1, mute = "#" + HEX.accent3;
  const dot = (t, y, fill, stroke) => `<circle cx="${X(t)}" cy="${Y(y)}" r="11" fill="${fill}" stroke="${stroke}" stroke-width="4"/>`;
  const lab = (t, y, txt, dx, dy, col) => `<text x="${X(t) + dx}" y="${Y(y) + dy}" font-family="Helvetica, Arial" font-size="30" fill="${col}">${txt}</text>`;
  const xe = X(ts[3]) + 60;
  // 接線方向 (ḋ) の矢印
  const vx = 1, vy = Math.cos(ts[3]); const L = 0.22;
  const ax2 = ts[3] + L * vx / Math.hypot(vx, vy), ay2 = dp1 + L * vy / Math.hypot(vx, vy);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <marker id="ha" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="${amber}"/></marker>
    <marker id="ht" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="${teal}"/></marker>
  </defs>
  <line x1="50" y1="${Y(0)}" x2="${W - 20}" y2="${Y(0)}" stroke="${mute}" stroke-width="2"/>
  <text x="${W - 40}" y="${Y(0) + 40}" font-family="Helvetica, Arial" font-size="30" fill="${mute}" font-style="italic">t</text>
  ${ts.map(t => `<line x1="${X(t)}" y1="${Y(0) - 8}" x2="${X(t)}" y2="${Y(0) + 8}" stroke="${mute}" stroke-width="2"/>`).join("")}
  <path d="${curve}" fill="none" stroke="${teal}" stroke-width="5"/>
  <path d="${par}" fill="none" stroke="${amber}" stroke-width="4" stroke-dasharray="12 9"/>
  <line x1="${xe}" y1="${Y(dp1)}" x2="${xe}" y2="${Y(pred)}" stroke="${amber}" stroke-width="4" marker-start="url(#ha)" marker-end="url(#ha)"/>
  <line x1="${X(ts[3])}" y1="${Y(dp1)}" x2="${xe + 10}" y2="${Y(dp1)}" stroke="${amber}" stroke-width="2" stroke-dasharray="4 4"/>
  <line x1="${X(ts[3])}" y1="${Y(pred)}" x2="${xe + 10}" y2="${Y(pred)}" stroke="${amber}" stroke-width="2" stroke-dasharray="4 4"/>
  <text x="${xe + 14}" y="${(Y(dp1) + Y(pred)) / 2 + 12}" font-family="Helvetica, Arial" font-size="36" font-style="italic" font-weight="bold" fill="${amber}">e</text>
  <line x1="${X(ts[3])}" y1="${Y(dp1)}" x2="${X(ax2)}" y2="${Y(ay2)}" stroke="${teal}" stroke-width="4" marker-end="url(#ht)" opacity="0.9"/>
  ${dot(ts[0], dm2, teal, teal)}${dot(ts[1], dm1, teal, teal)}${dot(ts[2], d0, teal, teal)}
  ${dot(ts[3], dp1, "#ffffff", teal)}${dot(ts[3], pred, amber, amber)}
  ${lab(ts[0], dm2, "d", -22, 52, ink)}${lab(ts[0], dm2, "n−2", -2, 64, ink).replace('font-size="30"', 'font-size="21"')}
  ${lab(ts[1], dm1, "d", -22, 52, ink)}${lab(ts[1], dm1, "n−1", -2, 64, ink).replace('font-size="30"', 'font-size="21"')}
  ${lab(ts[2], d0, "d", 18, 44, ink)}${lab(ts[2], d0, "n", 38, 56, ink).replace('font-size="30"', 'font-size="21"')}
  ${lab(ts[3], dp1, "d", -20, 56, ink)}${lab(ts[3], dp1, "n+1", 0, 68, ink).replace('font-size="30"', 'font-size="21"')}
  ${lab(ts[3], pred, "d", -96, 4, amber)}${lab(ts[3], pred, "pred", -76, 16, amber).replace('font-size="30"', 'font-size="21"')}
</svg>`;
}

// ---------- 数式ヘルパ: "^{..}" / "_{..}" を上付き・下付きの run にする ----------
function eq(src, base = {}) {
  const runs = [];
  let i = 0, buf = "";
  const push = (text, extra) => { if (text) runs.push({ text, options: { fontFace: "Cambria Math", ...base, ...extra } }); };
  while (i < src.length) {
    const c = src[i];
    if ((c === "^" || c === "_") && src[i + 1] === "{") {
      push(buf); buf = "";
      const j = src.indexOf("}", i);
      push(src.slice(i + 2, j), c === "^" ? { superscript: true } : { subscript: true });
      i = j + 1;
    } else { buf += c; i++; }
  }
  push(buf);
  return runs;
}

(async () => {
  const png = await sharp(Buffer.from(diagramSVG())).png().toBuffer();

  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 × 5.625 in
  pres.title = "3 点外挿の誤差のサイン波解析";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 8.6, h: 1.5, fontSize: 34, align: "left", bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.25, w: 8.6, h: 1.2, fontSize: 18, color: "CADCEB", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, align: "left", bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.1, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3 },
  });

  // ---- 1. タイトル ----
  pres.addSection({ title: "導入" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "導入" });
  s.addText("3 点外挿の誤差をサイン波で解析する", { placeholder: "title" });
  s.addText("シェル変位を 1 つの振動モードとみなし，予測誤差の大きさと向きを厳密に求める", { placeholder: "body" });
  s.addNotes("シェル変位 d(t) を単一モードの複素振動として扱い，3 点外挿 d_pred = 3d_n − 3d_{n−1} + d_{n−2} の誤差を閉じた形で求める．");

  // ---- 2. 設定 ----
  pres.addSection({ title: "解析" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "解析" });
  s.addText("設定：1 つのモードを複素振動で表す", { placeholder: "title" });
  const rows = [
    ["シェル変位", "d(t) = Re[A e^{iωt}]"],
    ["時刻", "t_{n} = n·dt"],
    ["1 ステップの位相", "θ = ω·dt"],
    ["z = e^{iθ} とおくと", "d_{n−j} = Re[A e^{iωtₙ} z^{−j}]"],
  ];
  rows.forEach(([lbl, ex], k) => {
    const y = 1.3 + k * 0.72;
    s.addText(eq(lbl, { fontFace: undefined }), { x: 0.5, y, w: 4.6, h: 0.28, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, objectName: `label-${k}` });
    s.addText(eq(ex), { x: 0.5, y: y + 0.27, w: 4.6, h: 0.38, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: `eq-${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.3, w: 4.6, h: 0.82, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" }, objectName: "pred-box" });
  s.addText([{ text: "3 点外挿（2 次の多項式で外挿）", options: { fontSize: 12, color: C.text2, breakLine: true } },
             ...eq("d_{pred} = 3d_{n} − 3d_{n−1} + d_{n−2}", { fontSize: 18, color: C.text1, bold: true })],
            { x: 0.7, y: 4.36, w: 4.3, h: 0.7, margin: 0, valign: "middle", isTextBox: true, objectName: "pred-eq" });
  s.addImage({ data: "image/png;base64," + png.toString("base64"), x: 5.35, y: 1.3, w: 4.2, h: 2.8, altText: "サイン波上の 3 点 d_{n−2}, d_{n−1}, d_n から 2 次式で外挿した d_pred が，真の d_{n+1} より進行方向に行きすぎる様子．差が誤差 e．", objectName: "diagram" });
  s.addText("θ を誇張して描画（θ = 0.7）．破線が 3 点を通る 2 次式，矢印 e が予測誤差，青い矢印が速度 ḋ の向き", { x: 5.35, y: 4.2, w: 4.2, h: 0.9, fontSize: 11, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "diagram-caption" });

  // ---- 3. 誤差の計算 ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "解析" });
  s.addText("誤差を厳密に計算する", { placeholder: "title" });
  const steps = [
    ["予測誤差 e = d_pred − d_n+1 をまとめる", "e = Re[A e^{iωtₙ} (3 − 3z^{−1} + z^{−2} − z)]"],
    ["3 次の差分として因数分解", "  = Re[−A e^{iωtₙ} z^{−2} (z − 1)^{3}]"],
    ["z − 1 = 2i sin(θ/2) e^{iθ/2} を代入", "  = Re[8 sin^{3}(θ/2) · iA e^{iω(tₙ − dt/2)}]"],
    ["iA e^{iωt} = ḋ/ω を使う", "e = (8 sin^{3}(θ/2) / ω) · ḋ(t_{n} − dt/2)  ≈  ω^{2}dt^{3} · ḋ"],
  ];
  steps.forEach(([lbl, ex], k) => {
    const y = 1.22 + k * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.08, w: 0.42, h: 0.42, fill: { color: k === 3 ? C.accent1 : C.accent2 }, line: { type: "none" }, objectName: `step-dot-${k}` });
    s.addText(String(k + 1), { x: 0.5, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `step-num-${k}` });
    s.addText(eq(lbl.replace("d_pred", "d_{pred}").replace("d_n+1", "d_{n+1}"), { fontFace: undefined }), { x: 1.15, y, w: 5.2, h: 0.28, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, objectName: `step-label-${k}` });
    s.addText(eq(ex, k === 3 ? { bold: true } : {}), { x: 1.15, y: y + 0.27, w: 5.5, h: 0.4, fontSize: 17, color: C.text1, margin: 0, isTextBox: true, objectName: `step-eq-${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 1.22, w: 2.6, h: 2.9, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "check-card" });
  s.addText([
    { text: "検算", options: { fontSize: 14, bold: true, color: C.text1, breakLine: true } },
    { text: "一般式（テイラー展開）", options: { fontSize: 12, color: C.text2, breakLine: true } },
    ...eq("e = −dt^{3} d^{(3)}", { fontSize: 16, color: C.text1 }), { text: "", options: { breakLine: true } },
    { text: "に，サイン波の関係", options: { fontSize: 12, color: C.text2, breakLine: true } },
    ...eq("d^{(3)} = −ω^{2} ḋ", { fontSize: 16, color: C.text1 }), { text: "", options: { breakLine: true } },
    { text: "を代入すると ステップ 4 と一致する", options: { fontSize: 12, color: C.text2, breakLine: true } },
    ...eq("（d^{(3)} は 3 階の時間微分）", { fontFace: undefined, fontSize: 11, color: C.text2 }),
  ], { x: 7.1, y: 1.4, w: 2.25, h: 2.6, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "check-text" });

  // ---- 4. 結論 ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "解析" });
  s.addText("誤差は小さいが，速度と同じ向きにそろう", { placeholder: "title" });
  const cards = [
    { big: eq("θ^{3}"), head: "大きさ", body: [...eq("振幅に対する相対誤差は 8 sin^{3}(θ/2) ≈ θ^{3}"), { text: "", options: { breakLine: true } }, { text: "θ = 0.047 なら約 1×10⁻⁴", options: {} }], col: C.accent2 },
    { big: eq("3θ/2"), head: "向き", body: eq("ḋ と同じ位相．ずれは 3θ/2 ≈ 0.07 rad だけ"), col: C.accent1 },
  ];
  cards.forEach((c, k) => {
    const x = 0.5 + k * 4.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 4.4, h: 2.55, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `card-${k}` });
    s.addText(c.big.map(r => ({ text: r.text, options: { ...r.options, fontSize: 54, bold: true, color: c.col } })), { x: x + 0.3, y: 1.4, w: 3.8, h: 1.0, margin: 0, valign: "middle", isTextBox: true, objectName: `card-big-${k}` });
    s.addText(c.head, { x: x + 0.3, y: 2.45, w: 3.8, h: 0.35, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `card-head-${k}` });
    s.addText(c.body.map(r => ({ text: r.text, options: { ...r.options, fontFace: undefined, fontSize: 14, color: C.text2 } })), { x: x + 0.3, y: 2.85, w: 3.8, h: 0.85, margin: 0, valign: "top", isTextBox: true, objectName: `card-body-${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.05, w: 9.0, h: 0.85, rectRadius: 0.08, fill: { color: C.text1 }, line: { type: "none" }, objectName: "takeaway-box" });
  s.addText("予測はシェルが進む方向へ少し行きすぎる（先回りする）", { x: 0.8, y: 4.05, w: 8.4, h: 0.85, fontSize: 20, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true, objectName: "takeaway" });
  s.addText("θ = ω·dt ≈ 0.047 は 15 THz のモード・dt = 0.5 fs の場合", { x: 0.5, y: 5.0, w: 8.4, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true, objectName: "footnote" });

  require("fs").mkdirSync(path.dirname(OUT), { recursive: true });
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  require("fs").writeFileSync(path.join(__dirname, "diagram.png"), png);
  console.log("wrote", OUT);
})();
