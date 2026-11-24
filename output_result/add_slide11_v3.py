"""進捗報告11_24.pptx の最後のスライド (11) を「3 点外挿と 4 点外挿の ⟨e·Δd⟩」の計算に差し替える．
入力は slide 11 が空の版 (変更前のバックアップ)．"""
import copy
import re
import sys

from pptx import Presentation
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
s10, s11 = prs.slides[9], prs.slides[10]
MATH = "Cambria Math"
TX2, AC1, AC2, BG1, BG2 = (MSO_THEME_COLOR.TEXT_2, MSO_THEME_COLOR.ACCENT_1, MSO_THEME_COLOR.ACCENT_2,
                           MSO_THEME_COLOR.BACKGROUND_1, MSO_THEME_COLOR.BACKGROUND_2)


def parse(s):
    """'A^{2} d_{n+1}' → [(text, kind)]．^{..} 上付き，_{..} 下付き．"""
    out, pos = [], 0
    for m in re.finditer(r"([\^_])\{([^}]*)\}", s):
        if m.start() > pos:
            out.append((s[pos:m.start()], "n"))
        out.append((m.group(2), "sup" if m.group(1) == "^" else "sub"))
        pos = m.end()
    if pos < len(s):
        out.append((s[pos:], "n"))
    return out


def write(p, s, size, math=False, color=None, bold=False):
    for text, kind in parse(s):
        r = p.add_run()
        r.text = text
        f = r.font
        f.size = Pt(size)
        f.bold = bold
        if color is not None:
            f.color.theme_color = color
        rPr = r._r.get_or_add_rPr()
        if kind == "sup":
            rPr.set("baseline", "30000")
        elif kind == "sub":
            rPr.set("baseline", "-25000")
        if math:
            for tag in ("a:latin", "a:ea", "a:cs"):
                rPr.append(rPr.makeelement(qn(tag), {"typeface": MATH}))


def textbox(x, y, w, h, name, anchor=MSO_ANCHOR.TOP):
    tb = s11.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


def card(x, y, w, h, name):
    box = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    box.adjustments[0] = 0.05
    box.fill.solid()
    box.fill.fore_color.theme_color = BG2
    box.line.fill.background()
    box.shadow.inherit = False
    return box


# ---- タイトル ----
ttl = next(sh for sh in s11.shapes if sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type))
ttl.text_frame.paragraphs[0].add_run().text = "誤差 × シェルの動き：3 点外挿と 4 点外挿の平均"

# ---- 設定 (2 行) ----
tf = textbox(0.667, 1.42, 12.0, 0.42, "setup", MSO_ANCHOR.MIDDLE)
p = tf.paragraphs[0]
write(p, "d_{n} = Re[A e^{iωtₙ}]，z = e^{iθ}，シェルの動き Δd = Re[A e^{iωtₙ}(z − 1)] = Re[b e^{iωtₙ}]，b = 2i sin(θ/2) A e^{iθ/2}", 16, math=True)
p = tf.add_paragraph()
write(p, "e = Re[a e^{iωtₙ}] と書くと 1 周期平均は ⟨e·Δd⟩ = ½ Re[a b^{*}]．相関 = cos(arg a − arg b)．漏れ ΔE ≈ K e·Δd：⟨e·Δd⟩ ＞ 0 で加熱", 13, color=TX2)

# ---- 2 枚のカード ----
cards = [
    (0.667, "3 点外挿 (k = 2)", "d_{pred} = 3d_{n} − 3d_{n−1} + d_{n−2}", AC2, [
        ("誤差（z − 1 = 2i sin(θ/2) e^{iθ/2} を代入）", "e = Re[−A e^{iωtₙ} z^{−2}(z − 1)^{3}] = Re[a e^{iωtₙ}]"),
        ("係数：a に i がつく → ḋ（= Re[iωA e^{iωt}]）と同じ位相", "a = 8i sin^{3}(θ/2) A e^{−iθ/2}"),
        ("1 周期平均 ½ Re[a b^{*}]（a b^{*} = 16 sin^{4}(θ/2)|A|^{2} e^{−iθ}）", "⟨e·Δd⟩ = 8|A|^{2} sin^{4}(θ/2) cos θ  ≈  ½ |A|^{2}θ^{4}"),
    ], "相関 = cos θ ≈ 1 → ほぼ同じ向き → 加熱"),
    (6.833, "4 点外挿 (k = 3)", "d_{pred} = 4d_{n} − 6d_{n−1} + 4d_{n−2} − d_{n−3}", AC1, [
        ("誤差（(z − 1)^{4} = 16 sin^{4}(θ/2) e^{2iθ}）", "e = Re[−A e^{iωtₙ} z^{−3}(z − 1)^{4}] = Re[a e^{iωtₙ}]"),
        ("係数：a は実数 × A → d と同じ位相（θ だけずれる）", "a = −16 sin^{4}(θ/2) A e^{−iθ}"),
        ("1 周期平均 ½ Re[a b^{*}]（a b^{*} = 32i sin^{5}(θ/2)|A|^{2} e^{−3iθ/2}）", "⟨e·Δd⟩ = 16|A|^{2} sin^{5}(θ/2) sin(3θ/2)  ≈  ¾ |A|^{2}θ^{6}"),
    ], "相関 = sin(3θ/2) ≈ 3θ/2 → ほぼ直交"),
]
CW, CY, CH = 5.833, 2.02, 3.62
for j, (x, head, pred, col, rows, verdict) in enumerate(cards):
    card(x, CY, CW, CH, f"card-{j}")
    ix, iw = x + 0.25, CW - 0.5
    tf = textbox(ix, CY + 0.16, iw, 0.36, f"card-head-{j}", MSO_ANCHOR.MIDDLE)
    write(tf.paragraphs[0], head, 18, color=col, bold=True)
    tf = textbox(ix, CY + 0.52, iw, 0.32, f"card-pred-{j}", MSO_ANCHOR.MIDDLE)
    write(tf.paragraphs[0], pred, 14, math=True, color=TX2)
    y = CY + 0.98
    for k, (lab, eq) in enumerate(rows):
        tf = textbox(ix, y, iw, 0.26, f"card-{j}-lab-{k}", MSO_ANCHOR.MIDDLE)
        write(tf.paragraphs[0], lab, 12, color=TX2)
        tf = textbox(ix, y + 0.27, iw, 0.36, f"card-{j}-eq-{k}", MSO_ANCHOR.MIDDLE)
        write(tf.paragraphs[0], eq, 16, math=True)
        y += 0.72
    # 判定 (白い角丸の札)
    pill = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(ix), Inches(CY + CH - 0.5),
                                Inches(iw), Inches(0.36))
    pill.name = f"card-verdict-{j}"
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.theme_color = BG1
    pill.line.fill.background()
    pill.shadow.inherit = False
    ptf = pill.text_frame
    ptf.margin_left = ptf.margin_right = Inches(0.15)
    ptf.margin_top = ptf.margin_bottom = 0
    ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
    ptf.paragraphs[0].alignment = PP_ALIGN.LEFT
    write(ptf.paragraphs[0], verdict, 13, color=col, bold=True)

# ---- 下段: 数値比較 ----
tf = textbox(0.667, 5.80, 12.0, 0.95, "numbers")
p = tf.paragraphs[0]
write(p, "θ = 0.053（実効周波数 16.8 THz，dt = 0.5 fs）で数値にすると", 13, color=TX2)
p = tf.add_paragraph()
write(p, "k = 2： ⟨e·Δd⟩ = 3.9×10^{−6} |A|^{2}（相関 0.999）     k = 3： ⟨e·Δd⟩ = 1.6×10^{−8} |A|^{2}（相関 0.079）", 16,
      math=True)
p = tf.add_paragraph()
write(p, "→ k = 3 は k = 2 の 1.5θ^{2} ≈ 1/240．符号は同じ（正）でも実質ゼロ．実測では CG ノイズが勝ち，相関 ≈ 0（shell_cos −0.003）", 13,
      color=AC1, bold=True)

# ---- ページ番号 ----
for shp in s10.shapes:
    if shp.name.startswith("Slide Number"):
        el = copy.deepcopy(shp._element)
        for t in el.iter(qn("a:t")):
            if t.text.strip() == "10":
                t.text = "11"
        s11.shapes._spTree.append(el)

prs.save(DST)
print("saved", DST)
