"""進捗報告11_24.pptx の最後の (空の) スライド 11 に「誤差 × シェルの動き の平均」の解説を入れる．
slide 10 と同じ部品 (番号付き丸 + ラベル + Cambria Math の式) で組む．"""
import copy
import math
import sys

from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
s10, s11 = prs.slides[9], prs.slides[10]
MATH = "Cambria Math"

# ---- タイトル ----
ttl = next(sh for sh in s11.shapes if sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type))
ttl.text_frame.paragraphs[0].add_run().text = "誤差 × シェルの動き：平均は 0 か正か"


def runs(tf, parts, size, math=False, color=None, bold=False, align=None):
    """parts: [(text, 'n'|'sub'|'sup'), ...] を 1 段落に書く．"""
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    for text, kind in parts:
        r = p.add_run()
        r.text = text
        f = r.font
        f.size = Pt(size)
        f.bold = bold
        if color is not None:
            f.color.theme_color = color
        rPr = r._r.get_or_add_rPr()
        if kind == "sub":
            rPr.set("baseline", "-25000")
        elif kind == "sup":
            rPr.set("baseline", "30000")
        if math:
            for tag in ("a:latin", "a:ea", "a:cs"):
                el = rPr.makeelement(qn(tag), {"typeface": MATH})
                rPr.append(el)


def textbox(slide, x, y, w, h, name, anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


# ---- 左列: slide 10 と同じ座標の 4 ステップ ----
DOT_X, LAB_X, DOT = 609600, 1402080, 512064
YS = [1584960, 2633472, 3681984, 4730496]
LAB_W, EQ_W = 5852160, 5852160
steps = [
    ([("誤差を d 方向と ḋ 方向に分ける", "n")],
     [("e = a·d + b·ḋ     (d = A sin ψ,  ḋ = Aω cos ψ)", "n")]),
    ([("1 ステップの漏れ ＝ 誤差 × シェルの動き", "n")],
     [("ΔE ≈ K e·ḋ dt = K (a ", "n"), ("d·ḋ", "n"), (" + b ", "n"), ("ḋ", "n"), ("2", "sup"), (") dt", "n")]),
    ([("d 成分：d·ḋ = ½ d(d", "n"), ("2", "sup"), (")/dt は時間微分の形 → 1 周期で元に戻る", "n")],
     [("d·ḋ = ½A", "n"), ("2", "sup"), ("ω sin 2ψ   →   ⟨d·ḋ⟩ = 0", "n")]),
    ([("ḋ 成分：ḋ", "n"), ("2", "sup"), (" は常に正 → 打ち消し合わない", "n")],
     [("ḋ", "n"), ("2", "sup"), (" = A", "n"), ("2", "sup"), ("ω", "n"), ("2", "sup"), (" cos", "n"), ("2", "sup"),
      ("ψ   →   ⟨dE/dt⟩ = ½ K b A", "n"), ("2", "sup"), ("ω", "n"), ("2", "sup")]),
]
for k, ((lab, eq), y) in enumerate(zip(steps, YS)):
    dot = s11.shapes.add_shape(MSO_SHAPE.OVAL, Emu(DOT_X), Emu(y), Emu(DOT), Emu(DOT))
    dot.name = f"step-dot-{k}"
    dot.fill.solid()
    dot.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1 if k == 3 else MSO_THEME_COLOR.ACCENT_2
    dot.line.fill.background()
    dot.shadow.inherit = False
    tf = dot.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    runs(tf, [(str(k + 1), "n")], 16, color=MSO_THEME_COLOR.BACKGROUND_1, bold=True, align=PP_ALIGN.CENTER)
    runs(textbox(s11, LAB_X, y - 97536, LAB_W, 341376, f"step-label-{k}"), lab, 16, color=MSO_THEME_COLOR.TEXT_2)
    runs(textbox(s11, LAB_X, y + 231648, EQ_W, 487680, f"step-eq-{k}"), eq, 20, math=True)

# ---- 右列: 1 周期のグラフ 2 枚 ----
N = 97
psi = [2 * math.pi * i / (N - 1) for i in range(N)]
CH_X, CH_W, CH_H = 7406640, 4175760, 1920240
specs = [
    (1417320, "d 成分：d·ḋ ∝ ½ sin 2ψ", [0.5 * math.sin(2 * p) for p in psi], 0.0, "平均 0"),
    (3474720, "ḋ 成分：ḋ² ∝ cos²ψ", [math.cos(p) ** 2 for p in psi], 0.5, "平均 ½ > 0"),
]
for j, (y, title, ys, mean, mlab) in enumerate(specs):
    cd = XyChartData()
    s = cd.add_series(title)
    for p, v in zip(psi, ys):
        s.add_data_point(p / math.pi, v)
    m = cd.add_series(mlab)
    m.add_data_point(0, mean)
    m.add_data_point(2, mean)
    gf = s11.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_SMOOTH_NO_MARKERS,
                              Emu(CH_X), Emu(y), Emu(CH_W), Emu(CH_H), cd)
    gf.name = f"chart-{j}"
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    tp = ch.chart_title.text_frame.paragraphs[0]
    for txt, col in ((title + "   ", MSO_THEME_COLOR.TEXT_2), ("--- " + mlab, MSO_THEME_COLOR.ACCENT_2)):
        r = tp.add_run()
        r.text = txt
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.theme_color = col
    ser0, ser1 = ch.plots[0].series
    ser0.format.line.width = Pt(2.25)
    ser0.format.line.color.theme_color = MSO_THEME_COLOR.ACCENT_1
    ser0.smooth = True
    ser1.format.line.width = Pt(1.75)
    ser1.format.line.color.theme_color = MSO_THEME_COLOR.ACCENT_2
    ser1.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ser1.smooth = False
    xa, ya = ch.category_axis, ch.value_axis
    xa.minimum_scale, xa.maximum_scale, xa.major_unit = 0, 2, 0.5
    ya.minimum_scale, ya.maximum_scale, ya.major_unit = -0.5, 1.0, 0.5
    xa.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    for ax in (xa, ya):
        ax.tick_labels.font.size = Pt(10)
        ax.tick_labels.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        ax.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    ya.has_major_gridlines = True
    ya.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
    xa.has_major_gridlines = False
    xa.tick_labels.number_format = '0.0"π"'
    xa.tick_labels.number_format_is_linked = False
    ya.tick_labels.number_format = "0.0"
    ya.tick_labels.number_format_is_linked = False

# ---- 下段: まとめ (薄い背景のカード) ----
box = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(609600), Emu(5577840), Emu(10972800), Emu(667512))
box.name = "summary"
box.adjustments[0] = 0.18
box.fill.solid()
box.fill.fore_color.theme_color = MSO_THEME_COLOR.BACKGROUND_2
box.line.fill.background()
box.shadow.inherit = False
tf = box.text_frame
tf.margin_left = tf.margin_right = Emu(182880)
tf.margin_top = tf.margin_bottom = 0
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.word_wrap = True
runs(tf, [("d 方向の誤差（偶数階：d, d", "n"), ("(2)", "sup"), (", d", "n"), ("(4)", "sup"),
          ("）：平均 0 → 保存量は揺れるだけ（k = 1, 3）", "n")], 15, color=MSO_THEME_COLOR.TEXT_2, bold=True,
     align=PP_ALIGN.LEFT)
p2 = tf.add_paragraph()
p2.alignment = PP_ALIGN.LEFT
for text, kind in (("ḋ 方向の誤差（奇数階：d", "n"), ("(1)", "sup"), (", d", "n"), ("(3)", "sup"),
                   ("）：平均 ≠ 0 → ドリフト（k = 0：遅れて冷却，k = 2：先回りして加熱）", "n")):
    r = p2.add_run()
    r.text = text
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.theme_color = MSO_THEME_COLOR.ACCENT_1
    if kind == "sup":
        r._r.get_or_add_rPr().set("baseline", "30000")

# ---- ページ番号: slide 10 のものを複製して 11 に ----
for shp in s10.shapes:
    if shp.name.startswith("Slide Number"):
        el = copy.deepcopy(shp._element)
        for t in el.iter(qn("a:t")):
            if t.text.strip() == "10":
                t.text = "11"
        s11.shapes._spTree.append(el)

prs.save(DST)
print("saved", DST)
