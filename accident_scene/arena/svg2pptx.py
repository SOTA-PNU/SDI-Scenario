"""Turn the baked exec-path SVGs into a PowerPoint deck of native shapes.

Not an image import: every rect, line, circle and label becomes a real
PowerPoint object, so the diagram stays editable (move a box, recolour it,
retype a label) inside the deck.  Only the subset of SVG the diagrams use is
handled - presentation attributes, no CSS, no paths.

    python svg2pptx.py OUT.pptx  SVG:제목:부제  [SVG:제목:부제 ...]
"""

import re
import sys
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

SVG_NS = "{http://www.w3.org/2000/svg}"
EMU_IN = 914400
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN_TOP = Inches(1.45)          # room for the slide header
MARGIN_SIDE = Inches(0.75)
MARGIN_BOTTOM = Inches(0.55)

INK = "101820"
MUTED = "5A6A79"
SANS = "맑은 고딕"
MONO = "Consolas"


def hex_of(v, default=None):
    if not v or v == "none":
        return default
    m = re.match(r"#([0-9a-fA-F]{6})", v.strip())
    return m.group(1).upper() if m else default


def strip_font(v):
    """'IBM Plex Mono',Consolas,monospace -> pick a face PowerPoint will have."""
    if not v:
        return SANS
    return MONO if "Mono" in v or "Consolas" in v else SANS


class Placer:
    """SVG user units -> slide EMU, aspect preserved, centred under the header."""

    def __init__(self, vb_w, vb_h):
        avail_w = SLIDE_W - 2 * MARGIN_SIDE
        avail_h = SLIDE_H - MARGIN_TOP - MARGIN_BOTTOM
        self.s = min(avail_w / vb_w, avail_h / vb_h)
        self.ox = MARGIN_SIDE + (avail_w - vb_w * self.s) / 2
        self.oy = MARGIN_TOP + (avail_h - vb_h * self.s) / 2

    def x(self, v):
        return Emu(int(self.ox + float(v) * self.s))

    def y(self, v):
        return Emu(int(self.oy + float(v) * self.s))

    def d(self, v):
        return Emu(int(float(v) * self.s))

    def pt(self, px):
        return Pt(float(px) * self.s / EMU_IN * 72.0)


def add_rect(slide, p, el):
    x, y = float(el.get("x", 0)), float(el.get("y", 0))
    w, h = float(el.get("width", 0)), float(el.get("height", 0))
    rx = float(el.get("rx", 0) or 0)
    shape_kind = MSO_SHAPE.ROUNDED_RECTANGLE if rx else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_kind, p.x(x), p.y(y), p.d(w), p.d(h))
    sh.shadow.inherit = False
    if rx:
        sh.adjustments[0] = min(0.5, rx / min(w, h))
    fill = hex_of(el.get("fill"))
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        sh.fill.background()
    stroke = hex_of(el.get("stroke"))
    if stroke:
        sh.line.color.rgb = RGBColor.from_string(stroke)
        sh.line.width = p.pt(float(el.get("stroke-width", 1)))
        if el.get("stroke-dasharray"):
            ln = sh.line._get_or_add_ln()
            dash = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
            ln.append(dash)
    else:
        sh.line.fill.background()
    sh.text_frame.text = ""
    return sh


def add_circle(slide, p, el):
    cx, cy, r = float(el.get("cx")), float(el.get("cy")), float(el.get("r"))
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, p.x(cx - r), p.y(cy - r),
                                p.d(2 * r), p.d(2 * r))
    sh.shadow.inherit = False
    fill = hex_of(el.get("fill"), INK)
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    sh.line.fill.background()
    return sh


def add_line(slide, p, el, arrow):
    x1, y1 = float(el.get("x1")), float(el.get("y1"))
    x2, y2 = float(el.get("x2")), float(el.get("y2"))
    from pptx.enum.shapes import MSO_CONNECTOR
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                    p.x(x1), p.y(y1), p.x(x2), p.y(y2))
    stroke = hex_of(el.get("stroke"), INK)
    cn.line.color.rgb = RGBColor.from_string(stroke)
    cn.line.width = p.pt(float(el.get("stroke-width", 1.4)))
    if arrow:
        ln = cn.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"),
                                 {"type": "triangle", "w": "med", "len": "med"}))
    return cn


def add_text(slide, p, el):
    txt = (el.text or "").strip()
    if not txt:
        return None
    x, y = float(el.get("x")), float(el.get("y"))
    fs = float(el.get("font-size", 12))
    anchor = el.get("text-anchor", "start")
    fill = hex_of(el.get("fill"), INK)
    face = strip_font(el.get("font-family"))
    bold = el.get("font-weight") in ("600", "700", "bold")

    # SVG y is the baseline; a PowerPoint textbox is anchored top. Place a box
    # of ~2x the em height centred on the baseline and middle-anchor the text.
    box_h = fs * 2.0
    box_w = max(len(txt) * fs * 1.15, 60)
    left = p.x(x - box_w / 2) if anchor == "middle" else p.x(x - fs * 0.25)
    top = p.y(y - fs * 1.32)
    tb = slide.shapes.add_textbox(left, top, p.d(box_w), p.d(box_h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    par = tf.paragraphs[0]
    par.alignment = PP_ALIGN.CENTER if anchor == "middle" else PP_ALIGN.LEFT
    run = par.add_run()
    run.text = txt
    run.font.size = p.pt(fs)
    run.font.name = face
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(fill)
    return tb


def header(slide, title, subtitle):
    tb = slide.shapes.add_textbox(MARGIN_SIDE, Inches(0.5),
                                  SLIDE_W - 2 * MARGIN_SIDE, Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = 0
    p1 = tf.paragraphs[0]
    r1 = p1.add_run()
    r1.text = title
    r1.font.size = Pt(26)
    r1.font.bold = True
    r1.font.name = SANS
    r1.font.color.rgb = RGBColor.from_string(INK)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = subtitle
    r2.font.size = Pt(13)
    r2.font.name = SANS
    r2.font.color.rgb = RGBColor.from_string(MUTED)


def build(out, jobs):
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    for path, title, subtitle in jobs:
        root = ET.parse(path).getroot()
        vb_w, vb_h = (float(v) for v in root.get("viewBox").split()[2:])
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        header(slide, title, subtitle)
        p = Placer(vb_w, vb_h)
        arrows = set()
        for el in root.iter():
            if el.tag == SVG_NS + "line" and el.get("marker-end"):
                arrows.add(id(el))
        n = 0
        for el in root:
            tag = el.tag.replace(SVG_NS, "")
            if tag == "defs":
                continue
            if tag == "rect":
                # the first rect is the white ground; the slide already is white
                if float(el.get("width")) >= vb_w - 1:
                    continue
                add_rect(slide, p, el)
            elif tag == "circle":
                add_circle(slide, p, el)
            elif tag == "line":
                add_line(slide, p, el, id(el) in arrows)
            elif tag == "text":
                add_text(slide, p, el)
            else:
                continue
            n += 1
        print(f"  {path}: {n} 도형")
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = []
    for spec in sys.argv[2:]:
        path, title, subtitle = spec.split(":", 2)
        jobs.append((path, title, subtitle))
    build(out, jobs)
