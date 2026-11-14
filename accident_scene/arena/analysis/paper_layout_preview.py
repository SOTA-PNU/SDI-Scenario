"""Lay out PAPER_IEMEK.md in the IEMEK (대한임베디드공학회) template geometry with Firefox and
report the page count; also writes a PDF preview.

Geometry from the template PDF (see PAPER_IEMEK_NOTES.md): page 516x729 pt (B5), text area
x 57-459 pt and y 89-662 pt, two columns of 191 pt with a 20 pt gutter; body 8 pt serif on a
12.8 pt pitch (160%), section heads 10 pt bold sans, sub-heads 8 pt bold sans, title 16 pt bold,
English title 13 pt, abstract/keywords full width. The browser does the line breaking (Korean
breaks between words, as in HWP's 어절 setting) and the table heights; a small paginator then
fills the columns, splitting paragraphs at line boundaries. Fonts are stand-ins (Noto Serif/Sans
CJK KR for 바탕/돋움), so this approximates, and does not replace, setting the text in the HWP file.

Two placement models are reported:
  float  - a figure/table that does not fit at the end of a column moves to the top of the next
           column while text keeps filling the current one (how an author places objects);
  strict - every block stays in source order and the gap is left empty (worst case).

Usage:
  python3 analysis/paper_layout_preview.py PAPER_IEMEK.md paper_preview/PAPER_IEMEK_layout.pdf \
      [float|strict] [build_dir]
  python3 analysis/paper_layout_preview.py --paste PAPER_IEMEK.md paper_preview/PAPER_IEMEK_paste.html
      (single-column copy for pasting into the HWP template; tables paste as tables)
Needs Firefox and geckodriver. The Ubuntu snap provides both; its binaries are called directly
(/snap/firefox/current/usr/lib/firefox/...) because the snap wrapper refuses to start from a
system-service cgroup. Override with $FIREFOX_BIN / $GECKODRIVER.
"""
import base64
import html
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE_W_PT, PAGE_H_PT, COL_W_PT = 516, 729, 191


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("`", "")


def png_aspect(path):
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return h / w


def is_cap(s, kinds):
    return any(s.startswith("**" + k) for k in kinds)


def parse(md, root, reserve=True):
    head, body = md.split("## Ⅰ. 서 론", 1)
    body = "## Ⅰ. 서 론" + body
    hd = {"authors": [], "foot": []}
    rules = 0
    for line in head.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "---":
            rules += 1
            continue
        if rules == 1:                       # between the two rules: page-1 footnote
            hd["foot"].append(s)
        elif s.startswith("# "):
            hd["title"] = s[2:]
        elif s.startswith("(") and "etitle" not in hd:
            hd["etitle"] = s
        elif s.startswith("**Abstract"):
            hd["abstract"] = s
        elif s.startswith("**Keywords"):
            hd["keywords"] = s
        else:
            hd["authors"].append(s)
    # The author block and the page-1 footnote are typed straight into the HWP template, so the
    # manuscript may omit them; reserve the template's space (4 author lines, 4 footnote lines)
    # so the page count still includes them.
    authors, foot_lines, cls = hd["authors"], hd["foot"], ""
    if reserve and not authors:
        authors, cls = ["저자 이름 (HWP 양식에서 입력)", "소속", "영문 이름", "영문 소속"], " reserved"
    if reserve and not foot_lines:
        foot_lines = ["*Corresponding Author (교신저자 이메일, HWP에서 입력)", "저자: 소속", "저자: 소속",
                      "※ 사사 문구 (없으면 삭제)"]
    header = (f'<div class="title">{inline(hd["title"])}</div>'
              f'<div class="etitle">{inline(hd["etitle"])}</div>'
              + "".join(f'<div class="auth{cls}">{inline(a)}</div>' for a in authors)
              + f'<p class="abs">{inline(hd["abstract"])}</p>'
              + f'<p class="kw">{inline(hd["keywords"])}</p>')
    fcls = ' class="reserved"' if not hd["foot"] else ""
    foot = ('<div class="rule"></div>' + "".join(f"<p{fcls}>{inline(x)}</p>" for x in foot_lines)
            if foot_lines else "")

    units, figs = [], []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("## "):
            units.append({"kind": "head", "html": f"<h2>{inline(s[3:])}</h2>"})
        elif s.startswith("### "):
            units.append({"kind": "head", "html": f"<h3>{inline(s[4:])}</h3>"})
        elif s.startswith("!["):
            src = re.search(r"\((.*?)\)", s).group(1)
            figs.append(src)
            j, caps = i + 1, []
            while j < len(lines) and (not lines[j].strip() or is_cap(lines[j].strip(), ("그림", "Fig"))):
                if lines[j].strip():
                    caps.append(lines[j].strip())
                j += 1
            hpt = COL_W_PT * png_aspect(root / src)
            units.append({"kind": "float", "html":
                          f'<div class="fig"><img src="{src}" style="width:{COL_W_PT}pt;height:{hpt:.2f}pt">'
                          f'<div class="cap">{"<br>".join(inline(c) for c in caps)}</div></div>'})
            i = j
            continue
        elif is_cap(s, ("표", "Table")):
            j, caps = i, []
            while j < len(lines) and (not lines[j].strip() or is_cap(lines[j].strip(), ("표", "Table"))):
                if lines[j].strip():
                    caps.append(lines[j].strip())
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j].strip())
                j += 1
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            note = None
            if k < len(lines) and lines[k].strip().startswith("주:"):
                note, j = lines[k].strip(), k + 1
            rows = [r for r in rows if not re.match(r"^\|[\s:|-]+\|$", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            span = [1] * len(cells)          # an empty first cell merges with the cell above
            for ri in range(len(cells) - 1, 0, -1):
                if ri > 1 and not cells[ri][0]:
                    span[ri - 1] += span[ri]
                    span[ri] = 0
            t = [f'<div class="cap">{"<br>".join(inline(c) for c in caps)}</div><table>']
            for ri, r in enumerate(cells):
                tag = "th" if ri == 0 else "td"
                tds = [f'<{tag}{f" rowspan={span[ri]}" if span[ri] > 1 else ""}>{inline(r[0])}</{tag}>'] if span[ri] else []
                tds += [f"<{tag}>{inline(c)}</{tag}>" for c in r[1:]]
                t.append("<tr>" + "".join(tds) + "</tr>")
            t.append("</table>")
            if note:
                t.append(f'<div class="note">{inline(note)}</div>')
            units.append({"kind": "float", "html": "".join(t)})
            i = j
            continue
        elif s.startswith("- "):
            units.append({"kind": "text", "html": f'<p class="li">-&nbsp;{inline(s[2:])}</p>'})
        elif re.match(r"^\[\d+\]", s):
            units.append({"kind": "text", "html": f'<p class="ref">{inline(s)}</p>'})
        else:
            units.append({"kind": "text", "html": f"<p>{inline(s)}</p>"})
        i += 1
    return header, foot, units, figs


CSS = """
@page { size: 516pt 729pt; margin: 0 }
html, body { margin: 0; padding: 0; background: #fff; color: #000; }
body { font-family: 'Noto Serif CJK KR', serif; }
.page { position: relative; width: 516pt; height: 729pt; overflow: hidden; break-after: page; }
.page:last-child { break-after: auto; }
.col { position: absolute; width: 191pt; }
#stage, #hstage { position: absolute; left: -3000pt; top: 0; visibility: hidden; }
#stage { width: 191pt; } #hstage { width: 402pt; }
.unit { display: flow-root; }
p { margin: 0; font-size: 8pt; line-height: 12.8pt; text-align: justify; text-indent: 8pt;
    word-break: keep-all; overflow-wrap: break-word; }
p.li { text-indent: -6pt; padding-left: 8pt; }
p.ref { text-indent: -10pt; padding-left: 10pt; margin-bottom: 2pt; text-align: left; }
h2 { font: bold 10pt/16pt 'Noto Sans CJK KR', sans-serif; margin: 6pt 0 4pt; }
h3 { font: bold 8pt/12.8pt 'Noto Sans CJK KR', sans-serif; margin: 4pt 0 2pt; text-indent: 10pt; }  /* template: 들여쓰기 10pt */
.fig img { display: block; }
.cap { font-size: 8pt; line-height: 12.8pt; text-align: center; word-break: keep-all; }
.fig .cap { margin: 3pt 0 6pt; }
table { width: 100%; border-collapse: collapse; font-size: 7pt; line-height: 1.45; margin: 2pt 0 4pt;
        word-break: keep-all; }
th, td { border: 0.4pt solid #000; padding: 1.5pt 2pt; text-align: center; vertical-align: middle; }
th { font-weight: bold; }
.note { font-size: 7pt; line-height: 11.2pt; text-align: justify; word-break: keep-all; margin-bottom: 6pt; }
#header { position: absolute; left: 57pt; top: 89pt; width: 402pt; }
.title { font: bold 16pt/25.6pt 'Noto Sans CJK KR', sans-serif; text-align: center; margin-bottom: 6pt;
         word-break: keep-all; }
.etitle { font: 13pt/20.8pt 'Noto Sans CJK KR', sans-serif; text-align: center; margin-bottom: 6pt;
          word-break: keep-all; }
.auth { font-size: 9pt; line-height: 14.4pt; text-align: center; }
p.abs { text-indent: 0; margin-bottom: 6pt; }
p.kw { text-indent: 0; margin-bottom: 10pt; }
#foot { position: absolute; left: 57pt; width: 191pt; }
#foot .rule { width: 60pt; border-top: 0.5pt solid #000; margin-bottom: 4pt; }
#foot p { text-indent: 0; text-align: left; }
.reserved { color: #a3a3a3; }
.stamp { position: absolute; left: 57pt; top: 688pt; width: 402pt; font: 6pt/8pt 'Noto Sans CJK KR', sans-serif;
         color: #888; text-align: center; }
"""

JS = r"""
window.addEventListener('load', () => document.fonts.ready.then(run));
function run() {
  const PX = 96 / 72, L = {colX: [57, 268], top: 89, colH: 573, lh: 12.8, eps: 0.3};
  const hpt = el => el.getBoundingClientRect().height / PX;
  const mk = (tag, cls, inner) => { const e = document.createElement(tag); if (cls) e.className = cls;
                                     if (inner !== undefined) e.innerHTML = inner; return e; };
  const stage = document.getElementById('stage'), hstage = document.getElementById('hstage');
  // flow-root so the keywords' bottom margin (the gap before the body) counts in the header height
  const hdr = mk('div', '', DATA.header); hdr.style.display = 'flow-root'; hstage.appendChild(hdr);
  const headerH = hpt(hdr);
  const headerParts = [...hdr.children].map(e => [e.className, +(hpt(e)).toFixed(1)]);
  hstage.innerHTML = '';
  const ft = mk('div', '', DATA.foot); stage.appendChild(ft);
  const footH = hpt(ft); stage.innerHTML = '';
  const warns = []; let units_i = 0;
  const units = DATA.units.map(u => {
    const el = mk('div', 'unit', u.html); stage.appendChild(el);
    const h = hpt(el); let lines = 0, extra = 0;
    let lastw = null;
    if (u.kind === 'text') {
      const pe = el.querySelector('p'), ph = hpt(pe); lines = Math.round(ph / L.lh); extra = h - lines * L.lh;
      const rg = document.createRange(); rg.selectNodeContents(pe);
      const rs = [...rg.getClientRects()], pb = pe.getBoundingClientRect();
      if (rs.length) { const bot = Math.max(...rs.map(r => r.top));
        lastw = +((Math.max(...rs.filter(r => r.top > bot - 2).map(r => r.right)) - pb.left) / PX).toFixed(0); }
      if (Math.abs(ph - lines * L.lh) > 0.3) warns.push(['line pitch', u.html.slice(0, 50), ph]);
    }
    const tb = el.querySelector('table');
    if (tb && tb.scrollWidth > el.clientWidth + 1) warns.push(['table wider than column', u.html.slice(0, 50), tb.scrollWidth / PX]);
    return {kind: u.kind, html: u.html, h, lines, extra, lastw, idx: units_i++};
  });
  stage.innerHTML = '';
  function layout(mode, render) {
    const pagesEl = document.getElementById('pages'); if (render) pagesEl.innerHTML = '';
    const cols = [], deferred = [], trace = []; let ci = 0, npages = 0;
    function addPage() {
      const first = npages === 0; npages++;
      let page = null;
      if (render) { page = mk('div', 'page'); pagesEl.appendChild(page); page.appendChild(mk('div', 'stamp', DATA.stamp)); }
      const top = first ? L.top + headerH : L.top;
      const caps = first ? [L.colH - headerH - footH, L.colH - headerH] : [L.colH, L.colH];
      if (render && first) {
        const h = mk('div', '', DATA.header); h.id = 'header'; page.appendChild(h);
        const f = mk('div', '', DATA.foot); f.id = 'foot'; f.style.top = (L.top + L.colH - footH) + 'pt'; page.appendChild(f);
      }
      for (let k = 0; k < 2; k++) {
        let el = null;
        if (render) { el = mk('div', 'col'); el.style.left = L.colX[k] + 'pt'; el.style.top = top + 'pt';
                      el.style.height = caps[k] + 'pt'; page.appendChild(el); }
        cols.push({el, cap: caps[k], used: 0});
      }
    }
    const cur = () => { while (ci >= cols.length) addPage(); return cols[ci]; };
    const put = (c, u) => { trace.push([u.idx, ci, +c.used.toFixed(1)]);
                            if (render) c.el.appendChild(mk('div', 'unit', u.html)); c.used += u.h; };
    function putFrag(c, u, from, n) {
      const last = from + n === u.lines;
      trace.push([u.idx, ci, +c.used.toFixed(1), from, n]);
      if (render) {
        const w = mk('div', 'frag'); w.style.height = (n * L.lh) + 'pt'; w.style.overflow = 'hidden';
        if (last && u.extra > 0) w.style.marginBottom = u.extra + 'pt';
        const inner = mk('div', 'unit', u.html); inner.style.marginTop = (-from * L.lh) + 'pt';
        w.appendChild(inner); c.el.appendChild(w);
      }
      c.used += n * L.lh + (last ? u.extra : 0);
    }
    function nextCol() {
      ci++; const c = cur();
      while (deferred.length && (deferred[0].h <= c.cap - c.used + L.eps || c.used === 0)) put(c, deferred.shift());
    }
    for (const u of units) {
      let c = cur();
      if (u.kind === 'float') {
        if (mode === 'float' && deferred.length) { deferred.push(u); continue; }
        if (u.h <= c.cap - c.used + L.eps) { put(c, u); continue; }
        if (mode === 'float') { deferred.push(u); continue; }
        nextCol(); put(cur(), u); continue;
      }
      if (u.kind === 'head') {                       // keep a heading with two lines of what follows
        if (u.h + 2 * L.lh > c.cap - c.used + L.eps) nextCol();
        put(cur(), u); continue;
      }
      let done = 0;
      for (;;) {
        c = cur();
        const room = c.cap - c.used, rem = u.lines - done;
        if (rem * L.lh + u.extra <= room + L.eps) { putFrag(c, u, done, rem); break; }
        const fit = Math.min(rem, Math.floor((room + L.eps) / L.lh));
        if (fit >= 1) { putFrag(c, u, done, fit); done += fit; }
        nextCol();
      }
    }
    while (deferred.length) nextCol();
    const last = cols[ci];
    return {pages: npages, columns: +(ci + last.used / last.cap).toFixed(3),
            fill: cols.map(c => [+c.used.toFixed(1), +c.cap.toFixed(1)]), trace};
  }
  const mode = new URLSearchParams(location.search).get('render') || 'float';
  const other = mode === 'float' ? 'strict' : 'float';
  const stats = {headerH: +headerH.toFixed(1), headerParts, footH: +footH.toFixed(1), warns};
  stats[other] = layout(other, false);
  stats[mode] = layout(mode, true);
  stats.rendered = mode;
  stats.paras = units.filter(u => u.kind === 'text').map(u => [u.idx, u.lines, u.lastw]);
  window.__stats = stats;
}
"""


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def wd(port, method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def write_paste_html(md_path, out_html):
    """Single-column HTML of the paper for copying into the HWP template (tables paste as tables)."""
    header, foot, units, figs = parse(md_path.read_text(), md_path.parent, reserve=False)
    body = "".join(u["html"] for u in units)
    for src in figs:   # point images at the real figure files
        body = body.replace(f'src="{src}"', f'src="{(md_path.parent / src).resolve().as_uri()}"')
    out_html.write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>PAPER_IEMEK (붙여넣기용)</title>'
        "<style>body{max-width:720px;margin:24px auto;padding:0 16px;font-family:'Noto Serif CJK KR',serif;"
        "font-size:15px;line-height:1.6;background:#fff;color:#000} table{border-collapse:collapse;margin:8px 0}"
        "th,td{border:1px solid #000;padding:2px 6px} .cap,.title,.etitle,.auth{text-align:center}"
        ".title{font-size:22px;font-weight:bold} .etitle{font-size:18px} .note{font-size:13px}"
        "img{width:254px} .rule{border-top:1px solid #000;width:80px;margin-top:16px}</style></head><body>"
        + header + body + ("<hr>" + foot if foot else "") + "</body></html>")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--paste":
        write_paste_html(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
        print("wrote", sys.argv[3])
        return
    md_path, out_pdf = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    mode = sys.argv[3] if len(sys.argv) > 3 else "float"
    build = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(tempfile.mkdtemp(prefix="paper_layout_"))
    root = md_path.parent
    header, foot, units, figs = parse(md_path.read_text(), root)
    build.mkdir(parents=True, exist_ok=True)
    for src in figs:
        (build / src).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / src, build / src)
    data = {"header": header, "foot": foot, "units": units,
            "stamp": "레이아웃 미리보기 — 대한임베디드공학회 양식 기하(B5, 2단 191 pt, 본문 8 pt·줄 간격 160%)에 "
                     "대체 글꼴(Noto CJK)로 조판. 실제 HWP 조판과 다를 수 있음."}
    (build / "index.html").write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>PAPER_IEMEK layout</title>'
        f"<style>{CSS}</style></head><body><div id=\"hstage\"></div><div id=\"stage\"></div><div id=\"pages\"></div>"
        f"<script>const DATA = {json.dumps(data, ensure_ascii=False)};</script><script>{JS}</script></body></html>")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(build)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); gport = s.getsockname()[1]; s.close()
    snap = Path("/snap/firefox/current/usr/lib/firefox")
    gecko = os.environ.get("GECKODRIVER") or (str(snap / "geckodriver") if (snap / "geckodriver").exists()
                                              else shutil.which("geckodriver"))
    firefox = os.environ.get("FIREFOX_BIN") or (str(snap / "firefox") if (snap / "firefox").exists() else None)
    env = dict(os.environ, TMPDIR=str(build), MOZ_HEADLESS="1")
    gd = subprocess.Popen([gecko, "--port", str(gport)], stdout=subprocess.DEVNULL,
                          stderr=open(build / "geckodriver.log", "w"), env=env)
    sid = None
    try:
        for _ in range(200):
            try:
                wd(gport, "GET", "/status")
                break
            except Exception:
                time.sleep(0.1)
        sid = wd(gport, "POST", "/session", {"capabilities": {"alwaysMatch": {
            "browserName": "firefox", "moz:firefoxOptions": dict(
                {"args": ["-headless"]}, **({"binary": firefox} if firefox else {}))}}})["value"]["sessionId"]
        wd(gport, "POST", f"/session/{sid}/url",
           {"url": f"http://127.0.0.1:{httpd.server_address[1]}/index.html?render={mode}"})
        stats = None
        for _ in range(600):
            stats = wd(gport, "POST", f"/session/{sid}/execute/sync",
                       {"script": "return window.__stats || null;", "args": []})["value"]
            if stats:
                break
            time.sleep(0.1)
        if not stats:
            sys.exit("layout did not finish")
        pdf = wd(gport, "POST", f"/session/{sid}/print", {
            "page": {"width": PAGE_W_PT / 72 * 2.54, "height": PAGE_H_PT / 72 * 2.54},
            "margin": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "background": True, "shrinkToFit": False, "scale": 1.0})["value"]
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(base64.b64decode(pdf))
    finally:
        if sid:
            try:
                wd(gport, "DELETE", f"/session/{sid}")
            except Exception:
                pass
        gd.terminate()
        httpd.shutdown()
    for m in ("float", "strict"):
        r = stats[m]
        print(f"{m:6s}: pages {r['pages']}, columns used {r['columns']:.2f} -> {r['columns'] / 2:.2f} pages")
    print(f"header {stats['headerH']} pt, footnote {stats['footH']} pt; rendered: {stats['rendered']} -> {out_pdf}")
    print("header parts (class, pt):", stats["headerParts"])
    for w in stats["warns"]:
        print("WARN", w)
    print("column fill (used/cap pt):", stats[stats["rendered"]]["fill"])
    if "--paras" in sys.argv:
        plain = [re.sub(r"<[^>]+>", "", u["html"])[:30] for u in units]
        for idx, n, lw in stats["paras"]:
            print(f"  #{idx:2d} {n:2d} lines, last line {lw}/191 pt | {plain[idx]}")
    if "--trace" in sys.argv:
        plain = [re.sub(r"<[^>]+>", "", u["html"])[:28] for u in units]
        for m in ("float", "strict"):
            print(f"-- {m} placement: unit, column, y(pt), [from-line, lines]")
            for e in stats[m]["trace"]:
                print(f"  #{e[0]:2d} {units[e[0]]['kind']:5s} col {e[1]} y {e[2]:6.1f} {e[3:] if len(e) > 3 else ''} {plain[e[0]]}")


if __name__ == "__main__":
    main()
