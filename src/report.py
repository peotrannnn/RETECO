"""Build a single local web page that compares every system measured so far.

    python src/report.py --results notebooks/results --open

It reads `results/index.json` and `results/scores/*.json`, bakes the numbers
into one self-contained HTML file, and writes `results/report.html`. Opening
that file needs no server and no network: the data is inside it.

Why a generated file rather than a small web server: a page served over
`file://` cannot read sibling JSON files, and asking a teammate to keep a
server running to look at a table is a worse deal than handing them one file
they can also email.

The palette matches the notebooks, so a figure copied out of a notebook and a
bar on this page mean the same colour. It has been checked for colour-vision
deficiency; the two hues used together on one chart are the green/orange pair,
and every bar also carries its number, so colour never carries meaning alone.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "teal": "#1baf7a",
    "amber": "#eda100", "pink": "#e87ba4", "violet": "#4a3aa7",
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_soft": "#52514e",
    "ink_muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
}


def collect(results_dir):
    results = Path(results_dir)
    index_path = results / "index.json"
    index = (json.loads(index_path.read_text(encoding="utf-8"))
             if index_path.exists() else {})

    systems = []
    for path in sorted((results / "scores").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = index.get(record["hash"], {})
        systems.append({
            "hash": record["hash"],
            "name": record.get("name", "unnamed"),
            "split": record.get("split", "train"),
            "macro": record.get("macro", {}),
            "fit": record.get("fit", {}).get("ndcg"),
            "check": record.get("check", {}).get("ndcg"),
            "n_topics": record.get("n_topics", 0),
            "n_gold": record.get("n_gold_queries", 0),
            "per_domain": {d: s.get("ndcg") for d, s
                           in record.get("per_domain", {}).items()},
            "per_domain_full": record.get("per_domain", {}),
            "score_version": record.get("score_version"),
            "system": entry.get("system", {}),
            "note": entry.get("system", {}).get("note", ""),
            "when": entry.get("when", ""),
            "aliases": entry.get("aliases", []),
        })
    systems.sort(key=lambda s: -(s["macro"].get("ndcg") or 0))
    return systems


CSS = """
:root {
  color-scheme: light;
  --blue:%(blue)s; --orange:%(orange)s; --teal:%(teal)s; --amber:%(amber)s;
  --violet:%(violet)s;
  --surface:%(surface)s; --ink:%(ink)s; --soft:%(ink_soft)s;
  --muted:%(ink_muted)s; --grid:%(grid)s; --axis:%(axis)s;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  padding: 28px 16px 80px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 34px 0 10px; font-weight: 650; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 22px; }
.flag {
  border-left: 3px solid var(--orange); background: #fdf3ee;
  padding: 10px 14px; margin: 0 0 20px; font-size: 13px; border-radius: 0 6px 6px 0;
}
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%%; min-width: 760px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 9px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th {
  font-weight: 600; font-size: 11.5px; color: var(--soft); cursor: pointer;
  text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;
  border-bottom: 1px solid var(--axis); user-select: none;
}
th:hover { color: var(--ink); }
th.sorted::after { content: " \\2193"; color: var(--blue); }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #f4f3ef; }
tbody tr.on { background: #eaf1fb; }
tbody tr.on td:first-child { box-shadow: inset 3px 0 0 var(--blue); }
.name { font-weight: 550; }
.split { font-size: 10px; letter-spacing: 0.05em; text-transform: uppercase;
         padding: 1px 6px; border-radius: 4px; margin-left: 8px;
         background: #f0eee7; color: var(--soft); vertical-align: 1px; }
.split.dev { background: #efe7fb; color: var(--violet); font-weight: 600; }
.hash { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11.5px; color: var(--muted); }
.best { font-weight: 700; color: var(--blue); }
.warn { color: var(--orange); font-weight: 600; }
.chart { margin: 6px 0 0; }
.row { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 10px; margin-bottom: 2px; }
.row .lab { font-size: 12px; color: var(--soft); text-align: right; white-space: nowrap; }
.track { position: relative; height: 19px; }
.bar { position: absolute; top: 0; height: 19px; border-radius: 0 4px 4px 0; }
.bar.neg { border-radius: 4px 0 0 4px; }
.val { position: absolute; top: 0; height: 19px; line-height: 19px; font-size: 11.5px; color: var(--soft); white-space: nowrap; }
.zero { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--axis); }
.panel { border: 1px solid var(--grid); border-radius: 8px; padding: 18px; margin-top: 10px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
@media (max-width: 820px) { .cols { grid-template-columns: 1fr; } body { padding: 20px 16px 60px; } }
pre { background: #f4f3ef; padding: 12px; border-radius: 6px; overflow-x: auto;
      font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; margin: 0; }
select { font: inherit; padding: 5px 8px; border: 1px solid var(--axis);
         border-radius: 6px; background: #fff; color: var(--ink); }
.pickers { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 4px; }
.key { display: flex; gap: 18px; font-size: 12px; color: var(--soft); margin: 10px 0 0; }
.key i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; margin-right: 6px; }
#tip { position: fixed; pointer-events: none; background: var(--ink); color: #fff;
       padding: 5px 9px; border-radius: 5px; font-size: 12px; opacity: 0;
       transition: opacity .1s; z-index: 9; white-space: nowrap; }
footer { color: var(--muted); font-size: 12px; margin-top: 44px;
         border-top: 1px solid var(--grid); padding-top: 14px; }
code { background: #f4f3ef; padding: 1px 5px; border-radius: 4px;
       font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
""" % PALETTE

JS = r"""
const S = DATA.systems, DOMS = DATA.domains;
const P = {blue:"#2a78d6", orange:"#eb6834", teal:"#1baf7a", muted:"#898781"};
const f4 = v => (v === null || v === undefined) ? "–" : v.toFixed(4);
const tip = document.getElementById("tip");

function hover(el, text) {
  el.addEventListener("mousemove", e => {
    tip.textContent = text; tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 220) + "px";
    tip.style.top = (e.clientY - 34) + "px";
  });
  el.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
}

/* ---- table ---- */
const COLS = [
  ["name", "system", s => s.name, false],
  ["hash", "hash", s => s.hash, false],
  ["fit", "fit", s => s.fit, true],
  ["check", "check", s => s.check, true],
  ["ndcg", "nDCG@10", s => s.macro.ndcg, true],
  ["map", "MAP", s => s.macro.map, true],
  ["precision", "P@10", s => s.macro.precision, true],
  ["recall", "R@100", s => s.macro.recall, true],
  ["ceiling", "ceiling", s => s.macro.ceiling, true],
  ["n_topics", "topics", s => s.n_topics, false],
];
let sortKey = "ndcg", chosen = 0;

function best(key, split) {
  const vals = S.filter(s => s.split === split)
    .map(s => COLS.find(c => c[0] === key)[2](s)).filter(v => v != null);
  return vals.length ? Math.max(...vals) : null;
}

function drawTable() {
  const col = COLS.find(c => c[0] === sortKey);
  // train rows first, then dev: a dev score and a train score are measured
  // on different queries and must not read as one ranking.
  const rows = S.map((s, i) => [s, i]).sort((a, b) => {
    if (a[0].split !== b[0].split) return a[0].split === "dev" ? 1 : -1;
    const x = col[2](a[0]), y = col[2](b[0]);
    if (x == null) return 1; if (y == null) return -1;
    return typeof x === "string" ? x.localeCompare(y) : y - x;
  });
  const bests = {};
  for (const sp of new Set(S.map(s => s.split)))
    COLS.forEach(c => { if (c[3]) (bests[sp] = bests[sp] || {})[c[0]] = best(c[0], sp); });

  let h = "<thead><tr>" + COLS.map(c =>
    `<th data-k="${c[0]}" class="${c[0] === sortKey ? "sorted" : ""}">${c[1]}</th>`
  ).join("") + "</tr></thead><tbody>";
  for (const [s, i] of rows) {
    h += `<tr data-i="${i}" class="${i === chosen ? "on" : ""}">`;
    for (const c of COLS) {
      const v = c[2](s);
      let cell;
      if (c[0] === "name") cell = `<span class="name">${s.name}</span>`
        + (s.split === "dev" ? `<span class="split dev">dev</span>` : "");
      else if (c[0] === "hash") cell = `<span class="hash">${s.hash}</span>`;
      else if (c[0] === "n_topics")
        cell = s.n_topics === s.n_gold ? v
             : `<span class="warn">${v}/${s.n_gold}</span>`;
      else cell = `<span class="${c[3] && v != null && v === bests[s.split][c[0]] ? "best" : ""}">${f4(v)}</span>`;
      h += `<td>${cell}</td>`;
    }
    h += "</tr>";
  }
  const t = document.getElementById("tbl");
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("th").forEach(th => th.onclick = () => {
    sortKey = th.dataset.k; drawTable();
  });
  t.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => {
    chosen = +tr.dataset.i;
    document.getElementById("pickA").value = chosen;
    drawTable(); drawOne(); drawDiff();
  });
}

/* ---- bars: one value per domain ---- */
function bars(node, rows, opts) {
  const max = Math.max(...rows.map(r => Math.abs(r.v)), 1e-9);
  const signed = opts.signed;
  node.innerHTML = rows.map(r => {
    const pct = Math.abs(r.v) / max * (signed ? 46 : 88);
    const left = signed ? (r.v >= 0 ? 50 : 50 - pct) : 0;
    const colour = signed ? (r.v >= 0 ? P.teal : P.orange) : P.blue;
    const lab = signed
      ? (r.v >= 0 ? `left:${50 + pct}%;padding-left:7px`
                  : `right:${50 + pct}%;padding-right:7px`)
      : `left:${pct}%;padding-left:7px`;
    return `<div class="row"><div class="lab">${r.k}</div><div class="track">`
      + (signed ? `<div class="zero" style="left:50%"></div>` : "")
      + `<div class="bar${signed && r.v < 0 ? " neg" : ""}" data-t="${r.t}"
           style="left:${left}%;width:${pct}%;background:${colour}"></div>`
      + `<div class="val" style="${lab}">${signed && r.v >= 0 ? "+" : ""}${r.v.toFixed(4)}</div>`
      + `</div></div>`;
  }).join("");
  node.querySelectorAll(".bar").forEach(b => hover(b, b.dataset.t));
}

function drawOne() {
  const s = S[chosen];
  document.getElementById("oneTitle").textContent =
    `Chi ti\u1ebft: ${s.name}`;
  const rows = DOMS.filter(d => s.per_domain[d] != null)
    .map(d => ({k: d, v: s.per_domain[d],
                t: `${d}: nDCG@10 ${f4(s.per_domain[d])}`}))
    .sort((a, b) => b.v - a.v);
  bars(document.getElementById("oneChart"), rows, {signed: false});
  document.getElementById("cfg").textContent =
    JSON.stringify(s.system, null, 1);
  const cmd = `P.submit(SYSTEMS / "${s.name}.json",\n`
    + `         "submissions/${s.name}.txt",\n`
    + `         tag="${s.name}", split="dev", checker=CHECKER)`;
  document.getElementById("cmd").textContent = cmd;
}

function drawDiff() {
  const a = S[+document.getElementById("pickA").value];
  const b = S[+document.getElementById("pickB").value];
  const rows = DOMS.filter(d => a.per_domain[d] != null && b.per_domain[d] != null)
    .map(d => ({k: d, v: a.per_domain[d] - b.per_domain[d],
                t: `${d}: ${a.name} ${f4(a.per_domain[d])} vs `
                 + `${b.name} ${f4(b.per_domain[d])}`}))
    .sort((x, y) => y.v - x.v);
  if (a.split !== b.split) {
    document.getElementById("diffChart").innerHTML = "";
    document.getElementById("diffSum").innerHTML =
      `<b>${a.name}</b> ch\u1ea1y tr\u00ean t\u1eadp <b>${a.split}</b> c\u00f2n `
      + `<b>${b.name}</b> tr\u00ean <b>${b.split}</b>. Hai t\u1eadp c\u00f3 c\u00e2u `
      + `truy v\u1ea5n kh\u00e1c nhau n\u00ean hai \u0111i\u1ec3m n\u00e0y `
      + `kh\u00f4ng so \u0111\u01b0\u1ee3c v\u1edbi nhau.`;
    return;
  }
  bars(document.getElementById("diffChart"), rows, {signed: true});
  const won = rows.filter(r => r.v > 0).length;
  const mean = rows.reduce((t, r) => t + r.v, 0) / (rows.length || 1);
  document.getElementById("diffSum").innerHTML =
    `<b>${a.name}</b> h\u01a1n \u1edf ${won}/${rows.length} nh\u00f3m. `
    + `Trung b\u00ecnh theo nh\u00f3m: <b>${mean >= 0 ? "+" : ""}${mean.toFixed(4)}</b>. `
    + `Ch\u00eanh l\u1ec7ch d\u01b0\u1edbi 0,01 th\u01b0\u1eddng l\u00e0 nhi\u1ec5u; `
    + `ch\u1ea1y <code>P.compare</code> \u0111\u1ec3 bi\u1ebft c\u00f3 th\u1eadt hay kh\u00f4ng.`;
}

function fill(id, initial) {
  const sel = document.getElementById(id);
  sel.innerHTML = S.map((s, i) =>
    `<option value="${i}">${s.name} (${s.hash})</option>`).join("");
  sel.value = initial;
  sel.onchange = () => {
    if (id === "pickA") { chosen = +sel.value; drawTable(); drawOne(); }
    drawDiff();
  };
}

drawTable();
fill("pickA", 0); fill("pickB", Math.min(1, S.length - 1));
drawOne(); drawDiff();
"""

PAGE = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RETECO — so sanh he thong</title>
<style>%(css)s</style></head>
<body><div class="wrap">
<h1>RETECO Track 1a — so sánh hệ thống</h1>
<p class="sub">%(n)s hệ thống · tập <b>%(split)s</b> · sinh lúc %(when)s ·
nguồn <code>%(src)s</code></p>
%(flags)s

<h2>Bảng kết quả</h2>
<p class="sub">Bấm tiêu đề cột để sắp xếp. Bấm một dòng để xem chi tiết bên dưới.
Số in đậm màu xanh là giá trị cao nhất của cột đó.</p>
<div class="scroll"><table id="tbl"></table></div>
<p class="sub" style="margin-top:10px">Cột <b>fit</b> là phần dùng để chọn,
<b>check</b> là phần chỉ đọc sau khi đã chọn. Sắp xếp theo <b>check</b> rồi
chọn theo nó là tự phá mất ý nghĩa của nó.</p>

<h2 id="oneTitle">Chi tiết</h2>
<div class="panel"><div class="cols">
  <div><div class="chart" id="oneChart"></div></div>
  <div>
    <p class="sub" style="margin:0 0 8px">Cấu hình</p>
    <pre id="cfg"></pre>
    <p class="sub" style="margin:14px 0 8px">Lệnh xuất bài nộp (đổi tên file cho khớp)</p>
    <pre id="cmd"></pre>
  </div>
</div></div>

<h2>So sánh hai hệ thống</h2>
<div class="panel">
  <div class="pickers">
    <select id="pickA"></select><span class="sub" style="margin:0">so với</span>
    <select id="pickB"></select>
  </div>
  <p class="sub" id="diffSum" style="margin:12px 0 14px"></p>
  <div class="chart" id="diffChart"></div>
  <div class="key">
    <span><i style="background:%(teal)s"></i>hệ thống bên trái hơn</span>
    <span><i style="background:%(orange)s"></i>hệ thống bên phải hơn</span>
  </div>
</div>

<footer>
Trang này chỉ đọc <code>results/scores/</code> và <code>results/index.json</code>.
Chạy lại <code>python src/report.py</code> sau mỗi lần thêm hệ thống mới.
Mọi con số đều là trung bình theo nhóm, khớp cách chấm của ban tổ chức.
</footer>
</div>
<div id="tip"></div>
<script>const DATA = %(data)s;</script>
<script>%(js)s</script>
</body></html>
"""


def build(results_dir, out_path=None):
    systems = collect(results_dir)
    if not systems:
        raise SystemExit(
            f"No scored systems found under {results_dir}/scores/.\n"
            f"Run a system through the pipeline first (notebook 06b).")

    domains = sorted({d for s in systems for d in s["per_domain"]})
    flags = []

    incomplete = [s for s in systems if s["n_topics"] != s["n_gold"]]
    if incomplete:
        names = ", ".join(f"{s['name']} ({s['n_topics']}/{s['n_gold']})"
                          for s in incomplete)
        flags.append(
            f"<div class='flag'><b>Thiếu câu trả lời:</b> {names}. "
            f"Bộ chấm điểm bỏ qua câu không có kết quả thay vì cho 0 điểm, "
            f"nên điểm của các hệ thống này không so được với hệ thống trả lời "
            f"đủ câu.</div>")

    stale = [s for s in systems if s.get("score_version") != 2]
    if stale:
        names = ", ".join(s["hash"] for s in stale)
        flags.append(
            f"<div class='flag'><b>Điểm cũ:</b> {names} được chấm trước khi có "
            f"P@10, F1 và MAP. Chạy <code>P.score(h, force=True)</code> để "
            f"điền nốt; việc đó đọc lại danh sách đã lưu chứ không truy xuất "
            f"lại.</div>")

    splits = {s["split"] for s in systems}
    html = PAGE % {
        "css": CSS, "js": JS,
        "n": len(systems),
        "split": " + ".join(sorted(splits)),
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "src": str(results_dir),
        "flags": "\n".join(flags),
        "teal": PALETTE["teal"], "orange": PALETTE["orange"],
        "data": json.dumps({"systems": systems, "domains": domains},
                           ensure_ascii=False),
    }

    out = Path(out_path or Path(results_dir) / "report.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(systems)} systems, {len(domains)} domains)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="notebooks/results")
    parser.add_argument("--out", default=None)
    parser.add_argument("--open", action="store_true",
                        help="open the page in the default browser when done")
    args = parser.parse_args()
    out = build(args.results, args.out)
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
