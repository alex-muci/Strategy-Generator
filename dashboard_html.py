"""
dashboard_html.py
-----------------
Renders the signals phase into one self-contained HTML page.

No external requests: no CDN, no font file, no image. The page is a single file
you can open from disk, keep in a folder, mail to yourself, or serve from a
Raspberry Pi with no internet. Charts are inline SVG built here.

Layout order is the order you need it in front of a broker:

  1. the verdict from the research phase -- whether this book is worth trading
     at all, restated every single run so a weak result cannot quietly become
     habit
  2. the numbers that decide position size: equity, gross and net exposure,
     money at risk if every stop fills at once
  3. the trades to send
  4. the orders to have working on the next bar, with their order TYPE
  5. what you are holding, and how much room is left before each stop
  6. context: the out-of-sample curve, per-slot detail, the full search, and the
     overfitting diagnostics

`render_dashboard(..., full_document=False)` emits just the title, style and
body, for embedding somewhere that supplies its own document shell.
"""

from __future__ import annotations

import html
import json
import numpy as np
import pandas as pd

from live import utcnow
from pipeline import sizing_text

# Palette: the data-viz reference instance, validated in both modes with
# scripts/validate_palette.js (2 categorical slots, all checks PASS).
#   series 1 blue   #2a78d6 light / #3987e5 dark   -- the strategy
#   series 2 orange #eb6834 light / #d95926 dark   -- buy and hold
STYLE = """
:root {
  color-scheme: light;
  --page:           #f9f9f7;
  --surface:        #fcfcfb;
  --ink:            #0b0b0b;
  --ink-2:          #52514e;
  --muted:          #898781;
  --grid:           #e1e0d9;
  --axis:           #c3c2b7;
  --hairline:       rgba(11, 11, 11, 0.10);
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  --good:           #0ca30c;
  --warning:        #fab219;
  --serious:        #ec835a;
  --critical:       #d03b3b;
  --good-ink:       #006300;
  --critical-ink:   #a82b2b;
  --wash:           rgba(42, 120, 214, 0.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:         #0d0d0d;
    --surface:      #1a1a19;
    --ink:          #ffffff;
    --ink-2:        #c3c2b7;
    --muted:        #898781;
    --grid:         #2c2c2a;
    --axis:         #383835;
    --hairline:     rgba(255, 255, 255, 0.10);
    --series-1:     #3987e5;
    --series-2:     #d95926;
    --good-ink:     #0ca30c;
    --critical-ink: #e66767;
    --wash:         rgba(57, 135, 229, 0.14);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:         #0d0d0d;
  --surface:      #1a1a19;
  --ink:          #ffffff;
  --ink-2:        #c3c2b7;
  --muted:        #898781;
  --grid:         #2c2c2a;
  --axis:         #383835;
  --hairline:     rgba(255, 255, 255, 0.10);
  --series-1:     #3987e5;
  --series-2:     #d95926;
  --good-ink:     #0ca30c;
  --critical-ink: #e66767;
  --wash:         rgba(57, 135, 229, 0.14);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 24px; padding-inline: 16px; }
h1 { font-size: 21px; margin: 0 0 2px; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 0 0 12px; letter-spacing: -0.005em; }
h3 { font-size: 13px; margin: 18px 0 8px; color: var(--ink-2); font-weight: 600; }
p  { margin: 0 0 10px; }
.sub { color: var(--ink-2); font-size: 13px; margin-bottom: 20px; }
.card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: 10px;
  padding: 18px;
  margin-bottom: 16px;
}
.muted { color: var(--muted); }
.ink2  { color: var(--ink-2); }
.num   { font-variant-numeric: tabular-nums; }
.pos   { color: var(--good-ink); }
.neg   { color: var(--critical-ink); }
.mono  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }

/* ---- verdict banner: colour never carries the meaning alone ---- */
.verdict { display: flex; gap: 12px; align-items: flex-start;
           border-left: 4px solid var(--edge); border-radius: 10px;
           background: var(--surface); border-top: 1px solid var(--hairline);
           border-right: 1px solid var(--hairline); border-bottom: 1px solid var(--hairline);
           padding: 16px 18px; margin-bottom: 16px; }
.verdict.critical { --edge: var(--critical); }
.verdict.warning  { --edge: var(--warning); }
.verdict.good     { --edge: var(--good); }
.verdict .mark { font-size: 18px; line-height: 1.3; flex: 0 0 auto; }
.verdict .label { font-weight: 700; letter-spacing: 0.04em; font-size: 11px;
                  text-transform: uppercase; color: var(--ink-2); }
.verdict ul { margin: 10px 0 0; padding-left: 20px; color: var(--ink-2); }
.verdict li { margin-bottom: 5px; }

/* ---- stat tiles ---- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
         margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px;
        padding: 14px 16px; }
.tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
           color: var(--muted); margin-bottom: 6px; }
.tile .v { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
.tile .n { font-size: 12px; color: var(--ink-2); margin-top: 4px; }

/* ---- tables ---- */
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 8px 10px; white-space: nowrap;
         border-bottom: 1px solid var(--grid); }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
     color: var(--muted); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--wash); }
td.l, th.l { text-align: left; }
.tag { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
       padding: 2px 7px; border-radius: 5px; border: 1px solid var(--hairline); }
.tag.buy  { color: var(--good-ink); }
.tag.sell { color: var(--critical-ink); }
.tag.hold { color: var(--muted); }
.empty { color: var(--muted); padding: 10px 0; }
td.wrap { white-space: normal; min-width: 240px; max-width: 430px; }

/* ---- room-to-stop meter ---- */
.meter { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
.meter .track { width: 84px; height: 7px; border-radius: 4px; background: var(--grid);
                overflow: hidden; flex: 0 0 auto; }
.meter .fill { height: 100%; border-radius: 4px; background: var(--series-1); }
.meter.warning .fill  { background: var(--warning); }
.meter.critical .fill { background: var(--critical); }
.meter .pct { font-variant-numeric: tabular-nums; font-size: 12px; min-width: 34px; }

/* ---- chart ----
   Only the lines are SVG (stretched to the box, with non-scaling strokes). Every
   label is HTML in a gutter beside the plot, so nothing shrinks to 4px on a phone
   and the axis labels cannot collide with the endpoint labels. */
.chart { position: relative; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; font-size: 12px;
          color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 auto; }
.plot { position: relative; height: clamp(190px, 23vw, 250px);
        margin: 0 52px 0 42px; touch-action: pan-y; }
.plot svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.ylab { position: absolute; left: -42px; width: 36px; text-align: right; font-size: 10px;
        color: var(--muted); transform: translateY(-50%); font-variant-numeric: tabular-nums; }
.endlab { position: absolute; right: -52px; width: 48px; font-size: 11px; font-weight: 600;
          transform: translateY(-50%); font-variant-numeric: tabular-nums; }
.dot { position: absolute; width: 9px; height: 9px; border-radius: 50%; opacity: 0;
       transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--surface); }
.xaxis { display: flex; justify-content: space-between; margin: 7px 52px 0 42px;
         font-size: 10px; color: var(--muted); }
.tip { position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
       background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
       padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12);
       font-variant-numeric: tabular-nums; z-index: 5; min-width: 136px; }
.tip .d { color: var(--muted); margin-bottom: 4px; }
.tip .r { display: flex; justify-content: space-between; gap: 12px; }

details { margin-top: 10px; }
summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
summary:hover { color: var(--ink); }
footer { color: var(--muted); font-size: 12px; margin-top: 8px; }
footer code { font-size: 11px; }
"""

_STATUS_MARK = {"critical": "✖", "warning": "⚠", "good": "✔"}


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------

def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _n(x, dp=2, dash="–") -> str:
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _e(x)
    return dash if not np.isfinite(v) else f"{v:,.{dp}f}"


def _pct(x, dp=0, dash="–", sign=False) -> str:
    if x is None:
        return dash
    v = float(x)
    if not np.isfinite(v):
        return dash
    return f"{v:{'+' if sign else ''}.{dp}%}"


def _money(x, dp=0) -> str:
    if x is None or not np.isfinite(float(x)):
        return "–"
    v = float(x)
    return f"{'-' if v < 0 else ''}${abs(v):,.{dp}f}"


def _signed_cls(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _ts(x) -> str:
    t = pd.Timestamp(x)
    return t.strftime("%Y-%m-%d %H:%M") if (t.hour or t.minute) else t.strftime("%Y-%m-%d")


def _table(headers, rows, left_cols=(0,), empty="nothing to show") -> str:
    """headers: list of str. rows: list of list of (text, css_class) or str."""
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    th = "".join(f'<th class="l">{_e(h)}</th>' if i in left_cols else f"<th>{_e(h)}</th>"
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = []
        for i, cell in enumerate(r):
            text, cls = cell if isinstance(cell, tuple) else (cell, "")
            cls = (cls + (" l" if i in left_cols else "")).strip()
            tds.append(f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return ('<div class="scroll"><table><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def _side_tag(side, kind="") -> str:
    if side == 1:
        return '<span class="tag buy">BUY</span>'
    if side == -1:
        return '<span class="tag sell">SELL</span>'
    return f'<span class="tag hold">{_e(kind or "FLAT")}</span>'


# --------------------------------------------------------------------------
# the equity chart: two series, one axis, growth of 1.0
# --------------------------------------------------------------------------

def _equity_chart(curves: dict) -> str:
    dates, a, b = curves.get("dates", []), curves.get("strategy", []), curves.get("buy_hold", [])
    if len(dates) < 3:
        return ('<p class="empty">No out-of-sample curve yet \u2013 the research phase had too '
                'little history to walk the selection forward.</p>')

    VB_W, VB_H = 1000.0, 300.0            # the SVG box IS the plot area
    lo, hi = min(min(a), min(b)), max(max(a), max(b))
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.08, hi + span * 0.08
    n = len(dates)

    def fy(v):                            # 0 at the top, 1 at the bottom
        return 1 - (v - lo) / (hi - lo)

    def path(vals):
        return "M" + " L".join(f"{VB_W * i / (n - 1):.1f},{fy(v) * VB_H:.1f}"
                               for i, v in enumerate(vals))

    ticks, step = [], _nice_step(hi - lo)
    t = np.ceil(lo / step) * step
    while t <= hi + 1e-9:
        ticks.append(round(float(t), 6))
        t += step
    grid = "".join(
        f'<line x1="0" x2="{VB_W}" y1="{fy(v) * VB_H:.1f}" y2="{fy(v) * VB_H:.1f}" '
        f'stroke="var(--grid)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
        for v in ticks)
    ylabs = "".join(f'<div class="ylab" style="top:{fy(v) * 100:.2f}%">{v:.2f}x</div>'
                    for v in ticks)
    # break-even reads differently from the grid
    base = (f'<line x1="0" x2="{VB_W}" y1="{fy(1.0) * VB_H:.1f}" y2="{fy(1.0) * VB_H:.1f}" '
            f'stroke="var(--axis)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
            if lo < 1.0 < hi else "")

    # direct endpoint labels, nudged apart if the two series finish close together
    ta, tb = fy(a[-1]) * 100, fy(b[-1]) * 100
    if abs(ta - tb) < 7:
        mid = (ta + tb) / 2
        ta, tb = (mid - 3.5, mid + 3.5) if ta <= tb else (mid + 3.5, mid - 3.5)
    ends = (f'<div class="endlab" style="top:{ta:.2f}%;color:var(--series-1)">'
            f'{a[-1]:.2f}x</div>'
            f'<div class="endlab" style="top:{tb:.2f}%;color:var(--series-2)">'
            f'{b[-1]:.2f}x</div>')

    rows = json.dumps([[dates[i][:10], a[i], b[i]] for i in range(n)])
    return f"""
<div class="legend">
  <span><i class="swatch" style="background:var(--series-1)"></i>This book, out-of-sample
    (nested walk-forward)</span>
  <span><i class="swatch" style="background:var(--series-2)"></i>Holding the same ETFs,
    equal weight</span>
</div>
<div class="chart" id="eqchart">
  <div class="plot" id="eqplot">
    <svg viewBox="0 0 {VB_W:.0f} {VB_H:.0f}" preserveAspectRatio="none" role="img"
         aria-label="Growth of 1.0 out-of-sample, this book against holding the ETFs">
      {grid}{base}
      <path d="{path(b)}" fill="none" stroke="var(--series-2)" stroke-width="2"
            stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      <path d="{path(a)}" fill="none" stroke="var(--series-1)" stroke-width="2"
            stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      <line id="eqcross" x1="0" x2="0" y1="0" y2="{VB_H:.0f}" stroke="var(--axis)"
            stroke-width="1" opacity="0" vector-effect="non-scaling-stroke"/>
    </svg>
    {ylabs}{ends}
    <div class="dot" id="eqdot1" style="background:var(--series-1)"></div>
    <div class="dot" id="eqdot2" style="background:var(--series-2)"></div>
    <div class="tip" id="eqtip"></div>
  </div>
  <div class="xaxis"><span>{_e(dates[0][:7])}</span><span>{_e(dates[n // 2][:7])}</span>
    <span>{_e(dates[-1][:7])}</span></div>
</div>
<script>
(function () {{
  var rows = {rows}, lo = {lo:.6f}, hi = {hi:.6f}, VB_W = {VB_W:.0f};
  var plot = document.getElementById('eqplot'), tip = document.getElementById('eqtip'),
      cross = document.getElementById('eqcross'), d1 = document.getElementById('eqdot1'),
      d2 = document.getElementById('eqdot2');
  function fy(v) {{ return 1 - (v - lo) / (hi - lo); }}
  function show(ev) {{
    var r = plot.getBoundingClientRect();
    var cx = ('touches' in ev ? ev.touches[0].clientX : ev.clientX) - r.left;
    var frac = Math.max(0, Math.min(1, cx / r.width));
    var i = Math.round(frac * (rows.length - 1)), row = rows[i], px = i / (rows.length - 1);
    cross.setAttribute('x1', px * VB_W); cross.setAttribute('x2', px * VB_W);
    cross.setAttribute('opacity', 1);
    d1.style.left = d2.style.left = (px * 100) + '%';
    d1.style.top = (fy(row[1]) * 100) + '%'; d2.style.top = (fy(row[2]) * 100) + '%';
    d1.style.opacity = d2.style.opacity = 1;
    tip.innerHTML = '<div class="d">' + row[0] + '</div>'
      + '<div class="r"><span>This book</span><b>' + row[1].toFixed(3) + 'x</b></div>'
      + '<div class="r"><span>Buy &amp; hold</span><b>' + row[2].toFixed(3) + 'x</b></div>';
    tip.style.opacity = 1;
    var left = px * r.width + 14;
    if (left + tip.offsetWidth > r.width) left = px * r.width - tip.offsetWidth - 14;
    tip.style.left = Math.max(-38, left) + 'px';
    tip.style.top = Math.max(0, fy(row[1]) * r.height - 12) + 'px';
  }}
  function hide() {{
    tip.style.opacity = 0; cross.setAttribute('opacity', 0);
    d1.style.opacity = d2.style.opacity = 0;
  }}
  plot.addEventListener('mousemove', show);
  plot.addEventListener('touchmove', show);
  plot.addEventListener('mouseleave', hide);
  plot.addEventListener('touchend', hide);
}})();
</script>
"""


def _nice_step(span: float) -> float:
    raw = span / 4.0
    if raw <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def _curve_table(curves: dict) -> str:
    dates, a, b = curves.get("dates", []), curves.get("strategy", []), curves.get("buy_hold", [])
    if not dates:
        return ""
    step = max(1, len(dates) // 24)
    rows = [[_e(dates[i][:10]), f"{a[i]:.3f}x", f"{b[i]:.3f}x"] for i in range(0, len(dates), step)]
    if rows and rows[-1][0] != dates[-1][:10]:
        rows.append([_e(dates[-1][:10]), f"{a[-1]:.3f}x", f"{b[-1]:.3f}x"])
    return ("<details><summary>Show the curve as numbers</summary>"
            + _table(["date", "this book", "buy & hold"], rows) + "</details>")


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _room_to_stop(st: dict) -> tuple[float | None, str, str]:
    """Fraction of the ORIGINAL stop distance still between price and stop.

    1.0 means the stop is as far away as when the trade opened; 0 means it is
    about to fill. This is the number that tells you whether a position is
    hanging by a thread, which the raw stop price alone does not.
    """
    stop = next((o["level"] for o in st["exit_orders"] if o["kind"] == "stop"), None)
    if stop is None or st["position"] is None or not np.isfinite(st.get("atr", np.nan)):
        return None, "", "–"
    price, side = st["last_close"], st["position"]
    room = (price - stop) if side == 1 else (stop - price)
    full = st["atr"] * float(st["atr_mult_stop"])
    frac = float(np.clip(room / full, 0.0, 1.0)) if full > 0 else 0.0
    level = "critical" if frac < 0.15 else ("warning" if frac < 0.33 else "")
    note = {"critical": " ✖ very close", "warning": " ⚠ close", "": ""}[level]
    return frac, level, f"{room / price:.1%} away{note}"


def _positions_section(states: list) -> str:
    rows = []
    for st in states:
        if not st["position"]:
            continue
        frac, level, note = _room_to_stop(st)
        stop = next((o["level"] for o in st["exit_orders"] if o["kind"] == "stop"), None)
        meter = "–" if frac is None else (
            f'<div class="meter {level}"><div class="track"><div class="fill" '
            f'style="width:{frac * 100:.0f}%"></div></div>'
            f'<span class="pct">{frac:.0%}</span></div>')
        other = [o for o in st["exit_orders"] if o["kind"] != "stop"]
        rows.append([
            _e(st["asset"]),
            _side_tag(st["position"]),
            f'<span class="num">{_n(st["shares"], 1)}</span>',
            f'<span class="num">{_n(st["entry_price"])}</span>',
            f'<span class="num">{_n(st["last_close"])}</span>',
            f'<span class="num">{_n(stop)}</span>',
            meter,
            f'<span class="num {_signed_cls(st["unrealized"])}">'
            f'{_money(st["unrealized"])}</span>',
            f'<span class="num">{st["bars_held"]}</span>',
            f'<span class="muted">{_e(other[0]["note"]) if other else "hard stop only"}</span>',
        ])
    tbl = _table(
        ["asset", "side", "shares", "entry", "last", "stop", "room to stop", "unrealized",
         "bars", "other exit"],
        rows, left_cols=(0, 1, 9), empty="flat – no slot holds a position right now")
    legend = ("" if not rows else
              '<p class="muted" style="margin-top:10px">"Room to stop" is how much of the '
              'original stop distance is still between the last close and the stop: 100% means '
              'as far away as at entry, 0% means about to fill.</p>')
    return tbl + legend


def _orders_section(states: list) -> str:
    rows = []
    for st in states:
        for o in st["entry_orders"]:
            level = o.get("level")
            price = "at the open" if level is None else _n(level)
            extra = ""
            if o["kind"] == "stop_then_limit":
                price = f'{_n(level)} → {_n(o["limit"])}'
                extra = " (two-step)"
            rows.append([
                _e(st["asset"]), _side_tag(o["side"]),
                f'<span class="mono">{_e(o["kind"].replace("_", " "))}{extra}</span>',
                f'<span class="num">{price}</span>',
                f'<span class="num">{_n(o["shares"], 1)}</span>',
                f'<span class="num">{_money(o["shares"] * (level or st["last_close"]))}</span>',
                f'<span class="muted">{_e(o["note"])}</span>',
            ])
    blocked = [(st["asset"], st["template"], st["blocked_by"]) for st in states
               if not st["position"] and not st["entry_orders"] and st["blocked_by"]]
    tbl = _table(["asset", "side", "order type", "level", "shares", "notional", "why"],
                 rows, left_cols=(0, 1, 2, 6),
                 empty="no entry order to work on the next bar")
    if blocked:
        items = "".join(f'<li><b>{_e(a)}</b> <span class="muted">{_e(t)}</span>: '
                        f'{_e("; ".join(b))}</li>' for a, t, b in blocked[:8])
        more = (f'<li class="muted">…and {len(blocked) - 8} more</li>'
                if len(blocked) > 8 else "")
        tbl += (f'<details><summary>{len(blocked)} flat slot(s) with no order, and why</summary>'
                f'<ul class="ink2">{items}{more}</ul></details>')
    return tbl


def _trades_section(trades: pd.DataFrame, run: dict) -> str:
    rows = []
    for asset, r in trades.iterrows():
        act = str(r["action"])
        rows.append([
            _e(asset),
            ('<span class="tag hold">HOLD</span>' if act == "hold"
             else _side_tag(1 if act == "BUY" else -1)),
            f'<span class="num">{_n(r["held"], 1)}</span>',
            f'<span class="num">{_n(r["target"], 1)}</span>',
            f'<span class="num">{_n(r["delta"], 1)}</span>',
            f'<span class="num">{"–" if act == "hold" else _n(r["order_shares"], 1)}</span>',
            f'<span class="num">{"–" if act == "hold" else _money(r["order_notional"])}</span>',
        ])
    todo = int((trades["action"] != "hold").sum()) if len(trades) else 0
    head = (f'<p class="ink2">{todo} order(s) to send. "Held" comes from '
            f'<b>{_e(run["holdings_source"])}</b>'
            + ("" if run["holdings_source"] == "holdings.json" else
               ', which <b>assumes every order from the last run was filled</b>. Put your '
               'real broker positions in <code>state/holdings.json</code> (e.g. '
               '<code>{"SPY": 120, "TLT": -50}</code>) to diff against reality instead')
            + ".</p>")
    return head + _table(["asset", "action", "held", "target", "delta", "order", "notional"],
                         rows, left_cols=(0, 1), empty="the book already matches the target")


def _slots_section(states: list) -> str:
    rows = []
    for st in states:
        r = st.get("research", {})
        params = ", ".join(f"{k}={v}" for k, v in (st.get("params") or {}).items()) or "–"
        refit = "due now" if st.get("refit_due") else f'{st.get("bars_since_refit", "–")} bars ago'
        rows.append([
            _e(st["asset"]),
            f'<span class="mono">{_e(st["template"])}</span>',
            _pct(st["weight"]),
            f'<span class="num">{_n(r.get("oos_sharpe"))}</span>',
            f'<span class="num">{_n(r.get("cpcv_mean"))}</span>',
            f'<span class="num">{_pct(r.get("cpcv_prob_negative"))}</span>',
            f'<span class="num">{_n(r.get("bootstrap_p"), 3)}</span>',
            (f'<span class="mono muted">{_e(params)}</span>', "wrap"),
            f'<span class="muted">{_e(refit)}</span>',
        ])
    return _table(["asset", "template", "weight", "OOS Sharpe", "CPCV mean", "P(CPCV<0)",
                   "boot p", "current parameters", "refitted"],
                  rows, left_cols=(0, 1, 7, 8), empty="no slots")


def _diagnostics_section(spec: dict) -> str:
    d, c = spec["diagnostics"], spec["config"]
    items = [
        ("Nested walk-forward Sharpe", _n(d["nested_sharpe"]),
         f'the honest number: out-of-sample for the parameters AND the slot selection, '
         f'over {d["n_reselections"]} re-selections'),
        ("Static-selection Sharpe", _n(d["static_sharpe"]),
         "biased upward – that selection saw the whole history"),
        ("Probability of Backtest Overfitting", _n(d["pbo_trials"]),
         f'CSCV over all {d["n_trials"]:,} parameter trials; near 0.5 means the in-sample '
         f'winner is a coin toss out-of-sample'),
        ("White's Reality Check p", _n(d["reality_check_p"], 3),
         f'for {_e(d["reality_check_best"])}; above 0.05–0.15 is consistent with data '
         f'snooping'),
        ("Deflated Sharpe Ratio", _n(d["dsr_best"]),
         f'best slot Sharpe {_n(d["best_oos_sharpe"])} vs expected max of {d["n_eff"]} '
         f'independent noise trials {_n(d["sr_star_annual"])}'),
        ("Effective independent trials", f'{d["n_eff"]} of {d["n_slots"]}',
         "correlation clusters – hundreds of templates are not hundreds of bets"),
        ("Minimum backtest length", f'{_n(d["min_btl_years"], 1)} y',
         f'needed for {d["n_eff"]} trials at this Sharpe; {_n(d["years_available"], 1)} y '
         f'available'),
        ("Nested max drawdown", _pct(d["nested_max_drawdown"], 1),
         "worst peak-to-trough of the honest curve"),
    ]
    rows = [[_e(k), f'<span class="num">{v}</span>', f'<span class="muted">{n}</span>']
            for k, v, n in items]
    cfg = (f'Research run {_e(spec["created"])} on {_e(", ".join(spec["assets"]))}, '
           f'{_e(c["interval"])} bars from {_e(c["start"])}, family '
           f'<b>{_e(c["family"])}</b>. Walk-forward train={c["train_bars"]} test='
           f'{c["test_bars"]} {"anchored" if c["anchored"] else "rolling"}, '
           f'{_e(c["selection"])} parameter selection, {_n(c["cost_bps"], 1)} bps/side costs, '
           f'{_e(sizing_text(c))} (per slot), '
           f'{_e(c["weighting"])} weights.')
    return f'<p class="ink2">{cfg}</p>' + _table(["measure", "value", "what it means"],
                                                 rows, left_cols=(0, 2))


def _universe_section(spec: dict) -> str:
    uni = spec.get("universe") or []
    if not uni:
        return ""
    rows = [[
        _e(r["slot"]),
        f'<span class="num">{_n(r["oos_sharpe"])}</span>',
        f'<span class="num">{_n(r["cpcv_mean"])}</span>',
        f'<span class="num">{_pct(r["cpcv_prob_negative"])}</span>',
        "yes" if r["pardo_pass"] else "no",
        "<b>selected</b>" if r["selected"] else '<span class="muted">rejected</span>',
    ] for r in uni[:60]]
    more = (f'<p class="muted">Showing the top 60 of {len(uni)} slots.</p>'
            if len(uni) > 60 else "")
    return ("<details><summary>The whole search: every (asset, template) slot, ranked"
            f" ({len(uni)} of them)</summary>"
            '<p class="muted">This is the denominator. The selected rows look good partly '
            'because they were picked from this many candidates – which is what the '
            'Deflated Sharpe and PBO figures above are correcting for.</p>'
            + _table(["slot", "OOS Sharpe", "CPCV mean", "P(CPCV<0)", "Pardo", ""],
                     rows, left_cols=(0, 4, 5)) + more + "</details>")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def render_dashboard(spec, states, targets, trades, run, path=None,
                     full_document: bool = True) -> str:
    v = spec["verdict"]
    mark = _STATUS_MARK.get(v["level"], "•")
    reasons = "".join(f"<li>{_e(r)}</li>" for r in v["reasons"])
    n_open = sum(1 for s in states if s["position"])
    n_orders = sum(len(s["entry_orders"]) for s in states)
    n_trades = int((trades["action"] != "hold").sum()) if len(trades) else 0
    stale = ""
    age = utcnow() - pd.Timestamp(run["as_of"])
    if age > pd.Timedelta(days=4):
        stale = (f' <span class="neg">⚠ the newest closed bar is {age.days} days old '
                 f'– check the data feed</span>')

    tiles = [
        ("account equity", _money(run["account_equity"]), "what the book is sized on"),
        ("gross exposure", _pct(targets["gross_exposure"]),
         (f'scaled to {_n(targets["scale_applied"])} for the '
          f'{_pct(run["max_gross"])} cap' if targets["scale_applied"] < 1
          else f'cap {_pct(run["max_gross"])}')),
        ("net exposure", _pct(targets["net_exposure"], sign=True), "long minus short"),
        ("risk if all stops hit", _pct(targets["open_risk_pct"], 1),
         _money(targets["open_risk"])),
        ("open positions", f'{n_open} / {len(states)}', "slots holding something"),
        ("nested OOS Sharpe", _n(spec["diagnostics"]["nested_sharpe"]),
         "the honest research number"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{val}</div>'
        f'<div class="n">{n}</div></div>' for k, val, n in tiles)

    notes = ("".join(f"<li>{_e(n)}</li>" for n in run["notes"]))
    notes_html = (f'<details><summary>{len(run["notes"])} note(s) from this run</summary>'
                  f'<ul class="ink2">{notes}</ul></details>' if run["notes"] else "")

    body = f"""
<div class="wrap">
  <h1>ETF strategy dashboard</h1>
  <p class="sub">Bars through <b>{_e(_ts(run["as_of"]))}</b>{stale} &middot; generated
    {_e(_ts(run["generated"]))} &middot; {_e(", ".join(spec["assets"]))} &middot;
    {_e(spec["config"]["interval"])} bars &middot; {n_open} position(s), {n_orders} working
    order(s), {n_trades} trade(s) to send</p>

  <div class="verdict {_e(v["level"])}">
    <div class="mark" aria-hidden="true">{mark}</div>
    <div>
      <div class="label">Research verdict &middot; {_e(v["level"])}</div>
      <div><b>{_e(v["headline"])}</b></div>
      {f"<ul>{reasons}</ul>" if reasons else ""}
    </div>
  </div>

  <div class="tiles">{tile_html}</div>

  <div class="card">
    <h2>Trades to send</h2>
    {_trades_section(trades, run)}
  </div>

  <div class="card">
    <h2>Orders to work on the next bar</h2>
    <p class="ink2">A breakout entry is a <b>stop</b> order (it fills as price runs through the
      level); fading a break is a <b>limit</b> order (it fills as price reaches it). Sizes come
      from the engine's own rule: {_e(sizing_text(spec["config"]))}.</p>
    {_orders_section(states)}
  </div>

  <div class="card">
    <h2>What you should be holding</h2>
    {_positions_section(states)}
  </div>

  <div class="card">
    <h2>Out-of-sample track record</h2>
    <p class="ink2">Growth of 1.0 on the research history, with both the parameters and the
      slot selection chosen only from data available at each point.</p>
    {_equity_chart(spec.get("curves", {}))}
    {_curve_table(spec.get("curves", {}))}
  </div>

  <div class="card">
    <h2>Slots</h2>
    <p class="ink2">Parameters are re-optimized once per test window
      ({spec["config"]["test_bars"]} bars), never per run: holding them for a whole window is
      what the walk-forward actually measured.</p>
    {_slots_section(states)}
    {notes_html}
  </div>

  <div class="card">
    <h2>Why you should not trust this too much</h2>
    {_diagnostics_section(spec)}
    {_universe_section(spec)}
  </div>

  <footer>
    Nothing here places an order. Bars are complete bars only – the forming bar is
    dropped, so entries are what you could actually have worked.
    Re-run <code>etf_dashboard.py research</code> when you want the slot list re-examined,
    <code>signals</code> as often as your bars close.
  </footer>
</div>
"""
    title = f"ETF dashboard – {pd.Timestamp(run['as_of']).strftime('%Y-%m-%d')}"
    inner = f"<title>{_e(title)}</title>\n<style>{STYLE}</style>\n{body}"
    out = inner if not full_document else (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
        "viewport-fit=cover\">\n" + inner.split("\n", 1)[0] + "\n<style>" + STYLE +
        "</style>\n</head>\n<body>" + body + "</body>\n</html>\n")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
    return out
