"""
replay.py
---------
Replay a finished `main.py` run and write down what it traded.

`main.py` ends with three headline curves -- the best template, the static
portfolio and the nested walk-forward portfolio -- but keeps only their
numbers: the trades behind them are counted in each walk-forward window and
dropped (walkforward.py), and the orders that produced them were never kept.
This module puts them back on the table, in two steps that stay out of the
research path:

  1. `save_run` (called by main.py once the report is written) records what a
     replay needs: every template with its per-window parameters, the two
     portfolios' selections and weights, the evaluation settings, the exact
     OHLC bars the run used (yfinance revises history; a replay on a fresh
     download is a different backtest) and the return series to check against.

  2. `replay_run` re-simulates ONLY the (template, window) pairs a target is
     made of, with the same `window_backtest` call the walk-forward used, so
     the replay is exact by construction and costs one backtest per window
     instead of one per grid point: seconds for a whole run. The order log is
     read out of the engine's own loop (`backtest(..., log_orders=True)`),
     not rebuilt from the rules, so it cannot drift from what the engine did.

Every target is verified before its lists are trusted: the replayed return
series must equal the stored one to the last digit, the Sharpe must match,
the trade count must match. `check.json` says so per target.

Usage:
  python replay.py --out outputs                    # best, static and nested
  python replay.py --out outputs --which nested
  python replay.py --out outputs --which best --template "<template name>"
  python main.py --replay all                       # run, then replay in one go

Outputs, per target, in <out>/replay/<target>/:
  trades.csv        executed trades (and positions still open at a window's end)
  orders.csv        the order lifecycle: what was working, from when to when,
                    at what level, and every fill / expiry / cancellation
  order_events.csv  the raw per-bar order log the lifecycle rows were built from
  returns.csv       replayed vs stored per-bar returns and their difference
  check.json        the verification verdict and the numbers behind it
"""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

import pipeline
from data import synthetic_ohlc
from pipeline import worker_pool, pool_map
from strategy import StrategyTemplate, annualized_sharpe, set_periods_per_year
from walkforward import window_backtest

MANIFEST_VERSION = 1
MANIFEST = "run.json"
DATA_FILE = "data.csv"
PORTFOLIO_RETURNS = "portfolio_returns.csv"
SELECTED_RETURNS = "selected_returns.csv"
TARGETS = ("best", "static", "nested")
REPLAY_DIR = "replay"
_DATA_KEY = "replay"      # the one asset in the worker pool's data dict

# order kinds that describe a resting order on a bar; consecutive bars at the
# same level collapse into one lifecycle row (see `collapse_orders`)
WORKING_KINDS = ("entry_working", "pullback_working", "stop_working", "target_working")


# --------------------------------------------------------------------------
# the run manifest
# --------------------------------------------------------------------------

def _jsonable(o):
    """Plain JSON: numpy scalars to Python, NaN/inf to null, stamps to ISO."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(o).isoformat()
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    return o


def template_from_dict(d: dict) -> StrategyTemplate:
    """`asdict(tpl)` back to a template; a NaN regime threshold went through
    JSON as null."""
    d = dict(d)
    if d.get("regime_threshold") is None:
        d["regime_threshold"] = np.nan
    return StrategyTemplate(**d)


def _window_record(w: dict) -> dict:
    stats = w.get("oos_stats") or {}
    return dict(
        train_start=w["train_start"], train_end=w["train_end"],
        test_start=w["test_start"], test_end=w["test_end"],
        params=w.get("params"), skipped=bool(w.get("skipped", False)),
        n_trades=int(stats.get("n_trades", 0)),
    )


def run_manifest(df: pd.DataFrame, results: dict, port: dict, nested: dict, fam: dict, args, cfg: dict) -> dict:
    """Everything a replay needs, as a plain dict (see `save_run`)."""
    data = dict(
        source="yfinance" if args.real else "synthetic", ticker=args.real, start=args.start,
        interval=cfg["interval"], bars=args.bars, seed=args.seed, trend_prob=args.trend_prob,
        trend_drift=args.trend_drift, n_bars=int(len(df)), first=df.index[0], last=df.index[-1],
        file=DATA_FILE,
    )
    first = next(iter(results.values()))
    templates = {
        name: dict(
            template=asdict(res["template"]), asset=res.get("asset"),
            oos_sharpe=res["summary"]["oos_sharpe"], n_trades_oos=res["summary"]["n_trades_oos"],
            windows=[_window_record(w) for w in res["windows"]],
        )
        for name, res in results.items()
    }
    static_r = port["portfolio_returns"]
    return dict(
        schema_version=MANIFEST_VERSION, created=pd.Timestamp.now().isoformat(),
        data=data, config=dict(cfg), initial_equity=100_000.0,
        selection=dict(min_sharpe=args.min_sharpe, max_strategies=args.max_strategies,
                       corr_ceiling=args.corr_ceiling, require_pardo=args.require_pardo,
                       select_method=args.select_method, weighting=args.weighting,
                       family=args.family, sides=args.sides, max_templates=args.max_templates),
        best_template=fam["best_template"],
        boundaries=list(first["boundaries"]),
        static=dict(selected=list(port["selected"]), weights=port["weights"].to_dict(),
                    sharpe=annualized_sharpe(static_r) if len(static_r) > 2 else 0.0, n_bars=int(len(static_r))),
        nested=dict(sharpe=nested["sharpe"], n_bars=int(len(nested["portfolio_returns"])),
                    selections=[dict(period_start=s["period_start"], selected=list(s["selected"]),
                                     weights=dict(s["weights"])) for s in nested["selections"]]),
        templates=templates,
    )


def _needed_templates(manifest: dict) -> list:
    names = [manifest["best_template"]] + list(manifest["static"]["selected"])
    for s in manifest["nested"]["selections"]:
        names += list(s["selected"])
    return list(dict.fromkeys(n for n in names if n in manifest["templates"]))


def save_run(out_dir: str, df: pd.DataFrame, results: dict, port: dict, nested: dict, fam: dict, args, cfg: dict) -> dict:
    """Write the manifest, the bars, and the return series the replay checks
    against, into `out_dir`. Returns the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = run_manifest(df, results, port, nested, fam, args, cfg)
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=1)
    # %.17g round-trips every double; pandas' default writer does too but the
    # reader needs float_precision="round_trip" (see load_run), stated once here
    df.to_csv(os.path.join(out_dir, DATA_FILE), float_format="%.17g")
    pd.DataFrame({"static": port["portfolio_returns"], "nested": nested["portfolio_returns"]}).to_csv(
        os.path.join(out_dir, PORTFOLIO_RETURNS), float_format="%.17g", index_label="date")
    sel = {n: results[n]["oos_returns"] for n in _needed_templates(manifest) if len(results[n]["oos_returns"])}
    (pd.DataFrame(sel) if sel else pd.DataFrame()).to_csv(
        os.path.join(out_dir, SELECTED_RETURNS), float_format="%.17g", index_label="date")
    return manifest


def _read_series_frame(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")


def load_run(out_dir: str) -> dict:
    """The manifest plus the bars and stored series of a run, with templates
    and timestamps rebuilt. Regenerates synthetic bars from the recorded seed
    when data.csv is missing."""
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run main.py first (it writes the run manifest)")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema_version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: schema version {m.get('schema_version')}, this code reads {MANIFEST_VERSION}")
    data_path = os.path.join(out_dir, m["data"].get("file", DATA_FILE))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True, float_precision="round_trip")
    elif m["data"]["source"] == "synthetic":
        d = m["data"]
        df = synthetic_ohlc(n_bars=d["bars"], seed=d["seed"], trend_prob=d["trend_prob"], trend_drift=d["trend_drift"])
    else:
        raise FileNotFoundError(f"{data_path} not found and the run used {m['data']['ticker']}: "
                                "a fresh download would not be the same backtest")
    for spec in m["templates"].values():
        for w in spec["windows"]:
            for k in ("train_start", "train_end", "test_start", "test_end"):
                w[k] = pd.Timestamp(w[k])
    for s in m["nested"]["selections"]:
        s["period_start"] = pd.Timestamp(s["period_start"])
    m["boundaries"] = [pd.Timestamp(b) for b in m["boundaries"]]
    templates = {name: template_from_dict(spec["template"]) for name, spec in m["templates"].items()}
    return dict(
        out_dir=out_dir, manifest=m, df=df, templates=templates,
        portfolio_returns=_read_series_frame(os.path.join(out_dir, PORTFOLIO_RETURNS)),
        selected_returns=_read_series_frame(os.path.join(out_dir, SELECTED_RETURNS)),
    )


# --------------------------------------------------------------------------
# re-simulating windows (in the pool)
# --------------------------------------------------------------------------

def _window_slice(df: pd.DataFrame, w: dict) -> tuple[int, int]:
    """(first bar, one past the last bar) of a window's test period."""
    i0 = df.index.get_loc(w["test_start"])
    i1 = df.index.get_loc(w["test_end"]) + 1
    return int(i0), int(i1)


def _replay_window(job):
    """One (template, window) pair: the walk-forward's own OOS call
    (`walkforward.walk_forward` line "apply OOS"), with the order log on."""
    name, k, tpl_dict, params, i0, i1, initial_equity, log_orders = job
    df = pipeline._DATA[_DATA_KEY]
    tpl = template_from_dict(tpl_dict).with_params(**params)
    res = window_backtest(df, tpl, i0, i1, initial_equity=initial_equity, log_orders=log_orders)
    trades = pd.DataFrame(res["trades"])
    if trades.empty:
        trades = pd.DataFrame(columns=["entry_date", "side", "entry_price", "shares", "cost", "exit_date",
                                       "exit_price", "reason", "pnl", "bars_held"])
    trades["closed"] = True
    pos = res.get("open_position")
    if pos is not None:
        # marked to market at the window's end and dropped there (README,
        # "Known approximations"): listed so the book can be read back in full
        trades = pd.concat([trades, pd.DataFrame([dict(
            entry_date=pos["entry_date"], side=pos["side"], entry_price=pos["entry_price"], shares=pos["shares"],
            cost=pos["entry_cost"], exit_date=pd.NaT, exit_price=np.nan, reason="open_at_window_end",
            pnl=pos["unrealized"], bars_held=pos["bars_held"], closed=False)])], ignore_index=True)
    orders = res["orders"] if log_orders else None
    return name, k, res["returns"], trades, orders, len(res["trades"])


def _jobs_for(run: dict, pairs: list, log_orders: bool) -> list:
    m, df = run["manifest"], run["df"]
    jobs = []
    for name, k in pairs:
        w = m["templates"][name]["windows"][k]
        if w["skipped"]:
            continue
        i0, i1 = _window_slice(df, w)
        jobs.append((name, k, m["templates"][name]["template"], w["params"], i0, i1,
                     float(m["initial_equity"]), log_orders))
    return jobs


def _run_jobs(run: dict, pairs: list, pool, log_orders: bool) -> dict:
    """{(template, window): (returns, trades, orders, n_trades)} for every pair;
    a skipped window (no tradeable in-sample fit) is flat and has no trades."""
    m, df = run["manifest"], run["df"]
    done = {}
    for name, k, r, trades, orders, n in pool_map(pool, _replay_window, _jobs_for(run, pairs, log_orders)):
        done[(name, k)] = (r, trades, orders, n)
    for name, k in pairs:
        if (name, k) not in done:
            w = m["templates"][name]["windows"][k]
            i0, i1 = _window_slice(df, w)
            done[(name, k)] = (pd.Series(0.0, index=df.index[i0:i1]), None, None, 0)
    return done


def _tag(frame, name: str, k: int, w: dict, **extra) -> pd.DataFrame | None:
    if frame is None:
        return None
    out = frame.copy()
    out.insert(0, "template", name)
    out.insert(1, "window", k)
    out.insert(2, "test_start", w["test_start"])
    for key, v in extra.items():
        out[key] = v
    out["params"] = json.dumps(w["params"], sort_keys=True) if w["params"] else ""
    return out


def _stitch(parts: list) -> pd.Series:
    """`walk_forward`'s stitching of window returns into one OOS series."""
    if not parts:
        return pd.Series(dtype=float)
    r = pd.concat(parts)
    return r[~r.index.duplicated(keep="last")].sort_index()


def _concat(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# the three targets
# --------------------------------------------------------------------------

def _template_pairs(run: dict, name: str) -> list:
    return [(name, k) for k in range(len(run["manifest"]["templates"][name]["windows"]))]


def _assemble_template(run: dict, name: str, done: dict, **extra) -> dict:
    """One template's stitched OOS returns, trades and orders from replayed windows."""
    windows = run["manifest"]["templates"][name]["windows"]
    parts, trades, orders, checks = [], [], [], []
    for k, w in enumerate(windows):
        r, t, o, n = done[(name, k)]
        parts.append(r)
        trades.append(_tag(t, name, k, w, **extra))
        orders.append(_tag(o, name, k, w, **extra))
        checks.append(dict(template=name, window=k, test_start=w["test_start"], skipped=w["skipped"],
                           n_trades_stored=w["n_trades"], n_trades_replayed=n))
    return dict(returns=_stitch(parts), trades=_concat(trades), orders=_concat(orders), windows=checks)


def _window_of(run: dict, name: str, period_start: pd.Timestamp) -> int | None:
    """The window of `name` whose test period starts the nested period; by
    date, never by position (periods short of history leave no selection)."""
    for k, w in enumerate(run["manifest"]["templates"][name]["windows"]):
        if w["test_start"] == period_start:
            return k
    return None


def _nested_periods(run: dict) -> list:
    """[(selection, window index k)] with k the boundary window of the period
    (the same window for every template on the same bars)."""
    first = next(iter(run["manifest"]["templates"]))
    out = []
    for s in run["manifest"]["nested"]["selections"]:
        k = _window_of(run, first, s["period_start"])
        if k is None:
            raise ValueError(f"nested period {s['period_start']} matches no walk-forward window")
        out.append((s, k))
    return out


def replay_target(run: dict, which: str, pool=None, *, template: str | None = None, log_orders: bool = True) -> dict:
    """Replay one target. Returns dict(which, names, returns, stored, trades,
    orders, windows, check)."""
    m, df = run["manifest"], run["df"]
    stored_port = run["portfolio_returns"]
    # every Sharpe and vol-target size below reads the run's annualization
    # (worker_pool sets it for the pool; this covers a direct call too)
    set_periods_per_year(m["config"]["periods_per_year"])
    if which == "best":
        name = template or m["best_template"]
        if name not in m["templates"]:
            raise KeyError(f"template {name!r} is not in this run ({len(m['templates'])} templates in run.json)")
        done = _run_jobs(run, _template_pairs(run, name), pool, log_orders)
        res = _assemble_template(run, name, done, weight=1.0)
        # the run keeps the series of the templates its targets are made of; any
        # other template is checked on its Sharpe and trade counts alone
        stored = run["selected_returns"][name] if name in run["selected_returns"] else None
        spec = m["templates"][name]
        res.update(which=which, names=[name], stored=stored, sharpe_stored=spec["oos_sharpe"],
                   n_trades_stored=spec["n_trades_oos"],
                   folder=which if template is None else "template_" + _safe_filename(name))
    elif which == "static":
        names = list(m["static"]["selected"])
        weights = pd.Series({n: m["static"]["weights"][n] for n in names}, dtype=float)
        pairs = [p for n in names for p in _template_pairs(run, n)]
        done = _run_jobs(run, pairs, pool, log_orders)
        per = {n: _assemble_template(run, n, done, weight=float(weights[n])) for n in names}
        if names:
            rets = pd.concat({n: per[n]["returns"] for n in names}, axis=1, join="inner").fillna(0.0)
            returns = (rets[names] * weights).sum(axis=1)   # portfolio.select_portfolio's combination
        else:
            returns = pd.Series(dtype=float)
        stored = stored_port["static"].dropna() if "static" in stored_port else pd.Series(dtype=float)
        res = dict(which=which, names=names, returns=returns, stored=stored,
                   trades=_concat([per[n]["trades"] for n in names]),
                   orders=_concat([per[n]["orders"] for n in names]),
                   windows=[c for n in names for c in per[n]["windows"]],
                   sharpe_stored=m["static"]["sharpe"],
                   n_trades_stored=sum(m["templates"][n]["n_trades_oos"] for n in names))
    elif which == "nested":
        periods = _nested_periods(run)
        pairs = list(dict.fromkeys((n, _window_of(run, n, s["period_start"])) for s, _ in periods for n in s["selected"]))
        done = _run_jobs(run, pairs, pool, log_orders)
        parts, trades, orders, checks, names, n_stored = [], [], [], [], [], 0
        first = next(iter(m["templates"]))
        for s, k0 in periods:
            w0 = m["templates"][first]["windows"][k0]
            i0, i1 = _window_slice(df, w0)
            block = pd.Series(0.0, index=df.index[i0:i1])
            for n in s["selected"]:
                k = _window_of(run, n, s["period_start"])
                w = m["templates"][n]["windows"][k]
                r, t, o, cnt = done[(n, k)]
                wt = float(s["weights"][n])
                block = block.add(wt * r, fill_value=0.0)
                trades.append(_tag(t, n, k, w, weight=wt, period_start=s["period_start"]))
                orders.append(_tag(o, n, k, w, weight=wt, period_start=s["period_start"]))
                checks.append(dict(template=n, window=k, test_start=w["test_start"], skipped=w["skipped"],
                                   n_trades_stored=w["n_trades"], n_trades_replayed=cnt, period_start=s["period_start"]))
                n_stored += w["n_trades"]
                names.append(n)
            parts.append(block)
        stored = stored_port["nested"].dropna() if "nested" in stored_port else pd.Series(dtype=float)
        res = dict(which=which, names=list(dict.fromkeys(names)), returns=_stitch(parts), stored=stored,
                   trades=_concat(trades), orders=_concat(orders), windows=checks,
                   sharpe_stored=m["nested"]["sharpe"], n_trades_stored=n_stored)
    else:
        raise ValueError(f"unknown target {which!r}: one of {TARGETS}")
    res.setdefault("folder", which)
    if len(res["orders"]):
        res["orders"]["bar"] = df.index.get_indexer(res["orders"]["date"])
    res["check"] = verify(res["returns"], res["stored"], res["sharpe_stored"], res["n_trades_stored"],
                          int(sum(c["n_trades_replayed"] for c in res["windows"])))
    res["check"]["windows_mismatched"] = [c for c in res["windows"] if c["n_trades_stored"] != c["n_trades_replayed"]]
    res["check"]["ok"] = res["check"]["ok"] and not res["check"]["windows_mismatched"]
    return res


def _safe_filename(name: str) -> str:
    """A template name as a folder name every OS accepts (as main.py does)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# --------------------------------------------------------------------------
# verification and the order lifecycle view
# --------------------------------------------------------------------------

def verify(replayed: pd.Series, stored: pd.Series | None, sharpe_stored: float, n_trades_stored: int,
           n_trades_replayed: int, atol: float = 1e-12) -> dict:
    """Does the replay reproduce the run? Same bars, same per-bar returns (to
    `atol`), same Sharpe, same trade count. `ok` is the verdict. Without a
    stored series (a template the run did not keep one for) the bar-by-bar
    check is skipped and reported as such: Sharpe and trade counts decide."""
    sharpe = annualized_sharpe(replayed) if len(replayed) > 2 else 0.0
    has_series = stored is not None
    empty = bool(len(replayed) == 0 and (stored is None or len(stored) == 0))
    if empty:
        # a portfolio nothing qualified for: no bars, no trades, and consistent
        same_index, max_diff, series_ok = True, 0.0, True
    elif has_series:
        same_index = bool(len(replayed) == len(stored) and replayed.index.equals(stored.index))
        aligned = replayed.reindex(stored.index).fillna(0.0)
        max_diff = float(np.abs(aligned.to_numpy() - stored.to_numpy()).max()) if len(stored) and len(replayed) else np.nan
        series_ok = same_index and len(stored) > 0 and max_diff <= atol
    else:
        same_index, max_diff, series_ok = None, None, True
    ok = series_ok and abs(sharpe - float(sharpe_stored)) <= 1e-9 and n_trades_stored == n_trades_replayed
    return dict(ok=bool(ok), empty=empty, stored_series=has_series, same_bars=same_index,
                n_bars_replayed=int(len(replayed)), n_bars_stored=int(len(stored)) if has_series else None,
                max_abs_return_diff=max_diff, sharpe_replayed=float(sharpe), sharpe_stored=float(sharpe_stored),
                n_trades_replayed=int(n_trades_replayed), n_trades_stored=int(n_trades_stored))


def collapse_orders(events: pd.DataFrame) -> pd.DataFrame:
    """The raw per-bar log as an order lifecycle: a resting order that sat at
    the same level over consecutive bars is one row (first_date, last_date,
    n_bars); fills, submissions, expiries and cancellations stay one row each,
    on their own date. Sorted by date, so the book reads top to bottom."""
    cols = ["template", "window", "kind", "side", "detail_text", "level", "qty", "aux", "first_date", "last_date",
            "n_bars", "expires", "weight", "period_start", "params"]
    if events is None or not len(events):
        return pd.DataFrame(columns=[c for c in cols if c != "period_start" and c != "weight"])
    ev = events.copy()
    ev["first_date"] = ev["date"]
    ev["last_date"] = ev["date"]
    ev["n_bars"] = 1
    working = ev["kind"].isin(WORKING_KINDS)
    single = ev[~working]
    w = ev[working].sort_values(["template", "window", "kind", "side", "bar"], kind="stable")
    if len(w):
        key = ["template", "window", "kind", "side", "detail", "level"]
        same = (w[key] == w[key].shift()).all(axis=1) & (w["bar"] == w["bar"].shift() + 1)
        run_id = (~same).cumsum()
        agg = dict(first_date=("first_date", "first"), last_date=("last_date", "last"), n_bars=("bar", "size"))
        for c in ev.columns:
            if c not in agg and c not in ("date", "bar"):
                agg[c] = (c, "first")
        w = w.groupby(run_id, sort=False).agg(**agg).reset_index(drop=True)
    out = pd.concat([w, single], ignore_index=True) if len(w) else single
    # within a bar: what was working, then what was submitted, then what happened
    rank = out["kind"].map(lambda k: 0 if k in WORKING_KINDS else (1 if k.endswith("submit") or k.endswith("market") else 2))
    out = out.assign(_rank=rank).sort_values(["first_date", "template", "window", "_rank", "kind"], kind="stable")
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def _order_columns(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or not len(events):
        return pd.DataFrame()
    lead = ["template", "window", "date", "bar", "kind", "side", "detail_text", "level", "qty", "aux", "expires"]
    rest = [c for c in events.columns if c not in lead]
    return events[[c for c in lead if c in events.columns] + rest]


def _trade_columns(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or not len(trades):
        return pd.DataFrame()
    lead = ["template", "window", "entry_date", "exit_date", "side", "shares", "entry_price", "exit_price",
            "reason", "pnl", "cost", "bars_held", "closed"]
    rest = [c for c in trades.columns if c not in lead]
    return trades[[c for c in lead if c in trades.columns] + rest].sort_values(
        ["entry_date", "template"], kind="stable").reset_index(drop=True)


def write_target(out_dir: str, res: dict) -> str:
    """Write one target's files to <out_dir>/replay/<which>/; returns the folder."""
    folder = os.path.join(out_dir, REPLAY_DIR, res.get("folder", res["which"]))
    os.makedirs(folder, exist_ok=True)
    for stale in ("trades.csv", "orders.csv", "order_events.csv", "returns.csv", "check.json"):
        if os.path.exists(os.path.join(folder, stale)):   # a previous replay's file must not survive this one
            os.remove(os.path.join(folder, stale))
    _trade_columns(res["trades"]).to_csv(os.path.join(folder, "trades.csv"), index=False)
    if res["orders"] is not None and len(res["orders"]):
        _order_columns(res["orders"]).to_csv(os.path.join(folder, "order_events.csv"), index=False)
        collapse_orders(res["orders"]).to_csv(os.path.join(folder, "orders.csv"), index=False)
    rets = pd.DataFrame({"replayed": res["returns"]})
    if res["stored"] is not None:
        rets["stored"] = res["stored"]
        rets["diff"] = rets["replayed"] - rets["stored"]
    rets.to_csv(os.path.join(folder, "returns.csv"), float_format="%.17g", index_label="date")
    check = dict(res["check"], target=res["which"], templates=res["names"],
                 n_windows=len(res["windows"]), n_open_at_window_end=int((~res["trades"]["closed"]).sum())
                 if len(res["trades"]) else 0)
    with open(os.path.join(folder, "check.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(check), f, indent=1)
    return folder


def _verdict_line(res: dict, folder: str) -> str:
    c = res["check"]
    status = "OK      " if c["ok"] else "MISMATCH"
    diff = ("empty target (nothing selected)" if c["empty"] else
            f"max |return diff| {c['max_abs_return_diff']:.1e}" if c["stored_series"] else
            "no stored series (Sharpe and trades only)")
    return (f"  {status} {res['which']:<7} Sharpe replayed {c['sharpe_replayed']:.4f} vs stored {c['sharpe_stored']:.4f}, "
            f"{diff}, trades {c['n_trades_replayed']} vs {c['n_trades_stored']}, "
            f"{len(res['names'])} template(s), {len(res['windows'])} window(s) -> {folder}{os.sep}")


def replay_run(out_dir: str, which="all", *, template: str | None = None, jobs: int = 1,
               orders: bool = True, verbose: bool = True) -> dict:
    """Replay the targets of the run in `out_dir` and write their files.
    `which`: one of TARGETS, "all", or a list. Returns {target: result}."""
    targets = list(TARGETS) if which == "all" or which is None else ([which] if isinstance(which, str) else list(which))
    if "all" in targets:
        targets = list(TARGETS)
    for t in targets:
        if t not in TARGETS:
            raise ValueError(f"unknown target {t!r}: one of {TARGETS} or 'all'")
    t0 = time.time()
    run = load_run(out_dir)
    if verbose:
        d = run["manifest"]["data"]
        print(f"Replaying {', '.join(targets)} of {out_dir}{os.sep}: {d['n_bars']} bars of "
              f"{d['ticker'] or 'synthetic data'} ({d['interval']}), {len(run['manifest']['templates'])} templates")
    out = {}
    with worker_pool(jobs, {_DATA_KEY: run["df"]}, run["manifest"]["config"]) as pool:
        for t in targets:
            res = replay_target(run, t, pool, template=template, log_orders=orders)
            folder = write_target(out_dir, res)
            out[t] = res
            if verbose:
                print(_verdict_line(res, folder))
    if verbose:
        bad = [t for t, r in out.items() if not r["check"]["ok"]]
        print(f"Replay done in {time.time() - t0:.1f}s: " + ("every target reproduced its run." if not bad else
              f"MISMATCH in {', '.join(bad)} -- see check.json before reading the trade lists."))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replay a main.py run: verify it and list its orders and trades")
    p.add_argument("--out", default="outputs", help="the run's output folder (main.py --out)")
    p.add_argument("--which", nargs="+", default=["all"], choices=list(TARGETS) + ["all"],
                   help="best = the top-Sharpe template, static = the static portfolio, nested = the nested "
                        "walk-forward portfolio")
    p.add_argument("--template", default=None, help="replay this template as the 'best' target instead")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-orders", action="store_true", help="trades only, skip the order log")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    which = "all" if "all" in args.which else args.which
    return replay_run(args.out, which, template=args.template, jobs=args.jobs, orders=not args.no_orders)


if __name__ == "__main__":
    main()
