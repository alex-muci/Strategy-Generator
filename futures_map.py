"""
futures_map.py
--------------
Turn the dashboard's ETF book into whole futures contracts.

The research and the signals run on ETFs: long, clean, dividend-adjusted Yahoo
histories that all close with the New York cash session, and -- for the ETFs
that hold futures themselves (USO, UNG) -- the ROLLED return a futures trader
actually earns. Yahoo's continuous futures (`=F`) are unadjusted splices with
a gap at every roll, so nothing is fitted on them. They are used here for one
thing only: the price that converts a dollar target into a contract count.

    ETF target notional  x  hedge ratio  /  (futures price x multiplier)  ->  contracts

The hedge ratio is the volatility of the ETF over the volatility of the future.
It is 1 when both track the same thing (SPY and MES) and is what makes TLT
(duration ~17) and the T-bond future (duration ~12) the same bet.

Contracts are whole numbers, which an ETF book is not, so every conversion
reports what the rounding cost. Judge a contract's size by the dollars it moves
in a year, not by its notional: a 10-year note future is $110k of notional and
about the same risk as one micro S&P.

Nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FuturesContract:
    etf: str            # the ETF the signals are computed on
    root: str           # the exchange symbol you trade
    name: str
    multiplier: float   # dollars per 1.0 of the quoted price
    tick: float         # minimum price increment, in quoted price units
    yahoo: str          # Yahoo symbol the conversion price comes from
    exchange: str = "CME"
    yahoo_scale: float = 1.0          # quoted price = Yahoo price x this
    notional_range: tuple = (0.0, np.inf)   # a contract value outside this means a bad quote

    def value(self, price: float) -> float:
        """Dollar notional of one contract at `price` (in quoted units)."""
        return float(price) * self.multiplier


# Multipliers and ticks are the exchanges' contract specifications. CHECK THEM
# against the exchange before trading: a spec changes rarely, but a wrong one
# sizes every order wrong. `notional_range` is a deliberately wide sanity band
# that catches a mis-scaled quote (Yahoo has shown JPY futures both per yen and
# per 100 yen), not a forecast.
CONTRACTS = {c.etf: c for c in [
    FuturesContract("SPY", "MES", "Micro E-mini S&P 500", 5.0, 0.25, "MES=F", notional_range=(5e3, 1e5)),
    FuturesContract("QQQ", "MNQ", "Micro E-mini Nasdaq-100", 2.0, 0.25, "MNQ=F", notional_range=(5e3, 2e5)),
    FuturesContract("IWM", "M2K", "Micro E-mini Russell 2000", 5.0, 0.10, "M2K=F", notional_range=(2e3, 5e4)),
    FuturesContract("DIA", "MYM", "Micro E-mini Dow", 0.5, 1.0, "MYM=F", "CBOT", notional_range=(5e3, 1e5)),
    FuturesContract("SHY", "ZT", "2-Year T-Note", 2000.0, 1 / 256, "ZT=F", "CBOT", notional_range=(1.5e5, 2.5e5)),
    FuturesContract("IEI", "ZF", "5-Year T-Note", 1000.0, 1 / 128, "ZF=F", "CBOT", notional_range=(8e4, 1.5e5)),
    FuturesContract("IEF", "ZN", "10-Year T-Note", 1000.0, 1 / 64, "ZN=F", "CBOT", notional_range=(8e4, 1.6e5)),
    FuturesContract("TLT", "ZB", "U.S. Treasury Bond", 1000.0, 1 / 32, "ZB=F", "CBOT", notional_range=(8e4, 2e5)),
    FuturesContract("GLD", "MGC", "Micro Gold (10 oz)", 10.0, 0.10, "MGC=F", "COMEX", notional_range=(5e3, 2e5)),
    FuturesContract("SLV", "SIL", "Micro Silver (1,000 oz)", 1000.0, 0.005, "SIL=F", "COMEX", notional_range=(5e3, 3e5)),
    # a micro shares the big contract's price and only the size differs; Yahoo
    # carries next to no history for MHG=F and MCL=F, and has no micro gas at all
    FuturesContract("CPER", "MHG", "Micro Copper (2,500 lb)", 2500.0, 0.0005, "HG=F", "COMEX", notional_range=(3e3, 5e4)),
    FuturesContract("USO", "MCL", "Micro WTI Crude (100 bbl)", 100.0, 0.01, "CL=F", "NYMEX", notional_range=(1e3, 3e4)),
    FuturesContract("UNG", "MNG", "Micro Henry Hub Gas (1,000 mmBtu)", 1000.0, 0.001, "NG=F", "NYMEX", notional_range=(5e2, 3e4)),
    FuturesContract("FXE", "M6E", "Micro EUR/USD (12,500 EUR)", 12500.0, 0.0001, "M6E=F", notional_range=(8e3, 2.5e4)),
    FuturesContract("FXB", "M6B", "Micro GBP/USD (6,250 GBP)", 6250.0, 0.0001, "M6B=F", notional_range=(5e3, 1.5e4)),
    FuturesContract("FXA", "M6A", "Micro AUD/USD (10,000 AUD)", 10000.0, 0.0001, "M6A=F", notional_range=(4e3, 1.2e4)),
    FuturesContract("FXC", "MCD", "Micro CAD/USD (10,000 CAD)", 10000.0, 0.0001, "MCD=F", notional_range=(5e3, 1.2e4)),
    FuturesContract("FXY", "MJY", "Micro JPY/USD (1,250,000 JPY)", 1250000.0, 0.000001, "6J=F", notional_range=(5e3, 2e4)),
]}

HEDGE_RATIO_BARS = 120          # returns in the volatility ratio
HEDGE_RATIO_MIN_BARS = 60
HEDGE_RATIO_BOUNDS = (0.5, 2.5)
REBALANCE_BUFFER = 0.6          # contracts the target must move before a held count changes


def _robust_returns(close: pd.Series) -> pd.Series:
    """Close-to-close returns with the tails pulled in to 3 robust sigmas: a
    roll gap in a spliced futures series is a price jump nobody earned, and one
    of them in a 120-bar window would move the volatility ratio by itself."""
    r = close.pct_change().dropna()
    if len(r) < 3:
        return r
    med = float(r.median())
    mad = float((r - med).abs().median()) * 1.4826
    if mad <= 0:
        return r
    return r.clip(med - 3 * mad, med + 3 * mad)


def hedge_ratio(etf_close: pd.Series, fut_close: pd.Series,
                n: int = HEDGE_RATIO_BARS) -> tuple[float, str | None]:
    """(ratio, warning): dollars of futures notional per dollar of ETF notional.

    vol(ETF) / vol(future) on the bars both have. With too little shared history
    the ratio is 1 and the warning says so; a low or negative correlation means
    the two series are not the same market (wrong symbol, inverted quote) and is
    reported rather than traded through silently."""
    a, b = _robust_returns(etf_close), _robust_returns(fut_close)
    idx = a.index.intersection(b.index)[-n:]
    if len(idx) < HEDGE_RATIO_MIN_BARS:
        return 1.0, f"only {len(idx)} shared bars for the hedge ratio: using 1.0"
    a, b = a.loc[idx], b.loc[idx]
    sa, sb = float(a.std()), float(b.std())
    if not (sa > 0 and sb > 0):
        return 1.0, "a flat series in the hedge-ratio window: using 1.0"
    raw = sa / sb
    ratio = float(np.clip(raw, *HEDGE_RATIO_BOUNDS))
    corr = float(np.corrcoef(a, b)[0, 1])
    warn = None
    if corr < 0.5:
        warn = f"correlation with the future is only {corr:.2f}: check the symbol"
    elif ratio != raw:
        warn = f"hedge ratio {raw:.2f} clipped to {ratio:.2f}"
    return ratio, warn


def target_contracts(raw: float, held: float, buffer: float = REBALANCE_BUFFER) -> int:
    """Whole contracts to hold for an unrounded target of `raw`.

    Round to nearest -- but keep what is held while the target stays within
    `buffer` contracts of it and on the same side. The hedge ratio and the
    prices drift every day; without the buffer a target wandering between 2.49
    and 2.51 would send an order every morning."""
    if not np.isfinite(raw) or raw == 0:
        return 0
    nearest = int(np.sign(raw) * np.floor(abs(raw) + 0.5))
    held = int(round(held))
    if held == 0 or np.sign(held) != np.sign(raw):
        return nearest
    return held if abs(raw - held) <= buffer else nearest


def fut_level(level, etf_close: float, fut_price: float, ratio: float, tick: float):
    """An ETF price level restated on the future: the same percentage move,
    divided by the hedge ratio, rounded to the tick."""
    if level is None or not np.isfinite(level) or not etf_close:
        return None
    px = fut_price * (1.0 + (float(level) / etf_close - 1.0) / ratio)
    return round(round(px / tick) * tick, 10)


def to_contracts(by_asset: pd.DataFrame, etf_data: dict, fut_data: dict,
                 held: dict | None = None, buffer: float = REBALANCE_BUFFER) -> dict:
    """The ETF book (`live.portfolio_targets(...)["by_asset"]`) in contracts.

    `etf_data` and `fut_data` map an ETF symbol to its OHLC frame and to the
    frame of its future's conversion price. `held` is signed contracts by ROOT
    ({"MES": 2, "ZN": -1}). Every mapped asset with a target or a holding gets a
    row; assets without a contract, or with an implausible quote, are reported
    in `notes` and left out rather than guessed.
    """
    held = {str(k): float(v) for k, v in (held or {}).items()}
    assets = [a for a in etf_data if a in CONTRACTS]
    notes, rows, conv = [], [], {}
    for a in by_asset.index:
        if a not in CONTRACTS:
            notes.append(f"{a}: no futures contract mapped -- left out of the futures book")
    for a in assets:
        c = CONTRACTS[a]
        notional = float(by_asset["notional"].get(a, 0.0)) if len(by_asset) else 0.0
        h = held.get(c.root, 0.0)
        if a not in fut_data or fut_data[a] is None or fut_data[a].empty:
            if notional or h:
                notes.append(f"{a}: no price for {c.yahoo} -- {c.root} left out")
            continue
        price = float(fut_data[a]["Close"].iloc[-1]) * c.yahoo_scale
        value = c.value(price)
        lo, hi = c.notional_range
        if not lo <= value <= hi:
            notes.append(f"{a}: one {c.root} would be worth ${value:,.0f} at {c.yahoo}={price:g}, "
                         f"outside ${lo:,.0f}-${hi:,.0f} -- quote looks mis-scaled, left out")
            continue
        ratio, warn = hedge_ratio(etf_data[a]["Close"], fut_data[a]["Close"])
        if warn:
            notes.append(f"{a}/{c.root}: {warn}")
        conv[a] = dict(contract=c, price=price, value=value, ratio=ratio,
                       etf_close=float(etf_data[a]["Close"].iloc[-1]))
        if not notional and not h:
            continue
        raw = notional * ratio / value
        tgt = target_contracts(raw, h, buffer)
        delta = tgt - int(round(h))
        rows.append(dict(
            asset=a, root=c.root, exchange=c.exchange, etf_notional=notional, hedge_ratio=ratio,
            fut_price=price, contract_value=value, raw=raw, held=int(round(h)), target=tgt,
            delta=delta, action="hold" if delta == 0 else ("BUY" if delta > 0 else "SELL"),
            order_contracts=abs(delta), fut_notional=tgt * value,
            # what the whole-contract book holds beyond the ETF book, in ETF dollars
            rounding_error=(tgt - raw) * value / ratio,
        ))
    cols = ["root", "exchange", "etf_notional", "hedge_ratio", "fut_price", "contract_value", "raw",
            "held", "target", "delta", "action", "order_contracts", "fut_notional", "rounding_error"]
    book = (pd.DataFrame(rows).set_index("asset") if rows
            else pd.DataFrame(columns=cols, index=pd.Index([], name="asset")))
    wanted = float(book["etf_notional"].abs().sum()) if len(book) else 0.0
    return dict(
        book=book, conversions=conv, notes=notes,
        rounding_error=float(book["rounding_error"].abs().sum()) if len(book) else 0.0,
        rounding_error_pct=(float(book["rounding_error"].abs().sum()) / wanted) if wanted else 0.0,
        fut_gross=float(book["fut_notional"].abs().sum()) if len(book) else 0.0,
    )


def translate_orders(states: list, conversions: dict) -> pd.DataFrame:
    """Every working level of every slot -- protective stops of open positions
    and next-bar entries -- restated as a futures price and a contract count.

    Counts are per SLOT and unrounded next to their rounding: slots in the same
    market net into one position, so what you actually work is the row's level
    with the size that brings the account to the book's whole-contract target.
    """
    rows = []
    for st in states:
        cv = conversions.get(st["asset"])
        if cv is None:
            continue
        c, ratio, close = cv["contract"], cv["ratio"], cv["etf_close"]

        def row(what, side, kind, level, shares, limit=None):
            ref = level if level is not None else close
            raw = shares * ref * ratio / cv["value"]
            return dict(
                asset=st["asset"], root=c.root, slot=st["slot"], what=what, side=side, kind=kind,
                etf_level=level, fut_level=fut_level(level, close, cv["price"], ratio, c.tick),
                fut_limit=fut_level(limit, close, cv["price"], ratio, c.tick),
                contracts_raw=raw, contracts=int(np.floor(abs(raw) + 0.5)),
            )

        if st["position"]:
            for o in st["exit_orders"]:
                if o.get("level") is not None:
                    rows.append(row("exit", o["side"], o["kind"], o["level"], st["shares"]))
        for o in st["entry_orders"]:
            rows.append(row("entry", o["side"], o["kind"], o.get("level"), o["shares"], o.get("limit")))
    return pd.DataFrame(rows, columns=["asset", "root", "slot", "what", "side", "kind", "etf_level",
                                       "fut_level", "fut_limit", "contracts_raw", "contracts"])
