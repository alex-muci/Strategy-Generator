"""
futures_map.py
--------------
Turn the dashboard's ETF book into whole futures contracts.

The research and the signals run on ETFs: long, clean, dividend-adjusted Yahoo
histories, and -- for the ETFs that hold futures themselves (USO, UNG, VIXY,
CORN, WEAT, SOYB, CANE) -- the ROLLED return a futures trader actually earns.
Yahoo's continuous futures (`=F`) are unadjusted splices with a gap at every
roll, so nothing is fitted on them. They are used here for one thing only: the
price that converts a dollar target into a contract count.

    ETF target notional  x  hedge ratio  /  (futures price x multiplier x FX)  ->  contracts

The hedge ratio is the volatility of the ETF over the volatility of the future.
It is 1 when both track the same thing (SPY and MES) and is what makes TLT
(duration ~17) and the T-bond future (duration ~12) the same bet.

Three kinds of market need more than a `=F` symbol, and the table says which:

- **Euro contracts** (FSXE, FGBL, FBTP) are quoted in EUR. Their ETFs (EXHD.DE,
  IITB.MI) trade in EUR too, so `LISTINGS` tells the loader to restate those
  histories in dollars at TODAY's rate -- the returns stay the local ones the
  future pays, only the notional is in dollars (what a hedged share class
  shows). Contract values use the same rate. A futures trader is hedged the
  same way: the P&L accrues in EUR and only the variation margin is exposed.
- **No Yahoo series for the future** (Eurex FGBL and FBTP). The price comes
  from `state/futures_quotes.json` (`{"FGBL": 129.55, "FBTP": 118.20}`), which
  you keep current, and the hedge ratio from a proxy series when one exists
  (S&P's rolled Bund-futures index for FGBL) or from `default_ratio` -- a
  duration-based guess you can override in the same file.
- **A cash index stands in for the future** (^STOXX50E for FSXE, the front
  month's end-of-day TWAP for the Mini VIX). The note on the contract says how
  far the stand-in can sit from the traded price.

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
    multiplier: float   # currency units per 1.0 of the quoted price
    tick: float         # minimum price increment, in quoted price units
    yahoo: str | None   # Yahoo symbol the conversion price comes from; None = futures_quotes.json
    exchange: str = "CME"
    yahoo_scale: float = 1.0          # quoted price = Yahoo price x this
    notional_range: tuple = (0.0, np.inf)   # a contract value (in USD) outside this means a bad quote
    currency: str = "USD"             # what the price and the multiplier are in
    series: str | None = None         # Yahoo symbol whose returns proxy the future's for the hedge ratio
    default_ratio: float | None = None   # hedge ratio when no series is usable (None: 1.0, with a warning)
    note: str = ""                    # how the conversion price relates to the traded contract

    def value(self, price: float, fx: float = 1.0) -> float:
        """Dollar notional of one contract at `price` (in quoted units); `fx` is
        dollars per unit of the contract's currency."""
        return float(price) * self.multiplier * fx

    @property
    def ratio_symbol(self) -> str | None:
        return self.series if self.series is not None else self.yahoo


@dataclass(frozen=True)
class Listing:
    """An ETF that does not trade in dollars in the New York session."""
    currency: str
    session_close: tuple[str, str]    # (HH:MM, IANA zone) of the regular close


# Yahoo symbols that give dollars per one unit of a currency.
FX_SYMBOLS = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "CHF": "CHFUSD=X", "JPY": "JPYUSD=X"}

XETRA = ("17:30", "Europe/Berlin")
MILAN = ("17:30", "Europe/Rome")

# ETFs the loader must restate in dollars (and whose forming bar closes on
# another exchange's clock). Anything not listed is a dollar ETF closing in
# New York.
LISTINGS = {
    "EXHD.DE": Listing("EUR", XETRA),   # iShares eb.rexx Government Germany 5.5-10.5yr (2003)
    "EXW1.DE": Listing("EUR", XETRA),   # iShares Core EURO STOXX 50 (DE) (2000), if preferred to FEZ
    "IITB.MI": Listing("EUR", MILAN),   # iShares Italy Govt Bond (2012)
}


# Multipliers and ticks are the exchanges' contract specifications. CHECK THEM
# against the exchange before trading: a spec changes rarely, but a wrong one
# sizes every order wrong. `notional_range` is a deliberately wide sanity band
# in dollars that catches a mis-scaled quote (Yahoo has shown JPY futures both
# per yen and per 100 yen), not a forecast.
CONTRACTS = {c.etf: c for c in [
    # ---- equity index ------------------------------------------------------
    FuturesContract("SPY", "MES", "Micro E-mini S&P 500", 5.0, 0.25, "MES=F", notional_range=(5e3, 1e5)),
    FuturesContract("QQQ", "MNQ", "Micro E-mini Nasdaq-100", 2.0, 0.25, "MNQ=F", notional_range=(5e3, 2e5)),
    FuturesContract("IWM", "M2K", "Micro E-mini Russell 2000", 5.0, 0.10, "M2K=F", notional_range=(2e3, 5e4)),
    FuturesContract("DIA", "MYM", "Micro E-mini Dow", 0.5, 1.0, "MYM=F", "CBOT", notional_range=(5e3, 1e5)),
    # FEZ is the unhedged SPDR EURO STOXX 50 (2002): its dollar return carries
    # EUR/USD on top of the index, which the vol ratio partly absorbs. The
    # future trades within carry (a percent or two) of the cash index.
    FuturesContract("FEZ", "FSXE", "Micro-EURO STOXX 50 (EUR 1 x index)", 1.0, 0.5, "^STOXX50E", "Eurex",
                    notional_range=(2e3, 2e4), currency="EUR",
                    note="priced off the cash index; the future sits within carry of it"),
    # ---- rates -------------------------------------------------------------
    FuturesContract("SHY", "ZT", "2-Year T-Note", 2000.0, 1 / 256, "ZT=F", "CBOT", notional_range=(1.5e5, 2.5e5)),
    FuturesContract("IEI", "ZF", "5-Year T-Note", 1000.0, 1 / 128, "ZF=F", "CBOT", notional_range=(8e4, 1.5e5)),
    FuturesContract("IEF", "ZN", "10-Year T-Note", 1000.0, 1 / 64, "ZN=F", "CBOT", notional_range=(8e4, 1.6e5)),
    FuturesContract("TLT", "ZB", "U.S. Treasury Bond", 1000.0, 1 / 32, "ZB=F", "CBOT", notional_range=(8e4, 2e5)),
    # Yahoo carries no Eurex bond futures: the price is yours to keep in
    # futures_quotes.json. The Bund's hedge ratio can be estimated against
    # S&P's rolled Euro-Bund futures index; the BTP has no such series on
    # Yahoo, so it starts from the duration guess below.
    FuturesContract("EXHD.DE", "FGBL", "Euro-Bund (EUR 100k, 8.5-10.5y)", 1000.0, 0.01, None, "Eurex",
                    notional_range=(1e5, 2e5), currency="EUR", series="^SPEUBDP", default_ratio=0.9,
                    note="price from futures_quotes.json; ratio ~ ETF duration 7 / CTD duration 7.7"),
    FuturesContract("IITB.MI", "FBTP", "Long-Term Euro-BTP (EUR 100k, 8.5-11y)", 1000.0, 0.01, None, "Eurex",
                    notional_range=(8e4, 1.8e5), currency="EUR", default_ratio=0.85,
                    note="price from futures_quotes.json; ratio ~ ETF duration 6.5 / CTD duration 7.7"),
    # ---- volatility --------------------------------------------------------
    # VIXY IS a rolled long position in the first two VIX futures, so its
    # dollars map about one-to-one onto futures notional. ^VFTW1 is Cboe's
    # end-of-day TWAP of the front month: the price the contract actually
    # trades at, not spot VIX (which is far more volatile than any future).
    FuturesContract("VIXY", "VXM", "Mini VIX ($100 x VIX)", 100.0, 0.01, "^VFTW1", "CFE",
                    notional_range=(5e2, 1e4), default_ratio=1.0,
                    note="priced off the front month's end-of-day TWAP; VIXY blends months 1 and 2"),
    # ---- metals ------------------------------------------------------------
    FuturesContract("GLD", "MGC", "Micro Gold (10 oz)", 10.0, 0.10, "MGC=F", "COMEX", notional_range=(5e3, 2e5)),
    FuturesContract("SLV", "SIL", "Micro Silver (1,000 oz)", 1000.0, 0.005, "SIL=F", "COMEX", notional_range=(5e3, 3e5)),
    # a micro shares the big contract's price and only the size differs; Yahoo
    # carries next to no history for MHG=F and MCL=F, and has no micro gas at all
    FuturesContract("CPER", "MHG", "Micro Copper (2,500 lb)", 2500.0, 0.0005, "HG=F", "COMEX", notional_range=(3e3, 5e4)),
    FuturesContract("PPLT", "PL", "Platinum (50 oz)", 50.0, 0.10, "PL=F", "NYMEX", notional_range=(2e4, 2e5)),
    FuturesContract("PALL", "PA", "Palladium (100 oz)", 100.0, 0.50, "PA=F", "NYMEX", notional_range=(4e4, 4e5)),
    # ---- energy ------------------------------------------------------------
    FuturesContract("USO", "MCL", "Micro WTI Crude (100 bbl)", 100.0, 0.01, "CL=F", "NYMEX", notional_range=(1e3, 3e4)),
    FuturesContract("UNG", "MNG", "Micro Henry Hub Gas (1,000 mmBtu)", 1000.0, 0.001, "NG=F", "NYMEX", notional_range=(5e2, 3e4)),
    # ---- agriculture -------------------------------------------------------
    # Teucrium's funds hold the 2nd, 3rd and a deferred contract, never the
    # front: the vol ratio against the front-month splice absorbs most of the
    # difference. CBOT grains are quoted in cents per bushel; a micro (500 bu,
    # Feb 2025) is $5 per cent with a half-cent tick.
    FuturesContract("CORN", "MZC", "Micro Corn (500 bu)", 5.0, 0.5, "ZC=F", "CBOT", notional_range=(1e3, 6e3),
                    note="priced off the front month; CORN holds the 2nd, 3rd and December contracts"),
    FuturesContract("WEAT", "MZW", "Micro Wheat (500 bu)", 5.0, 0.5, "ZW=F", "CBOT", notional_range=(1.5e3, 8e3),
                    note="priced off the front month; WEAT holds the 2nd, 3rd and December contracts"),
    FuturesContract("SOYB", "MZS", "Micro Soybean (500 bu)", 5.0, 0.5, "ZS=F", "CBOT", notional_range=(3e3, 1.2e4),
                    note="priced off the front month; SOYB holds the 2nd, 3rd and November contracts"),
    # ICE sugar is quoted in cents per pound; 112,000 lb = $1,120 per cent
    FuturesContract("CANE", "SB", "Sugar No. 11 (112,000 lb)", 1120.0, 0.01, "SB=F", "ICE", notional_range=(8e3, 5e4),
                    note="priced off the front month; CANE holds the 2nd, 3rd and March contracts"),
    # ---- FX ----------------------------------------------------------------
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
QUOTES_STALE_DAYS = 3           # a hand-kept futures price older than this is flagged


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
                n: int = HEDGE_RATIO_BARS, fallback: float = 1.0) -> tuple[float, str | None]:
    """(ratio, warning): dollars of futures notional per dollar of ETF notional.

    vol(ETF) / vol(future) on the bars both have. With too little shared history
    the ratio is `fallback` (1, or the contract's duration-based default) and
    the warning says so; a low or negative correlation means the two series are
    not the same market (wrong symbol, inverted quote) and is reported rather
    than traded through silently."""
    a, b = _robust_returns(etf_close), _robust_returns(fut_close)
    idx = a.index.intersection(b.index)[-n:]
    if len(idx) < HEDGE_RATIO_MIN_BARS:
        return fallback, f"only {len(idx)} shared bars for the hedge ratio: using {fallback:g}"
    a, b = a.loc[idx], b.loc[idx]
    sa, sb = float(a.std()), float(b.std())
    if not (sa > 0 and sb > 0):
        return fallback, f"a flat series in the hedge-ratio window: using {fallback:g}"
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


def parse_quotes(raw: dict) -> dict:
    """futures_quotes.json normalized to {ROOT: {"price": float, "hedge_ratio": float | None}}.
    An entry is a bare price (`"FGBL": 129.55`) or an object with `price` and,
    optionally, `hedge_ratio`. A malformed entry is an error: a wrong price
    sizes every order on that contract wrong."""
    out = {}
    for root, v in (raw or {}).items():
        root = str(root)
        if isinstance(v, dict):
            price, ratio = v.get("price"), v.get("hedge_ratio")
        else:
            price, ratio = v, None
        try:
            price = float(price)
            ratio = None if ratio is None else float(ratio)
        except (TypeError, ValueError):
            raise ValueError(f"futures_quotes.json: {root}: expected a price, got {v!r}")
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"futures_quotes.json: {root}: price must be positive, got {price!r}")
        if ratio is not None and not (np.isfinite(ratio) and ratio > 0):
            raise ValueError(f"futures_quotes.json: {root}: hedge_ratio must be positive, got {ratio!r}")
        out[root] = dict(price=price, hedge_ratio=ratio)
    return out


def to_contracts(by_asset: pd.DataFrame, etf_data: dict, fut_data: dict,
                 held: dict | None = None, buffer: float = REBALANCE_BUFFER,
                 fx: dict | None = None, quotes: dict | None = None,
                 series: dict | None = None) -> dict:
    """The ETF book (`live.portfolio_targets(...)["by_asset"]`) in contracts.

    `etf_data` and `fut_data` map an ETF symbol to its OHLC frame (in dollars)
    and to the frame of its future's conversion price (in the contract's
    currency). `fx` is dollars per unit of currency ({"EUR": 1.08}); `quotes`
    is the parsed futures_quotes.json, the price of a contract Yahoo does not
    carry; `series` maps an ETF to the frame the hedge ratio is estimated
    against when that is not the price frame. `held` is signed contracts by
    ROOT ({"MES": 2, "ZN": -1}). Every mapped asset with a target or a holding
    gets a row; assets without a contract, a price, a rate or a plausible quote
    are reported in `notes` and left out rather than guessed.
    """
    held = {str(k): float(v) for k, v in (held or {}).items()}
    fx = {"USD": 1.0} | {str(k): float(v) for k, v in (fx or {}).items()}
    quotes, series = quotes or {}, series or {}
    assets = [a for a in etf_data if a in CONTRACTS]
    notes, rows, conv = [], [], {}
    for a in by_asset.index:
        if a not in CONTRACTS:
            notes.append(f"{a}: no futures contract mapped -- left out of the futures book")
    for a in assets:
        c = CONTRACTS[a]
        notional = float(by_asset["notional"].get(a, 0.0)) if len(by_asset) else 0.0
        h = held.get(c.root, 0.0)
        wanted = bool(notional or h)

        price_frame = fut_data.get(a)
        has_feed = price_frame is not None and not price_frame.empty
        quote = quotes.get(c.root)
        if has_feed:
            price, source = float(price_frame["Close"].iloc[-1]) * c.yahoo_scale, c.yahoo or "supplied series"
        elif quote is not None:
            price, source = quote["price"], "futures_quotes.json"
        elif c.yahoo is None:
            # a missing hand-kept price is a setup gap, reported whether or not
            # the book wants the contract today: the day it does is too late
            notes.append(f"{a}: no price for {c.root} -- Yahoo has no series; put its price in "
                         f"futures_quotes.json ({{\"{c.root}\": <price>}}) -- {c.root} left out")
            continue
        else:
            if wanted:
                notes.append(f"{a}: no price for {c.root} -- no data for {c.yahoo} -- {c.root} left out")
            continue
        rate = fx.get(c.currency)
        if rate is None or not np.isfinite(rate) or rate <= 0:
            if wanted:
                notes.append(f"{a}: no {c.currency}/USD rate to value {c.root} in dollars -- left out")
            continue
        value = c.value(price, rate)
        lo, hi = c.notional_range
        if not lo <= value <= hi:
            notes.append(f"{a}: one {c.root} would be worth ${value:,.0f} at {source}={price:g}"
                         f"{'' if c.currency == 'USD' else f' {c.currency}'}, "
                         f"outside ${lo:,.0f}-${hi:,.0f} -- quote looks mis-scaled, left out")
            continue

        if a in series and series[a] is not None and not series[a].empty:
            ratio_frame, ratio_source = series[a], c.series or "supplied series"
        elif has_feed:
            ratio_frame, ratio_source = price_frame, source
        else:
            ratio_frame, ratio_source = None, "default"
        fallback = c.default_ratio if c.default_ratio is not None else 1.0
        if quote is not None and quote.get("hedge_ratio") is not None:
            ratio, warn = quote["hedge_ratio"], None
            ratio_source = "futures_quotes.json"
        elif ratio_frame is not None:
            ratio, warn = hedge_ratio(etf_data[a]["Close"], ratio_frame["Close"], fallback=fallback)
        else:
            ratio = fallback
            warn = (f"no series to estimate the hedge ratio: using the contract's default {fallback:g}"
                    if c.default_ratio is not None else
                    "no series to estimate the hedge ratio: using 1.0")
        if warn:
            notes.append(f"{a}/{c.root}: {warn}")
        conv[a] = dict(contract=c, price=price, value=value, ratio=ratio, fx=rate,
                       etf_close=float(etf_data[a]["Close"].iloc[-1]))
        if not wanted:
            continue
        raw = notional * ratio / value
        tgt = target_contracts(raw, h, buffer)
        delta = tgt - int(round(h))
        rows.append(dict(
            asset=a, root=c.root, exchange=c.exchange, currency=c.currency, etf_notional=notional,
            hedge_ratio=ratio, ratio_source=ratio_source, fut_price=price, price_source=source,
            fx=rate, contract_value=value, raw=raw, held=int(round(h)), target=tgt,
            delta=delta, action="hold" if delta == 0 else ("BUY" if delta > 0 else "SELL"),
            order_contracts=abs(delta), fut_notional=tgt * value,
            # what the whole-contract book holds beyond the ETF book, in ETF dollars
            rounding_error=(tgt - raw) * value / ratio,
        ))
    # held contracts the loop above never reached. A root whose ETF is not in
    # this run's universe (a research re-run dropped it) is a live position
    # nothing else would ever close: it gets a closing order, unpriced. A root
    # that is no contract of the table at all is most likely a typo in the
    # holdings file -- flagged, never guessed at.
    by_root = {c.root: etf for etf, c in CONTRACTS.items()}
    for root, h in held.items():
        n_held = int(round(h))
        if n_held == 0:
            continue
        etf = by_root.get(root)
        if etf is None:
            notes.append(f"held {root} {n_held:+d}: not a contract in futures_map.CONTRACTS -- "
                         f"check holdings_futures.json; it is NOT in the book below")
            continue
        if etf in assets:
            continue
        c = CONTRACTS[etf]
        notes.append(f"held {root} {n_held:+d}: {etf} is no longer in the portfolio -- "
                     f"{'SELL' if n_held > 0 else 'BUY'} {abs(n_held)} {root} to close it")
        rows.append(dict(
            asset=etf, root=root, exchange=c.exchange, currency=c.currency, etf_notional=0.0,
            hedge_ratio=np.nan, ratio_source="", fut_price=np.nan, price_source="not in the portfolio",
            fx=np.nan, contract_value=np.nan, raw=0.0, held=n_held, target=0, delta=-n_held,
            action="SELL" if n_held > 0 else "BUY", order_contracts=abs(n_held), fut_notional=0.0,
            rounding_error=0.0,
        ))
    cols = ["root", "exchange", "currency", "etf_notional", "hedge_ratio", "ratio_source", "fut_price",
            "price_source", "fx", "contract_value", "raw", "held", "target", "delta", "action",
            "order_contracts", "fut_notional", "rounding_error"]
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
