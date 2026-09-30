""" 
caveats:
-   No-trade days. 
    On days the held spread doesn't print, the bar is flat. The P&L then lands as a gap on the next print, which is realistic for an illiquid book but can hide a breakout.
-   Settlements vs last trades. 
    If your vendor has spread settlements, use those as the close instead of the last trade.
-   Sign convention. 
    Confirm the exchange's convention (normally front minus back) matches side.
-   Starting level and units.
    With k0=0 the series is the cumulative P&L of holding one rolled spread. The engine in this
    repository handles prices at or below zero (see README, "Futures and spreads"), so there is no
    need for a positive k0. Run the trick with point_value=1, contracts=1, side=+1 and k0=0: the
    output then stays in SPREAD POINTS (a pure shift of the listed spread between rolls, with each
    roll's gap and cost folded in), and the contract multiplier is given to the engine once, as
    --point-value, so it is not applied twice. Let the engine take the short side (side=-1 swaps
    High and Low, which the engine does itself). roll_cost is then in points too.

"""
import numpy as np
import pandas as pd


def etf_trick_listed_spreads(O, H, L, C, expiry=None, roll_days=10, point_value=1.0,
                             contracts=1.0, side=1, k0=0.0, roll_cost=0.0):
    """Continuous OHLC equity (currency) of a rolled listed calendar spread (ETF trick).

    O, H, L, C : DataFrames, index = common trading dates, columns = listed spreads
                 ordered by front expiry ('Z25-Z26', 'Z26-Z27', ...).
                 A no-trade day is all-zero OHLC or a NaN close (a genuine 0.00 close is kept).
    expiry     : {column: last trading day of the FRONT leg}. Strongly recommended; if None
                 it is inferred from the last print, and contracts still printing near the
                 end of the data are treated as unexpired.
    roll_days  : roll at the close of the first day with <= roll_days trading days left on
                 which both old and new spreads printed (or the old one has expired).
    side       : +1 long the listed spread (buy front / sell back), -1 short.
    roll_cost  : points per spread per roll (the whole roll: sell old + buy new).
    Holdings change only at closes, so equity is affine in the spread within each bar and
    the listed spread's OHLC maps exactly (High/Low swap when short).
    """
    cols, dates = list(C.columns), C.index
    o, h, l, c = (df.reindex(index=dates, columns=cols).to_numpy(float, copy=True)
                  for df in (O, H, L, C))
    miss = np.isnan(c) | ((o == 0) & (h == 0) & (l == 0) & (c == 0))
    for a in (o, h, l, c):
        a[miss] = np.nan
    o = np.where(np.isnan(o), c, o)                      # close-only days -> flat bar
    h, l = np.fmax(np.fmax(h, o), c), np.fmin(np.fmin(l, o), c)
    tr = ~miss

    if expiry is None:
        last = pd.Series({k: dates[tr[:, i]].max() if tr[:, i].any() else pd.NaT
                          for i, k in enumerate(cols)})
        expiry = last.where(last < dates[max(0, len(dates) - 1 - roll_days)])
    e = pd.to_datetime(pd.Series(expiry).reindex(cols))
    epos, ok = np.full(len(cols), np.inf), e.notna().to_numpy()
    epos[ok] = dates.searchsorted(e[ok])                 # expiry as a position in dates

    T, m = len(dates), side * contracts * point_value    # m = currency per spread point
    out, held = np.full((T, 4), np.nan), np.full(T, None, object)
    t0 = int(np.argmax(tr.any(1)))
    cand = [i for i in range(len(cols)) if tr[t0, i] and epos[i] - t0 > roll_days]
    j = cand[0] if cand else int(np.argmax(tr[t0]))
    K, ref = float(k0), np.nan                           # ref = NaN means flat

    for t in range(t0, T):
        held[t] = cols[j]
        if np.isnan(ref) or not tr[t, j]:                # flat, or held spread didn't trade
            out[t] = K
            if np.isnan(ref) and tr[t, j]:
                ref = c[t, j]                            # enter at this close
        else:
            hi, lo = (h[t, j], l[t, j]) if m > 0 else (l[t, j], h[t, j])
            out[t] = K + m * (np.array([o[t, j], hi, lo, c[t, j]]) - ref)
            K, ref = out[t, 3], c[t, j]

        if j + 1 < len(cols) and epos[j] - t <= roll_days:   # roll at the close
            expired = epos[j] <= t
            if tr[t, j + 1] and (tr[t, j] or expired or np.isnan(ref)):
                if not np.isnan(ref):
                    K -= roll_cost * abs(m)              # shows up in the next open's gap
                j, ref = j + 1, c[t, j + 1]              # re-anchor to the new spread's close
            elif expired:
                j, ref = j + 1, np.nan                   # go flat until the new spread prints

    res = pd.DataFrame(out, index=dates, columns=["Open", "High", "Low", "Close"])
    res["Held"] = held
    return res.iloc[t0:]


# Example:
# expiry = pd.Series({"Z25-Z26": "2025-12-19", "Z26-Z27": "2026-12-18", ...})
# etf = etf_trick_listed_spreads(O, H, L, C, expiry, roll_days=10,
#                                point_value=50, roll_cost=0.25, k0=0.0)