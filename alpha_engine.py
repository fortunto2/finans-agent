#!/usr/bin/env python3
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from models import AlphaSpec, AlphaSeries, AlphaPoint, AlphaReport
from market_data_tools import MarketDataCollector


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: (pd.Series(x).rank().iloc[-1]-1)/(len(x)-1) if len(x)>1 else np.nan, raw=False)


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    w = np.arange(1, window + 1, dtype=float)
    w = w / w.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, w), raw=True)


def _apply_ops(base: pd.Series, ops: List[Dict[str, Any]]) -> pd.Series:
    s = base.copy()
    for op in ops:
        name = op["name"]
        window = int(op.get("window", 20))
        k = int(op.get("k", 1))
        if name == "delay":
            s = s.shift(k)
        elif name == "delta":
            s = s.diff(k)
        elif name == "ts_mean":
            s = s.rolling(window).mean()
        elif name == "ts_std":
            s = s.rolling(window).std(ddof=0)
        elif name == "zscore":
            mu = s.rolling(window).mean()
            sd = s.rolling(window).std(ddof=0)
            s = (s - mu) / sd.replace(0.0, np.nan)
        elif name == "ts_rank":
            s = _ts_rank(s, window)
        elif name == "decay_linear":
            s = _decay_linear(s, window)
        else:
            raise ValueError(f"Unsupported op: {name}")
    return s


def _spearman_ic1d(factor_df: pd.DataFrame, next_ret_df: pd.DataFrame) -> Tuple[float, int]:
    """Average cross-sectional IC(1d) per date using rank correlation."""
    common = factor_df.index.intersection(next_ret_df.index)
    ics = []
    for dt in common:
        x = factor_df.loc[dt].dropna()
        y = next_ret_df.loc[dt].reindex(x.index).dropna()
        x = x.reindex(y.index)
        if len(x) >= 2 and len(y) >= 2:
            xr = x.rank()
            yr = y.rank()
            ic = xr.corr(yr)
            if pd.notna(ic):
                ics.append(ic)
    return (float(np.mean(ics)) if len(ics) > 0 else np.nan, len(ics))


def compute_alphas(symbols: List[str], specs: List[AlphaSpec], timeframe: str, period: str) -> Dict[str, Any]:
    collector = MarketDataCollector()
    mr = collector.get_market_data(symbols, timeframe, period)
    rows = []
    for md in mr.market_data:
        rows.append({"symbol": md.symbol, "timestamp": md.timestamp, "open": md.open,
                     "high": md.high, "low": md.low, "close": md.close, "volume": md.volume})
    df = pd.DataFrame(rows).sort_values(["timestamp", "symbol"])
    if df.empty:
        return {"success": False, "error": "no market data"}
    df["returns"] = df.groupby("symbol")["close"].pct_change()
    df["next_ret"] = df.groupby("symbol")["returns"].shift(-1)
    series_out: List[AlphaSeries] = []
    reports: List[AlphaReport] = []
    factor_panels: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        fac = []
        for sym, g in df.groupby("symbol"):
            base = g[spec.input].astype(float)
            values = _apply_ops(base, [op.model_dump() for op in spec.ops])
            tmp = pd.DataFrame({"timestamp": g["timestamp"], "value": values})
            tmp["symbol"] = sym
            fac.append(tmp)
        fac_df = pd.concat(fac, ignore_index=True)
        for sym, g in fac_df.groupby("symbol"):
            pts = [AlphaPoint(timestamp=row["timestamp"], value=float(row["value"])) for _, row in g.dropna(subset=["value"]).iterrows()]
            series_out.append(AlphaSeries(symbol=sym, factor=spec.name, points=pts[-200:]))
        wide_factor = fac_df.pivot(index="timestamp", columns="symbol", values="value")
        wide_nextret = df.pivot(index="timestamp", columns="symbol", values="next_ret")
        ic, cover = _spearman_ic1d(wide_factor, wide_nextret)
        last_vals = fac_df.sort_values("timestamp").groupby("symbol")["value"].last().dropna().to_dict()
        reports.append(AlphaReport(factor=spec.name, last_value={k: float(v) for k, v in last_vals.items()}, ic1d=(None if np.isnan(ic) else float(ic)), coverage=cover))
        factor_panels[spec.name] = wide_factor
    return {"success": True, "factors": [s.model_dump() for s in series_out], "reports": [r.model_dump() for r in reports], "timestamp": datetime.utcnow().isoformat()}
