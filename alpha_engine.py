#!/usr/bin/env python3
"""
Alpha Factory Engine for Financial Trading Agent

Implementation of WorldQuant Finding Alphas operators for factor research.
Supports declarative alpha specification with cross-sectional IC calculation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from models import AlphaSpec, AlphaSeries, AlphaPoint, AlphaReport
from market_data_tools import MarketDataCollector

logger = logging.getLogger(__name__)


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series rank operation (percentile rank within rolling window)"""

    def rank_func(x):
        if len(x) <= 1:
            return np.nan
        ranks = pd.Series(x).rank()
        return (ranks.iloc[-1] - 1) / (len(x) - 1)

    return series.rolling(window).apply(rank_func, raw=False)


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    """Linear decay weighted average (recent values weighted more heavily)"""
    if window <= 1:
        return series

    # Create linear weights: [1, 2, 3, ..., window]
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()

    def weighted_mean(x):
        if len(x) != window:
            return np.nan
        return np.dot(x, weights)

    return series.rolling(window).apply(weighted_mean, raw=True)


def _apply_ops(base: pd.Series, ops: List[Dict[str, Any]]) -> pd.Series:
    """Apply sequence of alpha operations to base series"""
    s = base.copy()

    for op in ops:
        name = op["name"]
        window = op.get("window", 20)
        k = op.get("k", 1)

        # Handle None values and set defaults
        if window is None:
            window = 20
        if k is None:
            k = 1

        window = int(window)
        k = int(k)

        try:
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
                logger.warning(f"Unsupported alpha operation: {name}")
                raise ValueError(f"Unsupported alpha operation: {name}")
        except Exception as e:
            logger.error(f"Error applying operation {name}: {e}")
            # Return NaN series on error to prevent complete failure
            return pd.Series(index=s.index, dtype=float)

    return s


def _spearman_ic1d(
    factor_df: pd.DataFrame, next_ret_df: pd.DataFrame
) -> Tuple[float, int]:
    """
    Calculate average cross-sectional Information Coefficient (1-day)

    IC is the rank correlation between factor values and next-day returns
    across all stocks for each date, then averaged over time.
    """
    logger.debug(
        f"IC calculation: factor_df shape {factor_df.shape}, next_ret_df shape {next_ret_df.shape}"
    )

    common_dates = factor_df.index.intersection(next_ret_df.index)
    logger.debug(f"Common dates for IC: {len(common_dates)}")

    if len(common_dates) == 0:
        logger.warning("No common dates between factor and return data")
        return np.nan, 0

    ics = []
    valid_dates = 0

    for date in common_dates:
        # Get factor values and next returns for this date
        factor_values = factor_df.loc[date]
        next_returns = next_ret_df.loc[date]

        # Find symbols with both factor and return data (not NaN)
        both_valid = factor_values.notna() & next_returns.notna()

        if both_valid.sum() < 2:  # Need at least 2 symbols for correlation
            continue

        valid_dates += 1
        factor_vals = factor_values[both_valid]
        next_rets = next_returns[both_valid]

        # Calculate rank correlation (Spearman)
        try:
            factor_ranks = factor_vals.rank()
            return_ranks = next_rets.rank()

            ic = factor_ranks.corr(return_ranks)

            if pd.notna(ic):
                ics.append(ic)
                logger.debug(
                    f"Date {date}: IC={ic:.4f} with {len(factor_vals)} symbols"
                )

        except Exception as e:
            logger.debug(f"Error calculating IC for {date}: {e}")
            continue

    logger.debug(
        f"IC calculation completed: {len(ics)} valid ICs from {valid_dates} dates"
    )

    if len(ics) > 0:
        return float(np.mean(ics)), len(ics)
    else:
        return np.nan, 0


def compute_alphas(
    symbols: List[str], specs: List[AlphaSpec], timeframe: str, period: str
) -> Dict[str, Any]:
    """
    Compute alpha factors for given symbols and specifications

    Returns:
        Dict with success status, factor series, and analysis reports
    """
    try:
        logger.info(f"Computing {len(specs)} alpha factors for {len(symbols)} symbols")

        # Get raw historical data for alpha computation
        import yfinance as yf

        all_data = []
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=timeframe)

            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                continue

            # Convert to our format
            for timestamp, row in hist.iterrows():
                all_data.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp.to_pydatetime(),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )

        if not all_data:
            return {"success": False, "error": "No historical data available"}

        # Convert to pandas DataFrame
        df = pd.DataFrame(all_data).sort_values(["timestamp", "symbol"])

        if df.empty:
            return {"success": False, "error": "Empty market data"}

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate returns and next-day returns
        df["returns"] = df.groupby("symbol")["close"].pct_change()
        df["next_ret"] = df.groupby("symbol")["returns"].shift(-1)

        series_output: List[AlphaSeries] = []
        reports: List[AlphaReport] = []

        # Process each alpha specification
        for spec in specs:
            logger.info(f"Processing alpha factor: {spec.name}")

            factor_data = []

            # Calculate factor for each symbol
            for symbol, group in df.groupby("symbol"):
                try:
                    # Get base input series
                    if spec.input == "returns":
                        base_series = group["returns"].astype(float)
                    else:
                        base_series = group[spec.input].astype(float)

                    # Apply operations
                    factor_values = _apply_ops(
                        base_series, [op.model_dump() for op in spec.ops]
                    )

                    # Store results - use group timestamps and values with same index
                    factor_series_with_timestamps = pd.DataFrame(
                        {"timestamp": group["timestamp"], "value": factor_values}
                    ).dropna()

                    for _, row in factor_series_with_timestamps.iterrows():
                        factor_data.append(
                            {
                                "timestamp": row["timestamp"],
                                "symbol": symbol,
                                "value": float(row["value"]),
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Error computing factor {spec.name} for {symbol}: {e}"
                    )
                    continue

            if not factor_data:
                logger.warning(f"No valid data for factor {spec.name}")
                continue

            factor_df = pd.DataFrame(factor_data)

            # Create AlphaSeries for each symbol
            for symbol, symbol_data in factor_df.groupby("symbol"):
                points = []
                for _, row in symbol_data.dropna(subset=["value"]).iterrows():
                    points.append(
                        AlphaPoint(
                            timestamp=row["timestamp"], value=float(row["value"])
                        )
                    )

                # Keep only last 200 points to manage memory
                if points:
                    series_output.append(
                        AlphaSeries(
                            symbol=symbol, factor=spec.name, points=points[-200:]
                        )
                    )

            # Calculate Information Coefficient (IC)
            try:
                # Merge factor data with next returns for proper alignment
                factor_with_returns = factor_df.merge(
                    df[["symbol", "timestamp", "next_ret"]],
                    on=["symbol", "timestamp"],
                    how="inner",
                )

                if len(factor_with_returns) == 0:
                    logger.warning(f"No aligned data for factor {spec.name}")
                    ic, coverage = np.nan, 0
                else:
                    # Pivot to wide format for IC calculation
                    wide_factor = factor_with_returns.pivot(
                        index="timestamp", columns="symbol", values="value"
                    )
                    wide_next_ret = factor_with_returns.pivot(
                        index="timestamp", columns="symbol", values="next_ret"
                    )

                    ic, coverage = _spearman_ic1d(wide_factor, wide_next_ret)

                # Get last values per symbol
                last_values = (
                    factor_df.sort_values("timestamp")
                    .groupby("symbol")["value"]
                    .last()
                    .dropna()
                    .to_dict()
                )

                reports.append(
                    AlphaReport(
                        factor=spec.name,
                        last_value={k: float(v) for k, v in last_values.items()},
                        ic1d=None if np.isnan(ic) else float(ic),
                        coverage=coverage,
                    )
                )

                logger.info(f"Factor {spec.name}: IC={ic:.4f}, Coverage={coverage}")

            except Exception as e:
                logger.error(f"Error calculating IC for factor {spec.name}: {e}")
                reports.append(
                    AlphaReport(factor=spec.name, last_value={}, ic1d=None, coverage=0)
                )

        return {
            "success": True,
            "factors": [s.model_dump() for s in series_output],
            "reports": [r.model_dump() for r in reports],
            "timestamp": datetime.utcnow().isoformat(),
            "symbols_processed": len(symbols),
            "factors_computed": len(specs),
        }

    except Exception as e:
        logger.error(f"Alpha computation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "symbols_requested": symbols,
            "factors_requested": [spec.name for spec in specs],
        }
