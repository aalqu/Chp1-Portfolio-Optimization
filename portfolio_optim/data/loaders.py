from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optim.core.config import DataConfig, periods_per_year_for_frequency
from portfolio_optim.core.types import MarketDataSplit


def _find_ticker_file(data_root: Path, ticker: str) -> Path:
    matches = list(data_root.glob(f"**/{ticker}"))
    if not matches:
        raise FileNotFoundError(f"Could not find local data for {ticker} under {data_root}")
    return matches[0]


def _read_price_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    normalized = {column.lower().strip("<>"): column for column in frame.columns}
    date_col = normalized.get("date")
    close_col = normalized.get("close") or normalized.get("adj close") or normalized.get("adj_close")
    if date_col is None or close_col is None:
        raise ValueError(f"Expected date/close columns in {path}")
    series = frame[[date_col, close_col]].copy()
    series.columns = ["date", "close"]
    series["date"] = pd.to_datetime(series["date"].astype(str), format="%Y%m%d")
    series = series.dropna().drop_duplicates("date").sort_values("date")
    return series.set_index("date")["close"].astype(float)


def load_price_frame(data_config: DataConfig, tickers: list[str]) -> pd.DataFrame:
    prices = []
    for ticker in tickers:
        path = _find_ticker_file(data_config.data_root, ticker)
        series = _read_price_series(path)
        prices.append(series.rename(ticker))
    frame = pd.concat(prices, axis=1).dropna()
    if len(frame) < data_config.min_history:
        raise ValueError(f"Need at least {data_config.min_history} observations, found {len(frame)}")
    return frame


def _resample_prices(price_frame: pd.DataFrame, rebalance_frequency: str) -> pd.DataFrame:
    frequency = rebalance_frequency.lower()
    if frequency == "daily":
        return price_frame
    if frequency == "weekly":
        return price_frame.resample("W-FRI").last().dropna(how="any")
    if frequency == "monthly":
        return price_frame.resample("ME").last().dropna(how="any")
    if frequency == "quarterly":
        return price_frame.resample("QE").last().dropna(how="any")
    supported = ", ".join(sorted(["daily", "weekly", "monthly", "quarterly"]))
    raise ValueError(f"Unsupported rebalance_frequency={rebalance_frequency!r}. Expected one of: {supported}")


def compute_log_returns(price_frame: pd.DataFrame, rebalance_frequency: str = "daily") -> pd.DataFrame:
    price_frame = _resample_prices(price_frame, rebalance_frequency)
    returns = np.log(price_frame / price_frame.shift(1)).dropna()
    return returns


def make_market_split(data_config: DataConfig, tickers: list[str], rebalance_frequency: str = "daily") -> MarketDataSplit:
    prices = load_price_frame(data_config, tickers)
    returns = compute_log_returns(prices, rebalance_frequency)
    n_obs = len(returns)
    train_end = int(n_obs * data_config.train_fraction)
    val_end = int(n_obs * (data_config.train_fraction + data_config.validation_fraction))
    train = returns.iloc[:train_end].to_numpy()
    validation = returns.iloc[train_end:val_end].to_numpy()
    test = returns.iloc[val_end:].to_numpy()
    return MarketDataSplit(
        tickers=tickers,
        train_returns=train,
        validation_returns=validation,
        test_returns=test,
        rebalance_frequency=rebalance_frequency,
        periods_per_year=periods_per_year_for_frequency(rebalance_frequency),
    )
