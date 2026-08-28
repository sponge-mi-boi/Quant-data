import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as v


PROJECT = Path (__file__).resolve().parent.parent

OUTPUT = PROJECT / "artifacts"
sys.path.insert(0, str(PROJECT / "src/quant_backtester"))

from quant_backtester import _port_sim, runner_multiple
from quant_backtester import get_time_period
from quant_backtester .strategies import (
    _get_signals_momentum_cross_asset,
    _get_signals_mv_cross_asset,
)


STRATEGY_NAME = os.environ.get("STANDALONE_STRATEGY", "cross_asset_mv")
STRATEGIES = {
    "cross_asset_mv": {"cross_asset_mv": {"z_threshold": 2.0}},
    "cross_asset_momentum_trending": {
        "cross_asset_momentum_trending": {"z_threshold": 1.9283, "roll": 35}},
}
if STRATEGY_NAME not in STRATEGIES:
    raise ValueError(f"Unsupported cross-sectional strategy: {STRATEGY_NAME}")
STRATEGY = STRATEGIES[STRATEGY_NAME]
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", f"{STRATEGY_NAME}_5_walk_forward_periods")
FEE = 0.0005
SLIPPAGE = 0.0005
METRICS = {
    "Total_Return": False,
    "Sharpe": False,
    "Alpha": False,
    "Number_of_Trades": False,
}
STARTS = [0, 375, 750, 1125, 1500]
TRAIN_LENGTH = 500
VALIDATION_LENGTH = 260
HELD_OUT_LENGTH = 260
BASKET_CAPITAL = 100_000


def simulate_singletons(assets, period):
    params = {
        "stock_list": list(assets), "time_period": period, "freq": "d",
        "strat_class": STRATEGY, "parameters_": STRATEGY[STRATEGY_NAME],
    }
    if STRATEGY_NAME == "cross_asset_mv":
        signals = _get_signals_mv_cross_asset(params)[assets]
    else:
        signals = _get_signals_momentum_cross_asset(params)[assets]
    prices = get_time_period(assets, time_peri=period).reindex(signals.index)
    target_weights = signals.shift(1).fillna(0.0)
    quantities = (target_weights * 1_000).div(prices).fillna(0.0).astype(int)
    portfolio = v.Portfolio.from_orders(
        close=prices, size=quantities, size_type="TargetAmount",
        init_cash=1_000, freq="d", cash_sharing=False,
        fees=FEE, slippage=SLIPPAGE,
    )
    sharpe = portfolio.sharpe_ratio()
    trades = portfolio.positions.count()
    result = pd.DataFrame(index=pd.MultiIndex.from_tuples([(asset,) for asset in assets]))
    result[f"{period} Sharpe"] = sharpe.reindex(assets).to_numpy()
    result[f"{period} Number of Trades"] = trades.reindex(assets).to_numpy(dtype=float)
    return result


def simulate_basket(assets, period):
    return runner_multiple(
        [tuple(assets)], [period], _port_sim,
        init_money=BASKET_CAPITAL, strat_class=STRATEGY, inputs=None, num_processes=1,
        output_metrics=METRICS, freq="d", weights_filter={}, graphs=False, parallel=False,
        fees=FEE, slippage=SLIPPAGE,
    )


def eligible(df, period, min_trades):
    sharpe = pd.to_numeric(df[f"{period} Sharpe"], errors="coerce")
    trades = pd.to_numeric(df[f"{period} Number of Trades"], errors="coerce")
    return df[np.isfinite(sharpe) & (trades >= min_trades)]


def metric(row, period, name):
    return float(row[f"{period} {name}"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PROJECT / "data/close_1d_10y.parquet")
    universe = [str(column) for column in prices.columns]
    records = []

    for cycle, start in enumerate(STARTS, 1):
        training_period = (start, start + TRAIN_LENGTH)
        validation_period = (training_period[1], training_period[1] + VALIDATION_LENGTH)
        held_period = (validation_period[1], validation_period[1] + HELD_OUT_LENGTH)
        if held_period[1] > len(prices):
            raise ValueError(f"Cycle {cycle} extends beyond available data")
        print(f"CYCLE {cycle}/5 TRAINING {training_period}", flush=True)

        training = eligible(simulate_singletons(universe, training_period), training_period, 5)
        training = training.sort_values(f"{training_period} Sharpe", ascending=False)
        basket = [index[0] for index in training.head(10).index]
        if len(basket) < 2:
            raise ValueError(f"Cycle {cycle} has fewer than two eligible assets")

        validation = eligible(simulate_basket(basket, validation_period), validation_period, 3)
        if validation.empty:
            raise ValueError(f"Cycle {cycle} basket fails mechanical validation eligibility")
        validation_row = validation.iloc[0]
        held = simulate_basket(basket, held_period)
        held_row = held.iloc[0]

        record = {
            "cycle": cycle,
            "training_period": list(training_period),
            "validation_period": list(validation_period),
            "held_out_period": list(held_period),
            "held_out_start_date": str(prices.index[held_period[0]]),
            "held_out_end_date": str(prices.index[held_period[1] - 1]),
            "basket": basket,
            "validation_total_return": metric(validation_row, validation_period, "Total Return"),
            "validation_sharpe": metric(validation_row, validation_period, "Sharpe"),
            "validation_alpha": metric(validation_row, validation_period, "Alpha"),
            "validation_trades": metric(validation_row, validation_period, "Number of Trades"),
            "held_out_total_return": metric(held_row, held_period, "Total Return"),
            "held_out_sharpe": metric(held_row, held_period, "Sharpe"),
            "held_out_alpha": metric(held_row, held_period, "Alpha"),
            "held_out_trades": metric(held_row, held_period, "Number of Trades"),
        }
        records.append(record)
        print(json.dumps(record, default=str), flush=True)

    held_metrics = ["held_out_total_return", "held_out_sharpe", "held_out_alpha", "held_out_trades"]
    averages = {f"average_{name}": float(np.mean([row[name] for row in records])) for name in held_metrics}
    returns = np.array([row["held_out_total_return"] for row in records], dtype=float) / 100.0
    averages["compounded_held_out_return"] = float((np.prod(1.0 + returns) - 1.0) * 100.0)
    averages["profitable_held_out_periods"] = int((returns > 0).sum())
    averages["positive_validation_periods"] = int(sum(row["validation_sharpe"] > 0 for row in records))

    summary = {
        "test": f"{STRATEGY_NAME} - five walk-forward cycles with costs",
        "strategy": STRATEGY,
        "costs": {"fee_per_order": FEE, "slippage_per_order": SLIPPAGE},
        "roll": 375,
        "period_lengths": [TRAIN_LENGTH, VALIDATION_LENGTH, HELD_OUT_LENGTH],
        "universe_size": len(universe),
        "basket_initial_capital": BASKET_CAPITAL,
        "cycles": records,
        "averages": averages,
    }
    prefix = OUTPUT_PREFIX
    pd.DataFrame(records).to_csv(OUTPUT / f"{prefix}_held_out_metrics.csv", index=False)
    (OUTPUT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"averages": averages}, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
