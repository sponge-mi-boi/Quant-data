import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT  = Path(r"path\to\project")
OUTPUT = Path(r"path\to\output")
sys.path.insert(0, str(PROJECT / "src"))

from src import _port_sim, runner_multiple


STRATEGY_NAME = os.environ.get("STANDALONE_STRATEGY", "momentum_trending")
STRATEGIES = {
    "momentum_trending": {"momentum_trending": {"z_threshold": 1.999, "roll": 30}},
    "mv": {"mv": {"z_threshold": 1.998897, "roll": 30}},
}
if STRATEGY_NAME not in STRATEGIES:
    raise ValueError(f"Unsupported singleton strategy: {STRATEGY_NAME}")
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


def simulate(assets, period, parallel=True):
    return runner_multiple(
        [(asset,) for asset in assets],
        [period],
        _port_sim,
        init_money=1_000,
        strat_class=STRATEGY,
        inputs=None,
        num_processes=min(16, max(1, mp.cpu_count() - 1)),
        output_metrics=METRICS,
        freq="d",
        weights_filter={},
        graphs=False,
        parallel=parallel,
        fees=FEE,
        slippage=SLIPPAGE,
    )


def eligible(df, period, min_trades):
    sharpe = pd.to_numeric(df[f"{period} Sharpe"], errors="coerce")
    trades = pd.to_numeric(df[f"{period} Number of Trades"], errors="coerce")
    return df[np.isfinite(sharpe) & (trades >= min_trades)]


def metric_value(row, period, metric):
    return float(row[f"{period} {metric}"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PROJECT / "data" / "processed" / "close_1d_10y.parquet")
    universe = [str(column) for column in prices.columns]
    period_records = []
    held_rows = []

    for cycle, start in enumerate(STARTS, 1):
        training_period = (start, start + TRAIN_LENGTH)
        validation_period = (training_period[1], training_period[1] + VALIDATION_LENGTH)
        held_period = (validation_period[1], validation_period[1] + HELD_OUT_LENGTH)
        if held_period[1] > len(prices):
            raise ValueError(f"Cycle {cycle} extends beyond available data")
        print(f"CYCLE {cycle}/5 TRAINING {training_period}", flush=True)

        training = eligible(simulate(universe, training_period), training_period, 5)
        training = training.sort_values(f"{training_period} Sharpe", ascending=False)
        top_assets = [index[0] for index in training.head(10).index]

        validation = eligible(simulate(top_assets, validation_period), validation_period, 3)
        validation = validation.sort_values(f"{validation_period} Sharpe", ascending=False)
        if validation.empty:
            raise ValueError(f"Cycle {cycle} has no validation-eligible asset")
        winner = validation.index[0][0]
        held = simulate([winner], held_period, parallel=False)
        held_row = held.iloc[0]

        record = {
            "cycle": cycle,
            "training_period": list(training_period),
            "validation_period": list(validation_period),
            "held_out_period": list(held_period),
            "held_out_start_date": str(prices.index[held_period[0]]),
            "held_out_end_date": str(prices.index[held_period[1] - 1]),
            "training_selected_assets": top_assets,
            "validation_winner": winner,
            "held_out_total_return": metric_value(held_row, held_period, "Total Return"),
            "held_out_sharpe": metric_value(held_row, held_period, "Sharpe"),
            "held_out_alpha": metric_value(held_row, held_period, "Alpha"),
            "held_out_trades": metric_value(held_row, held_period, "Number of Trades"),
        }
        period_records.append(record)
        held_rows.append(record)
        print(json.dumps(record, default=str), flush=True)

    metrics = ["held_out_total_return", "held_out_sharpe", "held_out_alpha", "held_out_trades"]
    averages = {f"average_{metric}": float(np.mean([row[metric] for row in held_rows])) for metric in metrics}
    returns = np.array([row["held_out_total_return"] for row in held_rows], dtype=float) / 100.0
    averages["compounded_held_out_return"] = float((np.prod(1.0 + returns) - 1.0) * 100.0)
    averages["profitable_held_out_periods"] = int((returns > 0).sum())

    summary = {
        "test": f"{STRATEGY_NAME} - five walk-forward cycles with costs",
        "strategy": STRATEGY,
        "costs": {"fee_per_order": FEE, "slippage_per_order": SLIPPAGE},
        "roll": 375,
        "period_lengths": [TRAIN_LENGTH, VALIDATION_LENGTH, HELD_OUT_LENGTH],
        "universe_size": len(universe),
        "cycles": period_records,
        "averages": averages,
    }
    prefix = OUTPUT_PREFIX
    pd.DataFrame(period_records).to_csv(OUTPUT / f"{prefix}_held_out_metrics.csv", index=False)
    (OUTPUT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"averages": averages}, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
