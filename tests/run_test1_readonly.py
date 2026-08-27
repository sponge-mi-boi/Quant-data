import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(r"path\to\project")
OUTPUT = Path(r"path\to\output")
sys.path.insert(0, str(PROJECT / "src"))

from src import _port_sim, runner_multiple


PERIODS = {
    "training": (0, 500),
    "validation": (500, 760),
    "held_out": (760, 1020),
}
STRATEGY_CONFIGS = {
    "momentum_trending": {"momentum_trending": {"z_threshold": 1.999, "roll": 30}},
    "mv": {"mv": {"z_threshold": 1.998897, "roll": 30}},
    "cross_asset_momentum_trending": {
        "cross_asset_momentum_trending": {"z_threshold": 1.9283, "roll": 35}
    },
    "cross_asset_mv": {"cross_asset_mv": {"z_threshold": 2.0}},
}
STRATEGY_NAME = os.environ.get("BACKTEST_STRATEGY", "momentum_trending")
if STRATEGY_NAME not in STRATEGY_CONFIGS:
    raise ValueError(f"Unknown BACKTEST_STRATEGY: {STRATEGY_NAME}")
STRATEGY = STRATEGY_CONFIGS[STRATEGY_NAME]
METRICS = {
    "Total_Return": False,
    "Sharpe": False,
    "Alpha": False,
    "Number_of_Trades": False,
}


def simulate(assets, period):
    portfolios = [(asset,) for asset in assets]
    return runner_multiple(
        portfolios,
        [period],
        _port_sim,
        init_money=1000,
        strat_class=STRATEGY,
        inputs=None,
        num_processes=1,
        output_metrics=METRICS,
        freq="d",
        weights_filter={},
        graphs=False,
        parallel=False,
    )


def sharpe_column(df, period):
    return f"{period} Sharpe"


def normalize_index(df):
    result = df.copy()
    result.index = [idx[0] if isinstance(idx, tuple) and len(idx) == 1 else str(idx) for idx in result.index]
    result.index.name = "asset"
    return result


def eligible(df, period, min_trades=1):
    sharpe = pd.to_numeric(df[sharpe_column(df, period)], errors='coerce')
    trades = pd.to_numeric(df[f"{period} Number of Trades"], errors='coerce')
    return df[np.isfinite(sharpe) & (trades >= min_trades)]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PROJECT / "data" / "processed" / "close_1d_10y.parquet")
    universe = [str(c) for c in prices.columns]

    training = simulate(universe, PERIODS["training"])
    training = eligible(training, PERIODS["training"], min_trades=5)
    training = training.sort_values(sharpe_column(training, PERIODS["training"]), ascending=False)
    top_training = [idx[0] for idx in training.head(10).index]

    validation = simulate(top_training, PERIODS["validation"])
    validation = eligible(validation, PERIODS["validation"], min_trades=3)
    validation = validation.sort_values(sharpe_column(validation, PERIODS["validation"]), ascending=False)
    winner = validation.index[0][0]

    held_out = simulate([winner], PERIODS["held_out"])

    prefix = f"test1_fixed_{STRATEGY_NAME}"
    normalize_index(training).to_csv(OUTPUT / f"{prefix}_training_all_assets.csv")
    normalize_index(validation).to_csv(OUTPUT / f"{prefix}_validation_top10.csv")
    normalize_index(held_out).to_csv(OUTPUT / f"{prefix}_held_out.csv")

    index = prices.index
    summary = {
        "test": f"Test 1 - {STRATEGY_NAME} on individual assets",
        "source_project_read_only": str(PROJECT.parent),
        "strategy": STRATEGY,
        "ranking_metric": "Sharpe",
        "training_top_n": 10,
        "universe_size": len(universe),
        "periods": {
            name: {
                "positions": list(period),
                "start_date": str(index[period[0]]),
                "end_date": str(index[period[1] - 1]),
            }
            for name, period in PERIODS.items()
        },
        "training_selected_assets": top_training,
        "validation_selected_asset": winner,
        "held_out_result": normalize_index(held_out).reset_index().to_dict(orient="records"),
    }
    (OUTPUT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
