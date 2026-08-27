import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path(r"path\to\work")
PROJECT = WORK / "PythonProject1_basicbacktester" / "Published"
OUTPUT = Path(r"path\to\output")

sys.path.insert(0, str(PROJECT / "src"))

from src import _port_sim, runner_multiple


PERIODS = {"training": (0, 500), "validation": (500, 760), "held_out": (760, 1020)}
STRATEGY_CONFIGS = {
    "cross_asset_mv": {"cross_asset_mv": {"z_threshold": 2.0}},
    "cross_asset_momentum_trending": {
        "cross_asset_momentum_trending": {"z_threshold": 1.9283, "roll": 35}
    },
}
STRATEGY_NAME = os.environ.get("CROSS_BASKET_STRATEGY", "cross_asset_mv")
if STRATEGY_NAME not in STRATEGY_CONFIGS:
    raise ValueError(f"Unknown CROSS_BASKET_STRATEGY: {STRATEGY_NAME}")
STRATEGY = STRATEGY_CONFIGS[STRATEGY_NAME]
METRICS = {
    "Total_Return": False,
    "Sharpe": False,
    "Alpha": False,
    "Number_of_Trades": False,
}
INITIAL_CAPITAL = 100_000


def simulate(portfolios, period, initial_capital):
    return runner_multiple(
        portfolios,
        [period],
        _port_sim,
        init_money=initial_capital,
        strat_class=STRATEGY,
        inputs=None,
        num_processes=1,
        output_metrics=METRICS,
        freq="d",
        weights_filter={},
        graphs=False,
        parallel=False,
    )


def normalize_index(df):
    result = df.copy()
    result.index = ["|".join(idx) if isinstance(idx, tuple) else str(idx) for idx in result.index]
    result.index.name = "portfolio"
    return result


def eligible(df, period, min_trades):
    sharpe = pd.to_numeric(df[f"{period} Sharpe"], errors="coerce")
    trades = pd.to_numeric(df[f"{period} Number of Trades"], errors="coerce")
    return df[np.isfinite(sharpe) & (trades >= min_trades)]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PROJECT / "data" / "processed" / "close_1d_10y.parquet")
    universe = [str(column) for column in prices.columns]

    training = simulate([(asset,) for asset in universe], PERIODS["training"], 1_000)
    training = eligible(training, PERIODS["training"], min_trades=5)
    training = training.sort_values(f"{PERIODS['training']} Sharpe", ascending=False)
    selected = [index[0] for index in training.head(10).index]
    if len(selected) < 2:
        raise ValueError("Cross-sectional validation requires at least two eligible assets")

    basket = tuple(selected)
    validation = simulate([basket], PERIODS["validation"], INITIAL_CAPITAL)
    validation_eligible = eligible(validation, PERIODS["validation"], min_trades=3)
    if validation_eligible.empty:
        raise ValueError("Selected cross-sectional basket failed validation eligibility")

    held_out = simulate([basket], PERIODS["held_out"], INITIAL_CAPITAL)

    prefix = f"test1_fixed_{STRATEGY_NAME}_basket"
    normalize_index(training).to_csv(OUTPUT / f"{prefix}_training_all_assets.csv")
    normalize_index(validation).to_csv(OUTPUT / f"{prefix}_validation.csv")
    normalize_index(held_out).to_csv(OUTPUT / f"{prefix}_held_out.csv")

    index = prices.index
    summary = {
        "test": f"Test 1 - {STRATEGY_NAME} basket",
        "strategy": STRATEGY,
        "training_universe_size": len(universe),
        "training_selected_assets": selected,
        "basket_initial_capital": INITIAL_CAPITAL,
        "execution_delay_bars": 1,
        "periods": {
            name: {
                "positions": list(period),
                "start_date": str(index[period[0]]),
                "end_date": str(index[period[1] - 1]),
            }
            for name, period in PERIODS.items()
        },
        "validation_result": normalize_index(validation).reset_index().to_dict(orient="records"),
        "held_out_result": normalize_index(held_out).reset_index().to_dict(orient="records"),
    }
    (OUTPUT / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
