import json
import multiprocessing as mp
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


PROJECT = Path (__file__).resolve().parent.parent


OUTPUT  = PROJECT / "artifacts"
CHECKPOINT = PROJECT / "artifacts" / "checkpoint_costs_cointegration_test1_all_pair_pvalues.csv"

sys.path.insert(0, str(PROJECT / "src/quant_backtester"))

from quant_backtester import _port_sim, runner_multiple


PERIODS = {"training": (0, 500), "validation": (500, 760), "held_out": (760, 1020)}
STRATEGY = {"cointegration": {"z_threshold": 1.92, "roll": 32}}
METRICS = {
    "Total_Return": False,
    "Sharpe": False,
    "Alpha": False,
    "Number_of_Trades": False,
}
INITIAL_CAPITAL = 100_000
FDR_ALPHA = 0.05
FEE = 0.0005
SLIPPAGE = 0.0005
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "test1_fixed_cointegration")
_LOG_PRICES = None


def _init_worker(data_path, start, end):
    global _LOG_PRICES
    prices = pd.read_parquet(data_path).iloc[start:end].drop(columns=["SPY"], errors="ignore")
    _LOG_PRICES = np.log(prices)


def _test_pair(pair):
    left, right = pair
    data = _LOG_PRICES[[left, right]].dropna()
    if len(data) < 100:
        return left, right, np.nan
    try:
        # Fixed one-lag residual ADF keeps all 105k tests deterministic and
        # computationally tractable while remaining an Engle-Granger test.
        p_value = float(coint(data[left], data[right], maxlag=1, autolag=None)[1])
    except (ValueError, np.linalg.LinAlgError):
        p_value = np.nan
    return left, right, p_value


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1].clip(0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def screen_all_pairs(prices):
    assets = [str(column) for column in prices.columns if column != "SPY"]
    pairs = list(combinations(assets, 2))
    if CHECKPOINT.exists():
        screening = pd.read_csv(CHECKPOINT)
        expected = {"asset_1", "asset_2", "p_value"}
        if expected <= set(screening.columns) and len(screening) == len(pairs):
            print(f"REUSED_SCREENING_CHECKPOINT pairs={len(screening)}", flush=True)
            return screening

    workers = min(16, max(1, mp.cpu_count() - 1))
    rows = []
    data_path = str(PROJECT / "data/close_1d_10y.parquet")
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(data_path, PERIODS["training"][0], PERIODS["training"][1])) as pool:
        for count, result in enumerate(pool.imap_unordered(_test_pair, pairs, chunksize=100), 1):
            rows.append(result)
            if count % 5000 == 0 or count == len(pairs):
                print(f"SCREENED {count}/{len(pairs)}", flush=True)

    screening = pd.DataFrame(rows, columns=["asset_1", "asset_2", "p_value"])
    screening.to_csv(CHECKPOINT, index=False)
    return screening


def simulate(portfolios, period, parallel=True):
    return runner_multiple(
        portfolios,
        [period],
        _port_sim,
        init_money=INITIAL_CAPITAL,
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


def normalize_index(df):
    result = df.copy()
    result.index = ["|".join(index) if isinstance(index, tuple) else str(index) for index in result.index]
    result.index.name = "pair"
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PROJECT / "data/close_1d_10y.parquet")
    screening = screen_all_pairs(prices)
    valid = screening.dropna(subset=["p_value"]).copy()
    valid["q_value"] = benjamini_hochberg(valid["p_value"])
    screened = valid[valid["q_value"] <= FDR_ALPHA].sort_values("q_value")
    screening.to_csv(OUTPUT / f"{OUTPUT_PREFIX}_all_pair_pvalues.csv", index=False)
    screened.to_csv(OUTPUT / f"{OUTPUT_PREFIX}_fdr_pairs.csv", index=False)
    if screened.empty:
        raise ValueError("No pairs survived FDR correction")
    pairs = [tuple(row) for row in screened[["asset_1", "asset_2"]].itertuples(index=False, name=None)]
    print(f"FDR_SURVIVORS {len(pairs)}", flush=True)

    training = simulate(pairs, PERIODS["training"])
    training = eligible(training, PERIODS["training"], min_trades=5)
    training = training.sort_values(f"{PERIODS['training']} Sharpe", ascending=False)
    top_pairs = list(training.head(10).index)
    if not top_pairs:
        raise ValueError("No screened pairs passed training eligibility")

    validation = simulate(top_pairs, PERIODS["validation"])
    validation = eligible(validation, PERIODS["validation"], min_trades=3)
    validation = validation.sort_values(f"{PERIODS['validation']} Sharpe", ascending=False)
    if validation.empty:
        raise ValueError("No training-selected pair passed validation eligibility")
    winner = tuple(validation.index[0])
    held_out = simulate([winner], PERIODS["held_out"], parallel=False)

    prefix = OUTPUT_PREFIX
    normalize_index(training).to_csv(OUTPUT / f"{prefix}_training_pairs.csv")
    normalize_index(validation).to_csv(OUTPUT / f"{prefix}_validation_top_pairs.csv")
    normalize_index(held_out).to_csv(OUTPUT / f"{prefix}_held_out.csv")

    index = prices.index
    summary = {
        "test": "Test 1 - cointegration pair singleton portfolios",
        "strategy": STRATEGY,
        "universe_assets_excluding_spy": len(prices.columns) - int("SPY" in prices.columns),
        "unique_pairs_screened": len(screening),
        "fdr_alpha": FDR_ALPHA,
        "fdr_surviving_pairs": len(screened),
        "training_top_pairs": [list(pair) for pair in top_pairs],
        "validation_selected_pair": list(winner),
        "initial_capital": INITIAL_CAPITAL,
        "costs": {"fee_per_order": FEE, "slippage_per_order": SLIPPAGE},
        "engle_granger_adf_maxlag": 1,
        "engle_granger_autolag": None,
        "execution_delay_bars": 1,
        "periods": {
            name: {
                "positions": list(period),
                "start_date": str(index[period[0]]),
                "end_date": str(index[period[1] - 1]),
            }
            for name, period in PERIODS.items()
        },
        "held_out_result": normalize_index(held_out).reset_index().to_dict(orient="records"),
    }
    (OUTPUT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
