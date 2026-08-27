import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as v

ROOT  = Path(r"path\to\root")
PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
OUTPUT = ROOT / "outputs"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(ROOT / "work"))

from src import get_time_period
from src import regime_estimator
from src import (
    build_causal_features,
    fit_ridge_allocator,
    future_risk_adjusted_targets,
    predict_sleeve_weights,
    realized_strategy_returns,
)
from run_regime_one_cycle import SLEEVES, sleeve_weights

FULL_PERIOD = (0, 1020)
TRAIN = (0, 500)
VALIDATION = (500, 760)
HELD_OUT = (760, 1020)
HORIZON = 5
INITIAL_CAPITAL = 100_000
FEE = 0.0005
SLIPPAGE = 0.0005
SLEEVE_NAMES = list(SLEEVES)


def build_sleeves():
    raw = {name: sleeve_weights(name, config, FULL_PERIOD) for name, config in SLEEVES.items()}
    index = get_time_period(["SPY"], time_peri=FULL_PERIOD).index
    all_assets = list(dict.fromkeys(column for frame in raw.values() for column in frame.columns))
    prices = get_time_period(all_assets, time_peri=FULL_PERIOD).reindex(index)
    asset_returns = prices.pct_change().fillna(0.0)
    aligned = {name: frame.reindex(index).fillna(0.0) for name, frame in raw.items()}
    sleeve_returns = pd.DataFrame({
        name: realized_strategy_returns(
            frame, asset_returns[frame.columns], execution_delay=1, fee=FEE, slippage=SLIPPAGE)
        for name, frame in aligned.items()
    }, index=index)
    return aligned, prices, asset_returns, sleeve_returns


def combine(sleeves, allocations, index):
    all_assets = list(dict.fromkeys(column for frame in sleeves.values() for column in frame.columns))
    target = pd.DataFrame(0.0, index=index, columns=all_assets)
    allocations = allocations.reindex(index).ffill().fillna(0.0)
    for name, frame in sleeves.items():
        target.loc[:, frame.columns] += frame.reindex(index).fillna(0.0).mul(allocations[name], axis=0)
    return target.shift(1).fillna(0.0)


def metrics(target, prices, period):
    index = get_time_period(["SPY"], time_peri=period).index
    target = target.reindex(index).fillna(0.0)
    close = prices.reindex(index)[target.columns]
    quantities = (target * INITIAL_CAPITAL).div(close).fillna(0.0).astype(int)
    portfolio = v.Portfolio.from_orders(
        close=close, size=quantities, size_type="TargetAmount",
        init_cash=INITIAL_CAPITAL, freq="d", cash_sharing=True,
        fees=FEE, slippage=SLIPPAGE)
    benchmark = get_time_period(["SPY"], time_peri=period).reindex(index).pct_change().squeeze()
    stats = portfolio.stats()
    returns_stats = portfolio.returns_stats(benchmark_rets=benchmark)
    alpha_key = next(key for key in returns_stats.index if "alpha" in str(key).lower())
    return {
        "total_return": float(stats["Total Return [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "alpha": float(returns_stats[alpha_key]),
        "position_records": int(len(portfolio.positions.records_readable)),
    }


def passed(result):
    return bool(np.isfinite(result["sharpe"]) and result["total_return"] > 0
                and result["sharpe"] > 0 and result["position_records"] >= 20)


def main():
    sleeves, prices, asset_returns, sleeve_returns = build_sleeves()
    index = prices.index
    spy_returns = get_time_period(["SPY"], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_causal_features(spy_returns, asset_returns, sleeve_returns)
    targets = future_risk_adjusted_targets(sleeve_returns, HORIZON)

    train_index = index[TRAIN[0]:TRAIN[1] - HORIZON]
    model = fit_ridge_allocator(
        features.reindex(train_index), targets.reindex(train_index), purge_bars=HORIZON)
    deployment_index = index[VALIDATION[0]:HELD_OUT[1]]
    ml_allocations = predict_sleeve_weights(
        model, features.reindex(deployment_index), rebalance_every=5, max_sleeve_weight=0.4)

    equal_allocations = pd.DataFrame(0.2, index=index, columns=SLEEVE_NAMES)
    regime_allocations = regime_estimator({
        "time_period": FULL_PERIOD, "freq": "d",
        "weights_filter": {"regime_estimator": {
            "roll": 20, "half_life": 20, "low_quantile": 0.3, "high_quantile": 0.7}},
    })
    allocators = {"equal_weight": equal_allocations, "heuristic_regime": regime_allocations,
                  "ml_ridge": ml_allocations}
    results = {}
    for name, allocations in allocators.items():
        target = combine(sleeves, allocations, index)
        validation = metrics(target, prices, VALIDATION)
        gate = passed(validation)
        results[name] = {
            "validation": validation,
            "validation_passed": gate,
            "held_out": metrics(target, prices, HELD_OUT) if gate else None,
            "held_out_deployment": "strategy" if gate else "cash",
            "average_validation_allocations": allocations.reindex(index[500:760]).mean().to_dict(),
        }

    serializable_model = {
        "selected_alpha": model["selected_alpha"],
        "cv_mse": model["cv_mse"],
        "training_observations": model["training_observations"],
        "purge_bars": model["purge_bars"],
        "target_clip_quantiles": list(model["target_clip_quantiles"]),
        "feature_count": len(model["feature_columns"]),
        "target_horizon_bars": HORIZON,
        "rebalance_every_bars": 5,
        "max_sleeve_weight": 0.4,
    }
    output = {
        "test": "One-cycle ML sleeve allocator comparison",
        "periods": {"training": list(TRAIN), "validation": list(VALIDATION), "held_out": list(HELD_OUT)},
        "costs": {"fee_per_order": FEE, "slippage_per_order": SLIPPAGE},
        "validation_gate": {"positive_return": True, "positive_finite_sharpe": True, "minimum_position_records": 20},
        "model": serializable_model,
        "results": results,
        "scientific_status": "The held-out interval was used in earlier experiments and is not pristine for the overall research process.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "ml_allocator_one_cycle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
