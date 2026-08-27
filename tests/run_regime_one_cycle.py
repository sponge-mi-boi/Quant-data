import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as v


ROOT = Path(r"path\to\root")
PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
OUTPUT = ROOT / "outputs"
sys.path.insert(0, str(PROJECT / "src"))

from src import _delay_target_weights
from src import get_time_period
from src import regime_estimator
from src import (
    _get_signals,
    _get_signals_momentum_cross_asset,
    _get_signals_momentum_tr,
    _get_signals_mv,
    _get_signals_mv_cross_asset,
    _weights_alloc,
)


PERIODS = {"training": (0, 500), "validation": (500, 760), "held_out": (760, 1020)}
INITIAL_CAPITAL = 100_000
WARMUP_BARS = 100
FEE = 0.0005
SLIPPAGE = 0.0005
REGIME = {"roll": 20, "half_life": 20, "low_quantile": 0.3, "high_quantile": 0.7}
NEUTRALITY_FILTERS = {}
SLEEVES = {
    "momentum_trending": {
        "assets": ["DECK", "ENPH", "LLY", "WM", "AVB", "IT", "VZ", "GOOGL", "CZR", "TKO"],
        "parameters": {"z_threshold": 1.999, "roll": 30},
    },
    "mv": {
        "assets": ["EXR", "TRMB", "OMC", "WSM", "WDC", "NXPI", "BKNG", "MDLZ", "RCL", "KO"],
        "parameters": {"z_threshold": 1.998897, "roll": 30},
    },
    "cross_asset_mv": {
        "assets": ["DG", "MCHP", "BKNG", "PAYC", "ED", "EQIX", "GDDY", "META", "TMO", "HRL"],
        "parameters": {"z_threshold": 2.0},
    },
    "cross_asset_momentum_trending": {
        "assets": ["AMAT", "BR", "GRMN", "GIS", "A", "NTAP", "O", "MAA", "FDS", "ADP"],
        "parameters": {"z_threshold": 1.9283, "roll": 35},
    },
    "cointegration": {
        "pairs": [["CCL", "HUM"], ["AES", "PYPL"], ["ETN", "KKR"], ["EMN", "TAP"]],
        "parameters": {"z_threshold": 1.92, "roll": 32},
    },
}


def sleeve_weights(name, config, period):
    assets = config.get("assets") or [asset for pair in config["pairs"] for asset in pair]
    prices = get_time_period(assets, time_peri=period)
    params = {
        "stock_list": assets,
        "time_period": period,
        "freq": "d",
        "strat_class": {name: config["parameters"]},
        "parameters_": config["parameters"],
        "weights_filter": NEUTRALITY_FILTERS,
    }
    if name == "momentum_trending":
        raw = _get_signals_momentum_tr(params, prices)
    elif name == "mv":
        raw = _get_signals_mv(params, prices)
    elif name == "cross_asset_mv":
        raw = _get_signals_mv_cross_asset(params)[assets]
    elif name == "cross_asset_momentum_trending":
        raw = _get_signals_momentum_cross_asset(params)[assets]
    elif name == "cointegration":
        raw = _get_signals(params, prices)[assets]
    else:
        raise ValueError(name)
    return _weights_alloc(params, raw, NEUTRALITY_FILTERS)


def combined_target_weights(period):
    calculation_period = (max(0, period[0] - WARMUP_BARS), period[1])
    weights_by_sleeve = {
        name: sleeve_weights(name, config, calculation_period) for name, config in SLEEVES.items()
    }
    all_assets = list(dict.fromkeys(
        asset for config in SLEEVES.values()
        for asset in (config.get("assets") or [item for pair in config["pairs"] for item in pair])
    ))
    regime_params = {
        "time_period": calculation_period,
        "freq": "d",
        "weights_filter": {"regime_estimator": REGIME},
    }
    regime = regime_estimator(regime_params)
    common_index = regime.index
    combined = pd.DataFrame(0.0, index=common_index, columns=all_assets)
    for name, sleeve in weights_by_sleeve.items():
        aligned = sleeve.reindex(common_index).fillna(0.0)
        combined.loc[:, aligned.columns] += aligned.mul(regime[name], axis=0)
    # Do not renormalize: an inactive sleeve's allocation correctly remains cash.
    delayed = _delay_target_weights(combined, 1)
    trading_index = get_time_period(["SPY"], time_peri=period).index
    return delayed.reindex(trading_index).fillna(0.0), regime.reindex(trading_index).dropna()


def simulate(period):
    weights, regime = combined_target_weights(period)
    prices = get_time_period(weights.columns, time_peri=period).reindex(weights.index)
    quantities = (weights * INITIAL_CAPITAL).div(prices).fillna(0).astype(int)
    portfolio = v.Portfolio.from_orders(
        close=prices,
        size=quantities,
        size_type="TargetAmount",
        init_cash=INITIAL_CAPITAL,
        freq="d",
        cash_sharing=True,
        fees=FEE,
        slippage=SLIPPAGE,
    )
    benchmark = get_time_period(["SPY"], time_peri=period).reindex(weights.index).pct_change().squeeze()
    stats = portfolio.stats()
    return_stats = portfolio.returns_stats(benchmark_rets=benchmark)
    alpha_keys = [key for key in return_stats.index if "alpha" in str(key).lower()]
    result = {
        "total_return": float(stats["Total Return [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "alpha": float(return_stats[alpha_keys[0]]) if alpha_keys else float("nan"),
        "trades": int(len(portfolio.positions.records_readable)),
        "start_date": str(weights.index.min()),
        "end_date": str(weights.index.max()),
        "observations_after_warmup": int(len(weights)),
        "average_regime_weights": {key: float(value) for key, value in regime.mean().items()},
    }
    return result


def main():
    validation = simulate(PERIODS["validation"])
    validation_passed = bool(
        np.isfinite(validation["sharpe"])
        and validation["total_return"] > 0
        and validation["sharpe"] > 0
        and validation["trades"] >= 20
    )
    result = {
        "test": "One-cycle five-sleeve regime-weighted portfolio",
        "periods": {key: list(value) for key, value in PERIODS.items()},
        "initial_capital": INITIAL_CAPITAL,
        "costs": {"fee_per_order": FEE, "slippage_per_order": SLIPPAGE},
        "historical_warmup_bars": WARMUP_BARS,
        "regime_parameters": REGIME,
        "sleeves": SLEEVES,
        "validation_gate": {"positive_return": True, "positive_finite_sharpe": True, "minimum_trades": 20},
        "validation": validation,
        "validation_passed": validation_passed,
        "held_out_policy": "run unchanged only if validation passes; otherwise remain in cash",
    }
    if validation_passed:
        result["held_out"] = simulate(PERIODS["held_out"])
        result["held_out_deployment"] = "strategy"
    else:
        result["held_out"] = None
        result["held_out_deployment"] = "cash"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "regime_one_cycle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
