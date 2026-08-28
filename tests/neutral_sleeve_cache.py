"""Persistent cache for expensive neutralized strategy-sleeve construction."""

import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import run_ml_allocator_comparison as pipeline
import run_regime_one_cycle as strategy_source
from src.quant_backtester import get_time_period
from src.quant_backtester.ml_allocator import realized_strategy_returns


CACHE_VERSION = 4
ROOT = Path (__file__).resolve().parent.parent
CACHE_PATH = ROOT / "artifacts" / "neutral_sleeves_v4.pkl"


def cache_spec(full_period, neutrality, strategy_names):
    return {
        "version": CACHE_VERSION,
        "full_period": list(full_period),
        "neutrality": neutrality,
        "strategies": {name: strategy_source.SLEEVES[name] for name in strategy_names},
        "fee": pipeline.FEE,
        "slippage": pipeline.SLIPPAGE,
        "execution_delay_bars": 1,
    }


def cache_fingerprint(spec):
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_one_sleeve(args):
    name, config, full_period, neutrality = args
    strategy_source.NEUTRALITY_FILTERS = neutrality
    return name, strategy_source.sleeve_weights(name, config, full_period)


def _build_sleeves_parallel(full_period, neutrality, strategy_names):
    tasks = [(name, config, full_period, neutrality)
             for name, config in strategy_source.SLEEVES.items() if name in strategy_names]
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        raw = dict(executor.map(_build_one_sleeve, tasks))
    index = get_time_period(["SPY"], time_peri=full_period).index
    all_assets = list(dict.fromkeys(column for frame in raw.values()
                                   for column in frame.columns))
    prices = get_time_period(all_assets, time_peri=full_period).reindex(index)
    asset_returns = prices.pct_change().fillna(0.0)
    aligned = {name: frame.reindex(index).fillna(0.0) for name, frame in raw.items()}
    sleeve_returns = pd.DataFrame({
        name: realized_strategy_returns(
            frame, asset_returns[frame.columns], execution_delay=1,
            fee=pipeline.FEE, slippage=pipeline.SLIPPAGE)
        for name, frame in aligned.items()
    }, index=index)
    return aligned, prices, asset_returns, sleeve_returns


def load_or_build(full_period, neutrality, strategy_names=None, force=False):
    strategy_names = tuple(strategy_names or strategy_source.SLEEVES)
    spec = cache_spec(full_period, neutrality, strategy_names)
    fingerprint = cache_fingerprint(spec)
    tag = hashlib.sha256('|'.join(strategy_names).encode('utf-8')).hexdigest()[:10]
    cache_path = CACHE_PATH.with_name(f'neutral_sleeves_v4_{tag}.pkl')
    if cache_path.exists() and not force:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("fingerprint") == fingerprint:
            return payload["data"], True

    data = _build_sleeves_parallel(full_period, neutrality, strategy_names)
    payload = {"fingerprint": fingerprint, "spec": spec, "data": data}
    temporary = cache_path.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)
    return data, False
