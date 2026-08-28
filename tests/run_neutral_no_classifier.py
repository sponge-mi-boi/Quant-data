import json
import sys
from pathlib import Path

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT 
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

import run_ml_allocator_comparison as pipeline
import run_regime_one_cycle as strategy_source

FULL_PERIOD = (0, 2572)
PERIODS = {'training': (0, 2060), 'validation': (2060, 2320), 'held_out': (2320, 2572)}
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30}, 'pc': {'roll': 30, 'n': 1}}


def main():
    strategy_source.NEUTRALITY_FILTERS = NEUTRALITY
    pipeline.FULL_PERIOD = FULL_PERIOD
    all_sleeves, prices, _, _ = pipeline.build_sleeves()
    index = prices.index
    target = all_sleeves['cross_asset_mv'].reindex(index).fillna(0.0).shift(1).fillna(0.0)
    results = {name: pipeline.metrics(target, prices, period) for name, period in PERIODS.items()}
    output = {
        'test': 'Neutral cross-sectional mean reversion without a regime classifier',
        'classifier': None,
        'strategy': 'cross_asset_mv',
        'allocation_rule': '100% of strategy allocation to the single CMV sleeve',
        'neutrality': NEUTRALITY,
        'periods': {name: list(period) for name, period in PERIODS.items()},
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005},
        'results': results,
        'held_out_evaluations_this_run': 1,
        'scientific_status': 'No parameters were selected on rows 2320:2572; the fixed CMV portfolio evaluated this previously untouched interval once.',
    }
    path = ROOT / 'artifacts' / 'checkpoint_neutral_no_classifier_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
