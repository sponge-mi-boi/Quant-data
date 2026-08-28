import json
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]

OUT=ROOT/'artifacts'
FEATURES=('corr','liq','ac','dis','var')
LABELS={'corr':'correlation','liq':'liquidity','ac':'autocorrelation','dis':'dispersion','var':'volatility'}
results=[]
for combo in [*combinations(FEATURES,4),FEATURES]:
    slug='combo_'+'_'.join(combo)
    runs=[]
    for run in range(1,6):
        source=json.loads((OUT/f'checkpoint_preset_{slug}_adam_nested_outer_run_{run}_summary.json').read_text())
        runs.append({'run':run,**source['held_out'],'passed':source['held_out_passed']})
    metrics=('total_return','sharpe','alpha','max_drawdown')
    results.append({
        'features':[LABELS[x] for x in combo],
        'feature_codes':list(combo),
        'state_count':2**len(combo),
        'optimizer':'Adam with SPSA validation gradients for allocation controls only',
        'classifier':None,
        'learned_regime_mapping':False,
        'runs':runs,
        'averages':{metric:mean(row[metric] for row in runs) for metric in metrics},
        'passed_runs':sum(row['passed'] for row in runs),
        'total_runs':5,
    })
results.sort(key=lambda row:row['averages']['sharpe'],reverse=True)
output={
    'setup':'No-learning combinatorial preset regimes with nested chronological validation',
    'configuration_count':len(results),
    'feature_subset_sizes':[4,5],
    'execution':{'execution_delay_bars':1,'fee_per_order':.0005,'slippage_per_order':.0005},
    'results':results,
    'scientific_status':'Diagnostic: these historical held-out intervals were viewed earlier.',
}
path=OUT/'checkpoint_combinatorial_presets_4plus_features_nested_five_outer_summary.json'
path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(output,indent=2,allow_nan=False))
