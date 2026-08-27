import json
import os
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]; outdir=ROOT/'outputs'; runs=[]
slug=os.environ.get('PRESET_SUMMARY_SLUG','ac_corr_vol')
labels={
    'ac_corr_vol':'AC/correlation/volatility',
    'corr_liq_disp':'correlation/liquidity/dispersion',
    'corr_liq_disp__discrete8':'correlation/liquidity/dispersion discrete eight-state',
}
label=labels.get(slug,slug.replace('_','/'))
for number in range(1,6):
    source=json.loads((outdir/f'checkpoint_preset_{slug}_adam_nested_outer_run_{number}_summary.json').read_text())
    runs.append({'run':number,**source['held_out'],'passed':source['held_out_passed']})
metrics=('total_return','sharpe','alpha','max_drawdown')
result={'setup':f'No-learning preset {label} rules with nested chronological validation','optimizer':'Adam with SPSA validation gradients for allocation controls only','classifier':None,'learned_regime_mapping':False,'runs':runs,'averages':{m:mean(r[m] for r in runs) for m in metrics},'passed_runs':sum(r['passed'] for r in runs),'total_runs':5,'purge_bars':0,'purge_note':'No forward-return regime labels are used by the preset rules.','scientific_status':'Diagnostic: these historical held-out intervals were viewed earlier.'}
path=outdir/f'checkpoint_preset_{slug}_adam_nested_five_outer_summary.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2,allow_nan=False))
