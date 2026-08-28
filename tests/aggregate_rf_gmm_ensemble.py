import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]; outdir=ROOT/'artifacts'; runs=[]
for number in range(1,6):
    source=json.loads((outdir/f'checkpoint_rf_gmm_corr_liq_disp_nested_outer_run_{number}_summary.json').read_text())
    runs.append({'run':number,**source['held_out'],'passed':source['held_out_passed'],'random_forest_weight':source['selected_ensemble']['random_forest_weight'],'gaussian_mixture_weight':source['selected_ensemble']['gaussian_mixture_weight']})
metrics=('total_return','sharpe','alpha','max_drawdown')
result={'setup':'Random forest + Gaussian mixture correlation/liquidity/dispersion nested ensemble','runs':runs,'averages':{m:mean(r[m] for r in runs) for m in metrics},'passed_runs':sum(r['passed'] for r in runs),'total_runs':5,'scientific_status':'Diagnostic: these historical held-out intervals were viewed earlier.'}
path=outdir/'checkpoint_rf_gmm_corr_liq_disp_nested_five_outer_summary.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2,allow_nan=False))
