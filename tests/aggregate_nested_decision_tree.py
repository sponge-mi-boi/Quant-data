import json
import os
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "artifacts"


def main():
    model_kind = os.environ.get("NESTED_MODEL_KIND", "decision_tree")
    feature_slug = os.environ.get("NESTED_FEATURE_SLUG", "")
    beta_neutral = feature_slug.endswith("_beta_neutral")
    pc_neutral = feature_slug.endswith("_pc_neutral")
    historical_mega = "_historical_mega" in feature_slug
    historical_large = "_historical_large" in feature_slug
    time_series_momentum = "_time_series_momentum" in feature_slug
    file_prefix = f"{model_kind}_{feature_slug}" if feature_slug else model_kind
    model_label = {
        "random_forest": "Random forest",
        "elastic_net_logistic": "Elastic-net logistic",
        "decision_tree": "Decision tree",
    }.get(model_kind, model_kind.replace("_", " ").title())
    runs = []
    universe_snapshots = []
    for run_number in range(1, 6):
        path = OUTPUTS / f"checkpoint_{file_prefix}_nested_purged_outer_run_{run_number}_summary.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("universe_snapshot"):
            universe_snapshots.append(result["universe_snapshot"])
        runs.append(
            {
                "run": run_number,
                **result["held_out"],
                "passed": result["held_out_passed"],
                "outer_held_out": result["outer_held_out"],
            }
        )

    averages = {
        metric: mean(run[metric] for run in runs)
        for metric in ("total_return", "sharpe", "alpha", "max_drawdown")
    }
    summary = {
        "setup": f"{model_label} with {'time-series momentum versus cash' if time_series_momentum else 'three-strategy regime allocation'}{', historical pre-training large-cap universe' if historical_large else (', historical pre-training mega-cap universe' if historical_mega else '')}{', rolling beta neutrality' if beta_neutral else (', rolling leading-PC neutrality' if pc_neutral else '')} and nested purged chronological validation",
        "strategy_set": "time_series_momentum" if time_series_momentum else "three",
        "universe_filter": "historical_large" if historical_large else ("historical_mega" if historical_mega else "none"),
        "universe_snapshots": universe_snapshots,
        "neutrality": ["rolling_market_beta"] if beta_neutral else (["rolling_leading_principal_component"] if pc_neutral else []),
        "neutrality_lookback_bars": 60 if beta_neutral or pc_neutral else None,
        "runs": runs,
        "averages": averages,
        "passed_runs": sum(run["passed"] for run in runs),
        "total_runs": len(runs),
        "purge_bars": 5,
        "execution": {
            "execution_delay_bars": 1,
            "fee_per_order": 0.0005,
            "slippage_per_order": 0.0005,
        },
        "scientific_status": ("Diagnostic: these historical held-out intervals were viewed earlier; "
                              "the source universe is the May 2026 constituent file and therefore still has survivorship bias."
                              if historical_mega or historical_large else
                              "Diagnostic: these historical held-out intervals were viewed earlier."),
    }
    destination = OUTPUTS / f"checkpoint_{file_prefix}_nested_purged_five_outer_summary.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
