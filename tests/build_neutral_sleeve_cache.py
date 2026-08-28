"""Build the shared neutralized strategy-sleeve cache once."""

import sys
from pathlib import Path

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT
sys.path[:0] = [str(PROJECT / "src/quant_backtester"), str(ROOT / "artifacts")]

from neutral_sleeve_cache import CACHE_PATH, load_or_build


FULL_PERIOD = (0, 2060)
NEUTRALITY = {
    "dollar": {"param": None},
    "beta": {"roll": 30},
    "pc": {"roll": 30, "n": 1},
}


if __name__ == "__main__":
    _, loaded = load_or_build(FULL_PERIOD, NEUTRALITY)
    action = "loaded existing" if loaded else "built new"
    print(f"{action} neutralized sleeve cache: {CACHE_PATH}", flush=True)
