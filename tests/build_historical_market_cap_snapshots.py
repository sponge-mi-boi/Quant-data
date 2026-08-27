"""Cache point-in-time shares and build pre-training market-cap snapshots."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
CACHE = ROOT / 'work' / 'historical_market_cap'
SHARES_PATH = CACHE / 'yahoo_historical_shares.parquet'
SPLITS_PATH = CACHE / 'yahoo_splits.parquet'
SNAPSHOTS_PATH = CACHE / 'pretraining_market_caps.parquet'
FAILURES_PATH = CACHE / 'historical_shares_failures.csv'
MEGA_THRESHOLD = 200e9
LARGE_THRESHOLD = 10e9
SEC_FALLBACK_CIKS = {'META': '0001326801'}


def yahoo_symbol(symbol):
    return {'BRK.B': 'BRK-B', 'BF.B': 'BF-B'}.get(symbol, symbol)


def fetch_shares(symbol):
    for attempt in range(3):
        try:
            series = yf.Ticker(yahoo_symbol(symbol)).get_shares_full(
                start='2015-01-01', end='2022-04-01')
            if series is None or series.empty:
                raise ValueError('no historical shares returned')
            series = pd.to_numeric(series, errors='coerce').dropna()
            series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
            series = series[~series.index.duplicated(keep='last')].sort_index()
            return symbol, series, None
        except Exception as exc:
            if attempt == 2:
                return symbol, None, f'{type(exc).__name__}: {exc}'
            time.sleep(1.5 * (attempt + 1))


def fetch_splits(symbol):
    try:
        series = yf.Ticker(yahoo_symbol(symbol)).get_splits(period='max')
        if series is None or series.empty:
            return symbol, pd.Series(dtype=float)
        series = pd.to_numeric(series, errors='coerce').dropna()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        return symbol, series[~series.index.duplicated(keep='last')].sort_index()
    except Exception:
        return symbol, pd.Series(dtype=float)


def load_existing():
    if not SHARES_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(SHARES_PATH).sort_index()


def save_shares(series_by_symbol):
    frame = pd.concat(series_by_symbol, axis=1).sort_index()
    frame.columns.name = 'asset'
    frame.to_parquet(SHARES_PATH)
    return frame


def sec_reported_shares(symbol, cik):
    """Return shares indexed by SEC filing date, so availability is causal."""
    response = requests.get(
        f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
        headers={'User-Agent': 'historical-market-cap research example@example.com'},
        timeout=30)
    response.raise_for_status()
    facts = response.json()['facts']['us-gaap']['WeightedAverageNumberOfSharesOutstandingBasic']['units']['shares']
    rows = pd.DataFrame(row for row in facts if row.get('form') in ('10-K', '10-Q'))
    rows['filed'] = pd.to_datetime(rows['filed'])
    rows['start'] = pd.to_datetime(rows['start'])
    # For duplicate facts in one filing, prefer the latest-starting period
    # (the standalone quarter rather than a year-to-date average).
    rows = rows.sort_values(['filed', 'start']).drop_duplicates('filed', keep='last')
    return pd.Series(rows['val'].to_numpy(dtype=float), index=rows['filed'], name=symbol)


def sec_cik_by_ticker():
    response = requests.get(
        'https://www.sec.gov/files/company_tickers.json',
        headers={'User-Agent': 'historical-market-cap research example@example.com'},
        timeout=30)
    response.raise_for_status()
    return {row['ticker'].upper(): f"{int(row['cik_str']):010d}"
            for row in response.json().values()}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / 'yfinance_cache'))
    prices = pd.read_parquet(PROJECT / 'data' / 'processed' / 'close_1d_10y.parquet')
    universe = list(prices.columns)
    existing = load_existing()
    collected = {c: existing[c].dropna() for c in existing.columns}
    failures = []
    pending = [symbol for symbol in universe if symbol not in collected]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_shares, symbol): symbol for symbol in pending}
        for count, future in enumerate(as_completed(futures), 1):
            symbol, series, error = future.result()
            if series is not None:
                collected[symbol] = series
            else:
                failures.append({'asset': symbol, 'error': error})
            if count % 25 == 0:
                save_shares(collected)
                print(f'historical shares {count}/{len(pending)} complete', flush=True)
    for symbol, cik in SEC_FALLBACK_CIKS.items():
        if symbol not in collected:
            collected[symbol] = sec_reported_shares(symbol, cik)
            failures = [item for item in failures if item['asset'] != symbol]
            print(f'{symbol}: using causal SEC-filed share history fallback', flush=True)
    cik_map = sec_cik_by_ticker()
    for item in list(failures):
        symbol = item['asset']
        if symbol not in cik_map:
            continue
        try:
            collected[symbol] = sec_reported_shares(symbol, cik_map[symbol])
            failures.remove(item)
            print(f'{symbol}: using causal SEC-filed share history fallback', flush=True)
        except (KeyError, ValueError, requests.RequestException):
            pass
    shares = save_shares(collected)
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    if SPLITS_PATH.exists():
        splits = pd.read_parquet(SPLITS_PATH)
    else:
        split_series = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch_splits, symbol) for symbol in universe]
            for count, future in enumerate(as_completed(futures), 1):
                symbol, series = future.result()
                split_series[symbol] = series
                if count % 50 == 0:
                    print(f'split histories {count}/{len(universe)} complete', flush=True)
        splits = pd.concat(split_series, axis=1).sort_index()
        splits.columns.name = 'asset'
        splits.to_parquet(SPLITS_PATH)

    # Each universe is frozen using the close immediately preceding training.
    training_starts = [0, 260, 520, 780, 1040]
    rows = []
    for run, start in enumerate(training_starts, 1):
        if start == 0:
            cutoff = prices.index[0] - pd.offsets.BDay(1)
            cutoff_prices = yf.download(
                [yahoo_symbol(s) for s in universe], start=cutoff,
                end=cutoff + pd.Timedelta(days=4), auto_adjust=False,
                progress=False, threads=True)['Close'].iloc[0]
            cutoff_prices.index = [universe[[yahoo_symbol(x) for x in universe].index(x)]
                                   if x in [yahoo_symbol(y) for y in universe] else x
                                   for x in cutoff_prices.index]
            cutoff_prices = cutoff_prices.reindex(universe)
            # BK later changed its Yahoo symbol to BNY. EA is no longer served
            # for this date; its first local close is used only after verifying
            # that a +/-20% move cannot change its large-cap bucket.
            bk = yf.Ticker('BNY').history(start=cutoff - pd.Timedelta(days=1),
                                          end=cutoff + pd.Timedelta(days=2),
                                          auto_adjust=False)
            if not bk.empty:
                cutoff_prices.loc['BK'] = float(bk['Close'].iloc[-1])
            if pd.isna(cutoff_prices.get('EA')):
                cutoff_prices.loc['EA'] = float(prices.iloc[0]['EA'])
        else:
            cutoff = prices.index[start - 1]
            cutoff_prices = prices.iloc[start - 1]
        known_shares = shares.loc[:cutoff].ffill().iloc[-1].reindex(universe)
        # The stored close series is split-adjusted. Restate contemporaneous
        # shares onto the same basis using only corporate actions, not future
        # prices or fundamentals.
        future_splits = splits.loc[(splits.index > cutoff) & (splits.index <= prices.index[-1])]
        split_factor = future_splits.fillna(1.0).prod().reindex(universe).fillna(1.0)
        adjusted_shares = known_shares * split_factor
        caps = cutoff_prices * adjusted_shares
        caps.loc['SPY'] = np.nan
        if run == 1 and np.isfinite(caps.get('EA', np.nan)):
            ea_low, ea_high = .8 * caps['EA'], 1.2 * caps['EA']
            if not (LARGE_THRESHOLD <= ea_low and ea_high < MEGA_THRESHOLD):
                raise ValueError('EA first-close fallback is too close to a market-cap boundary')
        for asset, cap in caps.items():
            rows.append({'outer_run': run, 'training_start': prices.index[start],
                         'information_cutoff': cutoff, 'asset': asset,
                         'market_cap': cap,
                         'is_large': bool(np.isfinite(cap) and LARGE_THRESHOLD <= cap < MEGA_THRESHOLD),
                         'is_mega': bool(np.isfinite(cap) and cap >= MEGA_THRESHOLD)})
        run_rows = [row for row in rows if row['outer_run'] == run]
        print(f'outer run {run}: {sum(r["is_large"] for r in run_rows)} large, '
              f'{sum(r["is_mega"] for r in run_rows)} mega caps', flush=True)
    snapshots = pd.DataFrame(rows)
    snapshots.to_parquet(SNAPSHOTS_PATH, index=False)
    print(f'wrote {SNAPSHOTS_PATH}', flush=True)
    print(f'share histories: {len(collected)}/{len(universe)}; failures: {len(failures)}', flush=True)


if __name__ == '__main__':
    main()
