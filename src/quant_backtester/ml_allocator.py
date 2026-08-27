"""Leakage-safe machine-learning allocation across strategy sleeves."""

import numpy as np
import pandas as pd


def realized_strategy_returns(raw_weights: pd.DataFrame, asset_returns: pd.DataFrame,
                              execution_delay=1, fee=0.0, slippage=0.0) -> pd.Series:
    """Causal close-to-close sleeve returns from close-derived target weights."""
    execution_delay = int(execution_delay)
    if execution_delay < 0:
        raise ValueError('execution_delay must be non-negative')
    weights = pd.DataFrame(raw_weights, dtype=float)
    returns = pd.DataFrame(asset_returns, dtype=float).reindex(index=weights.index, columns=weights.columns)
    executed = weights.shift(execution_delay).fillna(0.0)
    # Orders execute at the current close; those holdings earn the next
    # close-to-close return, represented by shifting executed weights again.
    gross = (executed.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    costs = turnover * (float(fee) + float(slippage))
    return gross - costs


def future_risk_adjusted_targets(sleeve_returns: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward return divided by forward realized risk, excluding the current bar."""
    horizon = int(horizon)
    if horizon < 2:
        raise ValueError('target horizon must be at least 2')
    future = sleeve_returns.shift(-1)
    forward_return = future.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    forward_risk = future.iloc[::-1].rolling(horizon, min_periods=horizon).std(ddof=1).iloc[::-1]
    return forward_return.div(forward_risk.replace(0, np.nan))


def build_causal_features(market_returns: pd.Series, asset_returns: pd.DataFrame,
                          sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    """Features known at the current close; execution must occur on a later bar."""
    market = pd.Series(market_returns, dtype=float)
    assets = pd.DataFrame(asset_returns, dtype=float).reindex(market.index)
    sleeves = pd.DataFrame(sleeve_returns, dtype=float).reindex(market.index)
    wealth = (1.0 + market.fillna(0.0)).cumprod()
    features = pd.DataFrame(index=market.index)
    features['market_return_5'] = market.rolling(5).sum()
    features['market_return_20'] = market.rolling(20).sum()
    features['market_vol_20'] = market.rolling(20).std(ddof=1)
    features['market_trend_strength'] = market.ewm(halflife=20, adjust=False).mean().div(
        market.pow(2).ewm(halflife=20, adjust=False).mean().pow(0.5).replace(0, np.nan))
    features['market_autocorrelation_1'] = market.rolling(20).corr(market.shift(1))
    features['market_drawdown'] = wealth.div(wealth.cummax()) - 1.0
    features['dispersion'] = assets.std(axis=1, ddof=1)
    features['breadth'] = (assets > 0).mean(axis=1)
    for sleeve in sleeves.columns:
        features[f'{sleeve}_return_5'] = sleeves[sleeve].rolling(5).sum()
        features[f'{sleeve}_return_20'] = sleeves[sleeve].rolling(20).sum()
        features[f'{sleeve}_vol_20'] = sleeves[sleeve].rolling(20).std(ddof=1)
    return features.replace([np.inf, -np.inf], np.nan)


def _ridge_fit(x, y, alpha):
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _ridge_predict(x, coefficients):
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def fit_ridge_allocator(features: pd.DataFrame, targets: pd.DataFrame,
                        alphas=(0.01, 0.1, 1.0, 10.0, 100.0), purge_bars=5,
                        target_clip_quantiles=(0.01, 0.99)) -> dict:
    """Select ridge strength with expanding, ordered, purged validation blocks."""
    joined = features.join(targets, how='inner', lsuffix='_feature', rsuffix='_target').dropna()
    x = joined[features.columns].to_numpy(dtype=float)
    y = joined[targets.columns].to_numpy(dtype=float)
    if len(x) < 90:
        raise ValueError('at least 90 complete training observations are required')
    purge_bars = int(purge_bars)
    if purge_bars < 0:
        raise ValueError('purge_bars must be non-negative')
    lower_q, upper_q = map(float, target_clip_quantiles)
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError('target clip quantiles must satisfy 0 <= lower < upper <= 1')
    split_points = [int(len(x) * fraction) for fraction in (0.5, 0.65, 0.8)]
    scores = {}
    for alpha in alphas:
        errors = []
        for start, end in zip(split_points, split_points[1:] + [len(x)]):
            train_end = start - purge_bars
            if train_end < 30:
                raise ValueError('purge leaves too few observations in an inner training fold')
            train_x, valid_x = x[:train_end], x[start:end]
            train_y, valid_y = y[:train_end], y[start:end]
            lower = np.quantile(train_y, lower_q, axis=0)
            upper = np.quantile(train_y, upper_q, axis=0)
            train_y = np.clip(train_y, lower, upper)
            mean, std = train_x.mean(axis=0), train_x.std(axis=0)
            std[std == 0] = 1.0
            coefficients = _ridge_fit((train_x - mean) / std, train_y, alpha)
            prediction = _ridge_predict((valid_x - mean) / std, coefficients)
            errors.append(float(np.mean((prediction - valid_y) ** 2)))
        scores[float(alpha)] = float(np.mean(errors))
    selected_alpha = min(scores, key=scores.get)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    target_lower = np.quantile(y, lower_q, axis=0)
    target_upper = np.quantile(y, upper_q, axis=0)
    clipped_y = np.clip(y, target_lower, target_upper)
    coefficients = _ridge_fit((x - mean) / std, clipped_y, selected_alpha)
    return {
        'feature_columns': list(features.columns),
        'target_columns': list(targets.columns),
        'mean': mean,
        'std': std,
        'coefficients': coefficients,
        'selected_alpha': selected_alpha,
        'cv_mse': scores,
        'training_observations': len(x),
        'purge_bars': purge_bars,
        'target_clip_quantiles': (lower_q, upper_q),
        'target_lower': target_lower,
        'target_upper': target_upper,
    }


def _capped_positive_weights(scores, cap):
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    if scores.sum() == 0:
        return np.zeros_like(scores)
    cap = float(cap)
    if not 0 < cap <= 1 or cap * len(scores) < 1:
        raise ValueError('max sleeve weight is infeasible')
    weights = np.zeros_like(scores)
    active = scores > 0
    remaining = 1.0
    while active.any() and remaining > 1e-15:
        proposal = remaining * scores[active] / scores[active].sum()
        saturated = proposal > cap
        active_indices = np.flatnonzero(active)
        if not saturated.any():
            weights[active_indices] = proposal
            break
        saturated_indices = active_indices[saturated]
        weights[saturated_indices] = cap
        active[saturated_indices] = False
        remaining = 1.0 - weights.sum()
    return weights


def predict_sleeve_weights(model: dict, features: pd.DataFrame, rebalance_every=5,
                           max_sleeve_weight=0.4, minimum_score=0.0) -> pd.DataFrame:
    """Predict nonnegative capped allocations; unconfident rows remain cash."""
    x_frame = features.reindex(columns=model['feature_columns']).dropna()
    x = (x_frame.to_numpy(dtype=float) - model['mean']) / model['std']
    predictions = _ridge_predict(x, model['coefficients'])
    predictions[predictions <= float(minimum_score)] = 0.0
    weights = pd.DataFrame(
        [_capped_positive_weights(row, max_sleeve_weight) for row in predictions],
        index=x_frame.index, columns=model['target_columns'])
    update = np.arange(len(weights)) % int(rebalance_every) == 0
    weights.loc[~update] = np.nan
    return weights.ffill().fillna(0.0)
