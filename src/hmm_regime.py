"""Gaussian hidden-Markov regime model with causal out-of-sample filtering."""

import numpy as np
import pandas as pd

from .market_filters_analysis import _ewm_average_autocorrelation, _rolling_percentile


def build_variance_correlation_trend_features(
        market_returns, asset_returns, roll=20, half_life=20):
    """Build causal variance, average-correlation, and directional-trend features."""
    market = pd.Series(market_returns, dtype=float)
    assets = pd.DataFrame(asset_returns, dtype=float).reindex(market.index)
    variance = market.pow(2).ewm(halflife=half_life, adjust=False).mean()
    asset_mean = assets.ewm(halflife=half_life, adjust=False).mean()
    second = assets.pow(2).ewm(halflife=half_life, adjust=False).mean()
    asset_scale = (second - asset_mean.pow(2)).clip(lower=0).pow(.5).replace(0, np.nan)
    standardized = (assets - asset_mean).div(asset_scale)
    count = standardized.notna().sum(axis=1)
    pair_product = (standardized.sum(axis=1).pow(2) - standardized.pow(2).sum(axis=1)).div(
        (count * (count - 1)).where(count >= 2))
    correlation = pair_product.ewm(halflife=half_life, adjust=False).mean().clip(-1, 1)
    trend_mean = market.ewm(halflife=half_life, adjust=False).mean()
    trend = trend_mean.div(variance.pow(.5).replace(0, np.nan)).clip(-5, 5)
    return pd.DataFrame({
        'var': _rolling_percentile(variance, roll),
        'corr': _rolling_percentile(correlation, roll),
        'trend': trend,
    }).replace([np.inf, -np.inf], np.nan).dropna()


def build_variance_dispersion_trend_features(
        market_returns, asset_returns, roll=20, half_life=20):
    """Build causal variance, cross-sectional-dispersion, and trend features."""
    market = pd.Series(market_returns, dtype=float)
    assets = pd.DataFrame(asset_returns, dtype=float).reindex(market.index)
    variance = market.pow(2).ewm(halflife=half_life, adjust=False).mean()
    dispersion = assets.std(axis=1, ddof=1)
    trend_mean = market.ewm(halflife=half_life, adjust=False).mean()
    trend = trend_mean.div(variance.pow(.5).replace(0, np.nan)).clip(-5, 5)
    return pd.DataFrame({
        'var': _rolling_percentile(variance, roll),
        'dis': _rolling_percentile(dispersion, roll),
        'trend': trend,
    }).replace([np.inf, -np.inf], np.nan).dropna()


def variance_correlation_trend_allocations(
        features, allocation_half_life=5, rebalance_every=5,
        max_sleeve_weight=.4):
    """Map the three causal features to fixed, non-learned strategy weights."""
    frame = pd.DataFrame(features).reindex(columns=['var', 'corr', 'trend']).dropna()
    if rebalance_every < 1 or allocation_half_life <= 0:
        raise ValueError('rebalance_every and allocation_half_life must be positive')
    if not 0 < max_sleeve_weight <= 1:
        raise ValueError('max_sleeve_weight must be in (0, 1]')
    high_var = ((frame['var'] - .5) / .5).clip(0, 1)
    low_var = ((.5 - frame['var']) / .5).clip(0, 1)
    middle_var = (1 - (frame['var'] - .5).abs() / .5).clip(0, 1)
    high_corr = ((frame['corr'] - .5) / .5).clip(0, 1)
    low_corr = ((.5 - frame['corr']) / .5).clip(0, 1)
    middle_corr = (1 - (frame['corr'] - .5).abs() / .5).clip(0, 1)
    trend = frame['trend'].abs().clip(0, 1)
    flat = 1 - trend
    scores = pd.DataFrame(.05, index=frame.index, columns=[
        'momentum_trending', 'mv', 'cross_asset_mv',
        'cross_asset_momentum_trending', 'cointegration', 'cash'])
    scores['momentum_trending'] += 1.2 * trend + .6 * high_corr
    scores['cross_asset_momentum_trending'] += 1.0 * trend + .8 * low_corr
    scores['mv'] += 1.0 * flat + .6 * low_var + .3 * low_corr
    scores['cross_asset_mv'] += 1.0 * flat + .8 * low_corr + .4 * high_var
    scores['cointegration'] += .8 * flat + .7 * middle_corr + .4 * middle_var
    scores['cash'] += 1.2 * high_var + .4 * high_var * high_corr
    weights = scores.div(scores.sum(axis=1), axis=0)
    weights = weights.clip(upper=max_sleeve_weight)
    weights = weights.div(weights.sum(axis=1), axis=0)
    weights = weights.ewm(halflife=allocation_half_life, adjust=False).mean()
    update = np.arange(len(weights)) % int(rebalance_every) == 0
    weights.loc[~update] = np.nan
    return weights.ffill().dropna()


def build_hmm_features(market_returns, asset_returns, roll=20, half_life=20):
    """Build the four causal regime observations used by the HMM."""
    market = pd.Series(market_returns, dtype=float)
    assets = pd.DataFrame(asset_returns, dtype=float).reindex(market.index)
    variance = market.pow(2).ewm(halflife=half_life, adjust=False).mean()
    dispersion = assets.std(axis=1, ddof=1)
    asset_mean = assets.ewm(halflife=half_life, adjust=False).mean()
    second = assets.pow(2).ewm(halflife=half_life, adjust=False).mean()
    standardized = (assets - asset_mean).div((second - asset_mean.pow(2)).clip(lower=0).pow(.5).replace(0, np.nan))
    count = standardized.notna().sum(axis=1)
    pair_product = (standardized.sum(axis=1).pow(2) - standardized.pow(2).sum(axis=1)).div(
        (count * (count - 1)).where(count >= 2))
    correlation = pair_product.ewm(halflife=half_life, adjust=False).mean().clip(-1, 1)
    return pd.DataFrame({
        'var': _rolling_percentile(variance, roll),
        'dis': _rolling_percentile(dispersion, roll),
        'corr': _rolling_percentile(correlation, roll),
        'ac': _ewm_average_autocorrelation(market, half_life),
    }).replace([np.inf, -np.inf], np.nan).dropna()


def _logsumexp(values, axis=None):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.squeeze()


def _gaussian_log_density(x, means, covariances):
    observations, dimensions = x.shape
    output = np.empty((observations, len(means)))
    for state, (mean, covariance) in enumerate(zip(means, covariances)):
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError('state covariance must be positive definite')
        difference = x - mean
        quadratic = np.einsum('ij,jk,ik->i', difference, np.linalg.inv(covariance), difference)
        output[:, state] = -0.5 * (dimensions * np.log(2 * np.pi) + log_det + quadratic)
    return output


def _forward_backward(log_emissions, initial, transition):
    observations, states = log_emissions.shape
    log_transition = np.log(np.clip(transition, 1e-15, 1.0))
    alpha = np.empty((observations, states))
    alpha[0] = np.log(np.clip(initial, 1e-15, 1.0)) + log_emissions[0]
    for time in range(1, observations):
        alpha[time] = log_emissions[time] + _logsumexp(
            alpha[time - 1][:, None] + log_transition, axis=0)
    log_likelihood = _logsumexp(alpha[-1])
    beta = np.zeros((observations, states))
    for time in range(observations - 2, -1, -1):
        beta[time] = _logsumexp(
            log_transition + log_emissions[time + 1][None, :] + beta[time + 1][None, :], axis=1)
    gamma = np.exp(alpha + beta - log_likelihood)
    xi_sum = np.zeros((states, states))
    for time in range(observations - 1):
        xi_sum += np.exp(
            alpha[time][:, None] + log_transition + log_emissions[time + 1][None, :]
            + beta[time + 1][None, :] - log_likelihood)
    return log_likelihood, gamma, xi_sum


def _kmeans_initialization(x, states, seed):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), states, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(50):
        new_labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for state in range(states):
            if np.any(labels == state):
                centers[state] = x[labels == state].mean(axis=0)
    return labels, centers


def fit_gaussian_hmm(features: pd.DataFrame, states=3, max_iterations=200,
                     tolerance=1e-5, covariance_floor=1e-4, seed=7):
    """Fit a regularized full-covariance Gaussian HMM using Baum-Welch EM."""
    clean = pd.DataFrame(features, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    states = int(states)
    if states < 2 or len(clean) < max(60, states * 20):
        raise ValueError('insufficient complete observations for requested HMM states')
    mean = clean.mean().to_numpy()
    scale = clean.std(ddof=0).replace(0, 1.0).to_numpy()
    x = (clean.to_numpy() - mean) / scale
    labels, state_means = _kmeans_initialization(x, states, seed)
    covariances = []
    for state in range(states):
        subset = x[labels == state]
        covariance = np.cov(subset.T) if len(subset) > 1 else np.eye(x.shape[1])
        covariances.append(np.atleast_2d(covariance) + np.eye(x.shape[1]) * covariance_floor)
    covariances = np.asarray(covariances)
    initial = np.full(states, 1.0 / states)
    transition = np.full((states, states), 0.1 / max(1, states - 1))
    np.fill_diagonal(transition, 0.9)
    previous_likelihood = -np.inf
    for iteration in range(int(max_iterations)):
        emissions = _gaussian_log_density(x, state_means, covariances)
        likelihood, gamma, xi_sum = _forward_backward(emissions, initial, transition)
        weights = gamma.sum(axis=0).clip(1e-12)
        initial = gamma[0] / gamma[0].sum()
        transition = xi_sum + 1e-3
        transition /= transition.sum(axis=1, keepdims=True)
        state_means = gamma.T @ x / weights[:, None]
        for state in range(states):
            difference = x - state_means[state]
            covariances[state] = (
                np.einsum('t,ti,tj->ij', gamma[:, state], difference, difference) / weights[state]
                + np.eye(x.shape[1]) * covariance_floor)
        if likelihood - previous_likelihood < tolerance and likelihood >= previous_likelihood:
            break
        previous_likelihood = likelihood
    # Stable labels: increasing standardized variance-feature mean, or first feature if unnamed.
    variance_position = list(clean.columns).index('var') if 'var' in clean.columns else 0
    order = np.argsort(state_means[:, variance_position])
    inverse = np.argsort(order)
    state_means = state_means[order]
    covariances = covariances[order]
    initial = initial[order]
    transition = transition[np.ix_(order, order)]
    return {
        'feature_columns': list(clean.columns), 'mean': mean, 'scale': scale,
        'initial': initial, 'transition': transition, 'state_means': state_means,
        'covariances': covariances, 'states': states, 'iterations': iteration + 1,
        'log_likelihood': float(previous_likelihood), 'training_index': clean.index,
        'label_inverse': inverse,
    }


def filtered_state_probabilities(model, features: pd.DataFrame, initial_probabilities=None):
    """One-sided HMM filtering: row t never uses observations after t."""
    clean = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    x = (clean.to_numpy(dtype=float) - model['mean']) / model['scale']
    emissions = _gaussian_log_density(x, model['state_means'], model['covariances'])
    probabilities = np.asarray(
        model['initial'] if initial_probabilities is None else initial_probabilities, dtype=float)
    probabilities /= probabilities.sum()
    rows = []
    for time in range(len(clean)):
        prior = probabilities if time == 0 else probabilities @ model['transition']
        log_posterior = np.log(np.clip(prior, 1e-15, 1.0)) + emissions[time]
        probabilities = np.exp(log_posterior - _logsumexp(log_posterior))
        rows.append(probabilities.copy())
    return pd.DataFrame(rows, index=clean.index, columns=[f'state_{i}' for i in range(model['states'])])


def state_sleeve_weights(state_probabilities: pd.DataFrame, sleeve_returns: pd.DataFrame,
                         max_sleeve_weight=0.4):
    """Map training-only state probabilities to positive risk-adjusted sleeve weights."""
    probabilities, returns = state_probabilities.align(sleeve_returns, join='inner', axis=0)
    state_weights = []
    for state in probabilities.columns:
        membership = probabilities[state]
        mean = returns.mul(membership, axis=0).sum() / membership.sum()
        centered = returns.sub(mean)
        risk = centered.pow(2).mul(membership, axis=0).sum().div(membership.sum()).pow(0.5)
        scores = mean.div(risk.replace(0, np.nan)).fillna(0.0).clip(lower=0).to_numpy()
        if scores.sum() == 0:
            scores = np.ones_like(scores)
        weights = scores / scores.sum()
        # Iterative cap with deterministic redistribution.
        for _ in range(len(weights)):
            over = weights > max_sleeve_weight
            if not over.any():
                break
            excess = (weights[over] - max_sleeve_weight).sum()
            weights[over] = max_sleeve_weight
            under = weights < max_sleeve_weight - 1e-12
            weights[under] += excess * weights[under] / weights[under].sum()
        state_weights.append(weights)
    return pd.DataFrame(state_weights, index=state_probabilities.columns, columns=returns.columns)


def learned_state_allocations(state_probabilities, sleeve_returns,
                              max_sleeve_weight=.4, confidence_hurdle=1.0):
    """Learn state-to-strategy/cash allocations from training observations only.

    Strategy weights use positive conditional Sharpe scores.  Exposure is
    reduced toward cash when the best conditional mean lacks statistical
    confidence, preventing every discovered state from becoming fully invested.
    """
    probabilities, returns = pd.DataFrame(state_probabilities).align(
        pd.DataFrame(sleeve_returns), join='inner', axis=0)
    if probabilities.empty or returns.empty:
        raise ValueError('state probabilities and sleeve returns must overlap')
    if not 0 < max_sleeve_weight <= 1 or confidence_hurdle <= 0:
        raise ValueError('invalid allocation cap or confidence hurdle')
    rows = []
    for state in probabilities.columns:
        membership = probabilities[state].clip(lower=0)
        total = membership.sum()
        effective_n = total ** 2 / membership.pow(2).sum() if total > 0 else 0
        mean = returns.mul(membership, axis=0).sum().div(total) if total > 0 else returns.mean() * 0
        variance = returns.sub(mean).pow(2).mul(membership, axis=0).sum().div(total) if total > 0 else returns.var() * 0
        risk = variance.clip(lower=0).pow(.5).replace(0, np.nan)
        sharpe_score = mean.div(risk).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        t_score = mean.div(risk.div(np.sqrt(max(effective_n, 1)))).replace(
            [np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        confidence = float(np.clip(t_score.max() / confidence_hurdle, 0, 1))
        strategy = sharpe_score / sharpe_score.sum() if sharpe_score.sum() > 0 else sharpe_score
        strategy = strategy.clip(upper=max_sleeve_weight)
        strategy *= confidence
        row = strategy.to_dict()
        row['cash'] = max(0., 1. - float(strategy.sum()))
        rows.append(row)
    return pd.DataFrame(rows, index=probabilities.columns).fillna(0.)


def describe_hmm_states(model):
    """Return learned state feature means in original feature units."""
    means = model['state_means'] * model['scale'] + model['mean']
    return pd.DataFrame(means, columns=model['feature_columns'],
                        index=[f'state_{i}' for i in range(model['states'])])


def probability_weighted_allocations(probabilities, mapping, rebalance_every=5,
                                     smoothing_half_life=5):
    allocations = probabilities @ mapping
    allocations = allocations.ewm(halflife=float(smoothing_half_life), adjust=False).mean()
    update = np.arange(len(allocations)) % int(rebalance_every) == 0
    allocations.loc[~update] = np.nan
    return allocations.ffill().dropna()
