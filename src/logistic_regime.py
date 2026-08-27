"""Multinomial logistic classifier for training-derived market regimes."""

import numpy as np
import pandas as pd


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_logistic_regime(features, regime_probabilities, l2=0.01,
                        learning_rate=0.05, max_iterations=5000, tolerance=1e-8):
    """Fit soft-label multinomial logistic regression using training rows only."""
    x_frame, y_frame = pd.DataFrame(features).align(
        pd.DataFrame(regime_probabilities), join='inner', axis=0)
    joined = x_frame.join(y_frame, lsuffix='_x', rsuffix='_y').dropna()
    x = joined[x_frame.columns].to_numpy(dtype=float)
    y = joined[y_frame.columns].to_numpy(dtype=float)
    if len(x) < 60:
        raise ValueError('at least 60 complete training observations are required')
    if l2 < 0 or learning_rate <= 0:
        raise ValueError('l2 must be non-negative and learning_rate positive')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.zeros((design.shape[1], y.shape[1]))
    previous_loss = np.inf
    for iteration in range(int(max_iterations)):
        probabilities = _softmax(design @ coefficients)
        penalty = coefficients.copy()
        penalty[0] = 0.0
        gradient = design.T @ (probabilities - y) / len(x) + float(l2) * penalty
        coefficients -= float(learning_rate) * gradient
        loss = -np.mean(np.sum(y * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=1))
        loss += 0.5 * float(l2) * np.sum(penalty ** 2)
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss
    return {
        'feature_columns': list(x_frame.columns), 'state_columns': list(y_frame.columns),
        'mean': mean, 'scale': scale, 'coefficients': coefficients,
        'iterations': iteration + 1, 'loss': float(loss), 'l2': float(l2),
    }


def predict_regime_probabilities(model, features):
    frame = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    x = (frame.to_numpy(dtype=float) - model['mean']) / model['scale']
    design = np.column_stack([np.ones(len(x)), x])
    return pd.DataFrame(
        _softmax(design @ model['coefficients']), index=frame.index,
        columns=model['state_columns'])
