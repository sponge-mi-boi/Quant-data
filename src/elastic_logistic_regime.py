"""Multinomial logistic regression with proximal elastic-net regularization."""

import numpy as np
import pandas as pd


def _softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_elastic_logistic_regime(features, labels, penalty=.1, l1_ratio=.5,
                                learning_rate=.05, max_iterations=5000, tolerance=1e-8):
    x_frame, y_frame = pd.DataFrame(features).align(pd.DataFrame(labels), join='inner', axis=0)
    valid = x_frame.notna().all(axis=1) & y_frame.notna().all(axis=1)
    x = x_frame.loc[valid].to_numpy(float)
    y = y_frame.loc[valid].to_numpy(float)
    if len(x) < 60 or penalty < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError('invalid training sample or elastic-net parameters')
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    coef = np.zeros((design.shape[1], y.shape[1]))
    previous = np.inf
    l1, l2 = penalty * l1_ratio, penalty * (1 - l1_ratio)
    for iteration in range(max_iterations):
        probability = _softmax(design @ coef)
        gradient = design.T @ (probability - y) / len(x)
        gradient[1:] += l2 * coef[1:]
        updated = coef - learning_rate * gradient
        updated[1:] = np.sign(updated[1:]) * np.maximum(np.abs(updated[1:]) - learning_rate * l1, 0)
        change = np.max(np.abs(updated - coef))
        coef = updated
        if change < tolerance:
            break
        previous = change
    return {'feature_columns': list(x_frame.columns), 'state_columns': list(y_frame.columns),
            'mean': mean, 'scale': scale, 'coefficients': coef, 'penalty': penalty,
            'l1_ratio': l1_ratio, 'iterations': iteration + 1}


def predict_elastic_probabilities(model, features):
    frame = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    design = np.column_stack([np.ones(len(frame)),
                              (frame.to_numpy(float) - model['mean']) / model['scale']])
    return pd.DataFrame(_softmax(design @ model['coefficients']), index=frame.index,
                        columns=model['state_columns'])
