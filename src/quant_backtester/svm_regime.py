"""Causal, time-series-tuned SVM classifier for strategy regimes."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _complete_training_data(features, regime_labels):
    x_frame, y_frame = pd.DataFrame(features).align(
        pd.DataFrame(regime_labels), join='inner', axis=0)
    valid = x_frame.notna().all(axis=1) & y_frame.notna().all(axis=1)
    x = x_frame.loc[valid].to_numpy(dtype=float)
    label_values = y_frame.loc[valid].to_numpy(dtype=float)
    if len(x) < 100:
        raise ValueError('at least 100 complete training observations are required')
    if not np.allclose(label_values.sum(axis=1), 1.0):
        raise ValueError('regime labels must be one-hot encoded')
    return x_frame, y_frame, x, label_values.argmax(axis=1)


def _expanding_splits(count, gap, folds):
    if gap < 0:
        raise ValueError('purge gap must be non-negative')
    boundaries = np.linspace(count // 2, count, folds + 1, dtype=int)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        train_stop = start - gap
        if train_stop >= 50 and stop > start:
            yield np.arange(train_stop), np.arange(start, stop)


def fit_svm_regime(features, regime_labels, c_values=(0.1, 1.0, 10.0),
                   gamma_values=('scale', 0.1, 1.0), purge_gap=5, folds=3):
    """Tune an RBF SVM on purged expanding folds, then refit on all training rows."""
    if purge_gap < 0:
        raise ValueError('purge gap must be non-negative')
    x_frame, y_frame, x, y = _complete_training_data(features, regime_labels)
    if len(np.unique(y)) < 2:
        raise ValueError('at least two regime classes are required')
    candidates = []
    for c_value, gamma_value in product(c_values, gamma_values):
        scores = []
        for train_rows, validation_rows in _expanding_splits(len(x), purge_gap, folds):
            fold_y = y[train_rows]
            if len(np.unique(fold_y)) < 2:
                continue
            scaler = StandardScaler().fit(x[train_rows])
            classifier = SVC(
                C=float(c_value), gamma=gamma_value, kernel='rbf',
                class_weight='balanced',
                decision_function_shape='ovr')
            classifier.fit(scaler.transform(x[train_rows]), fold_y)
            prediction = classifier.predict(scaler.transform(x[validation_rows]))
            scores.append(balanced_accuracy_score(y[validation_rows], prediction))
        if scores:
            candidates.append((float(np.mean(scores)), float(c_value), gamma_value, scores))
    if not candidates:
        raise ValueError('not enough class diversity for purged cross-validation')
    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    best_score, best_c, best_gamma, fold_scores = candidates[0]
    scaler = StandardScaler().fit(x)
    classifier = SVC(
        C=best_c, gamma=best_gamma, kernel='rbf', class_weight='balanced',
        decision_function_shape='ovr')
    classifier.fit(scaler.transform(x), y)
    return {
        'feature_columns': list(x_frame.columns),
        'state_columns': list(y_frame.columns),
        'scaler': scaler,
        'classifier': classifier,
        'classes': classifier.classes_.astype(int),
        'c': best_c,
        'gamma': best_gamma,
        'cv_balanced_accuracy': best_score,
        'cv_fold_scores': [float(value) for value in fold_scores],
        'purge_gap': int(purge_gap),
    }


def predict_svm_scores(model, features):
    """Map SVM decision scores to normalized allocation scores (not calibrated odds)."""
    frame = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    decision = model['classifier'].decision_function(
        model['scaler'].transform(frame.to_numpy(dtype=float)))
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    shifted = decision - decision.max(axis=1, keepdims=True)
    available = np.exp(shifted)
    available /= available.sum(axis=1, keepdims=True)
    result = pd.DataFrame(0.0, index=frame.index, columns=model['state_columns'])
    for position, class_index in enumerate(model['classes']):
        result.iloc[:, class_index] = available[:, position]
    return result
