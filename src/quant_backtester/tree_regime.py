"""Deterministic random-forest classifier for strategy-regime allocation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def fit_tree_regime(features, regime_labels, n_estimators=200, max_depth=4,
                    min_samples_leaf=20, seed=41):
    x_frame, y_frame = pd.DataFrame(features).align(
        pd.DataFrame(regime_labels), join='inner', axis=0)
    valid = x_frame.notna().all(axis=1) & y_frame.notna().all(axis=1)
    x = x_frame.loc[valid].to_numpy(dtype=float)
    one_hot = y_frame.loc[valid].to_numpy(dtype=float)
    if len(x) < 100:
        raise ValueError('at least 100 complete training observations are required')
    if not np.allclose(one_hot.sum(axis=1), 1.0):
        raise ValueError('regime labels must be one-hot encoded')
    y = one_hot.argmax(axis=1)
    if len(np.unique(y)) < 2:
        raise ValueError('at least two regime classes are required')
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf), class_weight='balanced_subsample',
        random_state=int(seed), n_jobs=-1)
    classifier.fit(x, y)
    return {'feature_columns': list(x_frame.columns),
            'state_columns': list(y_frame.columns), 'classifier': classifier,
            'classes': classifier.classes_.astype(int),
            'n_estimators': int(n_estimators), 'max_depth': max_depth,
            'min_samples_leaf': int(min_samples_leaf), 'seed': int(seed)}


def predict_tree_probabilities(model, features):
    frame = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    available = model['classifier'].predict_proba(frame.to_numpy(dtype=float))
    result = pd.DataFrame(0.0, index=frame.index, columns=model['state_columns'])
    for position, class_index in enumerate(model['classes']):
        result.iloc[:, class_index] = available[:, position]
    return result
