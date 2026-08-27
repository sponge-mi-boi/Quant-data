"""Deterministic single decision-tree regime classifier."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def fit_decision_tree_regime(features, labels, max_depth=3, min_samples_leaf=30):
    x_frame, y_frame = pd.DataFrame(features).align(pd.DataFrame(labels), join='inner', axis=0)
    valid = x_frame.notna().all(axis=1) & y_frame.notna().all(axis=1)
    x = x_frame.loc[valid].to_numpy(float)
    y = y_frame.loc[valid].to_numpy(float).argmax(axis=1)
    if len(x) < 60 or len(np.unique(y)) < 2:
        raise ValueError('insufficient complete observations or class diversity')
    classifier = DecisionTreeClassifier(max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight='balanced', random_state=41)
    classifier.fit(x, y)
    return {'feature_columns': list(x_frame.columns), 'state_columns': list(y_frame.columns),
            'classifier': classifier, 'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf}


def predict_decision_tree_probabilities(model, features):
    frame = pd.DataFrame(features).reindex(columns=model['feature_columns']).dropna()
    available = model['classifier'].predict_proba(frame.to_numpy(float))
    result = pd.DataFrame(0.0, index=frame.index, columns=model['state_columns'])
    for position, class_index in enumerate(model['classifier'].classes_.astype(int)):
        result.iloc[:, class_index] = available[:, position]
    return result
