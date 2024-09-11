from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, y_scores, n_bootstraps):
    """Bootstrap AUC

    Bootstraps with the test set to get the 95% CI for AUC

    Parameters
    ----------
    y_true : list
        Binary true
    y_scores : list
        NN scores
    n_bootstraps : int, optional
        how many bootstraps to do, by default 1000

    Returns
    -------
    float, float, float
        mean auc, CI lower bound, CI upper bound
    """
    n_bootstraps = 10000
    # Array to store the ROC AUC scores for each bootstrap sample
    bootstrapped_scores = []

    assert y_true is not None
    assert y_scores is not None

    for _ in range(0, n_bootstraps):
        list_of_all_indices = list(range(len(y_true)))

        # Resample with replacement
        indices = random.choices(
            list_of_all_indices, k=len(list_of_all_indices),
        )
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        # Skip if resample doesn't have both classes
        if len(np.unique(y_true[indices])) < 2:
            continue

        bootstrapped_score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(bootstrapped_score)

    # Calculate the 95% confidence interval
    confidence_lower = np.percentile(bootstrapped_scores, 2.5)
    confidence_upper = np.percentile(bootstrapped_scores, 97.5)

    mean_auc = np.mean(bootstrapped_scores)
    return mean_auc, confidence_lower, confidence_upper
