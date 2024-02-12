import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from autoemulate.utils import get_model_name


def run_cv(X, y, cv, model, metrics, n_jobs, logger):
    model_name = get_model_name(model)

    # The metrics we want to use for cross-validation
    scorers = {metric.__name__: make_scorer(metric) for metric in metrics}

    logger.info(f"Cross-validating {model_name}...")
    logger.info(f"Parameters: {model.named_steps['model'].get_params()}")

    try:
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scorers,
            n_jobs=n_jobs,
            return_estimator=True,
            return_indices=True,
        )

    except Exception as e:
        logger.error(f"Failed to cross-validate {model_name}")
        logger.error(e)

    return cv_results


def update_scores_df(scores_df, model, cv_results):
    """Updates the scores dataframe with the results of the cross-validation.

    Parameters
    ----------
        scores_df : pandas.DataFrame
            DataFrame with columns "model", "metric", "fold", "score".
        model_name : str
            Name of the model.
        cv_results : dict
            Results of the cross-validation.

    Returns
    -------
        None
            Modifies the self.scores_df DataFrame in-place.

    """
    # Gather scores from each metric
    # Initialise scores dataframe
    for key in cv_results.keys():
        if key.startswith("test_"):
            for fold, score in enumerate(cv_results[key]):
                scores_df.loc[len(scores_df.index)] = {
                    "model": get_model_name(model),
                    "metric": key.split("test_", 1)[1],
                    "fold": fold,
                    "score": score,
                }
    return scores_df


def split_data(X, test_size=0.2, random_state=None, param_search=False):
    """Splits the data into training and testing sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
    param_search : bool
        Whether to split the data for hyperparameter search.

    Returns
    -------
    train_idx : array-like
        Indices of the training set.
    test_idx : array-like
        Indices of the testing set.
    """

    if param_search:
        idxs = np.arange(X.shape[0])
        train_idxs, test_idxs = train_test_split(
            idxs, test_size=test_size, random_state=random_state
        )
    else:
        train_idxs, test_idxs = None, None
    return train_idxs, test_idxs


def single_split(X, test_idxs):
    """Create a single split for sklearn's `cross_validate` function.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    test_idxs : array-like
        Indices of the testing set.

    Returns
    -------
    split_index : sklearn.model_selection.PredefinedSplit
        An instance of the PredefinedSplit class.
    """
    split_index = np.full(X.shape[0], -1)
    split_index[test_idxs] = 0

    return PredefinedSplit(test_fold=split_index)
