from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred, multioutput="uniform_average"):
    """Returns the root mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    multioutput : str, {"raw_values", "uniform_average", "variance_weighted"}, default="uniform_average"
        Defines how to aggregate metric for each output.
    """
    return root_mean_squared_error(y_true, y_pred, multioutput=multioutput)


def r2(y_true, y_pred, multioutput="uniform_average"):
    """Returns the R^2 score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    multioutput : str, {"raw_values", "uniform_average", "variance_weighted"}, default="uniform_average"
        Defines how to aggregate metric for each output.
    """
    return r2_score(y_true, y_pred, multioutput=multioutput)


#: A dictionary of available metrics.
METRIC_REGISTRY = {
    "rmse": rmse,
    "r2": r2,
}
