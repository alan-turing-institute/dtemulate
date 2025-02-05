import numpy as np
import torch
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


def history_matching(obs, expectations, threshold=3.0, discrepancy=0.0, rank=1):
    """
    Perform history matching to compute implausibility and identify NROY and RO points.

    Parameters:
        obs (tuple): Observations as (mean, variance).
        expectations (tuple): Predicted (mean, variance).
        threshold (float): Implausibility threshold for NROY classification.
        discrepancy (float or ndarray): Discrepancy value(s).
        rank (int): Rank for implausibility calculation.

    Returns:
        dict: Contains implausibility (I), NROY indices, and RO indices.
    """
    obs_mean, obs_var = np.atleast_1d(obs[0]), np.atleast_1d(obs[1])
    pred_mean, pred_var = np.atleast_1d(expectations[0]), np.atleast_1d(expectations[1])

    discrepancy = np.atleast_1d(discrepancy)
    n_obs = len(obs_mean)
    rank = min(max(rank, 0), n_obs - 1)
    if discrepancy.size == 1:
        discrepancy = np.full(n_obs, discrepancy)

    Vs = pred_var + discrepancy + obs_var
    I = np.abs(obs_mean - pred_mean) / np.sqrt(Vs)

    NROY = np.where(I <= threshold)[0]
    RO = np.where(I > threshold)[0]

    return {"I": I, "NROY": list(NROY), "RO": list(RO)}


def max_likelihood(
    expectations, obs, cov_matrix=None, lr=0.01, epochs=1000, quantile_threshold=0.20
):
    """
    Perform Maximum Likelihood Estimation (MLE) using PyTorch to optimize parameters.

    Parameters:
        obs (tuple): Observations as (mean, variance).
        expectations (tuple): Predicted (mean, variance).
        lr (float): Learning rate for optimizer.
        epochs (int): Number of optimization epochs.

    Returns:
        dict: Contains the log-likelihoods and plausible region indices.
    """
    pred_mean, pred_var = expectations
    obs_mean, obs_var = obs
    model_means = torch.tensor(
        pred_mean, dtype=torch.float32, requires_grad=True
    )  # (n_samples, n_outputs)
    model_vars = torch.tensor(
        pred_var, dtype=torch.float32, requires_grad=True
    )  # (n_samples, n_outputs)
    obs_mean = torch.tensor(obs_mean, dtype=torch.float32)  # (n_outputs,)
    obs_var = torch.tensor(obs_var, dtype=torch.float32)  # (n_outputs,)

    # If no covariance matrix is provided, use a diagonal covariance matrix (diagonal of variances)
    if cov_matrix is None:
        cov_matrix = torch.diag(obs_var)  # (n_outputs, n_outputs)

    optimizer = torch.optim.Adam([model_means, model_vars], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()

        nll = 0.5 * torch.sum(
            torch.log(torch.det(cov_matrix))
            + torch.matmul(obs_mean - model_means, torch.linalg.inv(cov_matrix))
            * obs_mean
            - model_means,
            dim=1,  # Sum over outputs (columns)
        )
        nll.mean().backward()
        optimizer.step()

    final_nll = nll.detach()

    threshold = torch.quantile(
        final_nll, quantile_threshold
    )  # Get the quantile threshold based on NLL values
    plausible_indices = torch.where(final_nll <= threshold)[0].tolist()

    return {"nll_values": final_nll.tolist(), "plausible_indices": plausible_indices}
