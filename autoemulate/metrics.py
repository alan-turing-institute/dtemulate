import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from autoemulate.utils import select_kernel


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


def negative_log_likelihood(K, obs_mean, model_mean):
    """
    Compute the negative log-likelihood for a given set of parameters and observed values.

    Parameters:
        K (Tensor): The covariance matrix (size: N x N) based on the input-output relationship.
        obs_mean (Tensor): The observed mean values (1D tensor or scalar).
        model_mean (Tensor): The predicted mean values of the model (1D tensor or scalar).

    Returns:
        Tensor: The computed negative log-likelihood value.
    """
    Sigma = K  # Combine cross-output and input covariance
    noise_term = 1e-5 * torch.eye(Sigma.shape[0])  # Add numerical stability
    Sigma = noise_term + Sigma

    # Compute the inverse and log determinant of Σ(θ)
    Sigma_inv = torch.inverse(Sigma)
    log_det_Sigma = torch.logdet(Sigma)

    # Compute the difference between observed and predicted means
    diff = (obs_mean - model_mean).reshape(-1, 1)

    # Compute the quadratic term: (y - m(θ))^T Σ(θ)^(-1) (y - m(θ))
    quad_term = torch.matmul(diff.T, torch.matmul(Sigma_inv, diff))

    # Negative log-likelihood
    nll = 0.5 * (
        quad_term + log_det_Sigma + len(diff) * torch.log(torch.tensor(2 * torch.pi))
    )

    return nll


def max_likelihood(
    expectations, obs, lr=0.01, epochs=1000, quantile_threshold=0.10, kernel_name=None
):
    """
    Maximize the likelihood by optimizing the model parameters to fit the observed data.

    Args:
        expectations (tuple): A tuple containing two elements:
            - pred_mean (Tensor): The predicted mean values (could be 1D or 2D tensor).
            - pred_var (Tensor): The predicted variance values (could be 1D or 2D tensor).
        obs (list or tuple): A list or tuple containing:
            - obs_mean (float or Tensor): The observed mean(s).
        lr (float, optional): The learning rate for optimization. Defaults to 0.01.
        epochs (int, optional): Number of epochs to run for optimization. Defaults to 1000.
        quantile_threshold (float, optional): Threshold for defining plausible regions based on NLL. Defaults to 0.10.
        kernel_name (str, optional): The name of the kernel function to use (e.g., "RBF"). Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "LLs": A numpy array of negative log-likelihoods for each parameter set.
            - "plausible_indices": A list of indices for parameter sets with NLL less than or equal to the quantile threshold.
    """
    pred_mean, pred_var = expectations

    obs_mean, obs_var = np.array(obs)
    if obs_mean is not list:
        obs_mean = [obs_mean]
    if obs_var is not list:
        obs_var = np.array([obs_var])

    obs_mean = torch.tensor(obs_mean, dtype=torch.float32, requires_grad=True)
    obs_var = torch.tensor(obs_var, dtype=torch.float32, requires_grad=True)
    # Track negative log-likelihoods for each parameter set
    NLLs = []

    for mean, var in zip(pred_mean, pred_var):
        kernel = select_kernel(kernel_name, length_scale=5000.0)
        K = kernel(mean.reshape(-1, 1))  # X is the input data
        K = torch.tensor(K, dtype=torch.float32, requires_grad=True)
        C = torch.tensor(np.eye(K.size(0)), dtype=torch.float32)

        mean = (
            torch.tensor(mean, dtype=torch.float32)
            if not isinstance(mean, torch.Tensor)
            else mean
        )
        var = (
            torch.tensor(var, dtype=torch.float32)
            if not isinstance(var, torch.Tensor)
            else var
        )
        params = torch.cat((mean.view(-1), var.view(-1))).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)
        # Optimize parameters
        nll_sum = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            nll = negative_log_likelihood(K, obs_mean, mean)
            nll.backward()
            optimizer.step()
            nll_sum += nll.item()
        NLLs.append(nll)
    NLLs = torch.tensor(NLLs)

    # Define plausible regions: example : top 5% of likelihoods
    threshold = torch.quantile(
        NLLs, quantile_threshold
    )  # Get the quantile threshold based on NLL values
    plausible_indices = torch.where(NLLs <= threshold)[0].tolist()
    return {
        "LLs": NLLs.numpy(),
        "plausible_indices": plausible_indices,
    }
