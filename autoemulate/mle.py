import numpy as np
import torch
from utils import select_kernel


def negative_log_likelihood(Sigma, obs_mean, model_mean):
    """
    Compute the negative log-likelihood for a multivariate Gaussian using NumPy.

    Args:
        Sigma (numpy.ndarray): Covariance matrix (\(\Sigma(\theta)\)).
        obs_mean (numpy.ndarray): Observed mean values (\(y\)).
        model_mean (numpy.ndarray): Predicted mean values (\(m(\theta)\)).

    Returns:
        float: Negative log-likelihood value.
    """
    # Add numerical stability to Sigma
    noise_term = 1e-5 * np.eye(Sigma.shape[0])
    Sigma = Sigma + noise_term

    # Compute log determinant and inverse
    Sigma_inv = np.linalg.inv(Sigma)
    log_det_Sigma = np.linalg.slogdet(Sigma)[1]  # Use slogdet for numerical stability

    # Compute the difference vector
    diff = (obs_mean - model_mean).reshape(-1, 1)

    # Quadratic term: (y - m(θ))^T Σ(θ)^(-1) (y - m(θ))
    quad_term = np.dot(diff.T, np.dot(Sigma_inv, diff))

    # Negative log-likelihood
    nll = 0.5 * (quad_term + log_det_Sigma + len(diff) * np.log(2 * np.pi))

    return nll.item()


def max_likelihood(parameters, model, obs, lr=0.01, epochs=1000, kernel_name="RBF"):
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
    pred_mean = model.predict(parameters)

    obs_mean, obs_var = np.array(obs)
    if obs_mean is not list:
        obs_mean = [obs_mean]
    if obs_var is not list:
        obs_var = np.array([obs_var])

    # Track negative log-likelihoods for each parameter set
    NLLs = []
    kernel = select_kernel(kernel_name, length_scale=5000.0)

    for mean in pred_mean:
        K = kernel(mean.reshape(-1, 1))  # X is the input data
        nll = negative_log_likelihood(K, obs_mean, mean)
        NLLs.append(nll)

    id = np.argmin(NLLs)
    optimizable_params = torch.tensor(parameters[id : id + 1, :], requires_grad=True)
    optimizer = torch.optim.Adam([optimizable_params], lr=lr)
    NLLs = []
    for _ in range(epochs):
        # Optimize parameters
        optimizer.zero_grad()
        mean = model.predict(optimizable_params.detach().numpy())
        K = kernel(mean.reshape(-1, 1))  # X is the input data
        nll = torch.tensor(
            negative_log_likelihood(K, obs_mean, mean),
            dtype=torch.float32,
            requires_grad=True,
        )
        nll.backward()
        optimizer.step()
        NLLs.append(nll.item())

    return {
        "optimized_params": optimizable_params.detach().numpy(),
        "NLLs": NLLs,
    }
