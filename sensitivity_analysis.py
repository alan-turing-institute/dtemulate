import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample

from autoemulate.utils import _ensure_2d


def _sensitivity_analysis(
    model, problem=None, X=None, N=1024, conf_level=0.95, as_df=True
):
    """Perform Sobol sensitivity analysis on a fitted emulator.

    Parameters:
    -----------
    model : fitted emulator model
        The emulator model to analyze.
    problem : dict
        The problem definition, including 'num_vars', 'names', and 'bounds', optional 'output_names'.
        Example:
        ```python
        problem = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]],
        }
        ```
    N : int, optional
        The number of samples to generate (default is 1024).
    conf_level : float, optional
        The confidence level for the confidence intervals (default is 0.95).
    as_df : bool, optional
        If True, return a pandas DataFrame (default is True).

    Returns:
    --------
    pd.DataFrame or dict
        If as_df is True, returns a long-format DataFrame with the sensitivity indices.
        Otherwise, returns a dictionary where each key is the name of an output variable and each value is a dictionary
        containing the Sobol indices keys ‘S1’, ‘S1_conf’, ‘ST’, and ‘ST_conf’, where each entry
        is a list of length corresponding to the number of parameters.
    """
    Si = _sobol_analysis(model, problem, X, N, conf_level)

    if as_df:
        return _sobol_results_to_df(Si)
    else:
        return Si


def _check_problem(problem):
    """
    Check that the problem definition is valid.
    """
    if not isinstance(problem, dict):
        raise ValueError("problem must be a dictionary.")

    if "num_vars" not in problem:
        raise ValueError("problem must contain 'num_vars'.")
    if "names" not in problem:
        raise ValueError("problem must contain 'names'.")
    if "bounds" not in problem:
        raise ValueError("problem must contain 'bounds'.")

    if len(problem["names"]) != problem["num_vars"]:
        raise ValueError("Length of 'names' must match 'num_vars'.")
    if len(problem["bounds"]) != problem["num_vars"]:
        raise ValueError("Length of 'bounds' must match 'num_vars'.")

    return problem


def _get_output_names(problem, num_outputs):
    """
    Get the output names from the problem definition or generate default names.
    """
    # check if output_names is given
    if "output_names" not in problem:
        output_names = [f"y{i+1}" for i in range(num_outputs)]
    else:
        if isinstance(problem["output_names"], list):
            output_names = problem["output_names"]
        else:
            raise ValueError("'output_names' must be a list of strings.")

    return output_names


def _generate_problem(X):
    """
    Generate a problem definition from a design matrix.
    """
    if X.ndim == 1:
        raise ValueError("X must be a 2D array.")

    return {
        "num_vars": X.shape[1],
        "names": [f"x{i+1}" for i in range(X.shape[1])],
        "bounds": [[X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])],
    }


def _sobol_analysis(model, problem=None, X=None, N=1024, conf_level=0.95):
    """
    Perform Sobol sensitivity analysis on a fitted emulator.

    Parameters:
    -----------
    model : fitted emulator model
        The emulator model to analyze.
    problem : dict
        The problem definition, including 'num_vars', 'names', and 'bounds'.
    N : int, optional
        The number of samples to generate (default is 1000).

    Returns:
    --------
    dict
        A dictionary where each key is the name of an output variable and each value is a dictionary
        containing the Sobol indices keys ‘S1’, ‘S1_conf’, ‘ST’, and ‘ST_conf’, where each entry
        is a list of length corresponding to the number of parameters.
    """
    # get problem
    if problem is not None:
        problem = _check_problem(problem)
    elif X is not None:
        problem = _generate_problem(X)
    else:
        raise ValueError("Either problem or X must be provided.")

    # saltelli sampling
    param_values = sample(problem, N)

    # evaluate
    Y = model.predict(param_values)
    Y = _ensure_2d(Y)

    num_outputs = Y.shape[1]
    output_names = _get_output_names(problem, num_outputs)

    # single or multiple output sobol analysis
    results = {}
    for i in range(num_outputs):
        Si = analyze(problem, Y[:, i], conf_level=conf_level)
        results[output_names[i]] = Si

    return results


def _sobol_results_to_df(results):
    """
    Convert Sobol results to a (long-format)pandas DataFrame.

    Parameters:
    -----------
    results : dict
        The Sobol indices returned by sobol_analysis.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns: 'output', 'parameter', 'index', 'value', 'confidence'.
    """
    rows = []
    for output, indices in results.items():
        for index_type in ["S1", "ST", "S2"]:
            values = indices.get(index_type)
            conf_values = indices.get(f"{index_type}_conf")
            if values is None or conf_values is None:
                continue

            if index_type in ["S1", "ST"]:
                rows.extend(
                    {
                        "output": output,
                        "parameter": f"X{i+1}",
                        "index": index_type,
                        "value": value,
                        "confidence": conf,
                    }
                    for i, (value, conf) in enumerate(zip(values, conf_values))
                )

            elif index_type == "S2":
                n = values.shape[0]
                rows.extend(
                    {
                        "output": output,
                        "parameter": f"X{i+1}-X{j+1}",
                        "index": index_type,
                        "value": values[i, j],
                        "confidence": conf_values[i, j],
                    }
                    for i in range(n)
                    for j in range(i + 1, n)
                    if not np.isnan(values[i, j])
                )

    return pd.DataFrame(rows)


# plotting --------------------------------------------------------------------


def _validate_input(results, index):
    if not isinstance(results, pd.DataFrame):
        results = _sobol_results_to_df(results)
        # we only want to plot one index type at a time
    valid_indices = ["S1", "S2", "ST"]
    if index not in valid_indices:
        raise ValueError(
            f"Invalid index type: {index}. Must be one of {valid_indices}."
        )
    return results[results["index"].isin([index])]


def _calculate_layout(n_outputs, n_cols=None):
    if n_cols is None:
        n_cols = 3 if n_outputs >= 3 else n_outputs
    n_rows = int(np.ceil(n_outputs / n_cols))
    return n_rows, n_cols


def _create_bar_plot(ax, output_data, output_name):
    """Create a bar plot for a single output."""
    bar_color = "#4C4B63"
    x_pos = np.arange(len(output_data))

    bars = ax.bar(
        x_pos,
        output_data["value"],
        color=bar_color,
        yerr=output_data["confidence"].values / 2,
        capsize=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(output_data["parameter"], rotation=45, ha="right")
    ax.set_ylabel("Sobol Index")
    ax.set_title(f"Output: {output_name}")


def _plot_sensitivity_analysis(results, index="S1", n_cols=None, figsize=None):
    """
    Plot the sensitivity analysis results.

    Parameters:
    -----------
    results : pd.DataFrame
        The results from sobol_results_to_df.
    index : str, default "S1"
        The type of sensitivity index to plot.
        - "S1": first-order indices
        - "S2": second-order/interaction indices
        - "ST": total-order indices
    n_cols : int, optional
        The number of columns in the plot. Defaults to 3 if there are 3 or more outputs,
        otherwise the number of outputs.
    figsize : tuple, optional
        Figure size as (width, height) in inches.If None, automatically calculated.

    """
    with plt.style.context("fast"):
        # prepare data
        results = _validate_input(results, index)
        unique_outputs = results["output"].unique()
        n_outputs = len(unique_outputs)

        # layout
        n_rows, n_cols = _calculate_layout(n_outputs, n_cols)
        figsize = figsize or (4.5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        elif n_outputs == 1:
            axes = [axes]

        for ax, output in zip(axes, unique_outputs):
            output_data = results[results["output"] == output]
            _create_bar_plot(ax, output_data, output)

        # remove any empty subplots
        for idx in range(len(unique_outputs), len(axes)):
            fig.delaxes(axes[idx])

        index_names = {
            "S1": "First-Order",
            "S2": "Second-order/Interaction",
            "ST": "Total-Order",
        }

        # title
        fig.suptitle(
            f"{index_names[index]} indices and 95% CI",
            fontsize=14,
        )

        plt.tight_layout()
        # prevent double plotting in notebooks
        plt.close(fig)

    return fig
