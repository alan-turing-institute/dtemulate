import inspect

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from autoemulate.utils import _ensure_2d


def _validate_inputs(cv_results, model_name):
    """Validates cv_results and model_name for plotting.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    model_name : str
        The name of a model to plot.
    """
    if not cv_results:
        raise ValueError("Run .compare() first.")

    if model_name:
        if model_name not in cv_results:
            raise ValueError(
                f"Model {model_name} not found. Available models are: {cv_results.keys()}"
            )


def _check_multioutput(y, output_index):
    """Checks if y is multi-output and if the output_index is valid."""
    if y.ndim > 1:
        if (output_index > y.shape[1] - 1) | (output_index < 0):
            raise ValueError(
                f"Output index {output_index} is out of range. The index should be between 0 and {y.shape[1] - 1}."
            )
        print(
            f"""Plotting the output variable with index {output_index}. 
To plot other outputs, set `output_index` argument to the desired index."""
        )


def _predict_with_optional_std(model, X_test):
    """Predicts the output of the model with or without uncertainty."""
    # see whether the model is a pipeline or not
    if isinstance(model, Pipeline):
        predict_params = inspect.signature(
            model.named_steps["model"].predict
        ).parameters
    else:
        predict_params = inspect.signature(model.predict).parameters
    # see whether the model has return_std in its predict parameters
    if "return_std" in predict_params:
        y_test_pred, y_test_std = model.predict(X_test, return_std=True)
    else:
        y_test_pred = model.predict(X_test)
        y_test_std = None

    return y_test_pred, y_test_std


def _calculate_subplot_layout(n_plots, n_cols=3):
    """Calculate optimal number of rows and columns for subplots.

    Parameters
    ----------
    n_plots : int
        Number of plots to display
    n_cols : int, optional
        Maximum number of columns allowed, default is 3

    Returns
    -------
    tuple
        (n_rows, n_cols) for the subplot layout
    """
    if n_plots <= 0:
        return (1, 1)

    n_cols = min(n_plots, n_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    return n_rows, n_cols


def _plot_single_fold(
    cv_results,
    X,
    y,
    model_name,
    fold_index,
    ax,
    style="Xy",
    annotation=" ",
    output_index=0,
    input_index=0,
):
    """Plots a single cv fold for a given model.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : str
        The name of the model to plot.
    fold_index : int
        The index of the fold to plot.
    ax : matplotlib.axes.Axes
        The axes on which to plot the results.
    plot : str, optional
        The type of plot to draw:
        "Xy" draws the input features vs. the output variables, including predictions.
        “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual” draws the residuals, i.e. difference between observed and predicted values,
        (y-axis) vs. the predicted values (x-axis).
    annotation : str, optional
        The annotation to add to the plot title. Default is an empty string.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    input_index : int, optional
        The index of the input variable to plot. Default is 0.
    """
    # get cv fold test indices
    test_indices = cv_results[model_name]["indices"]["test"][fold_index]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # get model trained on cv fold train indices
    model = cv_results[model_name]["estimator"][fold_index]
    y_test_pred, y_test_std = _predict_with_optional_std(model, X_test)

    # make sure everything is 2D
    y_test = _ensure_2d(y_test)
    y_test_pred = _ensure_2d(y_test_pred)
    if y_test_std is not None:
        y_test_std = _ensure_2d(y_test_std)

    # check output_index is valid and select the correct column
    if output_index >= y_test.shape[1]:
        raise ValueError(
            f"output_index {output_index} is out of range. The index should be between 0 and {y.shape[1] - 1}."
        )
    y_test = y_test[:, output_index]
    y_test_pred = y_test_pred[:, output_index]
    if y_test_std is not None:
        y_test_std = y_test_std[:, output_index]

    plot_types = ["actual_vs_predicted", "residual_vs_predicted", "Xy"]
    if style not in plot_types:
        raise ValueError(f"Invalid plot type: {style}, must be one of {plot_types}")

    plot_type = style
    if plot_type == "Xy":
        if input_index >= X.shape[1]:
            raise ValueError(
                f"input_index {input_index} is out of range. The index should be between 0 and {X.shape[1] - 1}."
            )
        # if X is multi-dimensional, we need to select the correct column
        if X.ndim > 1:
            X_test = X_test[:, input_index]
        title_suffix = f"{annotation}: {fold_index}"
        _plot_Xy(
            X_test,
            y_test,
            y_test_pred,
            y_test_std,
            ax,
            title=f"{model_name} - {title_suffix}",
            input_index=input_index,
            output_index=output_index,
        )
    else:
        display = PredictionErrorDisplay.from_predictions(
            y_true=y_test,
            y_pred=y_test_pred,
            kind=plot_type,
            ax=ax,
            scatter_kwargs={"edgecolor": "black", "linewidth": 0.5},
            line_kwargs={"linewidth": 1, "color": "#36454F"},
        )
        title_suffix = f"{annotation}: {fold_index}"
        ax.set_title(f"{model_name} - {title_suffix}")


def _plot_best_fold_per_model(
    cv_results,
    X,
    y,
    n_cols=3,
    style="Xy",
    figsize=None,
    output_index=0,
    input_index=0,
):
    """Plots results of the best (highest R^2) cv-fold for each model in cv_results.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    n_cols : int, optional
        The maximum number of columns in the plot. Default is 3.
    plot : str, optional
        The type of plot to draw:
        “standard" or "residual”.
    figsize : tuple, optional
        Width, height in inches. Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    input_index : int, optional
        The index of the input to plot. Default is 0.
    """

    n_models = len(cv_results)
    n_rows, n_cols = _calculate_subplot_layout(n_models, n_cols)

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    # plt.figure(figsize=figsize)
    for i, model_name in enumerate(cv_results):
        best_fold_index = np.argmax(cv_results[model_name]["test_r2"])
        _plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            best_fold_index,
            axs[i],
            style=style,
            annotation="Best CV-fold",
            output_index=output_index,
            input_index=input_index,
        )

    # hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)
    plt.tight_layout()
    return fig


def _plot_model_folds(
    cv_results,
    X,
    y,
    model_name,
    n_cols=3,
    style="Xy",
    figsize=None,
    output_index=0,
    input_index=0,
):
    """Plots all the folds for a given model.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : str
        The name of the model to plot.
    n_cols : int, optional
        The maximum number of columns in the plot. Default is 3.
    plot : str, optional
        The type of plot to draw:
        “standard” or “residual”.
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    input_index : int, optional
        The index of the input to plot. Default is 0.
    """

    n_folds = len(cv_results[model_name]["estimator"])
    n_rows, n_cols = _calculate_subplot_layout(n_folds, n_cols)

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axs = axs.flatten()

    for i in range(n_folds):
        _plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            i,
            axs[i],
            style,
            annotation="CV-fold",
            output_index=output_index,
            input_index=input_index,
        )
    # hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    return fig


def _plot_cv(
    cv_results,
    X,
    y,
    model_name=None,
    n_cols=3,
    style="Xy",
    figsize=None,
    output_index=0,
    input_index=0,
):
    """Plots the results of cross-validation.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : (str, optional)
        The name of the model to plot. If None, the best (largest R^2) fold for each model will be plotted.
    n_cols : int, optional
        The maximum number of columns in the plot. Default is 3.
    plot : str, optional
        The type of plot to draw:
        “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        For multi-output: Index of the output variable to plot.
    input_index : int, optional
        For multi-output: Index of the input variable to plot.
    """

    _validate_inputs(cv_results, model_name)
    _check_multioutput(y, output_index)

    if model_name:
        figure = _plot_model_folds(
            cv_results,
            X,
            y,
            model_name,
            n_cols,
            style,
            figsize,
            output_index,
            input_index,
        )
    else:
        figure = _plot_best_fold_per_model(
            cv_results, X, y, n_cols, style, figsize, output_index, input_index
        )

    return _display_figure(figure)


def _plot_model(
    model,
    X,
    y,
    style="Xy",
    n_cols=3,
    figsize=None,
    input_index=None,
    output_index=None,
):
    """Plots the model predictions vs. the true values.

    Parameters
    ----------
    model : object
        A fitted model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    plot : str, optional
        The type of plot to draw:
        "standard" draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        "residual" draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        "Xy" draws the input features vs. the output variables, including predictions.
    n_cols : int, optional
        The maximum number of columns in the plot. Default is 3.
    figsize : tuple, optional
        Overrides the default figure size.
    input_index : int or list of int, optional
        The index(es) of the input feature(s) to plot for "Xy" plots. If None, all features are used.
    output_index : int or list of int, optional
        The index(es) of the output variable(s) to plot. If None, all outputs are used.
    """
    # Get predictions, with uncertainty if available
    y_pred, y_std = _predict_with_optional_std(model, X)

    n_samples, n_features = X.shape
    n_outputs = y.shape[1] if y.ndim > 1 else 1

    # Handle input and output indices
    if input_index is None:
        input_index = list(range(n_features))
    elif isinstance(input_index, int):
        input_index = [input_index]

    if output_index is None:
        output_index = list(range(n_outputs))
    elif isinstance(output_index, int):
        output_index = [output_index]

    # check that input_index and output_index are valid
    if any(idx >= n_features for idx in input_index):
        raise ValueError(
            f"input_index {input_index} is out of range. The index should be between 0 and {n_features - 1}."
        )
    if any(idx >= n_outputs for idx in output_index):
        raise ValueError(
            f"output_index {output_index} is out of range. The index should be between 0 and {n_outputs - 1}."
        )

    # Calculate number of subplots
    if style == "Xy":
        n_plots = len(input_index) * len(output_index)
    else:
        n_plots = len(output_index)

    # Calculate number of rows
    n_rows, n_cols = _calculate_subplot_layout(n_plots, n_cols)

    # Set up the figure
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axs = axs.flatten()

    # make sure everything is 2D
    y = _ensure_2d(y)
    y_pred = _ensure_2d(y_pred)
    if y_std is not None:
        y_std = _ensure_2d(y_std)

    plot_index = 0
    for out_idx in output_index:
        if style == "Xy":
            for in_idx in input_index:
                if plot_index < len(axs):
                    a = _plot_Xy(
                        X[:, in_idx],
                        y[:, out_idx],
                        y_pred[:, out_idx],
                        y_std[:, out_idx] if y_std is not None else None,
                        ax=axs[plot_index],
                        title=f"$X_{in_idx}$ vs. $y_{out_idx}$",
                        input_index=in_idx,
                        output_index=out_idx,
                    )
                    plot_index += 1
        else:
            if plot_index < len(axs):
                display = PredictionErrorDisplay.from_predictions(
                    y_true=y[:, out_idx],
                    y_pred=y_pred[:, out_idx],
                    kind="actual_vs_predicted"
                    if style == "actual_vs_predicted"
                    else "residual_vs_predicted",
                    ax=axs[plot_index],
                    scatter_kwargs={"edgecolor": "black", "alpha": 0.7},
                    # line_kwargs={"color": "red"},
                )
                axs[plot_index].set_title(
                    f"{style.capitalize().replace('_', ' ')} - Output {out_idx}"
                )
                plot_index += 1

    # Hide any unused subplots
    for ax in axs[plot_index:]:
        ax.set_visible(False)
    plt.tight_layout()

    return _display_figure(fig)


def _plot_Xy(
    X, y, y_pred, y_std=None, ax=None, title="Xy", input_index=0, output_index=0
):
    """
    Plots observed and predicted values vs. features, including 2σ error bands where available.
    """

    # Sort the data
    sort_idx = np.argsort(X).flatten()
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    if y_std is not None:
        y_std_sorted = y_std[sort_idx]

    org_points_color = "Goldenrod"
    pred_points_color = "#6A5ACD"
    pred_line_color = "#6A5ACD"
    ci_color = "lightblue"

    if y_std is not None:
        ax.fill_between(
            X_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            color=ci_color,
            alpha=0.25,
            label="95% Confidence Interval",
        )
    ax.plot(
        X_sorted,
        y_pred_sorted,
        color=pred_line_color,
        label="pred.",
        alpha=0.8,
        linewidth=1,
    )  # , linestyle='--'
    ax.scatter(
        X_sorted,
        y_sorted,
        color=org_points_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="data",
    )
    ax.scatter(
        X_sorted,
        y_pred_sorted,
        color=pred_points_color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        label="pred.",
    )

    ax.set_xlabel(f"$X_{input_index}$", fontsize=13)
    ax.set_ylabel(f"$y_{output_index}$", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Get the handles and labels for the scatter plots
    handles, labels = ax.get_legend_handles_labels()

    # Add legend
    if y_std is not None:
        ax.legend(
            handles[-2:],
            ["data", "pred.(±2σ)"],
            loc="best",
            handletextpad=0,
            columnspacing=0,
            ncol=2,
        )
    else:
        ax.legend(
            handles[-2:],
            ["data", "pred."],
            loc="best",
            handletextpad=0,
            columnspacing=0,
            ncol=2,
        )

    # Calculate R2 score
    r2 = r2_score(y, y_pred)

    ax.text(
        0.05,
        0.05,
        f"R\u00B2 = {r2:.2f}",
        transform=ax.transAxes,
        verticalalignment="bottom",
    )


def _display_figure(fig):
    """
    Display a matplotlib figure appropriately based on the environment (Jupyter notebook or terminal).

    Args:
        fig: matplotlib figure object to display

    Returns:
        fig: the input figure object
    """
    # Are we in Jupyter?
    try:
        is_jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        is_jupyter = False

    if is_jupyter:
        # we don't show otherwise it will double plot
        plt.close(fig)
        return fig
    else:
        # in terminal, show the plot
        plt.show()
        plt.close(fig)
        return fig
