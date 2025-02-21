import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.plotting import _check_multioutput
from autoemulate.plotting import _plot_cv
from autoemulate.plotting import _plot_model
from autoemulate.plotting import _plot_single_fold
from autoemulate.plotting import _predict_with_optional_std
from autoemulate.plotting import _validate_inputs


@pytest.fixture(scope="module")
def ae_single_output():
    X, y = make_regression(n_samples=50, n_features=2, noise=0.5, random_state=42)
    em = AutoEmulate()
    em.setup(X, y, models=["gp", "rbf", "sop"])
    em.compare()
    return em


@pytest.fixture(scope="module")
def ae_multi_output():
    X, y = make_regression(
        n_samples=50, n_features=2, n_targets=2, noise=0.5, random_state=42
    )
    em = AutoEmulate()
    em.setup(X, y, models=["gp", "rbf", "sop"])
    em.compare()
    return em


# ------------------------------ test validate_inputs ------------------------------
def test_validate_inputs_with_empty_cv_results():
    cv_results = {}
    model_name = "model1"
    try:
        _validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Run .compare() first."


def test_validate_inputs_with_invalid_model_name():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model2"
    try:
        _validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert (
            str(e)
            == "Model model2 not found. Available models are: dict_keys(['model1'])"
        )


def test_validate_inputs_with_valid_inputs():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model1"
    _validate_inputs(
        cv_results, model_name
    )  # No exception should be raisedfrom autoemulate.plotting import check_multioutput


# ------------------------------ test check_multioutput ------------------------------
def test_check_multioutput_with_single_output():
    y = np.array([1, 2, 3, 4, 5])
    output_index = 0
    try:
        _check_multioutput(y, output_index)
    except ValueError as e:
        assert False, f"Unexpected ValueError: {str(e)}"


def test_check_multioutput_with_multioutput():
    y = np.array([[1, 2, 3], [4, 5, 6]])
    output_index = 1
    try:
        _check_multioutput(y, output_index)
    except ValueError as e:
        assert False, f"Unexpected ValueError: {str(e)}"


def test_check_multioutput_with_invalid_output_index():
    y = np.array([[1, 2, 3], [4, 5, 6]])
    output_index = 3
    try:
        _check_multioutput(y, output_index)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert (
            str(e)
            == "Output index 3 is out of range. The index should be between 0 and 2."
        )


# ------------------------------ test _predict_with_optional_std --------------------
def test_predict_with_optional_std(ae_single_output):
    # test whether the function correctly returns None for rbf's std
    rbf = ae_single_output.get_model(name="rbf")
    X = ae_single_output.X
    y_pred, y_std = _predict_with_optional_std(rbf, X)
    assert y_pred.shape == (X.shape[0],)
    assert y_std is None

    # test whether the function correctly returns the std for gp
    gp = ae_single_output.get_model(name="gp")
    y_pred, y_std = _predict_with_optional_std(gp, X)
    assert y_pred.shape == (X.shape[0],)
    assert y_std.shape == (X.shape[0],)
    assert np.all(y_std >= 0)


# ------------------------------ test plot_single_fold ------------------------------
def test_plot_single_fold_with_single_output():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a mock cv_results dictionary
    cv_results = {
        "model1": {
            "indices": {"test": [[0, 1, 2, 3, 4]]},
            "estimator": [model],
            "test_r2": [0.9],
        }
    }

    # Create a mock axes object
    fig, ax = plt.subplots()

    # Call the plot_single_fold function
    _plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        style="actual_vs_predicted",
        annotation="Test",
        output_index=0,
    )

    # Assert that the plot is displayed correctly
    assert ax.get_title() == "model1 - Test: 0"
    # assert ax.texts[0].get_text() == "$R^2$ = 0.900"


def test_plot_single_fold_with_multioutput():
    # Generate synthetic data
    X, y = make_regression(
        n_samples=100, n_features=1, n_targets=2, noise=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a mock cv_results dictionary
    cv_results = {
        "model1": {
            "indices": {"test": [[0, 1, 2, 3, 4]]},
            "estimator": [model],
            "test_r2": [0.9],
        }
    }

    # Create a mock axes object
    fig, ax = plt.subplots()

    # Call the plot_single_fold function
    _plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        style="residual_vs_predicted",
        annotation="Test",
        output_index=1,
    )

    # Assert that the plot is displayed correctly
    assert ax.get_title() == "model1 - Test: 0"
    # assert ax.texts[0].get_text() == "$R^2$ = 0.900"


# ------------------------------ test _plot_cv ------------------------------


def test__plot_cv(ae_single_output, monkeypatch):
    # Mock plt.show to do nothing
    monkeypatch.setattr(plt, "show", lambda: None)

    cv_results = ae_single_output.cv_results
    X, y = ae_single_output.X, ae_single_output.y

    # without model name
    fig = _plot_cv(cv_results, X, y)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3

    # with model name
    fig = _plot_cv(cv_results, X, y, model_name="RadialBasisFunctions")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6  # 5 cv folds, but three columns so 6 subplots are made


def test__plot_cv_output_range(ae_multi_output, monkeypatch):
    # Mock plt.show to do nothing
    monkeypatch.setattr(plt, "show", lambda: None)
    cv_results = ae_multi_output.cv_results
    X, y = ae_multi_output.X, ae_multi_output.y

    # check that output index 1 works
    fig_0 = _plot_cv(cv_results, X, y, output_index=0)
    fig_1 = _plot_cv(cv_results, X, y, output_index=1)
    assert isinstance(fig_1, plt.Figure)
    assert len(fig_1.axes) == 3

    # check that fig_0 and fig_1 are different
    assert fig_0 != fig_1

    # check that output index 2 raises an error
    with pytest.raises(ValueError):
        _plot_cv(cv_results, X, y, output_index=2)


def test__plot_cv_input_range(ae_multi_output, monkeypatch):
    # Mock plt.show to do nothing
    monkeypatch.setattr(plt, "show", lambda: None)
    cv_results = ae_multi_output.cv_results
    X, y = ae_multi_output.X, ae_multi_output.y

    # check that input index 1 works
    fig_0 = _plot_cv(cv_results, X, y, input_index=0)
    fig_1 = _plot_cv(cv_results, X, y, input_index=1)
    assert isinstance(fig_0, plt.Figure)
    assert isinstance(fig_1, plt.Figure)
    assert len(fig_1.axes) == 3

    # check that fig_0 and fig_1 are different
    assert fig_0 != fig_1

    # check that input index 2 raises an error (2 features)
    with pytest.raises(ValueError):
        _plot_cv(cv_results, X, y, input_index=2)


# # ------------------------------ most important tests, does it work? ----------------
# # ------------------------------ test plot_cv ----------------------------------


# # test plots with best cv per model, Xy plot
def test_plot_cv(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_single_output.plot_cv(style="Xy")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_plot_cv_input_index(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_single_output.plot_cv(input_index=1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_plot_cv_input_index_out_of_range(ae_single_output):
    with pytest.raises(ValueError):
        ae_single_output.plot_cv(input_index=2)


def test_plot_cv_output_index(ae_multi_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_multi_output.plot_cv(output_index=1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_plot_cv_output_index_out_of_range(ae_multi_output):
    with pytest.raises(ValueError):
        ae_multi_output.plot_cv(output_index=2)


# test plots with best cv per model, standard [;pt]
def test_plot_cv_actual_vs_predicted(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_single_output.plot_cv(style="actual_vs_predicted")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_plot_cv_output_index_actual_vs_predicted(ae_multi_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_multi_output.plot_cv(style="actual_vs_predicted", output_index=1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_plot_cv_output_index_actual_vs_predicted_out_of_range(ae_multi_output):
    with pytest.raises(ValueError):
        ae_multi_output.plot_cv(style="actual_vs_predicted", output_index=2)


# test plots with all cv folds for a single model
def test_plot_cv_model(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_single_output.plot_cv(model="gp")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6  # 5 cv folds, but three columns so 6 subplots are made


def test_plot_cv_model_input_index(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_single_output.plot_cv(model="gp", input_index=1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6


def test_plot_cv_model_output_index(ae_multi_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = ae_multi_output.plot_cv(model="gp", output_index=1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6


def test_plot_cv_model_input_index_out_of_range(ae_single_output):
    with pytest.raises(ValueError):
        ae_single_output.plot_cv(model="gp", input_index=2)


def test_plot_cv_model_output_index_out_of_range(ae_multi_output):
    with pytest.raises(ValueError):
        ae_multi_output.plot_cv(model="gp", output_index=2)


# # ------------------------------ test _plot_model ------------------------------
def test__plot_model_int(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = _plot_model(
        ae_single_output.get_model(name="gp"),
        ae_single_output.X,
        ae_single_output.y,
        style="Xy",
        input_index=0,
        output_index=0,
    )
    assert isinstance(fig, plt.Figure)
    assert all(term in fig.axes[0].get_title() for term in ["X", "y", "vs."])


def test__plot_model_list(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = _plot_model(
        ae_single_output.get_model(name="gp"),
        ae_single_output.X,
        ae_single_output.y,
        style="Xy",
        input_index=[0, 1],
        output_index=[0],
    )
    assert isinstance(fig, plt.Figure)
    assert all(term in fig.axes[1].get_title() for term in ["X", "y", "vs."])


def test__plot_model_int_out_of_range(ae_single_output):
    with pytest.raises(ValueError):
        _plot_model(
            ae_single_output.get_model(name="gp"),
            ae_single_output.X,
            ae_single_output.y,
            style="Xy",
            input_index=3,
            output_index=2,
        )


def test__plot_model_actual_vs_predicted(ae_single_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = _plot_model(
        ae_single_output.get_model(name="gp"),
        ae_single_output.X,
        ae_single_output.y,
        style="actual_vs_predicted",
        input_index=0,
        output_index=0,
    )
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_title() == "Actual vs predicted - Output 0"
