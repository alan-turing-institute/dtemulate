import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from autoemulate.emulators import RandomForest
from autoemulate.experimental_design import LatinHypercube
from autoemulate.sensitivity_analysis import _calculate_layout
from autoemulate.sensitivity_analysis import _check_problem
from autoemulate.sensitivity_analysis import _generate_problem
from autoemulate.sensitivity_analysis import _get_output_names
from autoemulate.sensitivity_analysis import _sobol_analysis
from autoemulate.sensitivity_analysis import _sobol_results_to_df
from autoemulate.sensitivity_analysis import _validate_input
from autoemulate.simulations.projectile import simulate_projectile
from autoemulate.simulations.projectile import simulate_projectile_multioutput


@pytest.fixture
def Xy_1d():
    lhd = LatinHypercube([(-5.0, 1.0), (0.0, 1000.0)])
    X = lhd.sample(100)
    y = np.array([simulate_projectile(x) for x in X])
    return X, y


@pytest.fixture
def Xy_2d():
    lhd = LatinHypercube([(-5.0, 1.0), (0.0, 1000.0)])
    X = lhd.sample(100)
    y = np.array([simulate_projectile_multioutput(x) for x in X])
    return X, y


@pytest.fixture
def model_1d(Xy_1d):
    X, y = Xy_1d
    rf = RandomForest()
    rf.fit(X, y)
    return rf


@pytest.fixture
def model_2d(Xy_2d):
    X, y = Xy_2d
    rf = RandomForest()
    rf.fit(X, y)
    return rf


# test problem checking ----------------------------------------------------------------
def test_check_problem():
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    problem = _check_problem(problem)
    assert problem["num_vars"] == 2
    assert problem["names"] == ["c", "v0"]
    assert problem["bounds"] == [(-5.0, 1.0), (0.0, 1000.0)]


def test_check_problem_invalid():
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
    }
    with pytest.raises(ValueError):
        _check_problem(problem)


def test_check_problem_bounds():
    problem = {"num_vars": 2, "names": ["c", "v0"], "bounds": [(-5.0, 1.0)]}
    with pytest.raises(ValueError):
        _check_problem(problem)


# test output name retrieval --------------------------------------------------
def test_get_output_names_default():
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    output_names = _get_output_names(problem, 1)
    assert output_names == ["y1"]


def test_get_output_names_custom():
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
        "output_names": ["lol1", "lol2"],
    }
    output_names = _get_output_names(problem, 2)
    assert output_names == ["lol1", "lol2"]


def test_get_output_names_invalid():
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
        "output_names": "lol",
    }
    with pytest.raises(ValueError):
        _get_output_names(problem, 1)


# test Sobol analysis ------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sobol_analysis(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }

    Si = _sobol_analysis(model_1d, problem)
    assert isinstance(Si, dict)
    assert "y1" in Si
    assert all(
        key in Si["y1"] for key in ["S1", "S1_conf", "S2", "S2_conf", "ST", "ST_conf"]
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sobol_analysis_2d(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    Si = _sobol_analysis(model_2d, problem)
    assert isinstance(Si, dict)
    assert ["y1", "y2"] == list(Si.keys())


@pytest.fixture
def sobol_results_1d(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    return _sobol_analysis(model_1d, problem)


# # test conversion to DataFrame --------------------------------------------------
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "problem, expected_names",
    [
        (
            {
                "num_vars": 2,
                "names": ["c", "v0"],
                "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
            },
            ["c", "v0", "c-v0"],
        ),
        (None, ["X1", "X2", "X1-X2"]),
    ],
)
def test_sobol_results_to_df(sobol_results_1d, problem, expected_names):
    df = _sobol_results_to_df(sobol_results_1d, problem)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "output",
        "parameter",
        "index",
        "value",
        "confidence",
    ]
    assert expected_names in df["parameter"].unique()
    assert all(isinstance(x, float) for x in df["value"])
    assert all(isinstance(x, float) for x in df["confidence"])


# test plotting ----------------------------------------------------------------


# test _validate_input ----------------------------------------------------------
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_validate_input(sobol_results_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    with pytest.raises(ValueError):
        _validate_input(sobol_results_1d, problem=problem, index="S3")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_validate_input_valid(sobol_results_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    Si = _validate_input(sobol_results_1d, problem=problem, index="S1")
    assert isinstance(Si, pd.DataFrame)


# test _calculate_layout ------------------------------------------------------
def test_calculate_layout():
    n_rows, n_cols = _calculate_layout(1)
    assert n_rows == 1
    assert n_cols == 1


def test_calculate_layout_3_outputs():
    n_rows, n_cols = _calculate_layout(3)
    assert n_rows == 1
    assert n_cols == 3


def test_calculate_layout_custom():
    n_rows, n_cols = _calculate_layout(3, 2)
    assert n_rows == 2
    assert n_cols == 2


# test _generate_problem -----------------------------------------------------


def test_generate_problem():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    problem = _generate_problem(X)
    assert problem["num_vars"] == 2
    assert problem["names"] == ["X1", "X2"]
    assert problem["bounds"] == [[0, 2], [0, 2]]
