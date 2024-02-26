import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Categorical
from skopt.space import Integer

from ..types import ArrayLike
from ..types import List
from ..types import Literal
from ..types import MatrixLike
from ..types import Self
from ..types import Union


class SecondOrderPolynomial(BaseEstimator, RegressorMixin):
    """Second order polynomial emulator.

    Creates a second order polynomial emulator. This is a linear model
    including all main effects, interactions and quadratic terms.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial.
    """

    def __init__(self, degree: int = 2):
        """Initializes a SecondOrderPolynomial object."""
        self.degree = degree

    def fit(self, X: MatrixLike, y: ArrayLike) -> Self:
        """Fits the emulator to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.model_ = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree)),
                ("model", LinearRegression()),
            ]
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: MatrixLike) -> ArrayLike:
        """Predicts the output of the emulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            The predicted values.
        """
        X = check_array(X, dtype=np.float64)
        check_is_fitted(self)
        predictions = self.model_.predict(X)
        return predictions

    def get_grid_params(
        self, search_type: Literal["random", "bayes"] = "random"
    ) -> Union[dict, List[tuple[dict[str, Categorical], int]]]:
        """Get the parameter grid for the model.

        Parameters
        ----------
        search_type : str, optional
            The type of parameter search to perform. Can be either 'random' or 'grid'.
            Defaults to 'random'.

        Returns
        -------
        dict
            The parameter grid for the model.
        """
        if search_type == "random":
            param_space = {}
        elif search_type == "bayes":
            param_space = [({"degree": Categorical([2])}, 1)]

        # TODO: Should this raise an error if the search type is not recognised?

        return param_space

    def _more_tags(self):
        """Returns more tags for the estimator.

        Returns
        -------
        dict
            The tags for the estimator.
        """
        return {"multioutput": True}
