import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from skopt.space import Categorical, Integer, Real

from autoemulate.utils import denormalise_y, normalise_y


class SupportVectorMachines(BaseEstimator, RegressorMixin):
    """Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        normalise_y=True,
    ):
        """Initializes a SupportVectorMachines object."""
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.normalise_y = normalise_y

    def fit(self, X, y):
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
        X, y = check_X_y(
            X,
            y,
            multi_output=self._more_tags()["multioutput"],
            y_numeric=True,
            ensure_min_samples=2,
        )

        # required for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = self.max_iter if self.max_iter > 0 else 1

        if self.normalise_y:
            y, self.y_mean_, self.y_std_ = normalise_y(y)
        else:
            y = y
            self.y_mean_ = np.zeros(y.shape[1]) if y.ndim > 1 else 0
            self.y_std_ = np.ones(y.shape[1]) if y.ndim > 1 else 1

        self.model_ = SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predicts the output for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted output values.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = self.model_.predict(X)

        if self.normalise_y:
            y_pred = denormalise_y(y_pred, self.y_mean_, self.y_std_)

        return y_pred

    def get_grid_params(self, search_type="random"):
        """Returns the grid paramaters for the emulator."""
        param_space_random = {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "degree": randint(2, 6),
            "gamma": ["scale", "auto"],
            "coef0": uniform(0.0, 1.0),
            "tol": uniform(1e-5, 1e-3),
            "C": uniform(1.0, 3.0),
            "epsilon": uniform(0.1, 0.3),
            "shrinking": [True, False],
            "cache_size": randint(200, 401),
            "verbose": [False],
            "max_iter": [-1],
        }

        param_space_bayes = {
            "kernel": Categorical(["rbf", "linear", "poly", "sigmoid"]),
            "degree": Integer(2, 5),
            "gamma": Categorical(["scale", "auto"]),
            "coef0": Real(0.0, 1.0),
            "tol": Real(1e-5, 1e-3),
            "C": Real(1.0, 4.0),
            "epsilon": Real(0.1, 0.4),
            "shrinking": Categorical([True, False]),
            "cache_size": Integer(200, 400),
            "verbose": Categorical([False]),
            "max_iter": Categorical([-1]),
        }

        if search_type == "random":
            param_space = param_space_random
        elif search_type == "bayes":
            param_space = param_space_bayes

        return param_space

    def _more_tags(self):
        return {"multioutput": False}
