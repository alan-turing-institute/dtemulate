from sklearn.model_selection import GridSearchCV
import logging


class HyperparamSearch:
    """Performs hyperparameter search for a given model."""

    def __init__(self, X, y, cv, n_jobs, param_grid=None, logger=None):
        """Initializes a HyperparamSearch object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        cv : int
            Number of folds in the cross-validation.
        n_jobs : int
            Number of jobs to run in parallel.
        param_grid : dict, default=None
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such dictionaries,
            in which case the grids spanned by each dictionary in the list are
            explored. This enables searching over any sequence of parameter
            settings.
        logger : logging.Logger
            Logger object.
        """
        self.X = X
        self.y = y
        self.cv = cv
        self.n_jobs = n_jobs
        self.param_grid = param_grid
        self.logger = logger or logging.getLogger(__name__)
        self.best_params = {}

    def search(self, model):
        """Performs hyperparameter search for a given model."""
        model_name = type(model.named_steps["model"]).__name__
        self.logger.info(f"Performing grid search for {model_name}...")

        try:
            # TODO: checks that parameters
            param_grid = self.prepare_param_grid(model, self.param_grid)

            grid_search = GridSearchCV(
                model, param_grid, cv=self.cv, n_jobs=self.n_jobs
            )
            grid_search.fit(self.X, self.y)

            best_params = self.extract_best_params(grid_search)
            self.logger.info(f"Best parameters for {model_name}: {best_params}")

            self.best_params = best_params
        except Exception as e:
            self.logger.error(f"Error during grid search for {model_name}: {e}")
            raise
        return model

    @staticmethod
    def prepare_param_grid(model, param_grid=None):
        """Prepares the parameter grid with prefixed parameters."""
        if param_grid is None:
            param_grid = model.named_steps["model"].get_grid_params()
        print(f"param_grid: {param_grid}")
        return {f"model__{key}": value for key, value in param_grid.items()}

    @staticmethod
    def extract_best_params(grid_search):
        """Extracts and formats best parameters from the grid search."""
        best_params = grid_search.best_params_
        return {key.split("model__", 1)[1]: value for key, value in best_params.items()}
