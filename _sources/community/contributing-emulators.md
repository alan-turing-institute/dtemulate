# Contributing Emulators

This guide explains how to contribute new emulator models to `AutoEmulate`.

## Emulator structure

All emulators in AutoEmulate are implemented as `scikit-learn` estimators, making them compatible with scikit-learn's cross-validation, grid-search, and pipeline functionality. Have a look at the [scikit-learn estimator developer guide](https://scikit-learn.org/1.5/developers/develop.html#rolling-your-own-estimator) for more details on how to implement a new emulator.

**Note**: Keep in mind when contributing emuulators that AutoEmulate doesn't currently support time-series or spatial data.

### Core Requirements

Each emulator class must:

1. Live in `autoemulate/emulators/`
2. Inherit from `sklearn.base`'s `BaseEstimator` and `RegressorMixin`
3. Implement the `fit` and `predict` methods
4. Include these additional methods/properties:

   - `get_grid_params()`: Returns a dictionary of parameter values for grid search over hyperparameters
   - `model_name`: Property that returns the emulator name (usually `self.__class__.__name__`)
   - `_more_tags()`: Defines emulator properties like multioutput support

### Getting Started

The easiest way to create a new emulator is to:

1. Look at existing emulators in `autoemulate/emulators/` as templates
2. Run the scikit-learn estimator tests `tests/test_estimators.py` early to catch any implementation issues
3. Add your own tests in `tests/models/`

### Naming Conventions

The `model_name` property allows the emulator to be accessed with both long and short names:

- Long name: The class name (e.g., "RadialBasisFunctions")
- Short name: Uppercase letters from long name (e.g., "rbf")

Make sure your chosen class name:

- Doesn't conflict with existing emulators
- Contains some uppercase letters for the short name
- Is descriptive of the emulation technique

## Testing emulators

We use two types of tests:

1. **Scikit-learn Test Suite**: Add your emulator to `tests/test_estimators.py` to verify scikit-learn compatibility. Not all tests need to pass - use `_more_tags()` to skip incompatible tests. See the [estimator tags overview](https://scikit-learn.org/1.5/developers/develop.html#estimator-tags) for details.

2. **Custom Tests**: Add specific tests for your emulator in `tests/models/` to verify its core functionality (e.g., validating end-to-end functionality of components such as parameter search etc).

## Registering an emulator

After your emulator passes tests:

1. Add it to `model_registry` in `autoemulate/emulators/__init__.py`
2. Set `is_core=False` to make it available but not a default model

## PyTorch emulators

PyTorch emulators require special handling:

1. Put the model architecture in `autoemulate/emulators/neural_networks/`
2. Put the main emulator class in `autoemulate/emulators/`
3. Use [skorch](https://skorch.readthedocs.io/) for scikit-learn compatibility:
   - Create `self.model_` as `NeuralNetRegressor` instance
   - Pass model architecture as first argument
   - Use `self.model_` in `fit` and `predict` methods

See existing PyTorch emulators like `conditional_neural_process.py` for examples.
