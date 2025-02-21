# First-Time Users' Frequently Asked Questions

## General Questions

1. What is `AutoEmulate`?
   <!-- A brief description of what the package does, its main features, and its intended use case. -->
   - A Python package that makes it easy to create emulators for complex simulations. It takes a set of simulation inputs `X` and outputs `y`, and automatically fits, optimises and evaluates various machine learning models to find the best emulator model. The emulator model can then be used as a drop-in replacement for the simulation, but will be much faster and computationally cheaper to evaluate. We have also implemented global sensitivity analysis as a common emulator application and working towards making `AutoEmulate` a true end-to-end package for building emulators.

2. How do I know whether `AutoEmulate` is the right tool for me?
   - You need to build an emulator for a simulation.
   - You want to do global sensitivity analysis
   - Your inputs `X` and outputs `y` are numeric and complete (we don't support missing data yet).
   - You have one or more input parameters and one or more output variables.
   - You have a small-ish dataset in the order of hundreds to few thousands of samples. All default emulator parameters and search spaces are optimised for smaller datasets.

3. Does `AutoEmulate` support multi-output data?
   - Yes, all models support multi-output data. Some do so natively, others are wrapped in a `MultiOutputRegressor`, which fits one model per target variable.

4. Does `AutoEmulate` support temporal or spatial data?
   - Not explicitly. The train-test split just takes a random subset as a test set, so does KFold cross-validation.

5. `AutoEmulate` takes a long time to run on my dataset, why?
   - The package fits a lot of models, in particular when hyperparameters are optimised. With say 8 default models and 5-fold cross-validation, this amounts to 40 model fits. With the addition of hyperparameter optimisation (n_iter=20), this results in 800 model fits. Some models such as Gaussian Processes and Neural Processes will take a long time to run on a CPU. However, don't despair! There is a [speeding up AutoEmulate guide](../../tutorials/02_speed.ipynb). As a rule of thumb, if your dataset is smaller than 1000 samples, you should be fine, if it's larger and you want to optimise hyperparameters, you might want to read the guide.

## Usage Questions

1. What data do I need to provide to `AutoEmulate` to build an emulator?
   <!-- A simple example to get a new user started, possibly pointing to more detailed tutorials or documentation. -->
   - You'll need two input objects: `X` and `y`. `X` is an ndarray / Pandas DataFrame of shape `(n_samples, n_parameters)` and `y` is an ndarray / Pandas DataFrame of shape `(n_samples, n_outputs)`. Each sample here is a simulation run, so each row of `X` corresponds to a set of input parameters and each row of `y` corresponds to the corresponding simulation output. You'll usually have created `X` using Latin hypercube sampling or similar methods, and `y` by running the simulation on these `X` inputs.

2. Can I use `AutoEmulate` for commercial purposes?
   <!-- Information on licensing and any restrictions on use. -->
   - Yes. It's licensed under the MIT license, which allows for commercial use. See the [license](../../../LICENSE) for more information.

## Advanced Usage

1. Does AutoEmulate support parallel processing or high-performance computing (HPC) environments?
   <!-- Details on the software's capabilities to leverage multi-threading, distributed computing, or HPC resources to speed up computations. -->
   - Yes, [AutoEmulate.setup()](../../reference/compare.rst) has an `n_jobs` parameter which allows to parallelise cross-validation and hyperparameter optimisation. We are also working on GPU support for some models.

2. Can AutoEmulate be integrated with other data analysis or simulation tools?
   <!-- Information on APIs, file formats, or protocols that facilitate the integration of AutoEmulate with other software ecosystems. -->
   - `AutoEmulate` takes simple `X` and `y` ndarrays as input, and returns emulators which are [scikit-learn estimators](https://scikit-learn.org/1.5/developers/develop.html), that can be saved and loaded, and used like any other scikit-learn model.

## Data Handling

1. What are the best practices for data preprocessing before using `AutoEmulate`?
   <!-- Tips and recommendations on preparing data, including normalisation, dealing with missing values, or data segmentation. -->
   - The user will typically run their simulation on a selected set of input parameters (-> experimental design) using a latin hypercube or other sampling method. `AutoEmulate` currently needs all inputs to be numeric and we don't support missing data. By default, `AutoEmulate` will scale the input data to zero mean and unit variance, and for some models it will also scale the output data. There's also the option to do dimensionality reduction in `setup()`.

## Troubleshooting

1. What common issues might I encounter when using `AutoEmulate`, and how can I solve them?
   <!-- A list of frequently encountered problems with suggested solutions, possibly linked to a more extensive troubleshooting guide. -->
   - `AutoEmulate.setup()` has a `log_to_file` option to log all warnings/errors to a file. It also has a `verbose` option to print more information to the console. If you encounter an error, please open an issue (see below).
   - One common issue is that the Jupyter notebook kernel crashes when running `compare()` in parallel, often due to `LightGBM`. In this case, we recommend either specifying `n_jobs=1` or selecting specific (non-LightGBM) models in `setup()` with the `models` parameter.
2. How can I report a bug or request a feature in `AutoEmulate`?
   <!-- Instructions on the proper channels for reporting issues or suggesting enhancements, including any templates or information to include. -->
   - You can report a bug or request a new feature through the [issue templates](https://github.com/alan-turing-institute/autoemulate/issues/new/choose) in our GitHub repository. Head on over there and choose one of the templates for your purpose and get started.

## Community and Learning Resources

1. Are there any community projects or collaborations using `AutoEmulate` I can join or learn from?
   <!-- Information on community-led projects, study groups, or collaborative research initiatives involving AutoEmulate. -->
   - Reach out to Martin ([email](mailto:mstoffel@turing.ac.uk)) or Sophie ([email](mailto:sarana@turing.ac.uk)) for more information.

2. Where can I find tutorials or case studies on using `AutoEmulate`?
   <!-- Directions to comprehensive learning materials, such as video tutorials (if we want to record that), written guides, or published research papers using AutoEmulate. -->
   - See the [tutorial](../../tutorials/01_start.ipynb) for a comprehensive guide on using the package. Case studies are coming soon.

3. How can I stay updated on new releases or updates to AutoEmulate?
   <!-- Guidance on subscribing to newsletters when/if we will have that, community calls if we start that, following the project on social media if we want to create those platforms, or joining community forums/Slack once we have that ready... -->
   - Watch the [AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate).

4. What support options are available if I need help with AutoEmulate?
   <!-- Overview of support resources, including documentation, community forums/Slack when we have that ready... -->
   - Please open an issue on GitHub or contact the maintainer ([email](mailto:mstoffel@turing.ac.uk)) directly.
