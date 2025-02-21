# Welcome to AutoEmulate

`AutoEmulate` is a Python library that makes it easy to create accurate and efficient emulators for complex simulations. Under the hood, the package runs a complete machine learning pipeline to compare and optimise a wide range of models, and provides functions for downstream tasks like prediction, sensitivity analysis and calibration.

```{button-ref} getting-started/index
:ref-type: doc
:color: primary
:class: sd-rounded-pill float-left


🚀 Get started
```

## ✨ Why AutoEmulate?

- 🛠️ **Diverse Emulators**: From classic Radial Basis Functions to cutting-edge Neural Processes
- 🪄 **Low-Code**: Data-processing, model comparison, cross-validation, hyperparameter search and more in few lines of code
- 🎯 **Optimized for Emulation**: Optimized for typical emulation scenarios with small to medium datasets (100s-1000s of points) with many inputs and outputs
- 🔌 **Easy Integration**: All emulators are `scikit-learn` compatible, and the underlying `PyTorch` models can be extracted for custom use
- 🔮 **Downstream Applications**: Still early days, but we've got prediction, sensitivity analysis, history matching and more

## 🎓 State-of-the-Art Models

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item} 📈 **Classical**

- Radial Basis Functions
- Second Order Polynomials

:::

:::{grid-item} 🌳 **Machine Learning**

- Random Forests
- Gradient Boosting
- Support Vector Machines
- LightGBM

:::

:::{grid-item} <img src="https://pytorch.org/assets/images/pytorch-logo.png" height="16"/> **Deep Learning**

- Multi-output / Multi-task Gaussian Processes
- Conditional Neural Processes

:::
::::

## 🔗 Quick Links

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} ⚡ Quickstart
:link: https://alan-turing-institute.github.io/autoemulate/getting-started/quickstart
Our quickstart guide will get you up and running in no time
:::

:::{grid-item-card} 📚 Tutorials
:link: https://alan-turing-institute.github.io/autoemulate/tutorials
Learn how to use AutoEmulate with our in-depth tutorials
:::

:::{grid-item-card} 👥 Contributing
:link: https://alan-turing-institute.github.io/autoemulate/community/contributing
Learn how to contribute to AutoEmulate
:::

:::{grid-item-card} 💻 GitHub Repository  
:link: https://github.com/alan-turing-institute/autoemulate
Check out our source code
:::

:::{grid-item-card} 🐛 Issue Tracker
:link: https://github.com/alan-turing-institute/autoemulate/issues
Report bugs or request new features
:::

:::{grid-item-card} 🔍 API Reference
:link: https://alan-turing-institute.github.io/autoemulate/reference
The AutoEmulate API
:::
::::
