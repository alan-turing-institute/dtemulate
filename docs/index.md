# Welcome to AutoEmulate

`AutoEmulate` is designed to be an easy, low-code pipeline to create emulators for complex simulations. At it's core, `AutoEmulate`'s `compare()` function implements a typical machine learning workflow including data processing, selecting a good emulator model, optimising model parameters and evaluating its test-set performance. Emulators range from classical models like *Radial Basis Functions* and *Second Order Polynomials* to popular machine learning methods like *Gradient Boosting* and *Support Vector Machines*, as well as modern PyTorch-based models like *Neural Processes* and *Multitask Gaussian Processes*. All default parameters and search spaces for hyperparameter optimisation are chosen to be appropriate for typical emulation problems, i.e. small-ish datasets (100s or 1000s of datapoints) with potentially many features and outputs. We've also implemented Global Sensitivity Analysis as a common use-case for emulators, and plan to add more applications in the future.

**Useful links**:
[Code repository](https://github.com/alan-turing-institute/autoemulate) |
[Issues](https://github.com/alan-turing-institute/autoemulate/issues) |

```{tableofcontents}
```
