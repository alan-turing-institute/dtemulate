# Installation instructions

`AutoEmulate` is a Python package that can be installed in a number of ways. In this section we will describe the main ways to install the package.

## Prerequisites

**Python Version:** `AutoEmulate` requires Python `>=3.10` and `<3.13`.

## Install from GitHub

This is the easiest way to install `AutoEmulate`.

Currently, because we are in active development, it's recommended to install the development version from GitHub:

```bash
pip install git+https://github.com/alan-turing-institute/autoemulate.git
```

## Install from PyPI

To get the latest release from PyPI:

```bash
pip install autoemulate
```

## Install using Poetry

If you'd like to contribute to `AutoEmulate`, you can install the package using Poetry.

* Ensure you have poetry installed. If not, install it following the [official instructions](https://python-poetry.org/docs/).

* Fork the repository on GitHub by clicking the "Fork" button at the top right of the [AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate)

* Clone your forked repository:

```bash
git clone https://github.com/YOUR-USERNAME/autoemulate.git
```

Navigate into the directory:

```bash
cd autoemulate
```

Set up poetry:

```bash
poetry install
```

Enter the poetry shell:

```bash
poetry shell
```
