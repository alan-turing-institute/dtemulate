# Contributing to the Docs

We welcome all documentation contributions, from fixing small typos to adding comprehensive tutorials. This guide will help you get started.

## Prerequisites

Before contributing, please read our [contributing guide](contributing.md) to set up your development environment and understand our workflow.

## Types of Documentation Contributions

### 1. Fixing typos and small changes

1. Navigate to the relevant file in the `docs/` directory
2. Make your changes
3. Build the docs locally to verify your changes:

   ```bash
   jupyter-book build docs --all
   ```

4. Open the generated file `docs/_build/html/index.html` in your browser to preview.

### 2. Adding tutorials

1. Create a new Jupyter notebook in `docs/tutorials/`
2. Include:
   - Clear introduction and objectives
   - Step-by-step instructions
   - Code examples
3. Add your tutorial to the table of contents:
   - Open `_toc.yml` in the docs/ directory
   - Add an entry for your new tutorial
4. Build and verify the docs as described above

### 3. Updating API documentation

The API documentation is generated from source code docstrings. There are two scenarios:

#### Modifying existing API docs

Simply update the docstring in the source code and rebuild:

```bash
jupyter-book build docs --all
```

#### Adding new API docs

1. Create a new `.rst` file in `docs/community/reference/`
2. Add the file to `_toc.yml`
3. Ensure your source code has comprehensive docstrings
4. Build the documentation
