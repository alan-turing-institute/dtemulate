# Releasing AutoEmulate

This guide explains how to create new releases of AutoEmulate for maintainers.

## Release Process

AutoEmulate uses GitHub Actions to automatically publish releases to PyPI when a new version tag is pushed. The process is as follows:

1. **Update Version Number**
First, give the package a new version. We recommend [this guide](https://py-pkgs.org/07-releasing-versioning.html) to decide on a version number.

Update the version in `pyproject.toml`:

```toml
[tool.poetry]
name = "autoemulate"
version = "X.Y.Z"  # Update this line
```

2. **Update CHANGELOG.md**
Add a new section to `CHANGELOG.md` describing the changes in this release. Follow the existing format:

```markdown
## [X.Y.Z] - YYYY-MM-DD

- Added feature X
- Fixed bug Y
- Changed Z
```

3. **Create and Push Tag**
Create a new git tag following semantic versioning (vX.Y.Z):

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

4. **Automated Release Process using GitHub Actions**

   When you push a tag matching the pattern 'vX.Y.Z', the release workflow `release.yaml` will automatically:
   - Check out the code
   - Set up Python
   - Install Poetry
   - Install dependencies
   - Build the package
   - Publish to PyPI

   The workflow requires a PyPI token stored in the repository secrets as `PYPI_TOKEN`.

## Prerequisites

Before creating a release, ensure:

1. All tests are passing on the main branch
2. Documentation is up to date
3. CHANGELOG.md is updated
4. You have appropriate permissions to push tags to the repository

## Troubleshooting

If the release fails:

1. Check the GitHub Actions logs for errors
2. Verify the PyPI token is correctly set in repository secrets
3. Ensure the version number in `pyproject.toml` matches the git tag
4. Make sure you haven't already published this version to PyPI

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Commit changes
- [ ] Create and push git tag
- [ ] Monitor GitHub Actions workflow
- [ ] Verify package is available on PyPI
- [ ] Test installation from PyPI

## Notes

- The release workflow only triggers on tags matching 'vX.Y.Z'
- Only maintainers with appropriate permissions can create releases
- Each version can only be published to PyPI once
- The workflow uses Python 3.10 for building and publishing