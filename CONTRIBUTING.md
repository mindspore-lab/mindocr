# MindOCR Contributing Guidelines

Contributions are welcome, and they are greatly appreciated! Every little
helps, and credit will always be given.

## Contributor License Agreement

It's required to sign CLA before your first code submission to MindOCR community.

For individual contributor, please refer to [ICLA online document](https://www.mindspore.cn/icla) for the detailed information.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/mindspore-lab/mindocr/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to fix it.

### Write Documentation

MindOCR could always use more documentation, whether as part of the
official MindOCR docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/mindspore-lab/mindocr/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Getting Started

Ready to contribute? Here's how to set up `mindocr` for local development.

1. Fork the `mindocr` repo on [GitHub](https://github.com/mindspore-lab/mindocr).
2. Clone your fork locally:

   ```shell
   git clone git@github.com:your_name_here/mindocr.git
   ```

   After that, you should add official repository as the upstream repository:

   ```shell
   git remote add upstream git@github.com:mindspore-lab/mindocr
   ```

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development:

   ```shell
   conda create -n mindocr python=3.8
   conda activate mindocr
   cd mindocr
   pip install -e .
   ```

4. Create a branch for local development:

   ```shell
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you finish making changes, check that your changes pass the linters and the tests:

   ```shell
   pre-commit run --show-diff-on-failure --color=always --all-files
   pytest
   ```

   If all static linting are passed, you will get output like:

   ![pre-commit-succeed](https://user-images.githubusercontent.com/74176172/221346245-ea868015-bb09-4e53-aa56-73b015e1e336.png)

   otherwise, you need to fix the warnings according to the output:

   ![pre-commit-failed](https://user-images.githubusercontent.com/74176172/221346251-7d8f531f-9094-474b-97f0-fd5a55e6d3de.png)

   To get pre-commit and pytest, just pip install them into your conda environment.

6. Commit your changes and push your branch to GitHub:

   ```shell
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.7, 3.8 and 3.9, and for PyPy. Check
   https://github.com/mindspore-lab/mindocr/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

You can install the git hook scripts instead of linting with `pre-commit run -a` manually.

run flowing command to set up the git hook scripts

```shell
pre-commit install
```

now `pre-commit` will run automatically on `git commit`!

## Releasing

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```shell
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Action will then deploy to PyPI if tests pass.
