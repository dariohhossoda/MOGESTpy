<div id="top"></div>

# Contributing to MOGESTpy

<!-- TABLE OF CONTENTS -->

We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug;
- Discussing the current code state;
- Submitting a fix;
- Proposing new features;
- Becoming a maintainer.

## Write Commits Consistently

We use the following set of conventions on how to write commits (based on [community development best practices suggestions](https://ruanbrandao.com.br/2020/02/04/como-fazer-um-bom-commit)):

### Only One Change per Commit

A commit exists to do one thing, so it's essential for a commit to make only one change. This helps better understand how the project has evolved over time and makes it easier to revert the commit if something goes wrong.

### Commit Title

- The **commit title** consists of a concise sentence that tells **what the commit does**. It should be concise, descriptive, and specific. Having a specific title helps differentiate similar commits by the message.
- Start the commit message title with an uppercase letter.
- Do not use a period at the end of the commit title.
- Use a blank line to separate the commit title and body.
- Limit the commit title line to 72 characters.
- Use the correct tense for the commit message title:
    - E.g.:
        - `Refactor system X to improve readability` ✅
        - `Update project installation documentation` ✅
        - `Remove obsolete methods` ✅
        - `Apply package updates` ❌
        - `Update links in README.md` ❌

### Commit Message

- A **commit message** can have a long description, or message body, and should focus on telling **what was done** in the commit **and why** this change happened.
- Limit the lines of the commit message body to 72 characters.

### Automated Tests

- The project's automated tests must pass without errors before making a commit.

<p align="right">(<a href="#top">back to top</a>)</p>

## We Develop with GitHub

We use GitHub to host the code, track issues and feature requests, as well as accept pull requests.

### We Use [GitHub Flow](https://guides.github.com/introduction/flow/index.html)

All code changes happen through pull requests. Pull requests are the best way to propose changes to the codebase (we use [GitHub Flow](https://guides.github.com/introduction/flow/index.html)):

1. Clone the repository and create your branch from `main` (for patches) or `development` (for new features in new releases);
2. If you added code that should be tested, add tests;
3. If you changed something in components, update the documentation;
4. Ensure the test suite passes;
5. Ensure your code follows the project's style;
6. Create your pull request!

<p align="right">(<a href="#top">back to top</a>)</p>

### Reporting Bugs Using [GitHub Issues](https://github.com/dariohhossoda/MOGESTpy/issues)

We use GitHub issues to track bugs. Report a bug by [opening a new issue](https://github.com/dariohhossoda/MOGESTpy/issues/new/choose).

<p align="right">(<a href="#top">back to top</a>)</p>

## Use Consistent Coding Style

We use the Python Enhancement Proposals (PEP 8) style guide.

```python
def do_something(parameter): # Comment
    """
    Function summary. Eg: This function is awesome

    param parameter: input parameter
    type parameter: str
    param return: function output
    type return: str
    """

    variable_name = parameter + '_foo'

    return variable_name
```
**Import Order**

As a rule of thumb, imports are divided as follows:

```python
import math # Native imports
import os

import pandas as pd # Third-party imports

import my_module # Project module imports


if __name__ == '__main__':
    print('Hello World')
```

<p align="right">(<a href="#top">voltar ao topo</a>)</p>
