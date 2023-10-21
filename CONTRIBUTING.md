# Contributing

Contributions to QuakeMigrate are welcomed. Whether you have identified a bug or would like to request a new feature, your first stop should be to reach out, either directly or—preferably—via the GitHub Issues panel, to discuss the proposed changes. Once we have had a chance to scope out the proposed changes you can proceed with making your contribution following the instructions below.

Please review and abide by the Code of Conduct, set out at the end of this file, when interacting with this project.

## Getting Started

 * Make sure you have a GitHub account
 * [Download](https://git-scm.com/downloads) and install git
 * Read the [git documentation](https://git-scm.com/book/en/Git-Basics)

## Submitting an Issue
There are currently two types of Issue template to choose from: a [Bug Report](https://github.com/QuakeMigrate/QuakeMigrate/blob/master/.github/ISSUE_TEMPLATE/bug_report.md); or a [Feature Request](https://github.com/QuakeMigrate/QuakeMigrate/blob/master/.github/ISSUE_TEMPLATE/feature_request.md). These contain specific guidance on the information we need to address the respective issue. Please endeavour to complete these templates in as much detail as possible.

## Pull Request Process

1. Fork the repo. If you have already worked on this project before, you may need to bring your personal fork up to date with the main project repo using `git fetch upstream`. Note: This requires you have set an `upstream` for the local copy of your fork. Check this using `git remote -v`. If there is no `upstream`, you can add one with `git remote add upstream https://github.com/QuakeMigrate/QuakeMigrate.git`. Finally, merge the `upstream` version of `master` into your local with `git checkout master; git merge upstream/master`.
2. Install the existing version of the code from source in a fresh virtual environment using `pip install .[dev]`. This will install additional packages that are used for development purposes, but that are not typically required to run QuakeMigrate. Note: If you use macOS or `zsh`, you may need to use `pip install '.[dev]'`.
3. Install the pre-commit config using `pre-commit install` from the base project directory.
4. Ensure all existing tests run with no (unexpected) errors.
5. Make a new branch from `master` using `git checkout -b <branch_name>`. Please try to conform to the suggested naming scheme below, using a short but helpful description:
    - For new features, name your branch `feature/<description>` e.g. `feature/kurtosis_onset`
    - For bugfixes, name your branch `bugfix/<description>`
    - For purely documentation changes, name your branch `docs/<description>`
    - For a new example contribution, name your branch `example/<description>`
6. Make your changes.
    - If you are fixing a bug, consider writing a test that fails due to this bug before making any changes. This way, you can validate your bugfix with a passing test.
    - Try to ensure your commits are atomic—if your contribution has an excessive number of commits for partially completed features/fixes, we will squash them during the merge process later.
    - Any commits that do not conform to the expected standards (defined by the `.pre-commit-config.yaml` file) will be flagged prior to any commit. Address any issues flagged. This process will automatically format your code using Black, which we do to ensure consistency in the code base.
    - Try to adhere to the commit message guidelines in our template. This can be installed locally using `git config commit.template .git-message-template` from the base QuakeMigrate source directory.
7. Any new features are accompanied by an appropriate test.
8. Ensure any and all contributions are properly documented. By this, we mean new functions have a docstring that describes the purpose of the function, as well as the arguments and outputs (including types). We use the NumPy style guide documentation. For examples, the user is encouraged to look at the other functions/classes in QuakeMigrate. We use a line-length of 88 for QuakeMigrate—try to stick to this for docstrings/comments. Code lines will be automatically formatted by Black at the commit stage.
9. Push your local changes to your remote fork, then open a Pull Request to `master` from the GitHub interface. Pull Request merging is handled by the QuakeMigrate project maintainers, whose responsibility it is to help you conform to the expected coding standards.

## Licensing contributions
All submitted contributions, whether features, fixes, or data, must be compatible with the GPLv3 and will be GPLv3 licensed as soon as they are part of QuakeMigrate. By submitting a Pull Request, you are acknowledging this and accepting the terms laid out in the LICENSE file.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at quakemigrate.developers@gmail.com. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
