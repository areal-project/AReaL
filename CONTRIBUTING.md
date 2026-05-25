# Contributing to AReaL

Thank you for your interest in contributing to AReaL! We welcome contributions from
everyone, whether you're fixing bugs, improving documentations, adding new features, or
helping with code reviews. This guide will help you get started.

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before participating and our
[Governance](GOVERNANCE.md) document to understand how the project is managed.

## Table of Contents

- [Quick Start](#quick-start)
- [Tips for Using AI-Assisted Coding](#tips-for-using-ai-assisted-coding)
- [CI/CD](#cicd)

## Quick Start

1. **Fork and Clone:**

   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/AReaL
   cd AReaL
   ```

1. **Install Development Dependencies:**

   Check our
   [installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation.html)
   for detailed setup instructions.

1. **Set Up Pre-commit Hooks:**

   ```bash
   # Install hooks (includes formatting, linting, and commit message checks)
   pre-commit install --install-hooks
   # Subsequent commits will automatically check your files and commit messages:
   git commit -a -m 'feat(engine): my change'
   ```

1. **Find an Issue:**

   - Browse
     [good first issues](https://github.com/areal-project/AReaL/labels/good%20first%20issue)
   - Check [help wanted](https://github.com/areal-project/AReaL/labels/help%20wanted)
     issues
   - Or create a new issue using our
     [issue templates](https://github.com/areal-project/AReaL/issues/new/choose)

1. **Make Your Changes:**

   - Create a branch: `git checkout -b your-feature-name`
   - Make your changes with proper formatting
   - Test your changes following the next step

1. **Test Your Changes:**

   ```bash
   # --sw: step-wise debugging
   # --lf: run the last failed test first
   pytest -sv --sw --lf tests/
   ```

   Our test suite includes:

   - Running all examples to ensure they can execute one RL step
   - Checking individual engine functionalities, including rollout, forward-backward,
     and weight updates
   - Verifying numerical consistency of our packed data format with HuggingFace padded
     input, with and without Ulysses
   - Testing staleness management functionality
   - Ensuring GSM8K SFT loss decreases and RL rewards increase
   - Running other unit tests for individual components

   Some unit tests require multiple GPUs. The entry point scripts are located under
   `tests/torchrun`. In the corresponding test files (e.g.,
   `test_data_redistribution.py`), we use subprocesses to launch distributed experiments
   with `torchrun` and wait for results.

   If you have modified documentation, prepare doc in English and Chinese (use
   [/translate-doc-zh](../en/reference/ai_assisted_dev.md#commands) if needed), then
   build the docs and preview locally:

   ```bash
   ./docs/build_all.sh
   ```

1. **Submit a Pull Request**

We suggest applying our provided agent harness command `/create-pr` whenever possible.
Use that in `claude`, `opencode`, or any other coding agent CLI.

**IMPORTANT**: For new features and code refactoring, please submit a corresponding
issue or open a draft PR to discuss with the core developers before making any code
changes. Directly opening a PR that conflicts with our future [roadmap](ROADMAP.md) may
waste your effort.

## Tips for Using AI-Assisted Coding

See the full
[AI-Assisted Development Guide](https://areal-project.github.io/AReaL/en/reference/ai_assisted_dev.html)
for detailed documentation.

## CI/CD

### Pre-commit Checks

Pre-commit checks run automatically on every PR. CI executes
`pre-commit run --all-files` to verify formatting (Ruff, clang-format, mdformat) and
linting. Commit messages are also validated against
[Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat: ...`,
`fix: ...`, `docs: ...`, `gov: ...`).

As long as you have `pre-commit install --install-hooks` set up locally, your code will
be checked before each commit and your commit messages will be validated automatically.

### Tests

Tests for PRs are triggered when the PR is manually tagged with `safe-to-test`. The test
suite runs on ephemeral GCP compute engines with 2 A100 GPUs (40GB memory).

> **IMPORTANT:** To re-run tests, **DO NOT** click the "Re-run workflow" button on
> GitHub. Instead, remove the `safe-to-test` tag and then add it back.

**Writing Tests for New Features:**

If you have implemented a new feature, we highly recommend writing tests and adding them
to our pytest workflow. Place your test files under `tests/test_*.py` and mark them with
our pre-defined pytest markers:

- `slow`: Tests that take more than 30 seconds to run. These will not run in the CI/CD
  workflow unless also marked with `ci`.
- `ci`: Tests that should run in the CI/CD workflow (only needed for `slow` tests).
- `gpu`: Tests that use a single GPU.
- `multi_gpu`: Tests that use more than one GPU.

Our CI/CD runs tests selected by `pytest -m "not slow or ci"`. Since our CI machines
only have two GPUs, please skip tests that require more than 2 GPUs to prevent CI
failures. For example:

```python
import pytest
from areal.infra.platforms import current_platform

# ordinary tests are supposed to run fast, and will run in CI
def test_fast_operation():
    ...

# slow operations that will NOT run in CI
@pytest.mark.slow
def test_slow_operation():
    ...

# slow operations BUT must be tested in CI
@pytest.mark.slow
@pytest.mark.ci
def test_slow_operation():
    ...

# skip tests for more than 2 GPUs
@pytest.mark.skipif(current_platform.device_count() < 4, reason="This test requires 4 GPUs")
def test_some_multi_gpu_functionality():
    ...
```

### Image Building

The image building workflow can be triggered manually from any branch by users with
write permissions to the repository.

**Triggering the Workflow:**

You can trigger the workflow from any branch using either method:

1. **Via GitHub UI:**

   - Go to **Actions** → **"Build and Test Docker Image"**
   - Click **"Run workflow"** dropdown
   - Select the branch you want to build from
   - Click **"Run workflow"**

1. **Via GitHub CLI:**

   ```bash
   # Build from main
   gh workflow run build-docker-image.yml --ref main

   # Build from a feature branch
   gh workflow run build-docker-image.yml --ref feature/my-changes

   # Build from current branch
   gh workflow run build-docker-image.yml --ref $(git branch --show-current)
   ```

**Pipeline Stages:**

The workflow executes the following stages sequentially:

1. **Build**: Builds the Docker image and pushes it with `:test` tag
1. **Test**: Automatically runs the full test suite using the `:test` image
1. **Promote**: If tests pass, promotes the image by retagging `:test` → `:dev`
1. **Cleanup**: Always deletes the `:test` image from the registry (success or failure)

Building the image from scratch takes approximately 1-2 hours, plus additional time for
running the test suite.

**Normal PR Testing:**

The PR-based test workflow (triggered by the `safe-to-test` label) remains unchanged and
uses the `:dev` image. This allows testing PRs against the last known-good image.

______________________________________________________________________

Thank you for contributing to AReaL! 🙏
