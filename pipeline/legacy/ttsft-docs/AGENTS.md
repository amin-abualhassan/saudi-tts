# Repository Guidelines

## Project Status

Read `PROJECT_STATUS.md` at the start of project work for the latest verified
results, current focus, open issues, and next steps. Update it after meaningful
milestones or changes of direction, linking to supporting reports and artifacts.
Keep live progress in each job's status/log files and summarize it here only when
useful. Keep `AGENTS.md` focused on working rules rather than progress history.

## Project Structure & Module Organization

This directory is a speech-development workspace containing separate Git repositories; run Git commands inside the relevant checkout.

- `fish-speech/`: Python models in `fish_speech/`, server and CLI utilities in `tools/`, regression tests in `tests/`, and documentation in `docs/`. The React/TypeScript studio lives in `aljawab_webapp/aljawab-voice-studio/`, with code in `src/` and static assets in `public/`.
- `OmniVoice/`: the `omnivoice/` package, training examples and configuration in `examples/`, and documentation in `docs/`.
- `chatter/`: Chatterbox and multilingual fine-tuning checkouts.
- `Splitter/`: diarization scripts and audio inputs/outputs.
- `training/`: preprocessing tools, a separate Fish Speech checkout, datasets, checkpoints, and experiment results. Confirm which checkout you intend to change.

## Build, Test, and Development Commands

Use separate Python environments for each project; their Torch and Transformers requirements differ. There is no workspace-wide build command.

- In `OmniVoice/`, run `uv sync` to install locked dependencies, then `uv run omnivoice-demo --ip 127.0.0.1 --port 8001` for the local demo.
- In `fish-speech/`, run `uv sync --extra cpu` for CPU dependencies; choose a supported CUDA extra from `pyproject.toml` for GPU work.
- In `fish-speech/`, run `python -m unittest discover -s tests` using its environment, and `pre-commit run --all-files` for formatting and file checks.
- In the studio directory, run `npm install`, then `npm run dev`. Use `npm run build`, `npm run lint`, and `npm test` for production compilation, ESLint, and Vitest.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Fish Speech uses Black and isort with the Black profile through pre-commit. Match the studio's two-space indentation and `PascalCase` React components. Keep changes consistent with neighboring code.

## Testing Guidelines

Fish Speech uses `unittest` files named `test_*.py`; studio tests use `*.test.ts` or `*.test.tsx`. Add regression tests for changed behavior. No coverage threshold is configured in the inspected projects. For model changes, document the checkpoint, hardware, sample input, and observed audio results.

## Commit & Pull Request Guidelines

Existing histories use short, descriptive subjects, sometimes prefixed with `fix:` or `docs:`. Follow the target checkout's conventions. Describe the behavior change, link relevant issues, report validation commands/results, and include screenshots for UI changes. Keep credentials, local environments, caches, datasets, generated audio, and checkpoints out of commits.
