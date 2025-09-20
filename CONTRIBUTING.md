Contributing to TokMem

Thanks for your interest in contributing! This guide outlines a simple workflow to help you get started quickly.

How to Contribute
- File an issue: Describe the problem, expected behavior, and context. Include repro steps and environment details when possible.
- Submit a PR: Fork the repo, create a feature branch, commit your changes, and open a pull request.

Development Setup
- Clone: `git clone https://github.com/khalilbalaree/TokMem.git`
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

Coding Guidelines
- Keep changes focused and minimal; avoid unrelated edits.
- Prefer clear names and small functions; add docstrings where useful.
- Format and lint if tools are configured in the repo.
- Include usage notes or comments in scripts when behavior is non-obvious.

Pull Requests
- Branch: `feature/<short-topic>` or `fix/<short-topic>`
- Commits: Write concise, imperative messages (e.g., "Add X", "Fix Y").
- Scope: One focused change per PR. Add a brief summary and testing notes.

Issue Reporting
- Provide: what happened, what you expected, steps to reproduce, logs or errors, environment (OS, Python, CUDA/GPU if relevant).

Code of Conduct
- Be respectful and constructive. Assume good intent. Review others’ work thoughtfully.

Thank you for helping improve TokMem!

