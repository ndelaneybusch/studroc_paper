
## Code style

- Use Python 3.12+ idioms (e.g., for type hints use | instead of Union[], use list instead of List, etc.).
- Prefer pathlib.Path over os where practical. For type hinting, use os.PathLike (or os.PathLike | str for unions in Python 3.12+), and note that pathlib.Path remains preferred for runtime paths.
- Prefer keyword arguments over positional arguments for any function with more than one argument.
- Use Google style docstrings for all functions.
- NEVER address the user through code comments. Comments must be evergreen - don't acknowledge prior code states or transient instructions.
- Pass data values through pydantic models where reasonable.
- Do not edit comments that are unrelated to the code you are changing. NEVER ADDRESS THE USER OR DESCRIBE YOUR CHANGES THROUGH COMMENTS. Comments MUST be "evergreen" and will not refer to changes from previous code states or user instructions.
- Always aim for maximal quality. Never make accomodations for backwards compatibility, just make sure the current code state is the best possible.

## Writing tests

- Approach testing like a red-teaming exercise. Try hard to elicit failure modes and surprising behavior.
- Every test should mitigate a specific risk to prod behavior. All tests must be well-motivated and a passing test must significantly reduce risk.
- Every test should include only meaningful, non-trivial assertions. For example, rather than testing the presence or absence of something, just directly test the correctness in a rigorous and opinionated way.
- Try hard to create interesting, meaningful, plausible test cases that cover a wide range of inputs and edge cases. Where possible, use existing general test cases in conftest.py.
- "Correct" behavior includes yielding correct values in the correct format, outputs that are sensitive to all inputs, outputs that are consistent with the code's documentation, and outputs that propagate nans (missing values) in expected/documented ways. Name tests to state the behavior being verified, and ensure assertions check outcomes that match the name/docs rather than mirroring implementation details.
- Prefer parametrization to cover input/output variants succinctly. Use pytest.mark.parametrize with readable ids, and parametrize fixtures for complex setup.
- Extract shared setup/teardown into fixtures. Move broadly useful fixtures to conftest.py, choose appropriate scopes (function/module/session), and use yield-fixtures or context managers for teardown.
- Optimize for readability while reducing duplication: prefer fewer, well‑structured tests over many near‑duplicates
- Unit tests of low-level functions should use realistic data cases and minimal mocking. Unit tests of high-level pipeline functions should primarily test routing, error handling, integration with dependencies, output format, and output contents, and should mock heavily (in order to make sure that called functions are receiving the correct information and are called in the correct circumstances, among other things).

## Locations

My huggingface hub location is:
"C:\\Users\\ndela\\.cache\\huggingface\\hub"

## Running commands

Run commands using `uv`, e.g. `uv run python <script.py>`
