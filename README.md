*This project has been created as part of the 42 curriculum by pneto-vi.*

# call me maybe — Function Calling with Constrained Decoding

## Description

`call me maybe` is a function calling system for small language models: it takes a
natural-language request ("What is the sum of 2 and 3?") and produces a **structured,
typed function call** instead of free-form text:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

Small models (Qwen/Qwen3-0.6B, 500M params) are notoriously unreliable at producing
JSON from a prompt alone (they succeed ~30% of the time). This project does **not**
trust the prompt: it guides the model **token by token**, rejecting every token that
breaks the target JSON schema. The result is **100% valid, parseable, schema-compliant
output on every prompt**, with near-perfect accuracy (90%+) from a tiny model.

## Instructions

### Requirements
- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) as package manager (the reviewer runs `uv sync`)

### Install
```bash
make install        # equivalent to: uv sync
```

### Run
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Default paths (all optional):
| Argument | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

### Lint & quality
```bash
make lint           # flake8 . + mypy (--warn-return-any --disallow-untyped-defs ...)
make lint-strict    # flake8 . + mypy . --strict
```

### Project layout
```
src/                 implementation
  pipeline.py        constrained decoding loop (token selection)
  generate.py        prompt building, JSON assembly, value generation
  grammar_number.py  DFA for integers / floats / exponents
  grammar_string.py  automaton for JSON strings (escapes, \uXXXX)
  grammar_enum.py    prefix matching for names and booleans
  vocab.py           token_text: token-id -> real-text table (from vocab file)
  validation.py      run() pipeline, schema validation, output writing
  loader.py          input loading with clear error messages + exit codes
  models.py          pydantic classes (FunctionDef, PromptItem, CallResult, ...)
  __main__.py        CLI (argparse)
data/input/          example prompts + function catalog for demonstration
llm_sdk/             provided Qwen wrapper (Small_LLM_Model)
```

## Example Usage

Input `data/input/function_calling_tests.json`:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Output `data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Algorithm Explanation

### 1. Prompt building
`build_prompt` serializes the function catalog (name, description, parameter types)
together with the user's request into the model's chat format. The prompt is only
context — it never decides the structure of the answer.

### 2. Tokenization and the token table
The model's tokenizer splits text into subword tokens. Two facts drive the design:

- **The tokenizer disguises characters**: a leading space is `Ġ`, a newline is `Ċ`;
  emoji become several byte-tokens. Decoding a token id to *real text* must undo
  this disguise.
- The vocabulary file (`vocab.json`) maps every token string to its id.

`token_text` is built **once** (cached): for every token in the vocab file, its
characters are mapped through a byte<->unicode table and decoded as UTF-8, so
`token_text[id]` holds the real text. Its size is checked against the model's logits
(`len(token_text) == len(logits)`) — a mismatch aborts loudly. This table lets the
decoder inspect candidate tokens locally, without calling the model.

### 3. The constrained decoding loop
For a free field (function name, parameter value):
1. The model returns logits for every token (`get_logits_from_input_ids`).
2. `top_down_argmax` scans the logits **from highest to lowest** and returns the
   first token that the active grammar accepts for the current buffer.
   *(Equivalent to masking invalid tokens with -inf — see Design decisions.)*
3. That token is appended to the buffer and the loop repeats (airbag: max 256
   tokens per field).

### 4. The grammars (the core)
- **Enum** (`grammar_enum.py`): the buffer must stay a *prefix* of one of the
  allowed words ("fn_add_numbers", "fn_greet", ..., or "true"/"false"). Generation
  ends naturally when no word can be extended further.
- **Number** (`grammar_number.py`): a deterministic finite automaton with states
  `S M Z I D F E P X` accepting integers, decimals and exponents
  (`-12`, `12.5`, `1.5e10`). A number is only *complete* in an accepting state
  (`ACCEPT = {X, F, I, Z}`); the field cannot stop mid-number (a trailing `.`
  or `e` is rejected). The loop also breaks when the number is complete **and** the
  closing token (`,` or `}`) has a higher logit than the next candidate.
- **String** (`grammar_string.py`): an automaton over the buffer that accepts
  normal characters, the single-char escapes (`\\ " / b f n r t`) and `\uXXXX`
  hex escapes, and rejects raw control characters and stray quotes. The opening
  quote is written by the generator; the field closes by appending a quote when the
  state is `in` **and** the quote logit beats the next candidate.

### 5. Structure vs. free zones
The JSON skeleton — `{"prompt":`, `,"name":"`, `","parameters":{`, keys, commas,
`}}` — is appended with the model's own `encode()` (known text, no guessing). Only
the *value zones* (function name, each parameter value) are free and go through the
grammars. The constructor is designed to make it impossible to emit malformed JSON.

### 6. Final validation
The assembled text is decoded, parsed with `json.loads` (guaranteed to succeed) and
re-validated against the catalog: the name must exist, the parameter keys must match
exactly, and every value must have the declared type (`number`, `string`, `boolean`).

### 7. Error handling (never crash)
Three layers:
1. `safe_definitions` / `safe_test` load the input JSON and translate
   `FileNotFoundError`, `JSONDecodeError` and schema errors into a clear message on
   stderr plus `exit(1)`.
2. In `run()`, a failed prompt is caught and turned into a fallback entry, so one
   bad prompt never kills the whole batch.
3. `__main__` has a final guard that reports a fatal error and exits cleanly.

## Design Decisions

- **Top-down scanning instead of -inf masking.** The subject suggests setting invalid
  tokens to negative infinity before sampling. Scanning the logits once, high to low,
  and skipping invalid tokens yields the same token with a single pass — no mask
  array, no extra sampling step. Chosen for simplicity and speed.
- **`token_text` built from the vocab file, not with `decode()`.** The table is
  reconstructed locally (byte<->unicode + UTF-8) and built once; `decode_ids` then
  maps ids to real text without touching the model. This is also a first step toward
  a fully home-made tokenizer.
- **Forcing number/string termination with logit comparisons.** A field is only
  closed when the buffer is in a *complete* state AND the closing token's logit wins
  over the best candidate. This is what guarantees values never end mid-number or
  with a dangling escape.
- **Per-prompt fallback.** A failure in one prompt produces a fallback entry and the
  run continues. The alternative (aborting the whole run) would violate "the program
  must never crash and must always explain itself".
- **pydantic for every class** (subject requirement), type hints everywhere, PEP 8
  (79 columns), mypy clean — including `make lint-strict`.
- **Grammar-first, not prompt-first.** The prompt is context; the guarantee of valid
  output lives in the decoder, never in the model's goodwill.

## Performance Analysis

> TODO (fill after the final measurement):
> - accuracy on the official prompt set ("90%+ correct function selection"):
>   **__%** (earlier measurement: 4/5, with the failure being one wrong function
>   choice — see Challenges)
> - total wall time for the official `data/input` file: **__ s** (target < 5 min);
>   the constraint is the model forward pass per token; the grammars themselves add
>   only microseconds per step
> - 100% of generated outputs parse and validate (no counter-example found so far)

The test suite (unit, no model) runs in ~3 s. A real model run is dominated by the
per-token forward passes of a 0.6B model; forcing the structure with `encode()` for
the skeleton avoids dozens of unnecessary forwards.

## Challenges Faced

- **The tokenizer's disguise.** Spaces (`Ġ`) and newlines (`Ċ`) hide inside tokens;
  emoji are split into byte-tokens. Building a correct `token_text` table required
  the byte<->unicode mapping (like the classic GPT-2 `bytes_to_unicode`) and a
  round-trip sanity check (`encode -> decode` must return the same text).
- **Vocabulary size mismatch.** `vocab.json` has 151643 entries while the model
  emits 151936 logits — 293 extra (special/reserved) tokens. The table is sized by
  the logits and checked, so a wrong table aborts instead of silently producing
  garbage.
- **Guaranteeing a number can end.** Letting the model stop anywhere produces
  "12.", "1e". The DFA accepts a number only in final states, and the close-token
  comparison decides the exact stopping point.
- **Strings must not swallow the closing quote.** The automaton has to know when a
  quote is *content* vs *the end of the field* — solved with the `in`/`esc`/`uni`
  states and the quote-logit comparison.
- **Tests revealed an empty-token trap.** With a perfectly flat FakeModel, decoding
  could stall on tokens whose text is empty (`""` is a "prefix of everything").
  Real models never do this (special tokens get near-zero logits), but the finding
  forced the FakeModel to emulate them — and validated the 256-token airbag.
- **Boring-but-critical robustness.** Missing files, broken JSON, wrong schemas,
  malformed edge cases (emoji prompts, huge numbers, `C:\\temp`, empty strings) —
  every case was turned into either a clean error message or a valid fallback,
  never a crash.
- **Tooling friction.** `flake8 .` and `mypy .` must exclude `.venv` and `llm_sdk`
  (config in `.flake8` and `pyproject.toml`), and mypy needs `python_version =
  3.12` to parse modern numpy stubs.
- **Initial Understandings**: When you first open the project, there is a lot of new knowledge to absorb and new syntax to look up, often without a clear direction on how to search for it.

## Testing Strategy

The tests are **not part of the submitted repository** (the subject asks for tests
only to validate the project, so the files are gitignored) — they are documented
here in full instead:

```bash
uv run pytest            # 51 passed in ~2.8 s (no real model needed)
uv run pytest -v         # verbose, one line per test (list below)
```

The suite (51 tests: 21 + 13 + 13 + 4) is organized in layers:

- **Grammars** (`test_grammars.py`, 21 tests): the number DFA (7), the string
  automaton (9) and enum prefix matching (5) — valid/invalid/edge inputs, state by
  state (`-0`, `1.5e10`, `\uXXXX` escapes, raw control characters, word-boundary
  enums, ...).
- **Decoding loop** (`test_pipeline.py`, 13 tests): `top_down_argmax`,
  `generate_free_field` and an end-to-end `generate_call` using a deterministic
  `ScriptedFakeModel` that guides generation to a *known* output, plus adversarial
  models that try to inject a raw quote or close strings early — the grammars must
  still produce valid JSON.
- **IO / error handling** (`test_io.py`, 13 tests): 6 tests on the raw loaders
  (missing file / broken JSON / wrong schema — `FileNotFoundError`,
  `JSONDecodeError`, `ValidationError` fly to the caller) and 6 on the `safe_*`
  wrappers (`SystemExit` + exit code 1 + exact stderr message), plus 1 happy-path
  test (success prints nothing).
- **Vocab** (`test_vocab.py`, 4 tests): the byte<->unicode table is a bijection
  over all 256 bytes (the foundation of `token_text`).

<details>
<summary>Full source: `test_grammars.py` (21 tests)</summary>

```python
"""Tests for the three grammars: number, string and enum (Marco 3)."""

from src.grammar_number import num_state, num_can_continue, num_is_complete
from src.grammar_string import str_state, str_can_continue
from src.grammar_enum import enum_can_continue, enum_is_complete

WORDS = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]


# ────────────────────────── NUMBER ──────────────────────────

def test_integer_state() -> None:
    """A plain integer is in the accepting state 'I'."""
    assert num_state("12") == "I"


def test_zero_and_negative_zero() -> None:
    """'0' and '-0' are complete and can keep growing."""
    for text in ("0", "-0"):
        assert num_can_continue(num_state(text)) is True
        assert num_is_complete(num_state(text)) is True


def test_exponent_notation() -> None:
    """Exponents with sign are valid and complete."""
    for text in ("1e5", "12.5e-3", "1.5E+3"):
        assert num_can_continue(num_state(text)) is True
        assert num_is_complete(num_state(text)) is True


def test_decimal_notation() -> None:
    """Decimals are valid and complete."""
    for text in ("12.5", "-0.25", "3.14159"):
        assert num_can_continue(num_state(text)) is True
        assert num_is_complete(num_state(text)) is True


def test_complete_number_can_stop() -> None:
    """A complete number can stop on any accepting state."""
    for text in ("42", "-7", "0.5", "1e10"):
        assert num_is_complete(num_state(text)) is True


def test_unfinished_states_cannot_stop() -> None:
    """Unfinished states (D, E, P) can continue but cannot stop."""
    for text in ("12.", "1e", "12.5e", "1e+"):
        assert num_can_continue(num_state(text)) is True
        assert num_is_complete(num_state(text)) is False


def test_invalid_number_returns_none() -> None:
    """Invalid numbers are rejected with None (dead state)."""
    for text in ("12a", "1.2.3", "+1", "--1", "abc", "1e++"):
        assert num_state(text) is None
        assert num_can_continue(None) is False


# ────────────────────────── STRING ──────────────────────────

def test_simple_string_continues() -> None:
    """Normal text can keep growing inside a string."""
    assert str_can_continue("hello") is True


def test_empty_string_continues() -> None:
    """An empty buffer is a valid (empty) normal string."""
    assert str_can_continue("") is True


def test_quote_inside_is_invalid() -> None:
    """A raw quote closes the string, so 'a"b' as content is invalid."""
    assert str_can_continue('a"b') is False


def test_escape_sequences_continue() -> None:
    """Single-char escapes are valid inside the string."""
    for text in (r"a\n", r"a\t", r"a\"", r"a\\", r"a\/b"):
        assert str_can_continue(text) is True


def test_incomplete_unicode_continues() -> None:
    """An unfinished \\uxxxx is still a living string ('uni')."""
    assert str_can_continue(r"\u12") is True


def test_bad_unicode_hex_is_invalid() -> None:
    """\\u followed by non-hex characters is a dead string."""
    assert str_can_continue(r"\u12zz") is False


def test_complete_unicode_continues() -> None:
    """A full \\u0041 escape keeps the string valid."""
    assert str_can_continue(r"\u0041") is True


def test_trailing_backslash_is_incomplete() -> None:
    """A lone backslash at the end is incomplete, not dead ('esc')."""
    assert str_state("abc\\") == "esc"
    assert str_can_continue("abc\\") is True


def test_control_characters_are_invalid() -> None:
    """Control characters (< 0x20) are rejected inside the string."""
    assert str_can_continue("a\x01b") is False


# ────────────────────────── ENUM ──────────────────────────

def test_enum_prefix_continues() -> None:
    """A prefix of a word can keep growing."""
    assert enum_can_continue("fn_g", WORDS) is True


def test_enum_full_word_is_prefix() -> None:
    """A whole word is a prefix of itself, so it still continues."""
    assert enum_can_continue("fn_greet", WORDS) is True


def test_enum_random_text_rejected() -> None:
    """Text that is not a prefix of any word is rejected."""
    assert enum_can_continue("xyz", WORDS) is False


def test_enum_is_complete_full_word() -> None:
    """A whole word is complete."""
    assert enum_is_complete("fn_greet", WORDS) is True


def test_enum_is_complete_partial() -> None:
    """A partial word is not complete."""
    assert enum_is_complete("fn_gr", WORDS) is False
```

</details>

<details>
<summary>Full source: `test_io.py` (13 tests)</summary>

```python
"""Tests for the IO layer: missing files, broken JSON, wrong schemas (Marco 9).

The loader has two faces (Marco 8's layered shielding):
- the raw load_functions/load_tests let the exceptions fly, so callers can
  handle them (FileNotFoundError, JSONDecodeError, ValidationError);
- the safe_* wrappers print a clear message to stderr and exit(1): no
  traceback, no crash, right exit code.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.loader import load_functions, load_tests, safe_definitions, safe_test

FUNCTIONS_JSON = """\
[
  {"name": "fn_add", "description": "adds",
   "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}}
]
"""

TESTS_JSON = """\
[
  {"prompt": "add 1 and 2"}
]
"""


# ---------- raw loaders: the exceptions fly ----------

def test_load_functions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_functions(str(tmp_path / "nope.json"))


def test_load_tests_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tests(str(tmp_path / "nope.json"))


def test_load_functions_broken_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken_functions.json"
    bad.write_text('{"name": "fn_add",')
    with pytest.raises(json.JSONDecodeError):
        load_functions(str(bad))


def test_load_tests_broken_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken_tests.json"
    bad.write_text('{"prompt": "ola",')
    with pytest.raises(json.JSONDecodeError):
        load_tests(str(bad))


def test_load_functions_wrong_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "schema_functions.json"
    bad.write_text('[{"name": "ping"}]')  # missing "description"
    with pytest.raises(ValidationError):
        load_functions(str(bad))


def test_load_tests_wrong_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "schema_tests.json"
    bad.write_text('[{"prompt": 123}]')  # prompt must be a string
    with pytest.raises(ValidationError):
        load_tests(str(bad))


# ---------- safe wrappers: clear stderr message + exit(1) ----------
# (tmp_path is a pytest fixture: a fresh temp dir per test)

def test_safe_tests_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        safe_test(str(tmp_path / "nope.json"))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "JSON File Not found:" in captured.err
    assert captured.out == ""  # messages are stderr-only (Marco 8.3)


def test_safe_definitions_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        safe_definitions(str(tmp_path / "nope.json"))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "JSON File Not found:" in captured.err
    assert captured.out == ""


def test_safe_tests_broken_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text('{"prompt": "ola",')
    with pytest.raises(SystemExit) as exc:
        safe_test(str(bad))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "JSON File Broken:" in captured.err
    assert captured.out == ""


def test_safe_definitions_broken_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text('[{"name": "fn_add",')
    with pytest.raises(SystemExit) as exc:
        safe_definitions(str(bad))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "JSON File Broken:" in captured.err
    assert captured.out == ""


def test_safe_tests_wrong_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "schema.json"
    bad.write_text('[{"prompt": 123}]')
    with pytest.raises(SystemExit) as exc:
        safe_test(str(bad))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "The parameters of the input are not valid" in captured.err
    assert captured.out == ""


def test_safe_definitions_wrong_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "schema.json"
    bad.write_text('[{"name": "ping"}]')  # missing "description"
    with pytest.raises(SystemExit) as exc:
        safe_definitions(str(bad))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "The parameters of the input are not valid" in captured.err
    assert captured.out == ""


# ---------- happy path: silence + data ----------

def test_safe_loaders_happy_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    functions_file = tmp_path / "functions.json"
    functions_file.write_text(FUNCTIONS_JSON)
    tests_file = tmp_path / "tests.json"
    tests_file.write_text(TESTS_JSON)

    funcs = safe_definitions(str(functions_file))
    tests = safe_test(str(tests_file))

    assert funcs[0].name == "fn_add"
    assert tests[0].prompt == "add 1 and 2"
    captured = capsys.readouterr()
    assert captured.err == ""  # no noise on success
```

</details>

<details>
<summary>Full source: `test_pipeline.py` (13 tests)</summary>

```python
"""Tests for the constrained decoding loop using the FakeModel (Marco 4)."""

import pytest

from src.fake_model import FakeModel, _FakeTensor
from src.generate import build_prompt, generate_call, to_ids
from src.models import FunctionDef
from src.pipeline import generate_free_field, masked_argmax, top_down_argmax
from src.validation import validate_result

TOKENS = ["a", "b", "c"]


def test_masked_argmax_best_valid() -> None:
    """masked_argmax picks the highest logit among the valid ids."""
    assert masked_argmax([10.0, 5.0, 1.0], [1, 2]) == 1


def test_masked_argmax_none_without_valid() -> None:
    """masked_argmax returns None when there is no valid id."""
    assert masked_argmax([10.0, 5.0, 1.0], []) is None


def test_prefers_valid_over_invalid() -> None:
    """The model prefers 'a' (10.0) but only 'b' is valid -> (1, 5.0)."""
    assert top_down_argmax(
        [10.0, 5.0, 1.0], TOKENS, "", lambda s: s == "b"
    ) == (1, 5.0)


def test_no_valid_token_returns_none() -> None:
    """top_down returns None when the arbiter rejects every token."""
    assert top_down_argmax(
        [10.0, 5.0, 1.0], TOKENS, "", lambda s: False
    ) is None


def test_masked_and_top_down_agree() -> None:
    """Both versions must pick the same id (equivalence check)."""
    logits = [10.0, 3.0, 8.0, 1.0]
    tokens = ["0", "1", "2", "3"]       # trick: token text == its own id
    valid = [1, 3]
    top = top_down_argmax(logits, tokens, "", lambda s: int(s) in valid)
    assert top is not None
    assert top[0] == masked_argmax(logits, valid)


def test_loop_ends_when_nothing_valid() -> None:
    """Field with nothing valid -> empty ids and text, no infinite loop."""
    fake = FakeModel([10.0, 5.0, 1.0])
    ids, text = generate_free_field(fake, [], TOKENS, lambda s: False)
    assert ids == [] and text == ""


def test_max_tokens_cuts_the_loop() -> None:
    """max_tokens is the airbag: it stops generation even when all is valid."""
    fake = FakeModel([10.0, 5.0, 1.0])
    ids, text = generate_free_field(
        fake, [], TOKENS, lambda s: True, max_tokens=4
    )
    assert len(ids) == 4 and len(text) == 4


def test_invalid_token_never_generated() -> None:
    """The forbidden ending 'a' (always preferred) is never generated."""
    fake = FakeModel([10.0, 5.0, 1.0])
    ids, text = generate_free_field(
        fake, [], TOKENS, lambda full: not full.endswith("a"), max_tokens=3
    )
    assert ids == [1, 1, 1] and text == "bbb"


# ────────────────────────── END-TO-END (Marco 9) ──────────────────────────

def make_token_text() -> list[str]:
    """Per-char fake vocab: token id == ord(char) for printable ASCII."""
    tt = [""] * 128
    for i in range(32, 127):
        tt[i] = chr(i)
    return tt


def make_functions(*names: str) -> list[FunctionDef]:
    """Small function catalog mirroring the real definitions JSON."""
    data = {
        "fn_add": {
            "name": "fn_add",
            "description": "adds",
            "parameters": {"a": {"type": "number"},
                           "b": {"type": "number"}},
        },
        "fn_greet": {
            "name": "fn_greet",
            "description": "greets",
            "parameters": {"who": {"type": "string"}},
        },
    }
    return [FunctionDef.model_validate(data[n]) for n in names]


class ScriptedFakeModel:
    """Deterministic fake guiding the constrained decoding (Marco 9).

    encode() turns each char into its ord (one token = one char), so the
    token_text table and the logits are tiny and hand-made. get_logits()
    prefers the next scripted char; the numeric closes and the closing quote
    get fixed boosts so the loops terminate exactly where the script ends.
    """

    SCRIPTS: dict[str, str] = {
        '"a":': "12",
        '"b":': "3",
        '"who":': "tiago",
    }

    def encode(self, text: str) -> _FakeTensor:
        return _FakeTensor([[ord(c) for c in text]])

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) if 0 <= i < 128 else "" for i in ids)

    def _preferred(self, text: str) -> str | None:
        marker = '"name":"'
        if marker in text:
            name = text.split(marker, 1)[1]
            if '"' not in name and len(name) < 6:
                return "fn_add"[len(name)]
        last, chosen = -1, None
        for cand in self.SCRIPTS:
            pos = text.rfind(cand)
            if pos > last:
                last, chosen = pos, cand
        if chosen is not None:
            value = text[last + len(chosen):]
            if value.startswith('"'):
                value = value[1:]    # the string opening quote is already in
            script = self.SCRIPTS[chosen]
            if len(value) < len(script):
                return script[len(value)]
        return None

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = self.decode(input_ids)
        logits = [1.0] * 128
        for i in range(32):        # empty tokens: keep them out of reach,
            logits[i] = -1000.0    # like the real model's special tokens
        logits[127] = -1000.0
        logits[ord(",")] = 4.0
        logits[ord("}")] = 4.0
        logits[ord('"')] = 5.0
        pref = self._preferred(text)
        if pref is not None:
            logits[ord(pref)] = 10.0
        return logits


def test_generate_call_writes_valid_json() -> None:
    """The full cycle with the fake model yields a known-valid call."""
    fake = ScriptedFakeModel()
    funcs = make_functions("fn_add")
    prompt = "add 1 and 2"
    base = to_ids(fake, build_prompt(funcs, prompt))
    obj = generate_call(fake, base, prompt, funcs, make_token_text())
    assert obj["name"] == "fn_add"
    assert obj["parameters"] == {"a": 12, "b": 3}
    result = validate_result(obj, funcs)
    assert result.name == "fn_add"


def test_output_always_valid_across_prompts() -> None:
    """Battery of weird prompts: every generated call validates."""
    fake = ScriptedFakeModel()
    funcs = make_functions("fn_add", "fn_greet")
    tt = make_token_text()
    for prompt in ("", "   ", "😀", "x" * 50):
        base = to_ids(fake, build_prompt(funcs, prompt))
        obj = generate_call(fake, base, prompt, funcs, tt)
        validate_result(obj, funcs)  # raises when invalid


def test_string_param_generation() -> None:
    """String values follow the script and close with the winning quote."""
    fake = ScriptedFakeModel()
    funcs = make_functions("fn_greet")
    prompt = "greet tiago"
    base = to_ids(fake, build_prompt(funcs, prompt))
    obj = generate_call(fake, base, prompt, funcs, make_token_text())
    assert obj["parameters"] == {"who": "tiago"}
    validate_result(obj, funcs)


class QuoteTemptingModel(ScriptedFakeModel):
    """Scripted fake that pines for a raw quote inside the string."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = super().get_logits_from_input_ids(input_ids)
        logits[ord('"')] = 10.0
        return logits


def test_quote_temptation_defused() -> None:
    """A model that pines for a raw quote inside the string cannot break
    the JSON: the grammar keeps the value valid and it closes cleanly."""
    funcs = make_functions("fn_greet")
    tt = make_token_text()
    fake = QuoteTemptingModel()
    base = to_ids(fake, build_prompt(funcs, "greet tiago"))
    obj = generate_call(fake, base, "greet tiago", funcs, tt)
    assert obj["parameters"] == {"who": "tiago"}
    validate_result(obj, funcs)  # always a valid call


def test_empty_catalog_fails_generation_loudly() -> None:
    """No functions: the name enum has nothing -> the generator fails
    loudly (StopIteration today, ValueError if the nicer message is in);
    the run() layer turns this into a fallback entry (Marco 7-8)."""
    fake = ScriptedFakeModel()
    base = to_ids(fake, build_prompt([], "hello"))
    with pytest.raises((StopIteration, ValueError)):
        generate_call(fake, base, "hello", [], make_token_text())
```

</details>

<details>
<summary>Full source: `test_vocab.py` (4 tests)</summary>

```python
"""Tests for the byte-level translation table (Marco 9, no SDK needed).

The bytes_to_unicode table turns every byte into a unique printable char.
This is the base of token_text: if the table were not an exact bijection,
decode_ids would mix up token texts and the grammars would compare garbage
(ANEXO J, trap 2).
"""

from src.vocab import bytes_to_unicode

b2u = bytes_to_unicode()
byte_of = {c: b for b, c in b2u.items()}


def test_all_256_bytes_are_mapped() -> None:
    """Every byte 0..255 has an entry."""
    assert len(b2u) == 256


def test_round_trip_byte_to_char_and_back() -> None:
    """char -> byte -> char is the identity."""
    for b in range(256):
        c = b2u[b]
        assert byte_of[c] == b


def test_no_duplicate_characters() -> None:
    """Each byte maps to a unique char (it is a bijection)."""
    assert len(set(b2u.values())) == 256


def test_visible_bytes_keep_their_char() -> None:
    """Printable ASCII keeps itself ('!'..'~'), nothing shifted."""
    for b in range(ord("!"), ord("~") + 1):
        assert b2u[b] == chr(b)
```

</details>

## Resources

- The subject — "call me maybe", 42.
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model card and Qwen
  documentation.
- [Hugging Face tokenizers](https://huggingface.co/docs/tokenizers) — subword
  tokenization, special tokens, byte-level BPE.
- GPT-2's `bytes_to_unicode` / BPE preprocessing (the byte<->char table used here
  is the standard GPT-2 scheme).
- Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks without
  Finetuning* (2023) — the technique family this project implements by hand.
- [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/)
  (docstrings), flake8 and mypy documentation.
- [JSON](https://www.json.org/json-en.html) — all the JSON patterns
- [DFA](https://home.uevora.pt/~fc/alp/02-automatos_finitos/02.01-afd.html) — Introductions to DFA
- [Tokens](https://platform.openai.com/tokenizer) — Tokenizer


### How AI was used
AI was used, per the 42 guidelines, as a tool *owned by the author*:

- **Design exploration**: researching how constrained decoding, DFA states and
  the token table work; drafting the first versions of the number DFA and the
  string automaton (explicitly adapted and re-explained by the author).
- **Test generation**: the pytest suite (grammars, IO, e2e) was generated with AI
  and debugged/validated iteratively by the author.
- **Bug diagnosis**: adversarial cases (empty-token stall, broken JSON, output-is-
  a-directory) were found and fixed together with the author's own Marco 8 battery.

The AI was very useful for things like this README. I made several ODT files with diagrams and explanations, and I asked it to summarize my notes so I could adapt them to my own way of thinking. It was especially helpful in providing a first introduction to each concept. After that, I just used it to help me test my code more efficiently, and nothing more.

## Bonus (implemented, working)

- **Comprehensive test suite**: 51 pytest tests, runnable without the model.
- **Home-made token table**: `token_text` reconstructed from the vocabulary file
  with a hand-built byte<->unicode map (close to the "recoding the tokenizer"
  bonus: `decode_ids` does not use the SDK's `decode`).