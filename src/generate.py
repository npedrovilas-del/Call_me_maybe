import json

import numpy as np

from llm_sdk import Small_LLM_Model
from src.grammar_enum import enum_can_continue
from src.pipeline import generate_free_field
from src.grammar_number import num_can_continue, num_state, num_is_complete
from src.pipeline import top_down_argmax
from src.grammar_string import str_can_continue, str_state
from src.models import FunctionDef

GRAMMARS = {
    "number": lambda full: num_can_continue(num_state(full)),
    "string": lambda full: str_can_continue(full),
    "boolean": lambda full: enum_can_continue(full, ["true", "false"]),
}

STRUCT_TAIL = "},]"
SCAN = 64


def to_ids(model: Small_LLM_Model, text: str) -> list[int]:
    """Encodes a text into token ids, ready to append to the stream.
    The structure text (keys, quotes, braces) is known beforehand, so unlike
    the free zones we don't have to guess it token by token — encode() does
    the splitting for us.
    """
    ids: list[int] = model.encode(text).tolist()[0]
    return ids


def decode_ids(ids: list[int], token_text: list[str]) -> str:
    """Turns token ids back into real text.
    Joins the decoded piece of each id (from the byte-level token_text table)
    into one string — this is what produces the final JSON text.
    """
    return "".join(token_text[i] for i in ids)


def build_prompt(functions: list[FunctionDef], prompt: str) -> str:
    """Builds the first context based on the prompt to start
    the machine generation output"""
    system = (
        "You will receive text to complete. Complete the JSON object:\n"
        '{"prompt": <user request>, "name": ..., "parameters": {...}}.\n'
        "Choose ONE of the functions above and fill the arguments correctly."
    )
    catalog = json.dumps([f.model_dump() for f in functions], indent=2)
    user = "USER: " + prompt
    final = system + "\n\n" + catalog + "\n\n" + user
    return final


def generate_call(
    model: Small_LLM_Model, base_ids: list[int], prompt: str,
    functions: list[FunctionDef], token_text: list[str],
) -> dict[str, object]:
    """Builds the output JSON for one prompt: forced structure + generated
    zones.
    The skeleton ('{"prompt":', keys, quotes, closing braces) is appended
    manually; the function name and each parameter value come from the model,
    constrained by the enum / number / string grammars.
    Returns the generated call as a dict, guaranteed to json.loads() cleanly.
    """
    ids = list(base_ids)
    ids += to_ids(model, '{"prompt":')
    ids += to_ids(model, json.dumps(prompt))
    names = [f.name for f in functions]
    ids += to_ids(model, ',"name":"')
    name_ids, _ = generate_free_field(
        model, ids, token_text,
        lambda full: enum_can_continue(full, names),
    )
    ids += name_ids
    generated_name = decode_ids(name_ids, token_text)
    func = next(f for f in functions if f.name == generated_name)
    ids += to_ids(model, '","parameters":{')
    for i, key in enumerate(func.parameters):
        if i > 0:
            ids += to_ids(model, ",")
        ids += to_ids(model, json.dumps(key) + ":")
        value_ids = generate_value(
            model, ids, token_text,
            func.parameters[key].type,
            last=(i == len(func.parameters) - 1),
        )
        ids += value_ids
    ids += to_ids(model, "}}")
    out_text = decode_ids(ids[len(base_ids):], token_text)
    obj = json.loads(out_text)
    assert isinstance(obj, dict)
    return obj


def first_raw_quote(text: str) -> int | None:
    """Index of the first *unescaped* double quote in text (None if absent)."""
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i
        i += 1
    return None


def is_content(buf: str, full: str) -> bool:
    """Legal content, never blank-only when the value has not started."""
    if str_state(full) is None:
        return False
    return not (buf == "" and not full.strip())


def string_move(
    logits: list[float], token_text: list[str], buf: str, quote_id: int,
) -> tuple[str, int, str] | None:
    """Decides the next step of a string value. One sentence of rule:
    **the best token that carries an unescaped quote closes the value —
    unless what follows the quote is real text, in which case the quote is
    content and gets escaped — but only if it beats the best content token.**
    Returns (kind, token_id, prefix):
      'content' -> keep writing token_id
      'escape'  -> write a literal quote (\\")
      'close'   -> the value ends here; prefix is the content written
                   before the quote ('}' from the token '}"')
    """
    if str_state(buf) != "in":
        # mid escape sequence: only plain content can continue it
        chosen = top_down_argmax(logits, token_text, buf, str_can_continue)
        return None if chosen is None else ("content", chosen[0], "")

    content = top_down_argmax(
        logits, token_text, buf, lambda full: is_content(buf, full))
    quote: tuple[float, int, int] | None = None
    for t in np.argsort(-np.asarray(logits))[:SCAN]:
        tid = int(t)
        text = token_text[tid]
        if not text:
            continue
        k = first_raw_quote(text)
        if k is None or str_state(buf + text[:k]) != "in":
            continue                       # no real quote in this token
        tail = text[k + 1:].lstrip()
        if k and tail and tail[0] not in STRUCT_TAIL:
            continue                       # quote in the middle, unusable
        quote = (float(logits[tid]), tid, k)
        break

    if quote is not None and (
            content is None or quote[0] > content[1]):
        text = token_text[quote[1]]
        tail = text[quote[2] + 1:].lstrip()
        if quote[2] == 0 and tail and tail[0] not in STRUCT_TAIL:
            return ("escape", quote_id, "")
        return ("close", quote_id, text[:quote[2]])
    if content is None:
        return None
    return ("content", content[0], "")


def generate_value(
    model: Small_LLM_Model, ids: list[int], token_text: list[str],
    value_type: str, last: bool = False,
) -> list[int]:
    """Generates the token ids of one parameter value, constrained by its type.
    A boolean stops on its own (the enum blocks everything once "true" or
    "false" is complete), a number stops when the model prefers the closing
    ','/'}', and a string stops at the first quote the model prefers over
    ordinary content. Every branch still guarantees a value the JSON parser
    accepts (half-written values are completed).
    Returns the ids of the generated value (including the quotes, for strings).
    """
    if value_type == "boolean":
        value_ids, buf = generate_free_field(
            model, ids, token_text,
            lambda full: enum_can_continue(full, ["true", "false"]),
        )
        if buf not in ("true", "false"):
            # dead end on a prefix ('t', 'fal'...): complete it, so the
            # emitted JSON never carries a half-written literal
            target = "true" if "true".startswith(buf) else "false"
            missing = target[len(buf):]
            if missing:
                value_ids = value_ids + to_ids(model, missing)
        return value_ids
    if value_type == "number":
        close_text = "}" if last else ","
        close_id = token_text.index(close_text)
        value_ids_n: list[int] = []
        buf = ""
        for _ in range(256):
            logits = model.get_logits_from_input_ids(ids)
            chosen = top_down_argmax(
                logits, token_text, buf,
                lambda full: num_can_continue(num_state(full)),
            )
            if chosen is None:
                break
            next_id, next_logit = chosen
            if (
                num_is_complete(num_state(buf))
                and logits[close_id] > next_logit
            ):
                break
            value_ids_n.append(next_id)
            buf += token_text[next_id]
            ids = ids + [next_id]
        if not num_is_complete(num_state(buf)):
            # dangling '-', '12.' or '12e': a trailing 0 always completes it
            value_ids_n.append(token_text.index("0"))
        return value_ids_n
    if value_type == "string":
        quote_id = token_text.index('"')
        backslash_id = token_text.index("\\")
        ids = ids + [quote_id]
        body: list[int] = []              # ids written inside the value
        buf = ""
        for _ in range(256):
            logits = model.get_logits_from_input_ids(ids)
            move = string_move(logits, token_text, buf, quote_id)
            if move is None:
                break
            kind, tid, prefix = move
            if kind == "close":
                if prefix:                # the '}' of a token like '}"}'
                    prefix_ids = to_ids(model, prefix)
                    body += prefix_ids
                    buf += prefix
                    ids = ids + prefix_ids
                break                     # the quote ends the value
            if kind == "escape":
                chunk, text = [backslash_id, quote_id], '\\"'
            else:
                chunk, text = [tid], token_text[tid]
            body += chunk
            buf += text
            ids = ids + chunk
        value = buf
        while str_state(value) != "in":   # drop a dangling escape
            value = value[:-1]
        value = value.lstrip()            # values never start with blanks
        if value != buf:
            body = to_ids(model, value)   # trimmed: re-encode the text
        return [quote_id] + body + [quote_id]   # always terminated
    raise ValueError(f"Unknown value_type {value_type!r}")
