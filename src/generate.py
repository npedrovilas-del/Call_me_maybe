import json

from src.grammar_enum import enum_can_continue
from src.pipeline import generate_free_field
from src.grammar_number import num_can_continue, num_state, num_is_complete
from src.pipeline import generate_free_field, top_down_argmax
from src.grammar_string import str_can_continue, str_state
from src.models import FunctionDef

GRAMMARS = {
    "number":  lambda full: num_can_continue(num_state(full)),
    "string":  lambda full: str_can_continue(full),
    "boolean": lambda full: enum_can_continue(full, ["true", "false"]),
}


def to_ids(model, text: str) -> list[int]:
    """Encodes a text into token ids, ready to append to the stream.
    The structure text (keys, quotes, braces) is known beforehand, so unlike
    the free zones we don't have to guess it token by token — encode() does
    the splitting for us.
    """
    return model.encode(text).tolist()[0]


def decode_ids(ids: list[int], token_text: list[str]) -> str:
    """Turns token ids back into real text.
    Joins the decoded piece of each id (from the byte-level token_text table)
    into one string — this is what produces the final JSON text.
    """
    return "".join(token_text[i] for i in ids)

def build_prompt(functions, prompt) -> str:
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

def generate_call(model, base_ids: list[int], prompt: str,
                  functions: list[FunctionDef], token_text: list[str]) -> dict[str, object]:
    """Builds the output JSON for one prompt: forced structure + generated zones.
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
    return json.loads(out_text)

def generate_value(model, ids, token_text, value_type, last=False) -> list[int]:
    """Generates the token ids of one parameter value, constrained by its type.
    Boolean stops on its own (once "true"/"false" is complete, the enum blocks
    everything). Number and string never stop alone — they need the terminator
    rule: the closing ','/'}'/'"' must win the vote against the next valid
    token while the value sits in a final state.
    Returns the ids of the generated value (including the quotes, for strings).
    """
    if value_type == "boolean":
        value_ids, _ = generate_free_field(
        model, ids, token_text,
        lambda full: enum_can_continue(full, ["true", "false"]),
        )
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
            if num_is_complete(num_state(buf)) and logits[close_id] > next_logit:
                break
            value_ids_n.append(next_id)
            buf += token_text[next_id]
            ids = ids + [next_id]
        return value_ids_n
    if value_type == "string":
        quote_id = token_text.index('"')
        value_ids_s: list[int] = [quote_id]
        ids = ids + [quote_id]
        buf = ""
        for _ in range(256):
            logits = model.get_logits_from_input_ids(ids)
            chosen = top_down_argmax(logits, token_text, buf, str_can_continue)
            if chosen is None:
                break
            next_id, next_logit = chosen
            if str_state(buf) == "in" and logits[quote_id] > next_logit:
                value_ids_s.append(quote_id)
                break
            value_ids_s.append(next_id)
            buf += token_text[next_id]
            ids = ids + [next_id]
        return value_ids_s