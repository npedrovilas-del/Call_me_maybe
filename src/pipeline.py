import math
from collections.abc import Callable

import numpy as np

def masked_argmax(logits: list[float], valid: list[int]) -> int | None:
    """Picks the valid token with the highest logit, ignoring the rest.
    A plain max over the allowed ids; None when nothing is valid, so the
    caller knows to stop. The simple, eager version of top_down_argmax.
    """
    best_id, best = None, -math.inf
    for t in valid:
        if logits[t] > best:
            best, best_id = logits[t], t
    return best_id

def top_down_argmax(logits: list[float], token_text: list[str], buffer: str, is_valid: Callable[[str], bool]) -> tuple[int, float] | None:
    """First valid token found when scanning from the highest logit down.
    The model's top vote wins as long as the grammar accepts it; otherwise we
    fall to the next candidate. Returns (id, logit) or None if nothing valid
    remains.
    """
    order = np.argsort(-np.asarray(logits))
    for t in order:
        if is_valid(buffer + token_text[int(t)]):
            return int(t), logits[int(t)]
    return None

def generate_free_field(model,ids: list[int], token_text: list[str], is_valid: Callable[[str], bool], max_tokens: int = 256,) -> tuple[list[int], str]:
    """Generates a free text field token by token, only allowing valid tokens.
    is_valid receives (buffer + candidate): the accumulated text plus the
    candidate token text. Returns (ids generated, text generated).
    """
    out_ids: list[int] = []
    out = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        chosen = top_down_argmax(logits, token_text, out, is_valid)
        if chosen is None:
            break
        next_id, _ = chosen
        out_ids.append(next_id)
        out += token_text[next_id]
        ids = ids + [next_id]
    return out_ids, out