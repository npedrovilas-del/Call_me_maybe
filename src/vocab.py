import json

from llm_sdk import Small_LLM_Model
from functools import lru_cache

def bytes_to_unicode():
    """Builds an ASCII-like table of all 256 bytes.
    Every byte becomes a printable character: the visible ones keep their
    normal character, and the non-visible ones are shifted to chr(256 + n).
    Returns a dict {byte: character}."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))  # Adds all the visible carachters to a dict of lists
    cs = bs[:] # Make a copy not connected to bs
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1 
    return dict(zip(bs, [chr(c) for c in cs]))


b2u = bytes_to_unicode()
byte_of = {c: b for b, c in b2u.items()}


def build_token_text(path: str) -> list[str]:
    """Builds the translation table: token_id -> real text.
    For each token in the vocab, converts its characters to bytes and decodes
    them as UTF-8, so spaces/newlines appear as real text instead of the Ġ/Ċ
    disguise. The list has one slot per model logit (special tokens stay "").
    Returns a list[str] where token_text[id] = real text.
    Works like a decoder of the whole vocab"""
    with open(path) as f:
        raw = json.load(f)
    m = Small_LLM_Model()
    tamanho = len(m.get_logits_from_input_ids([0]))
    token_text = [""] * tamanho
    for token, idx in raw.items():
        character = list(token)
        bytes_list = [byte_of[c] for c in character]
        text = bytes(bytes_list).decode("utf-8", "replace")
        token_text[idx] = text #passa para o final
    return token_text


@lru_cache(maxsize=None) # return the same value as the first time in this case the token_text of the get token text
def get_token_text(model) -> list[str]:
    """Builds token_text once (cached) and checks its size against the model. 
    The token_text is gonna be all the tokens already decoded to normal chr"""
    token_text = build_token_text(model.get_path_to_vocab_file())
    n_logits = len(model.get_logits_from_input_ids([0]))
    if len(token_text) != n_logits:
        raise ValueError(
            f"token_text tem {len(token_text)} posições, "
            f"mas o modelo devolve {n_logits} logits — abortar."
        )
    return token_text
