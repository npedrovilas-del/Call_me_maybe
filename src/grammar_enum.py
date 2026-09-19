def enum_can_continue(buf: str, words: list[str]) -> bool:
    """Is `buf` a prefix of any of the words?"""
    return any(w.startswith(buf) for w in words)

def enum_is_complete(buf: str, words: list[str]) -> bool:
    """Is `buf` one whole word?"""
    return buf in words