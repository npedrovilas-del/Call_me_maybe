from typing import Optional
BOARD: dict[str, dict[str, str]] = {
    "S": {
        "-": "M",   # Negative number
        "0": "Z",
        "1": "I", "2": "I", "3": "I", "4": "I", "5": "I",
        "6": "I", "7": "I", "8": "I", "9": "I",
    },
    "M": {
        "0": "Z",   # adds numbers after the S
        "1": "I", "2": "I", "3": "I", "4": "I", "5": "I",
        "6": "I", "7": "I", "8": "I", "9": "I",   # -5 -> I
    },

    "Z": {
        ".": "D",   # 0. -> D
        "e": "E",   # 0e -> E
        "E": "E",   # 0E -> E
    },
    "I": {
        "0": "I", "1": "I", "2": "I", "3": "I", "4": "I",
        "5": "I", "6": "I", "7": "I", "8": "I", "9": "I",  # 12->123
        ".": "D",   # 12. -> D
        "e": "E",   # 12e -> E
        "E": "E",   # 12E -> E
    },
    "D": {
        "0": "F", "1": "F", "2": "F", "3": "F", "4": "F",
        "5": "F", "6": "F", "7": "F", "8": "F", "9": "F",  # 12.5 -> F
    },
    "F": {
        "0": "F", "1": "F", "2": "F", "3": "F", "4": "F",
        "5": "F", "6": "F", "7": "F", "8": "F", "9": "F",  # 12.55 -> F
        "e": "E",   # 12.5e -> E
        "E": "E",   # 12.5E -> E
    },
    "E": {
        "0": "X", "1": "X", "2": "X", "3": "X", "4": "X",
        "5": "X", "6": "X", "7": "X", "8": "X", "9": "X",  # 12e5 -> X
        "+": "P",   # 12e+ -> P
        "-": "P",   # 12e- -> P
    },
    "P": {
        "0": "X", "1": "X", "2": "X", "3": "X", "4": "X",
        "5": "X", "6": "X", "7": "X", "8": "X", "9": "X",  # 12e+5 -> X
    },
    "X": {
        "0": "X", "1": "X", "2": "X", "3": "X", "4": "X",
        "5": "X", "6": "X", "7": "X", "8": "X", "9": "X",  # 12e+50 -> X
    },
}

ACCEPT: set[str] = {"X", "F", "I", "Z"}     # the possible finals


def num_state(text: str) -> Optional[str]:  # Optional or can be None
    """Returns and loops through the text to find if is valid the text"""
    state: Optional[str] = "S"
    for char in text:
        if state is None:
            return None
        state = BOARD[state].get(char)
        if state is None:
            return None
    return state


def num_can_continue(state: Optional[str]) -> bool:
    """Returns if its valid to continue in case of a invalid chr"""
    return state is not None


def num_is_complete(state: Optional[str]) -> bool:
    """This function returns if its valid to stop on this state"""
    return state in ACCEPT
