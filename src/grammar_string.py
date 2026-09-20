SIMPLE_ESCAPES = '"\\/bfnrt'          # 1 char escapes
HEX = "0123456789abcdefABCDEF"        # hex digits


def str_state(buf: str) -> "str | None":
    """None -> invalid; 'in' -> normal; 'esc'/'uni' -> incomplete escape."""
    i = 0
    while i < len(buf):
        ch = buf[i]
        if ch == "\\":
            after = buf[i + 1:]
            if not after:
                return "esc"
            if after[0] in SIMPLE_ESCAPES:
                i += 2
                continue
            if after[0] == "u":
                hex_part = after[1:5]
                if len(hex_part) < 4:
                    return "uni"
                if not all(c in HEX for c in hex_part):
                    return None
                i += 6
                continue
            return None
        if ch == '"' or ord(ch) < 0x20:
            return None
        i += 1
    return "in"


def str_can_continue(buf: str) -> bool:
    """Still possible to continue this string? (in/esc/uni, not dead)."""
    return str_state(buf) in ("in", "esc", "uni")
