SIMPLE_ESCAPES = '"\\/bfnrt'          # os escapes de 1 char
HEX = "0123456789abcdefABCDEF"        # dígitos hexadecimais

def str_state(buf: str) -> "str | None":
    """None -> inválida; 'in' -> normal; 'esc'/'uni' -> escape incompleto."""
    i = 0
    while i < len(buf):
        ch = buf[i]
        if ch == "\\":
            depois = buf[i + 1:]
            if not depois:
                return "esc"
            if depois[0] in SIMPLE_ESCAPES:
                i += 2
                continue
            if depois[0] == "u":
                hex_part = depois[1:5]
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
