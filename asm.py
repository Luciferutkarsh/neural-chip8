"""Tiny CHIP-8 assembler so the demo ROMs are our own code.

Syntax:  label:   MNEMONIC args   ; comment
Registers v0..vf, numbers as 12 / 0x0C / 0b1100, labels anywhere an address goes.
`db a, b, c` emits raw bytes.
"""
import re
import sys


def _num(tok, labels):
    tok = tok.strip()
    if tok in labels: return labels[tok]
    return int(tok, 0)


def _reg(tok):
    tok = tok.strip().lower()
    assert re.fullmatch(r"v[0-9a-f]", tok), f"expected register, got {tok!r}"
    return int(tok[1], 16)


def _is_reg(tok):
    return re.fullmatch(r"v[0-9a-f]", tok.strip().lower()) is not None


def _encode(m, a, L):
    R, N = _reg, lambda t: _num(t, L)
    if m == "cls": return 0x00E0
    if m == "ret": return 0x00EE
    if m == "jp":
        if len(a) == 2: return 0xB000 | N(a[1])
        return 0x1000 | N(a[0])
    if m == "call": return 0x2000 | N(a[0])
    if m == "se": return (0x5000 | R(a[0]) << 8 | R(a[1]) << 4) if _is_reg(a[1]) else (0x3000 | R(a[0]) << 8 | N(a[1]))
    if m == "sne": return (0x9000 | R(a[0]) << 8 | R(a[1]) << 4) if _is_reg(a[1]) else (0x4000 | R(a[0]) << 8 | N(a[1]))
    if m == "ld":
        d, s = a[0].strip().lower(), a[1].strip().lower()
        if d == "i": return 0xA000 | N(s)
        if d == "dt": return 0xF015 | R(s) << 8
        if d == "st": return 0xF018 | R(s) << 8
        if d == "f": return 0xF029 | R(s) << 8
        if d == "b": return 0xF033 | R(s) << 8
        if d == "[i]": return 0xF055 | R(s) << 8
        if s == "[i]": return 0xF065 | R(d) << 8
        if s == "dt": return 0xF007 | R(d) << 8
        if s == "k": return 0xF00A | R(d) << 8
        if _is_reg(s): return 0x8000 | R(d) << 8 | R(s) << 4
        return 0x6000 | R(d) << 8 | N(s)
    if m == "add":
        if a[0].strip().lower() == "i": return 0xF01E | R(a[1]) << 8
        if _is_reg(a[1]): return 0x8004 | R(a[0]) << 8 | R(a[1]) << 4
        return 0x7000 | R(a[0]) << 8 | N(a[1])
    alu = {"or": 1, "and": 2, "xor": 3, "sub": 5, "shr": 6, "subn": 7, "shl": 0xE}
    if m in alu:
        y = R(a[1]) if len(a) > 1 else 0
        return 0x8000 | R(a[0]) << 8 | y << 4 | alu[m]
    if m == "rnd": return 0xC000 | R(a[0]) << 8 | N(a[1])
    if m == "drw": return 0xD000 | R(a[0]) << 8 | R(a[1]) << 4 | N(a[2])
    if m == "skp": return 0xE09E | R(a[0]) << 8
    if m == "sknp": return 0xE0A1 | R(a[0]) << 8
    raise ValueError(f"unknown mnemonic {m}")


def assemble(src: str) -> bytes:
    lines = []
    for raw in src.splitlines():
        line = raw.split(";")[0].strip()
        while ":" in line.split()[0] if line else False:
            label, line = line.split(":", 1)
            lines.append(("label", label.strip()))
            line = line.strip()
        if line:
            parts = line.split(None, 1)
            args = [x for x in parts[1].split(",")] if len(parts) > 1 else []
            lines.append(("op", parts[0].lower(), args))
    labels, addr = {}, 0x200
    for item in lines:  # pass 1: addresses
        if item[0] == "label": labels[item[1]] = addr
        else: addr += len(item[2]) if item[1] == "db" else 2
    out = bytearray()
    for item in lines:  # pass 2: bytes
        if item[0] == "label": continue
        if item[1] == "db": out += bytes(_num(t, labels) for t in item[2])
        else:
            w = _encode(item[1], item[2], labels)
            out += bytes([w >> 8, w & 0xFF])
    return bytes(out)


if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    rom = assemble(src)
    open(sys.argv[2], "wb").write(rom)
    print(f"{sys.argv[2]}: {len(rom)} bytes")
