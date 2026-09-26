"""Reference CHIP-8 emulator. This is the 'teacher' the neural net learns to imitate.

Quirks chosen (the common "modern" set):
  - 8xy6 / 8xyE shift Vx in place (Vy ignored)
  - Fx55 / Fx65 leave I unchanged
  - Bnnn jumps to nnn + V0
  - sprites wrap their start position but clip at the screen edge
  - Fx0A: if any key is down, store the lowest one, otherwise re-run the instruction
Timers tick after the instruction when the external `tick` line is high.
"""
import numpy as np

FONT_ADDR = 0x050
FONT = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70,
    0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0, 0x10, 0xF0, 0x10, 0xF0,
    0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
    0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40,
    0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0, 0x10, 0xF0,
    0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
    0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0,
    0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80,
]


class State:
    def __init__(self):
        self.mem = np.zeros(4096, np.uint8)
        self.V = np.zeros(16, np.int64)
        self.I = 0
        self.PC = 0x200
        self.SP = 0
        self.stack = np.zeros(16, np.int64)
        self.DT = 0
        self.ST = 0
        self.disp = np.zeros((32, 64), np.uint8)

    @classmethod
    def boot(cls, rom: bytes):
        s = cls()
        s.mem[FONT_ADDR:FONT_ADDR + len(FONT)] = FONT
        s.mem[0x200:0x200 + len(rom)] = np.frombuffer(rom, np.uint8)
        return s

    def copy(self):
        c = State.__new__(State)
        c.mem, c.V, c.stack, c.disp = self.mem.copy(), self.V.copy(), self.stack.copy(), self.disp.copy()
        c.I, c.PC, c.SP, c.DT, c.ST = self.I, self.PC, self.SP, self.DT, self.ST
        return c

    def opcode(self):
        return (int(self.mem[self.PC]) << 8) | int(self.mem[(self.PC + 1) & 0xFFF])

    def window(self):
        """The 16 bytes of memory starting at I (what the neural net gets to see)."""
        return self.mem[(self.I + np.arange(16)) & 0xFFF]


def step(s: State, keys: int = 0, tick: bool = False, rng: int = 0):
    op = s.opcode()
    s.PC = (s.PC + 2) & 0xFFF
    x, y = (op >> 8) & 0xF, (op >> 4) & 0xF
    n, kk, nnn = op & 0xF, op & 0xFF, op & 0xFFF
    V = s.V
    top = op >> 12

    if op == 0x00E0:
        s.disp[:] = 0
    elif op == 0x00EE:
        s.SP = (s.SP - 1) & 0xF
        s.PC = int(s.stack[s.SP])
    elif top == 0x1:
        s.PC = nnn
    elif top == 0x2:
        s.stack[s.SP] = s.PC
        s.SP = (s.SP + 1) & 0xF
        s.PC = nnn
    elif top == 0x3:
        if V[x] == kk: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0x4:
        if V[x] != kk: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0x5 and n == 0:
        if V[x] == V[y]: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0x6:
        V[x] = kk
    elif top == 0x7:
        V[x] = (V[x] + kk) & 0xFF
    elif top == 0x8:
        a, b = int(V[x]), int(V[y])
        if n == 0x0: V[x] = b
        elif n == 0x1: V[x] = a | b
        elif n == 0x2: V[x] = a & b
        elif n == 0x3: V[x] = a ^ b
        elif n == 0x4: V[x] = (a + b) & 0xFF; V[0xF] = int(a + b > 0xFF)
        elif n == 0x5: V[x] = (a - b) & 0xFF; V[0xF] = int(a >= b)
        elif n == 0x6: V[x] = a >> 1; V[0xF] = a & 1
        elif n == 0x7: V[x] = (b - a) & 0xFF; V[0xF] = int(b >= a)
        elif n == 0xE: V[x] = (a << 1) & 0xFF; V[0xF] = a >> 7
    elif top == 0x9 and n == 0:
        if V[x] != V[y]: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0xA:
        s.I = nnn
    elif top == 0xB:
        s.PC = (nnn + int(V[0])) & 0xFFF
    elif top == 0xC:
        V[x] = rng & kk
    elif top == 0xD:
        x0, y0 = int(V[x]) % 64, int(V[y]) % 32
        sprite = s.window()
        hit = 0
        for r in range(n):
            yy = y0 + r
            if yy >= 32: break
            for b in range(8):
                xx = x0 + b
                if xx >= 64: break
                if (sprite[r] >> (7 - b)) & 1:
                    hit |= s.disp[yy, xx]
                    s.disp[yy, xx] ^= 1
        V[0xF] = hit
    elif top == 0xE and kk == 0x9E:
        if (keys >> (int(V[x]) & 0xF)) & 1: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0xE and kk == 0xA1:
        if not (keys >> (int(V[x]) & 0xF)) & 1: s.PC = (s.PC + 2) & 0xFFF
    elif top == 0xF:
        if kk == 0x07: V[x] = s.DT
        elif kk == 0x0A:
            if keys: V[x] = (keys & -keys).bit_length() - 1
            else: s.PC = (s.PC - 2) & 0xFFF
        elif kk == 0x15: s.DT = int(V[x])
        elif kk == 0x18: s.ST = int(V[x])
        elif kk == 0x1E: s.I = (s.I + int(V[x])) & 0xFFF
        elif kk == 0x29: s.I = FONT_ADDR + 5 * (int(V[x]) & 0xF)
        elif kk == 0x33:
            v = int(V[x])
            for k, d in enumerate((v // 100, (v // 10) % 10, v % 10)):
                s.mem[(s.I + k) & 0xFFF] = d
        elif kk == 0x55:
            for k in range(x + 1): s.mem[(s.I + k) & 0xFFF] = V[k]
        elif kk == 0x65:
            for k in range(x + 1): V[k] = s.mem[(s.I + k) & 0xFFF]
    # anything else (0nnn SYS, unknown) is a no-op

    if tick:
        s.DT = max(s.DT - 1, 0)
        s.ST = max(s.ST - 1, 0)
