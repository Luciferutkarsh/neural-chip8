"""How the machine state is shown to the network, and how its answer is wired back.

The state becomes 88 "tokens", each a row of up to 64 bits:

  OP  PC  I  SP  DT  ST  KEYS  AUX | V0..VF | STACK0..15 | MEM0..15 | ROW0..31
  OP = the opcode as four one-hot nibbles.
  AUX = the 60Hz tick line + one byte from a hardware random source.
  MEMk = memory[I + k]  (the harness reads 16 bytes at I, whatever the instruction)

The network answers for the same 88 slots: the *new value* of PC, I, SP, timers,
registers, stack and the 16 bytes at I, and an XOR flip mask for each screen row
(drawing on CHIP-8 is XOR, so that is the natural output there). The harness
writes those back. It never looks at the opcode: all the "what does this
instruction do" logic has to live in the weights.

(Version 1 used XOR flip masks for *everything*. It turned every plain copy,
like `LD Vx, kk`, into an XOR of two tokens, and the network got 0% on those.)
"""
import numpy as np

W = 64
OP, PC, I_, SP, DT, ST, KEYS, AUX = range(8)
V0, STK0, MEM0, ROW0 = 8, 24, 40, 56
NT = 88

IN_BITS = np.array([64, 12, 12, 4, 8, 8, 16, 9] + [8] * 16 + [12] * 16 + [8] * 16 + [64] * 32)
OUT_BITS = IN_BITS.copy()
OUT_BITS[[OP, KEYS, AUX]] = 0
IN_MASK = (np.arange(W)[None, :] < IN_BITS[:, None]).astype(np.uint8)    # [88, 64]
OUT_MASK = (np.arange(W)[None, :] < OUT_BITS[:, None]).astype(np.uint8)

_SH = np.arange(W, dtype=np.int64)


def _bits(vals):
    """int array [...]-> LSB-first bits [..., 64]"""
    return ((np.asarray(vals, np.int64)[..., None] >> _SH) & 1).astype(np.uint8)


def _unbits(b):
    return (b.astype(np.int64) << _SH[: b.shape[-1]]).sum(-1)


def encode(s, keys, tick, rng):
    scal = np.array([s.opcode(), s.PC, s.I, s.SP, s.DT, s.ST, keys, int(tick) | (rng << 1)])
    vals = np.concatenate([scal, s.V, s.stack, s.window()])
    t = np.zeros((NT, W), np.uint8)
    t[:ROW0] = _bits(vals)
    t[OP] = op_onehot(scal[0])
    t[ROW0:] = s.disp
    return t & IN_MASK


def op_onehot(op):
    """Opcode as four one-hot nibbles (4 x 16 = 64 bits), like a decoder's input lines.
    Raw 16 bits kept the network stuck behaving as a NOP machine (see DEVLOG)."""
    t = np.zeros(W, np.uint8)
    for k in range(4):
        t[16 * k + ((op >> (12 - 4 * k)) & 0xF)] = 1
    return t


def target(before, after):
    """What the network should output to turn `before` into `after`
    (memory measured at before.I)."""
    win = (before.I + np.arange(16)) & 0xFFF
    vals = np.concatenate([
        [0, after.PC, after.I, after.SP, after.DT, after.ST, 0, 0],
        after.V, after.stack, after.mem[win]])
    t = np.zeros((NT, W), np.uint8)
    t[:ROW0] = _bits(vals)
    t[ROW0:] = before.disp ^ after.disp
    return t & OUT_MASK


def apply(s, out):
    """The 'dumb wires': write the network's answer back into the machine."""
    f = out.astype(np.uint8) & OUT_MASK
    v = _unbits(f[:ROW0])
    win = (s.I + np.arange(16)) & 0xFFF   # window at the *old* I
    s.PC, s.I, s.SP = int(v[PC]), int(v[I_]), int(v[SP])
    s.DT, s.ST = int(v[DT]), int(v[ST])
    s.V = v[V0:V0 + 16].copy()
    s.stack = v[STK0:STK0 + 16].copy()
    s.mem[win] = v[MEM0:MEM0 + 16].astype(np.uint8)
    s.disp ^= f[ROW0:]
