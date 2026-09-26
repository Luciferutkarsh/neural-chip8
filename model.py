import os

import mlx.core as mx
import mlx.nn as nn

from encode import NT, W, ROW0


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiHeadAttention(d, heads)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def __call__(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h)
        return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))


class NeuralCPU(nn.Module):
    """88 state tokens in, 88 answers out. One forward pass = one clock cycle.

    bus=True: every token also sees a shared "bus" carrying the whole non-screen
    state (opcode, PC, I, SP, timers, keys, registers, stack, memory at I), like
    the wires a real CPU uses to route the instruction and register file around.
    Without it, attention spent ~1000+ steps just discovering where the opcode was.
    """

    def __init__(self, d=256, layers=8, heads=8, bus=False):
        super().__init__()
        self.inp = nn.Linear(W, d)
        self.slot = mx.random.normal((NT, d)) * float(os.environ.get("SLOT_STD", "0.02"))
        self.has_bus = bus
        if bus:
            self.bus = nn.Linear(ROW0 * W, d)
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, W)

    def __call__(self, bits, in_mask):
        v = (bits * 2.0 - 1.0) * in_mask
        x = self.inp(v) + self.slot
        if self.has_bus:
            x = x + self.bus(v[:, :ROW0].reshape(v.shape[0], -1))[:, None, :]
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln(x))


class GLUBlock(nn.Module):
    """Residual gated MLP block. The multiplicative gate makes 'pick register x' style
    selection much easier than a plain ReLU MLP."""

    def __init__(self, d, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.a, self.b = nn.Linear(d, hidden), nn.Linear(d, hidden)
        self.o = nn.Linear(hidden, d)

    def __call__(self, x):
        h = self.ln(x)
        return x + self.o(nn.silu(self.a(h)) * self.b(h))


class NeuralCPU2(nn.Module):
    """v2: a CPU core + a scanline unit, both plain learned networks.

    core:      reads the bus (all non-screen state + pooled collision signal) and
               writes the next non-screen state.
    scanline:  one small network shared by all 32 screen rows. Sees the bus, its own
               row and its row number; writes that row's XOR flips and a collision
               feature. Collision features are max-pooled into the core (that's
               how VF can learn about sprite collisions).
    """

    def __init__(self, d=1024, blocks=4, row_d=256, row_blocks=3, pool=32):
        super().__init__()
        import numpy as np
        from encode import IN_MASK
        self._bus_idx = mx.array(np.nonzero(IN_MASK[:ROW0].reshape(-1))[0])
        nbus = int(self._bus_idx.size)
        self.pool = pool
        self.row_bus = nn.Linear(nbus, row_d)
        self.row_in = nn.Linear(W, row_d)
        self.row_pos = mx.random.normal((NT - ROW0, row_d)) * 0.5
        self.row_blocks = [GLUBlock(row_d, 2 * row_d) for _ in range(row_blocks)]
        self.row_ln = nn.LayerNorm(row_d)
        self.row_out = nn.Linear(row_d, W + pool)
        self.core_in = nn.Linear(nbus + pool, d)
        self.core_blocks = [GLUBlock(d, 2 * d) for _ in range(blocks)]
        self.core_ln = nn.LayerNorm(d)
        self.core_out = nn.Linear(d, ROW0 * W)

    def __call__(self, bits, in_mask):
        B = bits.shape[0]
        v = (bits * 2.0 - 1.0) * in_mask
        bus = v[:, :ROW0].reshape(B, -1)[:, self._bus_idx]
        r = self.row_in(v[:, ROW0:]) + self.row_bus(bus)[:, None, :] + self.row_pos
        for blk in self.row_blocks:
            r = blk(r)
        r = self.row_out(self.row_ln(r))
        rows, feat = r[..., :W], r[..., W:]
        c = self.core_in(mx.concatenate([bus, feat.max(axis=1)], axis=-1))
        for blk in self.core_blocks:
            c = blk(c)
        core = self.core_out(self.core_ln(c)).reshape(B, ROW0, W)
        return mx.concatenate([core, rows], axis=1)
