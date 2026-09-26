"""Training data: random CHIP-8 machine states, one instruction each.

No game ROM is used here. Every sample is a made-up machine (random registers,
random screen, random stack...) about to execute one random instruction, and the
label is what the reference emulator says happens next.
"""
import multiprocessing as mp

import numpy as np

import emu
from encode import encode, target, NT

SPECIAL = np.array([0, 1, 2, 0x7F, 0x80, 0xFE, 0xFF])


def _byte(r):
    return int(r.choice(SPECIAL)) if r.random() < 0.25 else int(r.integers(256))


# (weight, generator(rng, state) -> opcode). Generators may nudge the state so
# that branches are taken ~half the time.
def _op_classes():
    X = lambda r: int(r.integers(16))
    def eq_kk(r, s, base):
        x = X(r); kk = int(s.V[x]) if r.random() < 0.5 else _byte(r)
        return base | x << 8 | kk
    def eq_xy(r, s, base):
        x, y = X(r), X(r)
        if r.random() < 0.5: s.V[y] = s.V[x]
        return base | x << 8 | y << 4
    def key(r, s, base):
        x = X(r)
        if r.random() < 0.5: s._keys |= 1 << (int(s.V[x]) & 0xF)
        return base | x << 8
    def alu(n): return lambda r, s: 0x8000 | X(r) << 8 | X(r) << 4 | n
    def fx(kk): return lambda r, s: 0xF000 | X(r) << 8 | kk
    return [
        (1, lambda r, s: 0x00E0),
        (2, lambda r, s: 0x00EE),
        (2, lambda r, s: 0x1000 | int(r.integers(4096))),
        (2, lambda r, s: 0x2000 | int(r.integers(4096))),
        (2, lambda r, s: eq_kk(r, s, 0x3000)),
        (2, lambda r, s: eq_kk(r, s, 0x4000)),
        (2, lambda r, s: eq_xy(r, s, 0x5000)),
        (1, lambda r, s: 0x6000 | X(r) << 8 | _byte(r)),
        (2, lambda r, s: 0x7000 | X(r) << 8 | _byte(r)),
        (1, alu(0)), (1, alu(1)), (1, alu(2)), (1, alu(3)),
        (3, alu(4)), (3, alu(5)), (2, alu(6)), (3, alu(7)), (2, alu(0xE)),
        (2, lambda r, s: eq_xy(r, s, 0x9000)),
        (1, lambda r, s: 0xA000 | int(r.integers(4096))),
        (2, lambda r, s: 0xB000 | int(r.integers(4096))),
        (2, lambda r, s: 0xC000 | X(r) << 8 | _byte(r)),
        (12, lambda r, s: 0xD000 | X(r) << 8 | X(r) << 4 | int(r.integers(16))),
        (2, lambda r, s: key(r, s, 0xE09E)),
        (2, lambda r, s: key(r, s, 0xE0A1)),
        (1, fx(0x07)), (2, fx(0x0A)), (1, fx(0x15)), (1, fx(0x18)),
        (2, fx(0x1E)), (2, fx(0x29)), (3, fx(0x33)), (3, fx(0x55)), (3, fx(0x65)),
    ]


CLASSES = _op_classes()
_W = np.array([w for w, _ in CLASSES], float); _W /= _W.sum()


def random_sample(r):
    s = emu.State()
    s.V = np.array([_byte(r) for _ in range(16)], np.int64)
    s.I = int(r.integers(0xFF0, 0x1000)) if r.random() < 0.1 else int(r.integers(4096))
    s.PC = int(r.integers(2048)) * 2 if r.random() < 0.9 else int(r.integers(4096))
    s.SP = int(r.integers(16))
    s.stack = r.integers(0, 4096, 16).astype(np.int64)
    s.DT = int(r.choice([0, 1, _byte(r)], p=[0.3, 0.2, 0.5]))
    s.ST = int(r.choice([0, 1, _byte(r)], p=[0.3, 0.2, 0.5]))
    density = r.choice([0.0, 0.02, 0.1, 0.3, 0.5])
    s.disp = (r.random((32, 64)) < density).astype(np.uint8)
    s._keys = 0 if r.random() < 0.5 else int(r.integers(65536)) & int(r.integers(65536)) & int(r.integers(65536))
    s.mem[(s.I + np.arange(16)) & 0xFFF] = r.integers(0, 256, 16)
    op = CLASSES[r.choice(len(CLASSES), p=_W)][1](r, s)
    s.mem[s.PC] = op >> 8
    s.mem[(s.PC + 1) & 0xFFF] = op & 0xFF
    keys, tick, rng = s._keys, bool(r.random() < 0.5), int(r.integers(256))
    return s, keys, tick, rng


def make_batch(seed, n):
    r = np.random.default_rng(seed)
    X = np.empty((n, NT, 8), np.uint8)
    Y = np.empty((n, NT, 8), np.uint8)
    for k in range(n):
        s, keys, tick, rng = random_sample(r)
        x = encode(s, keys, tick, rng)
        before = s.copy()
        emu.step(s, keys, tick, rng)
        X[k] = np.packbits(x, axis=-1, bitorder="little")
        Y[k] = np.packbits(target(before, s), axis=-1, bitorder="little")
    return X, Y


def unpack(a):
    return np.unpackbits(a, axis=-1, bitorder="little")


def batches(n, seed=0, workers=4, ahead=8, trace_frac=0.0):
    """Endless stream of fresh batches from a process pool, at most `ahead` in flight.
    (Pool.imap over an endless generator queues work forever and floods the main process.)"""
    from collections import deque
    pool = mp.Pool(workers)
    seeds = iter(range(seed * 10_000_000, 10**12))
    q = deque(pool.apply_async(mixed_batch, (next(seeds), n, trace_frac)) for _ in range(ahead))
    while True:
        out = q.popleft().get()
        q.append(pool.apply_async(mixed_batch, (next(seeds), n, trace_frac)))
        yield out




# ---- execution traces -------------------------------------------------------
# Each worker process keeps a few Pong games running between calls and samples
# states from them (with random key presses, random IPF and random resets).

_GAMES = None


def _new_game(r, rom):
    return dict(s=emu.State.boot(rom), ipf=int(r.integers(10, 41)), i=0, keys=0)


def trace_batch(seed, n, rom_path="roms/pong.ch8", games=8):
    global _GAMES
    r = np.random.default_rng(seed)
    rom = open(rom_path, "rb").read()
    if _GAMES is None:
        _GAMES = [_new_game(r, rom) for _ in range(games)]
        for g in _GAMES:  # start games at random points in time
            for _ in range(int(r.integers(0, 40000))):
                _advance(g, r)
    X = np.empty((n, NT, 8), np.uint8)
    Y = np.empty((n, NT, 8), np.uint8)
    for k in range(n):
        g = _GAMES[int(r.integers(len(_GAMES)))]
        for _ in range(int(r.integers(1, 30))):  # skip ahead a bit so samples aren't adjacent
            _advance(g, r)
        if r.random() < 1e-3:
            g.update(_new_game(r, rom))
        s, keys = g["s"], g["keys"]
        tick, rng = g["i"] % g["ipf"] == g["ipf"] - 1, int(r.integers(256))
        x = encode(s, keys, tick, rng)
        before = s.copy()
        emu.step(s, keys, tick, rng)
        g["i"] += 1
        X[k] = np.packbits(x, axis=-1, bitorder="little")
        Y[k] = np.packbits(target(before, s), axis=-1, bitorder="little")
    return X, Y


def _advance(g, r):
    if r.random() < 0.002:  # occasionally press / release a paddle key
        g["keys"] = int(r.choice([0, 0, 1 << 1, 1 << 4, 1 << int(r.integers(16))]))
    tick = g["i"] % g["ipf"] == g["ipf"] - 1
    emu.step(g["s"], g["keys"], tick, int(r.integers(256)))
    g["i"] += 1


def mixed_batch(seed, n, trace_frac):
    k = int(n * trace_frac)
    Xa, Ya = make_batch(seed, n - k)
    if k == 0:
        return Xa, Ya
    Xb, Yb = trace_batch(seed, k)
    return np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb])
