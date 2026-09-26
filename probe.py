"""Per-instruction accuracy of a checkpoint on fresh random states.

python3 probe.py out/run1.safetensors [n_samples]
"""
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np

import data
from encode import IN_MASK, OUT_MASK, OP, NT
from model import NeuralCPU, NeuralCPU2


def mnemonic(op):
    t, n, kk = op >> 12, op & 0xF, op & 0xFF
    if op == 0x00E0: return "CLS"
    if op == 0x00EE: return "RET"
    fixed = {1: "JP", 2: "CALL", 3: "SE xkk", 4: "SNE xkk", 5: "SE xy", 6: "LD xkk", 7: "ADD xkk",
             9: "SNE xy", 0xA: "LD I", 0xB: "JP V0", 0xC: "RND", 0xD: "DRW"}
    if t in fixed: return fixed[t]
    if t == 8: return {0: "LD xy", 1: "OR", 2: "AND", 3: "XOR", 4: "ADD xy", 5: "SUB", 6: "SHR", 7: "SUBN", 0xE: "SHL"}.get(n, "8??")
    if t == 0xE: return {0x9E: "SKP", 0xA1: "SKNP"}.get(kk, "E??")
    if t == 0xF: return {0x07: "LD x,DT", 0x0A: "LD x,K", 0x15: "LD DT", 0x18: "LD ST", 0x1E: "ADD I", 0x29: "LD F",
                         0x33: "BCD", 0x55: "STORE", 0x65: "LOAD"}.get(kk, "F??")
    return "SYS"


SLOTS = [("PC", [1]), ("I", [2]), ("SP", [3]), ("timers", [4, 5]), ("V", range(8, 24)),
         ("stack", range(24, 40)), ("mem", range(40, 56)), ("screen", range(56, 88))]


def decode_ops(x_bits):
    """Undo the one-hot nibble encoding of the OP token."""
    nib = x_bits[:, OP, :64].reshape(-1, 4, 16).argmax(-1).astype(np.int64)
    return (nib << np.array([12, 8, 4, 0])).sum(-1)


def load(path, d=256, layers=8):
    import json, os
    cfg = path.replace(".safetensors", ".cfg.json")
    for suffix in ("_1k", "_2k", "_3k", "_4k", "_6k", "_8k"):
        cfg = cfg.replace(suffix + ".cfg", ".cfg")
    c = json.load(open(cfg)) if os.path.exists(cfg) else dict(d=d, layers=layers, bus=False)
    m = (NeuralCPU2(c["d"], c["layers"]) if c.get("arch") == "v2"
         else NeuralCPU(c["d"], c["layers"], bus=c.get("bus", False)))
    m.load_weights(path)
    mx.eval(m.parameters())
    return m


def probe(model, n=8192, seed=424242, chunk=256):
    X, Y = data.make_batch(seed, n)
    x_bits, y_bits = data.unpack(X), data.unpack(Y)
    ops = decode_ops(x_bits)
    im = mx.array(IN_MASK, mx.float32)
    wrong = np.empty(y_bits.shape, bool)
    for k in range(0, n, chunk):
        pred = np.array(model(mx.array(x_bits[k:k + chunk], mx.float32), im) > 0)
        wrong[k:k + chunk] = (pred != y_bits[k:k + chunk].astype(bool)) & OUT_MASK.astype(bool)
    per = defaultdict(lambda: [0, 0])
    part = defaultdict(lambda: defaultdict(int))
    for i, op in enumerate(ops):
        name = mnemonic(int(op))
        ok = not wrong[i].any()
        per[name][0] += ok; per[name][1] += 1
        for sname, idx in SLOTS:
            if wrong[i, list(idx)].any(): part[name][sname] += 1
    return per, part, (~wrong.any((1, 2))).mean()


if __name__ == "__main__":
    model = load(sys.argv[1])
    per, part, overall = probe(model, int(sys.argv[2]) if len(sys.argv) > 2 else 8192)
    print(f"overall exact: {overall*100:.2f}%")
    for name, (ok, tot) in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        where = ", ".join(f"{k}:{v}" for k, v in sorted(part[name].items(), key=lambda kv: -kv[1]))
        print(f"  {name:9s} {ok/tot*100:6.2f}%  (n={tot:4d})  wrong in: {where}")
