"""Sanity check: can a small model learn just LD Vx,kk / JP / LD I in a few hundred steps?"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import data
from encode import IN_MASK, OUT_MASK
from model import NeuralCPU

keep = {"LD xkk": 7, "JP": 2, "LD I": 19}
w = np.zeros(len(data.CLASSES))
for idx in keep.values():
    w[idx] = 1
data._W = w / w.sum()


def main():
    d, layers = int(sys.argv[1]), int(sys.argv[2])
    model = NeuralCPU(d, layers, bus=os.environ.get("BUS") == "1")
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=float(sys.argv[3]) if len(sys.argv) > 3 else 5e-4)
    im, om = mx.array(IN_MASK, mx.float32), mx.array(OUT_MASK, mx.float32)

    def loss_fn(m, x, y):
        bce = nn.losses.binary_cross_entropy(m(x, im), y, with_logits=True, reduction="none")
        return (bce * om).sum() / (om.sum() * x.shape[0])

    vg = nn.value_and_grad(model, loss_fn)
    for step in range(1, int(sys.argv[4]) + 1 if len(sys.argv) > 4 else 401):
        X, Y = data.make_batch(step, 128)
        x, y = mx.array(data.unpack(X), mx.float32), mx.array(data.unpack(Y), mx.float32)
        loss, g = vg(model, x, y)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if step % 100 == 0:
            X, Y = data.make_batch(10**6, 512)
            x, y = mx.array(data.unpack(X), mx.float32), data.unpack(Y)
            wrong = (np.array(model(x, im) > 0) != y.astype(bool)) & OUT_MASK.astype(bool)
            print(f"step {step}  loss {loss.item():.4f}  exact {(~wrong.any((1, 2))).mean()*100:.1f}%", flush=True)
    inspect(model)




def inspect(model):
    from probe import SLOTS, decode_ops, mnemonic
    im = mx.array(IN_MASK, mx.float32)
    X, Y = data.make_batch(10**6 + 1, 512)
    x, y = data.unpack(X), data.unpack(Y)
    wrong = (np.array(model(mx.array(x, mx.float32), im) > 0) != y.astype(bool)) & OUT_MASK.astype(bool)
    ops = decode_ops(x)
    for name, idx in SLOTS:
        print(f"  {name:7s} samples wrong: {wrong[:, list(idx)].any((1, 2)).mean()*100:5.1f}%")
    for i in range(4):
        rows, cols = np.nonzero(wrong[i])
        print(f"  {mnemonic(int(ops[i])):8s} {ops[i]:04x}: wrong (slot,bit): {list(zip(rows.tolist(), cols.tolist()))[:12]}")


if __name__ == "__main__":
    main()
