"""Train the neural CPU on random single-instruction samples.

python3 train.py --steps 20000 --out out/model.safetensors
"""
import argparse
import json
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import data
from encode import IN_MASK, OUT_MASK, NT, ROW0
from model import NeuralCPU, NeuralCPU2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--out", default="out/model.safetensors")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", type=int, default=250)
    p.add_argument("--change-weight", type=float, default=10.0)
    p.add_argument("--bus", type=int, default=1)
    p.add_argument("--arch", default="v2")
    p.add_argument("--trace-frac", type=float, default=0.0, help="fraction of each batch from Pong traces")
    args = p.parse_args()

    mx.random.seed(args.seed)
    mx.set_memory_limit(4 << 30)   # 16GB laptop with a browser open: swapping made it 50x slower
    mx.set_cache_limit(1 << 30)
    model = (NeuralCPU2(args.d, args.layers) if args.arch == "v2"
             else NeuralCPU(args.d, args.layers, bus=bool(args.bus)))
    json.dump(dict(arch=args.arch, d=args.d, layers=args.layers, bus=bool(args.bus)),
              open(args.out.replace(".safetensors", ".cfg.json"), "w"))
    if args.resume:
        model.load_weights(args.resume)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"params: {nparams/1e6:.2f}M  d={args.d} layers={args.layers} batch={args.batch}", flush=True)

    warm = min(500, args.steps // 10)
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, warm),
         optim.cosine_decay(args.lr, args.steps - warm, args.lr * 0.02)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)

    in_mask = mx.array(IN_MASK, mx.float32)
    out_mask = mx.array(OUT_MASK, mx.float32)

    screen = mx.array((np.arange(NT) >= ROW0)[:, None], mx.float32)

    def loss_fn(model, x, y):
        logits = model(x, in_mask)
        bce = nn.losses.binary_cross_entropy(logits, y, with_logits=True, reduction="none")
        # Only a handful of the ~2600 output bits change per instruction. Weight those up,
        # otherwise "change nothing" is a very comfortable local minimum.
        changed = mx.where(screen > 0, y, mx.abs(y - x))
        w = out_mask * (1.0 + args.change_weight * changed)
        return (bce * w).sum() / (out_mask.sum() * x.shape[0])

    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    def to_mx(X, Y):
        return mx.array(data.unpack(X), mx.float32), mx.array(data.unpack(Y), mx.float32)

    def evaluate(X, Y, chunk=256):
        bad_bits, exact = 0.0, 0.0
        for k in range(0, len(X), chunk):
            x, y = to_mx(X[k:k + chunk], Y[k:k + chunk])
            wrong = mx.abs((model(x, in_mask) > 0).astype(mx.float32) - y) * out_mask
            bad_bits += wrong.sum().item()
            exact += (wrong.sum(axis=(1, 2)) == 0).sum().item()
        return bad_bits / (OUT_MASK.sum() * len(X)), exact / len(X)

    val = data.make_batch(999_999, 4096)
    stream = data.batches(args.batch, seed=args.seed + 1, trace_frac=args.trace_frac)
    tval = data.trace_batch(888_888, 2048) if args.trace_frac > 0 else None
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        X, Y = next(stream)
        loss = train_step(*to_mx(X, Y))
        mx.eval(state)
        if step % args.log == 0 or step == 1:
            be, ex = evaluate(*val)
            el = time.time() - t0
            mem = mx.get_peak_memory() / 2**30
            tex = f"  pong-trace exact {evaluate(*tval)[1]*100:6.2f}%" if tval else ""
            print(f"step {step:6d}  loss {loss.item():.5f}  val bit-err {be:.2e}  "
                  f"val exact {ex*100:6.2f}%{tex}  {el/60:5.1f} min  peak {mem:.1f}GB", flush=True)
            log.append(dict(step=step, loss=loss.item(), bit_err=be, exact=ex, minutes=el / 60))
        if step % 1000 == 0 or step == args.steps:
            model.save_weights(args.out)
            json.dump(log, open(args.out.replace(".safetensors", ".log.json"), "w"))
    print("done", flush=True)
    os._exit(0)  # don't wait on the endless worker pool


if __name__ == "__main__":  # multiprocessing on macOS re-imports this file in every worker
    main()
