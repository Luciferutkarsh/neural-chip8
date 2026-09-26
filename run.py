"""Run a ROM on the neural CPU, side by side with the real emulator.

Both machines get the exact same inputs (keys, 60Hz ticks, random bytes).
We record when (and in what part of the state) the neural machine first
disagrees with the real one, then keep both running to watch the drift.

python3 run.py out/run1.safetensors roms/pong.ch8 --frames 600 --gif out/pong.gif
"""
import argparse
import time

import mlx.core as mx
import numpy as np

import emu
import screen
from encode import encode, apply, IN_MASK
from probe import load, mnemonic


def same(a, b):
    diffs = []
    if a.PC != b.PC: diffs.append("PC")
    if a.I != b.I: diffs.append("I")
    if a.SP != b.SP: diffs.append("SP")
    if (a.DT, a.ST) != (b.DT, b.ST): diffs.append("timers")
    if (a.V != b.V).any(): diffs.append("V" + ",".join(f"{k:X}" for k in np.nonzero(a.V != b.V)[0]))
    if (a.stack[:max(a.SP, b.SP)] != b.stack[:max(a.SP, b.SP)]).any(): diffs.append("stack")
    if (a.mem != b.mem).any(): diffs.append("mem")
    if (a.disp != b.disp).any(): diffs.append(f"screen({int((a.disp != b.disp).sum())}px)")
    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("rom")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--ipf", type=int, default=30, help="instructions per 60Hz frame")
    p.add_argument("--gif", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--live", action="store_true", help="draw the neural screen in the terminal")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    args = p.parse_args()

    model = load(args.ckpt, args.d, args.layers)
    im = mx.array(IN_MASK, mx.float32)
    step_fn = mx.compile(lambda x: model(x, im) > 0)

    rom = open(args.rom, "rb").read()
    real, neural = emu.State.boot(rom), emu.State.boot(rom)
    r = np.random.default_rng(args.seed)
    first_div, events, frames = None, [], []
    t0, steps = time.time(), 0
    for f in range(args.frames):
        acc_r, acc_n = np.zeros((32, 64), np.uint8), np.zeros((32, 64), np.uint8)
        for i in range(args.ipf):
            keys, tick, rng = 0, i == args.ipf - 1, int(r.integers(256))
            op = neural.opcode()
            x = encode(neural, keys, tick, rng)
            flips = np.array(step_fn(mx.array(x[None], mx.float32))[0])
            apply(neural, flips)
            emu.step(real, keys, tick, rng)
            steps += 1
            acc_r |= real.disp; acc_n |= neural.disp
            if first_div is None:
                d = same(real, neural)
                if d:
                    first_div = steps
                    events.append((steps, f, hex(op), mnemonic(op), d))
        if args.gif:
            frames.append((acc_r.copy(), acc_n.copy()))
        if args.live:
            print("\033[H\033[2J" + screen.to_text(acc_n) + f"\nframe {f}  step {steps}  "
                  f"{steps/(time.time()-t0):.0f} steps/s  diverged: {first_div}", flush=True)
    el = time.time() - t0
    px = int((real.disp != neural.disp).sum())
    print(f"{steps} neural steps in {el:.1f}s ({steps/el:.0f} steps/s)")
    print(f"first divergence at step: {first_div}  {events[0] if events else ''}")
    print(f"end: {px} screen pixels differ; state diffs: {same(real, neural)}")
    print(f"scores real {real.V[5]}-{real.V[6]}  neural {neural.V[5]}-{neural.V[6]}")
    if args.gif:
        from PIL import Image, ImageDraw
        imgs = []
        for fr_r, fr_n in frames[::2]:
            a, b = screen.to_image(fr_r, 5), screen.to_image(fr_n, 5, on=(80, 220, 255))
            canvas = Image.new("RGB", (a.width * 2 + 12, a.height + 22), (0, 0, 0))
            canvas.paste(a, (0, 22)); canvas.paste(b, (a.width + 12, 22))
            dr = ImageDraw.Draw(canvas)
            dr.text((4, 4), "real emulator (code)", fill=(255, 176, 0))
            dr.text((a.width + 16, 4), "neural network (weights only)", fill=(80, 220, 255))
            imgs.append(canvas)
        screen.save_gif(imgs, args.gif, ms=66)
        print("wrote", args.gif)


if __name__ == "__main__":
    main()
