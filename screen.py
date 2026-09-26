"""Rendering helpers: terminal (half-block chars) and PNG/GIF frames."""
import numpy as np
from PIL import Image


def to_text(disp):
    rows = []
    for y in range(0, 32, 2):
        top, bot = disp[y], disp[y + 1]
        rows.append("".join(" ▀▄█"[t | (b << 1)] for t, b in zip(top, bot)))
    return "\n".join(rows)


ON, OFF = (255, 176, 0), (18, 14, 8)


def to_image(disp, scale=8, on=ON, off=OFF):
    rgb = np.where(disp[..., None].astype(bool), np.array(on, np.uint8), np.array(off, np.uint8))
    img = Image.fromarray(rgb.astype(np.uint8))
    return img.resize((64 * scale, 32 * scale), Image.NEAREST)


def save_gif(frames, path, ms=33):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
