"""Draws docs/architecture.png: how the neural CHIP-8 works (v2 architecture)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, PANEL = "#0f0d0a", "#1a1712"
AMBER, CYAN, GREY, TEXT, DIM = "#ffb000", "#50dcff", "#c9c4bb", "#f2eee8", "#8f887d"
MONO = "DejaVu Sans Mono"

fig = plt.figure(figsize=(16, 10), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160); ax.set_ylim(-16, 92); ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, edge, fill=PANEL, lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fill, ec=edge, lw=lw))


def text(x, y, s, size=11, color=TEXT, weight="normal", ha="left", family=None, va="center"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, family=family)


def arrow(p, q, color, rad=0.0, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# title
text(80, 87, "A neural network that is the CPU", 22, weight="bold", ha="center")
text(80, 82, "one forward pass = one CHIP-8 clock cycle  ·  no emulator code in the loop",
     12.5, DIM, ha="center")

# ---- machine state -----------------------------------------------------------
box(3, 12, 38, 64, GREY)
text(6, 72.5, "MACHINE STATE", 13, weight="bold")
text(6, 69, "88 tokens × up to 64 bits", 10.5, DIM)
chips = [
    ("OP", "opcode, 4 one-hot nibbles", AMBER),
    ("PC  I  SP  DT  ST", "program counter, index, timers", AMBER),
    ("KEYS  tick  rng", "keypad, 60 Hz tick, random byte", AMBER),
    ("V0 … VF", "16 registers", AMBER),
    ("STACK 0 … 15", "return addresses", AMBER),
    ("MEM[I … I+15]", "16 bytes of RAM at I", AMBER),
    ("ROW 0 … 31", "the screen, 64 px per row", CYAN),
]
y = 63
for name, desc, col in chips:
    h = 9.5 if col == CYAN else 6.2
    y -= h
    box(5.5, y + 0.6, 33, h - 1.2, col, fill=BG, lw=1.4)
    text(7.5, y + h / 2 + (1.3 if col == CYAN else 0.9), name, 10.5, col, "bold", family=MONO)
    text(7.5, y + h / 2 - (1.9 if col == CYAN else 1.4), desc, 8.8, DIM)
    if col == CYAN:
        for k in range(5):  # a few fake screen rows
            ax.plot([26.5 + k * 2.3, 27.9 + k * 2.3], [y + h / 2 - 1.8] * 2, color=CYAN, lw=2.5, alpha=0.35 + 0.1 * k)
bus_top, bus_bot = 63, 63 - 6 * 6.2
screen_mid = bus_bot - 9.5 / 2

# ---- bus -----------------------------------------------------------------------
box(49, 26, 13, 44, AMBER)
text(55.5, 64, "BUS", 14, AMBER, "bold", ha="center")
text(55.5, 58.5, "581 bits", 10, TEXT, ha="center", family=MONO)
for i, line in enumerate(["everything", "except the", "screen, wired", "to every", "part of the", "network", "(like a real", "CPU's bus)"]):
    text(55.5, 52 - i * 3, line, 9.2, DIM, ha="center")
arrow((39, (bus_top + bus_bot) / 2), (49, (bus_top + bus_bot) / 2), AMBER)

# ---- CPU core ------------------------------------------------------------------
box(72, 48, 48, 28, AMBER)
text(75, 72.3, "CPU CORE", 13.5, AMBER, "bold")
text(75, 68.3, "gated residual MLP · 4 blocks · width 1024", 10, DIM)
text(75, 62.5, "in:", 10.5, TEXT, "bold"); text(80, 62.5, "bus + collision wire", 10.5)
text(75, 57.5, "out:", 10.5, TEXT, "bold"); text(80.5, 57.5, "next PC, I, SP, timers,", 10.5)
text(80.5, 53.5, "V0…VF, stack, MEM[I…I+15]", 10.5)

# ---- scanline unit -------------------------------------------------------------
box(72, 8, 48, 28, CYAN)
text(75, 32.3, "SCANLINE UNIT  × 32", 13.5, CYAN, "bold")
text(75, 28.3, "one small gated MLP, same weights for every row", 10, DIM)
text(75, 22.5, "in:", 10.5, TEXT, "bold"); text(80, 22.5, "its row's 64 px + row number + bus", 10.5)
text(75, 17.5, "out:", 10.5, TEXT, "bold"); text(80.5, 17.5, "XOR flip mask for its row", 10.5)
text(80.5, 13.5, "+ a collision feature", 10.5)

# collision wire (scanline -> core)
arrow((113, 36), (113, 48), CYAN, lw=2.4)
text(111.5, 42, "max-pool over 32 rows\n= collision wire (VF)", 9.5, CYAN, ha="right")

# bus -> core / scanline, screen -> scanline
arrow((62, 58), (72, 62), AMBER)
arrow((62, 36), (72, 26), AMBER)
arrow((39, screen_mid), (72, 17), CYAN, rad=0.12)

# ---- harness -------------------------------------------------------------------
box(128, 22, 29, 46, GREY)
text(142.5, 63.5, "HARNESS", 13.5, GREY, "bold", ha="center")
text(142.5, 59.8, "\"dumb wires\"", 10.5, DIM, ha="center")
for i, line in enumerate(["• write the new", "   values back", "• XOR flip masks", "   onto the screen", "• fetch the opcode", "   at PC and 16", "   bytes at I"]):
    text(130.5, 53 - i * 3.3, line, 10)
text(142.5, 26.5, "never decodes the\ninstruction", 9.8, AMBER, "bold", ha="center")
arrow((120, 62), (128, 55), AMBER)
arrow((120, 22), (128, 35), CYAN)

# ---- clock loop ----------------------------------------------------------------
ax.plot([142.5, 142.5], [22, 3], color=GREY, lw=2.2)
ax.plot([142.5, 22], [3, 3], color=GREY, lw=2.2)
arrow((22, 3), (22, 12), GREY)
text(82, 4.8, "next clock cycle  (≈ 600 per second on a MacBook)", 10.5, GREY, ha="center")

# ---- training band -------------------------------------------------------------
box(3, -14, 154, 12.5, DIM, fill=PANEL, lw=1.2)
text(6, -4.6, "HOW IT'S TRAINED", 11.5, TEXT, "bold")
text(6, -9.8, "teacher: a normal emulator labels every sample   ·   data: random machine states, "
     "then 50% real Pong traces   ·   loss: per-bit cross-entropy, changed bits ×10   ·   31M params, MLX on an M5",
     9.8, DIM)

fig.savefig("docs/architecture.png", facecolor=BG)
print("wrote docs/architecture.png")
