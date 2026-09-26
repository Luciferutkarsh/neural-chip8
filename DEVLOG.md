# Dev log: neural CHIP-8

Raw notes, written as things happen. Numbers are real.

## 1. Reference emulator + our own Pong

- Wrote a ~150 line CHIP-8 emulator (`emu.py`) and a tiny assembler (`asm.py`), then Pong in CHIP-8 assembly (`roms/pong.asm`, 332 bytes). Both paddles are AI, so it plays itself.
- **Failure #1 (my bug, not the AI's):** frames kept coming out with no paddles and no ball. Printing pixel counts per frame showed them alternating between 41 and 28, and 28 is just the score digits. The game erased everything at the top of the loop, ran ~40 instructions of logic, *then* redrew. At 30 instructions per frame, every other frame was captured during the "everything erased" window.
  - Fix 1: render like a CRT with phosphor persistence (OR every display state inside a frame). Real emulators fight the same flicker.
  - Fix 2: rewrite the game loop so every object is erased and redrawn back-to-back. After that, min/max pixels per frame was 33/45 over 3000 frames, with no blank frames.

## 2. Data + encoding

- The state is 88 tokens: opcode, PC, I, SP, two timers, keys, a "tick + random byte" line, 16 registers, 16 stack slots, 16 bytes of memory at I, and 32 screen rows (64 bits each).
- The network outputs an XOR "flip mask" per token. The harness XORs it back into the machine. The harness never reads the opcode, so everything about *which* instruction does *what* has to live in the weights.
- Data is **only random machine states**, never Pong. Each sample is a made-up machine with random registers, random screen noise and a random stack, running one random instruction. Around 12k samples/sec/core.
- A round-trip test (apply the true flip mask and compare with the emulator) passed on 3000 random states before any training.

## 3. Training infrastructure: three failures before the model learned anything

- **Failure #2: I fork-bombed my laptop.** On macOS, Python multiprocessing starts workers by *re-importing the main script*. `train.py` had no `if __name__ == "__main__"` guard, so each data worker started its own training run, which started its own workers... The output log hit 300k lines of "params: 1.21M" and tracebacks.
- **Failure #3: an endless `Pool.imap`.** Feeding `imap` an infinite generator makes it queue work forever. 6 workers made about 140 batches/sec that the main process had to unpickle, fighting the training loop for the GIL. Fixed with a bounded queue of 8 batches in flight.
- **Failure #4: swap death.** The training process sat at **11GB** (MLX GPU memory counts as process memory) and Chrome held about 12GB on a 16GB Mac, so there was 10.5GB of swap in use. One step at batch 512 took minutes instead of ~1s. Fixed with batch 256, validation in chunks, and `mx.set_memory_limit(4GB)`. Peak is now 4.9GB at 0.6s/step.
- First sign of life: the loss dropped from 0.70 to 0.042 in 20 steps, then flattened. At that point the model predicts "nothing changes" for every bit, which is right ~99% of the time per bit and useless as a CPU.

## 4. Run 1: the network can't do the *easy* instructions

6.4M-param transformer (d=256, 8 layers), batch 256, lr 5e-4. After 2000 steps (19 min) it got ~9-10% of instructions exactly right, and the per-instruction breakdown was upside down:

| instruction | exact | what it is |
|---|---|---|
| `LD Vx, kk` | **0%** | put a constant in a register |
| `JP nnn` | **0%** | jump to an address |
| `LD I, nnn` | **0%** | set I to a constant |
| `CALL`, `RET`, `LD DT`, `LOAD`, `STORE` | 0% | |
| `AND` / `OR` | 21-29% | |
| `SE` / `SNE` / `SKP` (compare + skip) | 34-43% | |
| `CLS` | 80% | |

Instructions that just **copy** a value were the ones at zero. The reason is my design: I made the output an XOR flip mask ("which bits change") so "nothing changes" would be the default. But then loading `kk` into `Vx` isn't a copy. The network has to output `old_Vx XOR kk`, an XOR between two different tokens. I had rebuilt the XOR problem, the thing that stalled single-layer perceptrons in 1969, and wrapped it around every mov instruction.

**Fix:** registers, PC, I, SP, timers, stack and memory output their *new value*. Only the screen keeps XOR flips, because CHIP-8 drawing *is* XOR, so there the flip mask is the natural answer (the sprite, shifted). Stopped run 1 and restarted as run 2 with lr 1e-3.

## 5. Run 2: the NOP machine

The "new value" fix alone did not help. At step 3000 (28 min), `LD Vx,kk`, `JP`, `LD I`, `CALL`, `RET` were *still* 0%. So the XOR explanation in section 4 was at best half right. I'd blamed the output format too quickly.

Looking at actual predictions showed what the network was doing:

```
JP 8b9:    want PC=8b9  got 91c   (old PC was 91a -> it predicted old PC + 2)
JP 3ab:    want PC=3ab  got 9d2   (old PC 9d0 -> +2 again)
LD VC,ff:  got 97, old value d7   (roughly "leave the register alone")
LD V8,f0:  got a2, old value a2   (exactly "leave it alone")
```

**It had learned to be a CPU where every instruction is a NOP.** Go to the next instruction and change nothing. Per bit that's right almost all the time, so the loss looks good. Compare-and-skip instructions scored ~30% only because "don't skip" is right half the time.

Two changes (run 3):
1. **Opcode as four one-hot nibbles** (16x4 = 64 bits) instead of 16 raw bits. Basically giving it the input lines of an instruction decoder. It still has to learn what every instruction *does*; picking "which register" becomes a lookup instead of a puzzle.
2. **Loss weighting:** bits that should change get 10x weight. Out of ~2600 output bits per step, JP's 12 PC bits were a rounding error.

## 6. Run 3 and the bug hunt: the network couldn't find the instruction

Run 3 (one-hot opcode + loss weighting) crawled to ~7% exact by step 2500, and copy instructions (`LD Vx,kk`, `JP`, `LD I`) were *still* 0%. So I made the smallest test I could think of:

- only `LD I, nnn` samples, loss only on the I slot. The answer is literally "the low 12 bits of the opcode".
- **Result: loss sat at 0.693 = ln 2 for 300 steps. A coin flip.**

The data was fine (target I == nnn every time). The problem was attention. The I token has to *find* the opcode token among 88 tokens, and at init attention is ~uniform, so the opcode arrives diluted to 1/88 and the gradient is tiny. A 1500-step run on just 3 copy instructions only started to move around step 1100.

**Fix: give it a bus.** Every token now also sees a shared vector with the whole non-screen state (opcode, PC, I, timers, keys, registers, stack, memory at I), the same way a real CPU wires the instruction and register file to everything. The logic is still 100% learned. Same minimal test with the bus: **coin flip to 100% exact in 250 steps.**

Also dropped from 8 layers to 4 (attention has much less routing to figure out now), which is about 2x faster. Run 4: d=256, 4 layers, 4.1M params, lr 1e-3.

## 7. Run 4 → v2 architecture: stop paying for attention

Run 4 (transformer + bus) was 2x faster but still 0% on `LD I` at step 2000. In the full mix `LD I` is ~1.4% of samples, so it had seen ~7k examples where the minimal test had 25k. There are 35 instruction types to learn, so the fix is *more samples per second*.

**v2 = core + scanline unit**, both ordinary learned networks, no attention:
- *core*: a gated residual MLP (4 blocks, width 1024) that reads the bus and writes the next PC / I / SP / timers / registers / stack / memory.
- *scanline unit*: one small shared network run on each of the 32 screen rows. It sees the bus, its own row and its row number, and outputs that row's XOR flips plus a "collision" feature.
- the collision features are max-pooled across rows and wired into the core, which is how VF can find out about sprite collisions.

About 4x the samples/sec: 1000 steps in 1.4 min at batch 512 vs ~9 min before.

Silly bug on the way: MLX treats *every* array attribute on a module as a trainable parameter, including my integer index list. The optimizer turned it into floats and the gather crashed. Fix: name it `_bus_idx`.

**Run 5 (v2):** instructions get learned *one at a time*, like the classic "grokking" curves:

| instruction | step 4k | step 8k |
|---|---|---|
| `JP` | 7.6% | 94.9% |
| `LD I` | 6.2% | 83.1% |
| `CLS` | 30% | 100% |
| `LOAD` / `STORE` | 16% / 5% | 61% / 55% |
| `ADD`, `SUB`, `RET`, `CALL` | ~0% | ~0% |

First closed-loop Pong with the 8k checkpoint: it got **instruction #2** wrong (`LD V3, 13`, the paddle's starting position). After 600 steps the neural machine's score counter read **128-0**.

## 8. Run 5 final + run 6 (fine-tune on Pong traces)

- Run 5 finished 20k steps (27.5 min) at **35.1% exact on random states**, and was flattening as the learning rate decayed. Learning *every* instruction bit-exactly from random states alone would take many more hours on this laptop.
- Surprise: that model, which **had never seen Pong**, already got **51.4%** of real Pong instructions exactly right.
- Run 6: fine-tune from run 5 on batches that are 50% random states and 50% Pong execution traces (8 games per worker, random seeds, random key presses, random instructions-per-frame).
  - Pong-trace exact: 51% → 84% (1k steps) → 88% (2k) → 89% (2.5k).
  - Random-state exact fell **35% → 15%** in the first 500 steps and slowly recovered to ~25%. Classic catastrophic forgetting: the warmup pushed the learning rate back up and the new data overwrote the general skills.
- **Failure #9: the laptop went to sleep.** On battery with the lid closed, macOS suspends everything and `caffeinate` can't stop it. 250 training steps took 18 minutes of wall-clock time.
- Run 6 finished (12k steps): **Pong-trace exact 97.66%**, random-state exact 36.9% (it recovered past run 5's 35%).
  Trajectory: 91% (3k) → 93.3% (6k) → 94.3% (8k) → 97.0% (10k) → 97.7% (12k). Still climbing.
- Closed-loop Pong with the final model: the first **12 instructions match exactly**. Instruction 13 is a `DRW` that puts 2 pixels wrong, and it drifts from there (final frame: paddles moved, both "0"s melted into non-digits, neural score 0-1 vs real 0-0). The 4k checkpoint broke at instruction 2.
- Held-out bounce ROM: breaks at instruction 11 (`LD V8, 50`).
- 599 neural steps/sec at inference, fast enough for real-time CHIP-8 if it were correct.
