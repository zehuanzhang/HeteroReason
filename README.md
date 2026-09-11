# HeteroReason Artifact

This repository contains the algorithmic implementation and the hardware-side
simulation artifacts used for the AE results.

## Contents

- `AE_alg/`: algorithmic implementation and scripts for the speculative-
  decoding experiments.
- `AE_hardware/`: hardware simulator with the token-index
  communication model enabled. This is the U280-style projected configuration.
- `AE_hardware_V80/`: V80 projection based on the same
  communication-aware simulator, using the V80 draft-speed and power
  assumptions documented in that directory.

## Hardware relationship

`AE_hardware_V80` is a separate copy of the communication-aware
simulator. The original `AE_hardware` directory is preserved so
the U280-style and V80-style results can be reproduced independently.

The hardware directories include their simulator, inputs, and generated
outputs. The root-level `figure8/` directory contains the final four-bar plot.
The three main components are independent; changes in one do not update the
others.

## Entry points

Algorithmic reproduction:

```bash
cd AE_alg/Algorithm
# Follow AE_alg/README.md and run the supplied smoke/key/full scripts.
```

Hardware simulation and plotting:

```bash
cd AE_hardware
# Follow README.md for the communication-aware U280 commands.

cd ../AE_hardware_V80
# Follow README.md for the corresponding V80 commands.
```

The final normalized four-bar Figure 8 can be generated from the packaged
U280 and V80 metric summaries:

```bash
python figure8/plot_figure8.py \
  --normalized
```

The output is written to `figure8/outputs/`.
