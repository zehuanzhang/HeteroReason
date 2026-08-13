# Hardware Source and Input Manifest

## Portable Implementation

The files under `simulator/`, `scripts/run_config1.sh`, `scripts/reproduce_all.sh`, and `tests/` are the path-independent AE interface for the packaged v12 hardware simulator. The key hardware artifact focuses on Config1. `config1.json` replays the packaged with-backtracking BBeam1 trace and recomputes FPGA-GPU-GPU overlap inside the simulator.

## Simulator Provenance

The public artifact provides the portable simulator. The internal simulator snapshots used to derive the packaged model are listed below for provenance.

| Internal snapshot | Original path | SHA256 |
|---|---|---|
| `v12_rescue_serial_then_parallel` | portable `Hardware/simulator/` implementation | packaged source |

## Trace Directory Digests

The digest covers sorted JSON filenames and file contents within each dataset directory. The table below lists the Config1 key hardware traces used by `scripts/reproduce_all.sh`.

| Configuration | Mode | Dataset | Examples | SHA256 |
|---|---|---|---:|---|
| Config1 | with_bt | Math500 | 500 | `aa50c9ba5622667eaa9622b20b316d772595454bfbb81ca0e747465bdc0096e8` |
| Config1 | with_bt | GSM8K | 1319 | `4508ef792b416a1f5d58c1055c6737d4b0ee8299bf5055bd1fe58295c52b6a47` |
| Config1 | with_bt | Gaokao2023en | 385 | `0f1d989daf8bb8a99becc05a6def2443ace88894e178a72451a08d8ff246708a` |
| Config1 | with_bt | OlympiadBench | 675 | `3d0c76e79c7726afd53d7ec4c0a5ab90f863a439a9dc8a82eb041969a116e27c` |
