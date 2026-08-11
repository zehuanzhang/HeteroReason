# Hardware Source and Input Manifest

## Portable Implementation

The files under `simulator/`, `scripts/run_config1.sh`, `scripts/reproduce_all.sh`, and `tests/` are the path-independent AE interface for the packaged critical-token hardware simulator. The key hardware artifact focuses on Config1. `config1.json` replays the with-backtracking trace, and `config1_no_bt.json` replays the no-backtracking `beam1_prefetch_cache` trace.

## Simulator Provenance

The public artifact provides the portable simulator. The internal simulator snapshots used to derive the packaged model are listed below for provenance.

| Internal snapshot | Original path | SHA256 |
|---|---|---|
| `v11_05B7B7B_board.py` | `SimularMICRO/v11_05B7B7B_board.py` | `237dc8f9540e4901e418d5a63de311159fa606e5278443b4c40165273c1e5a95` |

## Trace Directory Digests

The digest covers sorted JSON filenames and file contents within each dataset directory. The table below lists the Config1 key hardware traces used by `scripts/reproduce_all.sh`.

| Configuration | Mode | Dataset | Examples | SHA256 |
|---|---|---|---:|---|
| Config1 | with_bt | Math500 | 500 | `515f77faced62b7b86fa66bfbb5368fee9a0dc47a85fe1d9e760e03336085dfd` |
| Config1 | with_bt | GSM8K | 1319 | `97caae2ee5c8433483f4f67051c6fdfeface31fb0be0045a56d60cfb25359349` |
| Config1 | with_bt | Gaokao2023en | 385 | `868fa2a2edecebd11653ea2200c87ac319ac3a670a26ffc9f9f747fe321b7507` |
| Config1 | with_bt | OlympiadBench | 675 | `31320016a4f3dce31c173905f84d3447e7d1613f18f7779b119e17aaec4b9622` |
| Config1 | no_bt | Math500 | 500 | `c9eb24d36854490b67f81eac5923c3c2be97122b0b8ef064b7991b69db71d96a` |
| Config1 | no_bt | GSM8K | 1319 | `60d7de8e8a86f92090eb829655a03fa106903a51b7c8cfd970adb5c3e1205b2d` |
| Config1 | no_bt | Gaokao2023en | 385 | `1e6bfecff63e674d88927f1f4e48d87dfb0623111c3132e896f8f7eab6270c5e` |
| Config1 | no_bt | OlympiadBench | 675 | `16270ac1cf26a1760eeab51800dc71389e2002433a6179c4fc563b572fc90a4f` |
