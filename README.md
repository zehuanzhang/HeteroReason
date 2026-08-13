# HeteroReason: Heterogeneous FPGA-GPU Acceleration for Disaggregated Speculative Reasoning

Artifact repository for **HeteroReason: Heterogeneous FPGA-GPU Acceleration for Disaggregated Speculative Reasoning**.
HeteroReason disaggregates speculative reasoning across an FPGA draft engine, a GPU process reward model (PRM), and a GPU target model. The artifact provides software implementations, experiment scripts, and hardware simulator. 

## Table of Contents

1. [Evaluation Guide](#1-evaluation-guide)
2. [Getting Started](#2-getting-started)
3. [Reproduction Experiments](#3-reproduction-experiments)
4. [Mapping to Manuscript Results](#4-mapping-to-manuscript-results)
5. [Project Structure](#5-project-structure)
6. [Variability and Interpretation](#6-variability-and-interpretation)

## 1. Evaluation Guide

| Experiments | Coverage | Expected time |
|---|---|---|
| Algorithmic smoke | Config1, 8 fixed Math500 examples for RSD, Backtracking-enhanced RSD, and Backtracking-enhanced RSD with optimizations | Hardware-dependent |
| Key Results | Config1 on Math500 with the main algorithmic optimization variants | About 1 h 47 min on 3x RTX 3090 |
| Full Results | Config1 and Config2 across all datasets and optimization methods | About 1-2 days on 3x RTX 3090 |
| Hardware simulations | Trace-driven latency and throughput simulation with and without backtracking; Config1 is the key hardware result | Several seconds |

Our evaluation platform uses an AMD EPYC 7543 32-Core processor and three NVIDIA GeForce RTX 3090 GPUs with 24 GB memory each. The Algorithm experiments use three GPUs, with one GPU each for the draft model, target model, and PRM; the Hardware simulator runs on the CPU only. Model checkpoints are not included in the repository and require additional local storage.

## 2. Getting Started

### 2.1 Environment

Install the environment:

```bash
pip install -r requirements.txt
```

Set the local model paths:

```bash
export DRAFT_MODEL=/path/to/Qwen2.5-0.5B-Instruct
export PRM_MODEL=/path/to/Qwen2.5-Math-PRM-7B
export CONFIG1_TARGET_MODEL=/path/to/Qwen2.5-7B-Instruct
export CONFIG2_TARGET_MODEL=/path/to/Qwen2.5-1.5B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2
```

### 2.2 Algorithmic Smoke

Smoke mode runs the three Table 3 methods on eight fixed Config1/Math500 examples. It validates the environment.

```bash
cd Algorithm
bash run_smoke.sh
```

## 3. Reproduction Experiments

### 3.1 Algorithmic Experiments

#### 3.1.1 Key Results

Full Algorithm experiments take about 1-2 days on the reference 3x RTX 3090 platform. Some key results can be reproduced with shorter time such as BRSD + optimizations on all 500 Config1/Math500 examples:

```bash
cd Algorithm
bash run_key_results.sh
```

This key run supports resuming from saved JSONL outputs and historically took 106 min 38 s on three NVIDIA GeForce RTX 3090 GPUs. Wall-clock latency depends on GPU hardware, drivers, CUDA, and the software stack. The historical Config1/Math500 reference used the complete 500-example dataset. 

All RSD, BRSD, and BRSD + optimizations methods can also be run for the same Config1/Math500 dataset column:

```bash
cd Algorithm
KEY_METHODS="rsd brsd brsd_optimized" bash run_key_results.sh
```

The observed sequential time for all three Config1/Math500 methods was approximately 6 h 51 min on 3x RTX 3090.

The complete machine-readable Table 3 reference is stored in [`Algorithm/reference_results/table3_accuracy_reference.json`](Algorithm/reference_results/table3_accuracy_reference.json).



#### 3.1.2 Full Results

The full experiment matrix reproduces all rows and columns of manuscript Table 3 for Config1 and Config2 on Math500, GSM8K, Gaokao2023en, and OlympiadBench. This reproduction takes about 1-2 days on the reference 3x RTX 3090 platform:

```bash
cd Algorithm
bash run_full_results.sh
```

Generated predictions, metrics, and logs are separated by configuration, method, run tag, and dataset under `Algorithm/results/` and `Algorithm/logs/`.

### 3.2 Hardware Simulation


#### 3.2.1 Cross-validation with on-board implementation

The FPGA performance model is cross-validated with the on-board FPGA performance at 200MHz (with rapidstream and tapa tool, it can be further improved to 300MHz). The packaged bitstream is stored in `Bitstream/decoding.xclbin`. If for Artifact Evalation purpose, reviewer need ready-to-run enviroment and FPGA devices, please contact authors to access the remote evaluation server to use the board packages prepared (slot booking is needed to schedule with reviewers' availability). 


#### 3.2.2 Trace-driven estimation

Trace-driven estimation replays the execution traces collected from the algorithm runs. The simulator applies the recorded accept, target-rescue, and backtracking events, including pipeline overlap between draft execution and PRM/communication.
Due to licensing issues with the RapidStream and TAPA tools, FPGA latency is scaled based on a projected operating frequency of 300 MHz and the higher number of KV ports supported by the FPGA board, incorporating the optimizations described in [r1]. The key Config1 performance model parameters used by the simulator are recorded in `Hardware/simulator/configs/config1.json`.

[r1] Zhang J, He Z, Fraser N, et al. FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design[J]. arXiv preprint arXiv:2601.15710, 2026.


#### 3.2.3 End-to-end simulation estimation

With the cross-validation with on-board FPGA performance, we build our end-to-end simulator as mentioned in our paper. The key hardware reproduction focuses on Config1. 

```bash
cd Hardware
./scripts/run_config1.sh with_bt
```


## 4. Generating Manuscript Results

### 4.1 Key result mapping

The packaged reference files contain the values used by the plotting/checking scripts. Config1 is reproduced as the key result.

**Table 3**

Reference data: `Algorithm/reference_results/table3_accuracy_reference.json`

Key reproduction command:

```bash
cd Algorithm
bash run_key_results.sh
```

Generated outputs are written under `Algorithm/results/.../`. The method directories are `rsd`, `brsd`, and `brsd_optimized`; each dataset directory contains generated predictions and a `*_metrics.json` file, whose `acc` field can be compared with the reference JSON.

**Figure 8**

Reference data: `Analysis/reference_results/figure_metrics.json`, field `figure8`. Figure 8 compares weak/strong GPU baselines with Ours; Ours is the FPGA-GPU-GPU simulator with backtracking.

Config1 key simulator command:

```bash
cd Hardware
./scripts/run_config1.sh with_bt
```

Plot command:

```bash
cd Analysis
python3 scripts/plot_figure8.py
```

Outputs: `Analysis/outputs/figure8_metrics.csv`, `Analysis/outputs/figure8.png`, and `Analysis/outputs/figure8.pdf`.

**Figure 9**

Reference data: `Analysis/reference_results/figure_metrics.json`, field `figure9`. Figure 9 computes latency speedup over the RSD framework; each bar is RSD latency divided by the corresponding system latency.

Config1 key simulator command:

```bash
cd Hardware
./scripts/run_config1.sh with_bt
```

Plot command:

```bash
cd Analysis
python3 scripts/plot_figure9.py
```

Outputs: `Analysis/outputs/figure9_speedup.csv`, `Analysis/outputs/figure9.png`, and `Analysis/outputs/figure9.pdf`.

**Figure 10**

Reference data: `Analysis/reference_results/figure_metrics.json`, field `figure10`. Figure 10 shows the ablation of T2, T1, and backtracking: Baseline is basic BBeam1, Baseline+T2 adds scheduling optimizations, Baseline+T2+T1 uses the simulator with backtracking, and the final bar uses the no-backtracking `beam1_prefetch_cache` trace.

Config1 key simulator commands:

```bash
cd Hardware
./scripts/run_config1.sh with_bt
```

Plot command:

```bash
cd Analysis
python3 scripts/plot_figure10.py
```

Outputs: `Analysis/outputs/figure10_ablation.csv`, `Analysis/outputs/figure10.png`, and `Analysis/outputs/figure10.pdf`.

For an exact-match hardware validation of the packaged Config1 case, run `cd Hardware && ./scripts/reproduce_all.sh`; this check compares `with_bt` with `Hardware/reference_results/simulator_reference.json`.


### 4.2 Data Flow

```text
Table 3:
  Algorithm datasets
    -> RSD / BRSD / BRSD + optimizations
    -> JSONL predictions and accuracy metrics
    -> Table 3

Figure 8:
  Packaged execution traces
    -> portable FPGA-GPU-GPU simulator
    -> latency and raw-throughput JSON/CSV
    -> measured GPU baselines + packaged Analysis reference metrics
    -> Figure 8

Figure 9:
  Beam1 and BBeam1 Algorithm metrics + with-backtracking simulator results
    -> collect per-dataset latency for Config1
    -> compute RSD latency / system latency
    -> Figure 9

Figure 10:
  BBeam1 metrics + optimized GPU-pipeline metrics
    -> with-backtracking simulator results
    -> compute adjacent-stage latency speedups
    -> Figure 10


```

The public Hardware package includes the paper-facing traces, so Hardware regression does not require the expensive Algorithm run.

## 5. Project Structure

```text
HeteroReason/
  README.md
  requirements.txt           top-level Algorithm and Analysis dependencies
  Algorithm/
    table3_src/               exact source snapshot for manuscript Table 3
    Config1/                  additional Config1 experiment entrypoints
    Config2/                  additional Config2 experiment entrypoints
    common/                   shared runner, evaluation code, and datasets
    reference_results/       complete Table 3 accuracy reference
    run_smoke.sh             8-example Table 3 smoke test
    run_key_results.sh       Config1/Math500 key-result reproduction
    run_full_results.sh      complete Table 3 reproduction
    README.md
  Hardware/
    simulator/               portable latency simulator
    inputs/                  packaged traces for Hardware simulation
    reference_results/       exact Hardware reference metrics
    scripts/                 one-command reproduction and validation
    tests/                   formula, synthetic, and regression tests
    README.md
  Analysis/
    reference_results/       packaged metrics and constants for figures
    scripts/                 Figure 8/9/10 reproduction scripts
    outputs/                 generated CSV/PNG/PDF files, ignored by git
    README.md
```

## 6. Variability and Interpretation

- The Hardware simulator is deterministic for the packaged traces and matches the reference JSON exactly.
- Algorithm wall-clock time is machine-dependent and should not be expected to match RTX 3090 measurements exactly.
- Algorithm runs use fixed seed `43` and greedy search temperature `0.0`. Accuracy should be close to the reference, although low-level GPU/software differences may affect individual generations.
- Smoke-test accuracy is not statistically comparable with complete-dataset results.
