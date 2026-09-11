# HeteroReason Algorithm AE Bundle

This bundle keeps only the algorithm-side artifacts needed to reproduce Table 3.

## 1. Layout

- `Algorithm/common/` shared runner, dataset loading, and evaluation plumbing
- `Algorithm/table3_src/` frozen source snapshot used by the Table 3 runs
- `Algorithm/Config1/` thin wrappers for the 7B target setup
- `Algorithm/Config2/` thin wrappers for the 1.5B target setup
- `Algorithm/reference_results/` machine-readable Table 3 reference values
- `Algorithm/reference_runs/config1/` bundled Config1 logs for inspection
- `Algorithm/run_key_results.sh` Config1 Math500 key-result entrypoint
- `Algorithm/run_full_results.sh` full Table 3 algorithm matrix
- `Algorithm/run_smoke.sh` 8-example smoke test

`table3_src/` is the exact manuscript source snapshot. `common/` is the shared launcher and runtime code used by every entrypoint.

## 2. Getting Started

### 2.1 Environment

Install the Python dependencies:

```bash
python3 -m pip install --upgrade pip
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

Use any three free GPUs for `CUDA_VISIBLE_DEVICES`; `0,1,2` is only an example.

### 2.2 Model Downloads

The scripts expect local Hugging Face checkpoints. If the models are not already available, install the Hugging Face CLI and download them to local folders:

```bash
pip install -U "huggingface_hub[cli]"

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /path/to/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-Math-PRM-7B --local-dir /path/to/Qwen2.5-Math-PRM-7B
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /path/to/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir /path/to/Qwen2.5-1.5B-Instruct
```

After downloading, export `DRAFT_MODEL`, `PRM_MODEL`, `CONFIG1_TARGET_MODEL`, and `CONFIG2_TARGET_MODEL` as shown above.

### 2.3 Runtime Expectations

Approximate historical runtimes on our 3x RTX 3090 setup are:

| Entry point | Scope | Estimated runtime |
| --- | --- | ---: |
| `bash run_smoke.sh` | Config1/Math500, 8 examples, three methods | 10-30 min |
| `KEY_METHODS="rsd" bash run_key_results.sh` | Config1/Math500 full dataset, RSD | 81:52 |
| `KEY_METHODS="brsd" bash run_key_results.sh` | Config1/Math500 full dataset, BRSD | 113:36 |
| `KEY_METHODS="brsd_optimized" bash run_key_results.sh` | Config1/Math500 full dataset, BRSD + optimizations | 106:50 |
| `bash run_full_results.sh` | All four datasets, three methods, Config1 and Config2 | Long run, roughly 1-2 days |

Runtime can vary substantially with GPU type, current GPU load, vLLM cache state, and model-loading/CUDA-graph overhead.

### 2.4 Algorithmic Smoke

Smoke mode runs the three Table 3 methods on eight fixed Config1/Math500 examples. It validates the environment.

```bash
cd Algorithm
bash run_smoke.sh
```

## 3. Key Result

From the repository root:

```bash
cd Algorithm
KEY_METHODS="rsd" bash run_key_results.sh
KEY_METHODS="brsd" bash run_key_results.sh
KEY_METHODS="brsd_optimized" bash run_key_results.sh
```

These are the three Table 3 methods. The default method is `brsd_optimized`.
Each command reproduces the Config1/Math500 key result for one method.

Reference results:

| Method | Paper label | Accuracy | Avg. latency / problem | Observed wall-clock |
| --- | --- | ---: | ---: | ---: |
| `rsd` | RSD baseline | 64.8% | 9.63 s | 81:52 |
| `brsd` | BRSD | 67.0% | 13.33 s | 113:36 |
| `brsd_optimized` | BRSD + optimizations | 67.2% | 12.56 s | 106:50 |

The accuracy values should match Table 3. Runtime can vary with GPU type, GPU load, and model-loading overhead.

## 4. Full Result

From the repository root:

```bash
cd Algorithm
bash run_full_results.sh
```

This runs `rsd`, `brsd`, and `brsd_optimized` on all four datasets for both Config1 and Config2.
