# HeteroReason Artifact

This repository contains the artifact for **HeteroReason**, including the
algorithmic implementation and the packaged inputs and scripts used to
reproduce the main algorithmic results and Figure 8.

## Contents

- `AE_alg/`: algorithmic implementation for the speculative-decoding
  experiments.
- `figure8/`: script and packaged metric summaries for the normalized,
  four-bar Figure 8 plot.
- `appendix.tex`: artifact appendix corresponding to the packaged layout.

## 1. Requirements

- Linux with Python 3.10 or a compatible Python environment.
- CUDA GPUs and a working PyTorch/vLLM installation for the algorithmic
  experiments.
- Local Hugging Face checkpoints for the draft, PRM, and target models.

The tested environment uses `vllm==0.9.2` and `transformers==4.53.3`.
Install the algorithm dependencies with:

```bash
cd AE_alg
python3 -m pip install -r requirements.txt
```

The scripts expect the following environment variables. Replace the paths
with the locations of the downloaded checkpoints on the local machine:

```bash
export DRAFT_MODEL=/path/to/Qwen2.5-0.5B-Instruct
export PRM_MODEL=/path/to/Qwen2.5-Math-PRM-7B
export CONFIG1_TARGET_MODEL=/path/to/Qwen2.5-7B-Instruct
export CONFIG2_TARGET_MODEL=/path/to/Qwen2.5-1.5B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2
```

For example, the models can be downloaded with the Hugging Face CLI:

```bash
python3 -m pip install -U "huggingface_hub[cli]"

hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /path/to/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-Math-PRM-7B \
  --local-dir /path/to/Qwen2.5-Math-PRM-7B
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /path/to/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /path/to/Qwen2.5-1.5B-Instruct
```

## 2. Algorithmic Reproduction

All algorithmic commands are run from `AE_alg/Algorithm` after setting the
environment variables above.

### Smoke test

The smoke test runs the three Table 3 methods on eight fixed Math500
examples using Config1:

```bash
cd AE_alg/Algorithm
bash run_smoke.sh
```

This is the recommended first check of the environment. It normally takes
about 10--30 minutes, depending on GPU type and model-loading overhead.

### Key result

The following commands reproduce the three Config1/Math500 key-result runs:

```bash
cd AE_alg/Algorithm
KEY_METHODS="rsd" bash run_key_results.sh
KEY_METHODS="brsd" bash run_key_results.sh
KEY_METHODS="brsd_optimized" bash run_key_results.sh
```

The default method is `brsd_optimized`. Historical runtimes on three RTX
3090 GPUs were approximately 82 minutes for `rsd`, 114 minutes for `brsd`,
and 107 minutes for `brsd_optimized`. Runtime varies with hardware, GPU
load, CUDA/vLLM versions, and cache state.

### Full algorithmic result

To run the complete Table 3 matrix across Config1, Config2, and the four
datasets (`math500`, `gsm8k`, `gaokao2023en`, and `olympiadbench`):

```bash
cd AE_alg/Algorithm
bash run_full_results.sh
```

The full run is substantially longer than the smoke and key-result runs.
Results and logs are written below `AE_alg/Algorithm/results/` and
`AE_alg/Algorithm/logs/`.

## 3. Figure 8

The repository includes the metric summaries required for the final
normalized Figure 8. No algorithmic or hardware experiment needs to be
rerun for this plot command:

```bash
python figure8/plot_figure8.py --normalized
```

The output is written to `figure8/outputs/`:

- `figure8_projected_compare_normalized.png`
- `figure8_projected_compare_normalized.pdf`

The figure contains four bars for each dataset/configuration:

1. Weak baseline
2. Strong baseline
3. Ours (U280 projected)
4. Ours (V80 projected)

The bars are normalized to the weak baseline for the corresponding
configuration and dataset. The packaged summaries are stored in
`figure8/inputs/`.

## 4. Project Structure

```text
.
├── AE_alg/
│   ├── Algorithm/
│   │   ├── run_smoke.sh
│   │   ├── run_key_results.sh
│   │   └── run_full_results.sh
│   ├── requirements.txt
│   └── README.md
├── figure8/
│   ├── inputs/
│   ├── outputs/
│   └── plot_figure8.py
└── appendix.tex
```

## Citation

If you use this artifact, please cite the accompanying HeteroReason paper.
