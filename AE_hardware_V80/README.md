# AE Hardware

This package contains the hardware/simulator side of the artifact in a
self-contained form. It can reproduce Figure 8 without rerunning model
inference.

Contents:

```text
simulator/        latency + energy simulator
inputs/           step-profile CSVs and goodput token logs
outputs/          precomputed simulator and energy outputs
```

The simulator package retains the following source data:

```text
latency: outputs/simulator_latency/*_per_problem.csv
goodput: inputs/token_logs/weak_config{1,2}/*.jsonl
energy:  outputs/energy/config{1,2}/*_per_problem.csv
```

The final four-bar Figure 8 uses the packaged metric summaries in
`../figure8/inputs/`.

## Final Figure 8

The final normalized four-bar Figure 8 is generated from the packaged metric
inputs in the parent `AE_camera/figure8` directory:

```bash
cd ..
python figure8/plot_figure8.py --normalized
```

## Regenerate Energy Outputs

The packaged energy outputs can also be regenerated from the included
step-profile CSVs. This does not run the models.

```bash
mkdir -p outputs/reproduced_energy/config1 outputs/reproduced_energy/config2

for DATASET in math500 gsm8k gaokao2023en olympiadbench; do
  python simulator/analyze_latency_energy_communication.py \
    -csv_path "inputs/step_profile_csv/config1/step-profile-config1-bbeam1-${DATASET}-exclusive0817_${DATASET}_step_profile.csv" \
    -o "outputs/reproduced_energy/config1/step-profile-config1-bbeam1-${DATASET}-exclusive0817_${DATASET}_step_profile" \
    -target_overlap \
    --target_size 7b
done

for DATASET in math500 gsm8k gaokao2023en olympiadbench; do
  python simulator/analyze_latency_energy_communication.py \
    -csv_path "inputs/step_profile_csv/config2/step-profile-config2-bbeam1-${DATASET}-exclusive0817_${DATASET}_step_profile.csv" \
    -o "outputs/reproduced_energy/config2/step-profile-config2-bbeam1-${DATASET}-exclusive0817_${DATASET}_step_profile" \
    -target_overlap \
    --target_size 1.5b
done
```
