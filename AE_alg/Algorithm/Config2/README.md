# Config2: 0.5B W8A8 Draft, 7B PRM, 1.5B Target

`baseline_src` is shared by Beam1 and BBeam1. `pipeline_src` is the later fixed source snapshot containing the multi-epoch bookkeeping correction used for the successful GSM8K rerun.

From the parent `Algorithm` directory:

```bash
bash Config2/run_beam1.sh smoke
bash Config2/run_bbeam1.sh smoke
bash Config2/run_pipeline.sh smoke
```

Replace `smoke` with `full` to evaluate all four datasets. Required model variables are documented in the parent README.
