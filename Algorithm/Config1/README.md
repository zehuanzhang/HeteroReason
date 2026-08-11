# Config1: 0.5B W8A8 Draft, 7B PRM, 7B Target

`baseline_src` is shared by Beam1 and BBeam1. `pipeline_src` preserves the source used for cancellable draft-pipeline experiments.

From the parent `Algorithm` directory:

```bash
bash Config1/run_beam1.sh smoke
bash Config1/run_bbeam1.sh smoke
bash Config1/run_pipeline.sh smoke
```

Replace `smoke` with `full` to evaluate all four datasets. Required model variables are documented in the parent README.
