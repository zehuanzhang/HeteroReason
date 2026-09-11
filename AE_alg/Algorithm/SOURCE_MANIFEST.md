# Source Manifest

The AE package preserves separate baseline and pipeline source snapshots. The original experiment directories remain unchanged.

| AE snapshot | Original directory | `main_online_tree_single.py` SHA256 | Original evaluation script SHA256 |
|---|---|---|---|
| Config1 baseline | `RSD_0.5Bw8a8_7B_beam1` | `54a3974964d789829aeb71ad8787473e5dbd07fe221b16773134a5033b7e43ca` | `237867020809adb9757343b34d64f266d0f3fd2036abdee3ef9319fbf1d859df` |
| Config1 pipeline | `RSD_0.5Bw8a8_7B_draftpipe_cancel0608` | `f197eaf7cd73774055a28fdf3c4b8313aed09a8ff32d82123aab0f340a4a44be` | `5fc90346d41da9e49648dabce68dd06fd83dfd943985efc1388abd266ca78372` |
| Config2 baseline | `RSD_0.5Bw8a8_1p5B_beam1` | `2e9de44bc8290aa669f09c983b65cc976e42216c9925209da05eee8c05523681` | `4a916f57c36ade3a8ece254920c6f733878ab9d62b9b3416b7990c9492491b4a` |
| Config2 pipeline | `RSD_0.5Bw8a8_1p5B_draftpipe_cancel0608_pure` | `f49b449d9fe4a76e56aca73033a50ee75026823881ceaa24cffe94a7408a6ad8` | `b191ea022573a0e7ef55b3c70e627f2c453567e66a6d83353736b4e0e503a5de` |

The pipeline snapshots include target critical-token trace instrumentation. The helper file `pipeline_src/target_trace.py` has SHA256 `4659f526621a00940effd1f01d3984efb299637a353839661e2323e579ec6390`. The Config2 pipeline snapshot is the later version containing the multi-epoch bookkeeping fix used for the successful GSM8K rerun.

## Manuscript Table 3 Source

`table3_src/` is the shared source snapshot used for the manuscript Table 3 accuracy experiments. Config1 and Config2 use the same implementation with different target-model paths.

| AE snapshot | Original directories | `main_online_tree_single.py` SHA256 |
|---|---|---|
| Table 3 RSD/BRSD/optimized | `RSD`, `RSD_0.5B_1.5B` | `4c6392b667fe9ce16f2ea159690118c20657091593c55c5ee7f2d5aa4136c3d4` |

The Table 3 method mapping is `beam1` for RSD, `bbeam1` for BRSD, and `bbeam1_prefetch_cache` for BRSD + optimizations.
