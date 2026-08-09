# Cross-fitted ET post-processing benchmark

An exploratory out-of-fold sweep varied the ET probability threshold and the
minimum connected-component size. For each evaluated fold, the operating point
was selected on the other folds. This procedure did not improve the default
hard-label reconstruction:

| Metric | Baseline mean | Cross-fitted mean | Paired difference [95% bootstrap interval] |
|---|---:|---:|---:|
| ET DSC | 0.873877 | 0.873154 | -0.000723 [-0.001224, -0.000317] |
| ET NSD | 0.915100 | 0.912848 | -0.002251 [-0.003345, -0.001370] |

The hard-label baseline exactly matched the exported OOF segmentations. The
analysis is exploratory and not fully nested: models used for operating-point
selection had been trained on some cases in the evaluated fold. It did not
inform the final inference configuration, which uses no tuned threshold or
connected-component filtering.
