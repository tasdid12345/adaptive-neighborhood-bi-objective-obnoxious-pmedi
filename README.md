# Adaptive vs fixed-order multi-objective VNS for the bi-objective obnoxious p-median

## Motivation
The paper "A multi-objective parallel variable neighborhood search for the bi-objective obnoxious p-median problem" (Sánchez-Oro et al., 2022) explores several neighborhoods for a bi-objective facility-dispersion problem. Here we ask whether learning online which neighborhood to try next helps a sequential multi-objective VNS under a fixed evaluation budget.

## Question
Under the same evaluation budget, can a multi-objective VNS that learns online which neighborhood to try next return a better-hypervolume trade-off set than a fixed-order multi-objective VNS on simulated bi-objective obnoxious p-median instances?

## Dataset
Seeded simulated Euclidean instances generated with numpy (100 sites, p=10). Tuning instances use uniform points; held-out test instances use a shifted clustered family. This is not the paper's dataset. The two objectives are assumed (the abstract does not state them): maximise the sum of each client's distance to its nearest chosen facility, and maximise the minimum distance between chosen facilities.

## Baseline vs improved
- Baseline: standard multi-objective VNS with a fixed neighborhood order (swap, double swap, shake + swap, larger shake + swap) and a non-dominated archive.
- Improved: same neighborhoods, archive, local search and evaluation budget, but the next neighborhood is chosen by probability matching on archive-improvement rewards (per-instance state). Its learning rate and minimum probability are tuned on the tuning instances only.

Metrics: normalised hypervolume on the test instances (main), archive size and per-instance win rate (secondary), with paired tests over seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to results.json (charts in figures/).

## Results (latest run)

On simulated bi-objective obnoxious p-median test instances (10 shifted clustered instances, 3 seeds), normalised archive hypervolume was 0.392 for a standard fixed-order multi-objective VNS versus 0.404 for the adaptive neighborhood-selection variant, a small, non-significant gain (paired t-test p=0.25).

| Metric | Value |
|---|---|
| baseline_score | 0.3918 |
| improved_score | 0.4042 |
| baseline_score_std | 0.0147 |
| improved_score_std | 0.01484 |
| improvement | 0.01235 |
| baseline_archive_size | 20.27 |
| improved_archive_size | 21.27 |
| improved_win_rate | 0.8 |
| paired_ttest_p | 0.2503 |
| paired_wilcoxon_p | 0.4316 |
| chosen_alpha | 0.1 |
| chosen_pmin | 0.05 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Objectives are assumed, not from the abstract. Instances (not rows) are the units; hypervolume is normalised per instance using the union of all compared fronts on that instance (reference = per-objective minimum). Only the adaptive learning rate/min probability were tuned, on tuning instances (the baseline has no hyper-parameters); the parameter score is a mean over independent tuning instances, as no model is fitted. Neighborhoods are random k-swap shakes followed by random swap local search; this is not the paper's parallel algorithm. p-values use per-instance seed-averaged HV (n=10) and are only indicative.

## Reference

This project is motivated by the following paper:

> J Sánchez-Oro, AD López-Sánchez, JM Colmenar (2022). *A multi-objective parallel variable neighborhood search for the bi-objective obnoxious p-median problem: J. Sánchez-Oro et al.*. Optimization Letters. https://link.springer.com/article/10.1007/s11590-020-01690-0

Paper author: Prof. Dr. Jesús Sánchez-Oro Cava, Universidad Rey Juan Carlos (jesus.sanchezoro@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
