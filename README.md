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

## Reference

This project is motivated by the following paper:

> J Sánchez-Oro, AD López-Sánchez, JM Colmenar (2022). *A multi-objective parallel variable neighborhood search for the bi-objective obnoxious p-median problem: J. Sánchez-Oro et al.*. Optimization Letters. https://link.springer.com/article/10.1007/s11590-020-01690-0

Paper author: Prof. Dr. Jesús Sánchez-Oro Cava, Universidad Rey Juan Carlos (jesus.sanchezoro@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
