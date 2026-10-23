"""Fixed-order vs adaptive-order multi-objective VNS on simulated bi-objective obnoxious p-median."""
import json, os, sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, N, P, BUDGET = 42, 100, 10, 6000   # BUDGET = objective evaluations per run
NS = [1, 2, 3, 5]                         # swaps per shake: swap, double swap, shake+swap, bigger shake+swap
K, LS_TRIES = len(NS), 6
N_TUNE, N_TEST, SEEDS = 30, 10, [42, 43, 44]
GRID = [(a, m) for a in (0.1, 0.3) for m in (0.05, 0.15)]  # (learning rate, min probability)
IU = np.triu_indices(P, 1)


def make_instance(i, clustered):
    """Uniform points (tuning family) or shifted Gaussian-cluster points (test family)."""
    rng = np.random.default_rng(SEED + i)
    if clustered:
        c = rng.uniform(0.5, 1.5, (5, 2))
        pts = c[rng.integers(5, size=N)] + rng.normal(0, 0.08, (N, 2))
    else:
        pts = rng.random((N, 2))
    return cdist(pts, pts)


def add(arch, f, s):
    """Insert into non-dominated archive (both objectives maximised); True if archive changed."""
    for g, _ in arch:
        if g[0] >= f[0] and g[1] >= f[1]:
            return False
    arch[:] = [a for a in arch if not (f[0] >= a[0][0] and f[1] >= a[0][1])]
    arch.append((f, s))
    return True


def run_mo_vns(D, seed, adaptive, alpha=0.1, pmin=0.05):
    """Same neighborhoods/archive/budget for both; only the neighborhood selection differs."""
    rng = np.random.default_rng(seed)
    used, s1, s2 = [0], N * D.max(), D.max()

    def ev(s):
        used[0] += 1
        return (D[:, s].min(1).sum(), D[np.ix_(s, s)][IU].min())

    def swap(s, m):
        s = list(s)
        for _ in range(m):
            while True:
                o = int(rng.integers(N))
                if o not in s:
                    break
            s[int(rng.integers(P))] = o
        return s

    arch = []
    for _ in range(5):                      # shared random starts
        s = list(rng.choice(N, P, replace=False))
        add(arch, ev(s), s)
    q, k = np.ones(K), 0
    while used[0] < BUDGET:
        if adaptive:                        # probability matching on archive-improvement reward
            pr = pmin + (1 - K * pmin) * q / q.sum()
            k = int(rng.choice(K, p=pr))
        cur = swap(arch[int(rng.integers(len(arch)))][1], NS[k])   # shake in N_k
        fc, ins, w = ev(cur), 0, rng.random()
        ins += add(arch, fc, cur)
        sc = lambda f: w * f[0] / s1 + (1 - w) * f[1] / s2
        for _ in range(LS_TRIES):           # swap local search on a random weighted scalarisation
            if used[0] >= BUDGET:
                break
            cand = swap(cur, 1)
            fk = ev(cand)
            ins += add(arch, fk, cand)
            if sc(fk) > sc(fc):
                cur, fc = cand, fk
        if adaptive:
            q[k] += alpha * (float(ins > 0) - q[k])
            q = np.maximum(q, 1e-3)
        else:                               # fixed order: restart at N_0 on success, else next
            k = 0 if ins else (k + 1) % K
    return np.array([a[0] for a in arch])


def hypervolume(F):
    """2-D hypervolume (maximisation) w.r.t. reference point (0, 0)."""
    F = F[np.argsort(-F[:, 0])]
    hv, prev = 0.0, 0.0
    for a, b in F:
        if b > prev:
            hv += a * (b - prev)
            prev = b
    return hv


def norm_hv(fronts):
    """Normalise objectives per instance with the union of all compared fronts, then HV."""
    U = np.vstack(fronts)
    lo, hi = U.min(0), U.max(0)
    return [hypervolume((F - lo) / (hi - lo + 1e-12)) for F in fronts]


def main():
    # 1) tune the adaptive rule on tuning (uniform) instances only; seed 42
    print("Tuning adaptive hyper-parameters on tuning instances...", flush=True)
    score = np.zeros(len(GRID))
    for i in range(N_TUNE):
        D = make_instance(i, False)
        fr = [run_mo_vns(D, SEED + i, True, a, m) for a, m in GRID]
        score += np.array(norm_hv(fr)) / N_TUNE
    alpha, pmin = GRID[int(np.argmax(score))]
    print(f"Chosen alpha={alpha}, pmin={pmin}", flush=True)

    # 2) single final evaluation on held-out clustered test instances
    fronts = {}
    for i in range(N_TEST):
        D = make_instance(1000 + i, True)
        for sd in SEEDS:
            for name, ad in (("base", False), ("imp", True)):
                fronts[name, sd, i] = run_mo_vns(D, sd * 1000 + i, ad, alpha, pmin)
        print(f"test instance {i + 1}/{N_TEST} done", flush=True)
    hv = {(n, sd): np.zeros(N_TEST) for n in ("base", "imp") for sd in SEEDS}
    size = {k: np.zeros(N_TEST) for k in hv}
    for i in range(N_TEST):
        keys = [(n, sd) for n in ("base", "imp") for sd in SEEDS]
        for key, h in zip(keys, norm_hv([fronts[key[0], key[1], i] for key in keys])):
            hv[key][i] = h
            size[key][i] = len(fronts[key[0], key[1], i])
    agg = lambda d, n: np.array([d[n, sd] for sd in SEEDS])   # (seeds, instances)
    hb, hi_, sb, si = agg(hv, "base"), agg(hv, "imp"), agg(size, "base"), agg(size, "imp")
    if not np.isfinite(hb).all() or not np.isfinite(hi_).all() or hb.mean() <= 0:
        sys.exit("No meaningful metrics")
    pb, pi = hb.mean(0), hi_.mean(0)           # per-instance means over seeds
    win = float(np.mean((pi > pb) + 0.5 * (pi == pb)))
    p_t = float(ttest_rel(pi, pb).pvalue)
    try:
        p_w = float(wilcoxon(pi, pb).pvalue)
    except ValueError:
        p_w = float("nan")
    m = {
        "baseline_score": float(hb.mean()), "improved_score": float(hi_.mean()),
        "baseline_score_std": float(hb.mean(1).std()), "improved_score_std": float(hi_.mean(1).std()),
        "improvement": float(hi_.mean() - hb.mean()),
        "baseline_archive_size": float(sb.mean()), "improved_archive_size": float(si.mean()),
        "improved_win_rate": win, "paired_ttest_p": p_t, "paired_wilcoxon_p": p_w,
        "chosen_alpha": alpha, "chosen_pmin": pmin,
    }
    print(json.dumps(m, indent=2), flush=True)

    os.makedirs("figures", exist_ok=True)
    plt.bar(["fixed-order", "adaptive"], [m["baseline_score"], m["improved_score"]],
            yerr=[m["baseline_score_std"], m["improved_score_std"]])
    plt.ylabel("normalised hypervolume (test)")
    plt.savefig("figures/hv.png", dpi=120, bbox_inches="tight")

    res = {
        "question": "Under the same evaluation budget, can a multi-objective VNS that learns online which neighborhood to try next return a better-hypervolume trade-off set than a fixed-order multi-objective VNS on simulated bi-objective obnoxious p-median instances?",
        "method": "Seeded simulated Euclidean instances (100 sites, p=10); tuning on uniform instances, final test on shifted clustered instances. Objectives (assumed): max sum of client distances to nearest chosen facility, max min distance between chosen facilities. Normalised hypervolume of final archive, 3 seeds, paired tests over test instances.",
        "dataset": {"name": "Simulated bi-objective obnoxious p-median instances", "source_url": "generated in script with numpy (no download)",
                    "n_rows": N_TUNE + N_TEST, "n_train": N_TUNE, "n_test": N_TEST,
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "normalised hypervolume of final archive (test instances)", "higher_is_better": True,
        "baseline": "Standard multi-objective VNS with fixed neighborhood order (restart at first neighborhood on archive improvement, else next) and non-dominated archive.",
        "improved": "Same neighborhoods, archive and evaluation budget, but the next neighborhood is drawn by probability matching on archive-improvement rewards with per-instance state.",
        "metrics": m,
        "notes": "Objectives are assumed, not from the abstract. Instances (not rows) are the units; hypervolume is normalised per instance using the union of all compared fronts on that instance (reference = per-objective minimum). Only the adaptive learning rate/min probability were tuned, on tuning instances (the baseline has no hyper-parameters); the parameter score is a mean over independent tuning instances, as no model is fitted. Neighborhoods are random k-swap shakes followed by random swap local search; this is not the paper's parallel algorithm. p-values use per-instance seed-averaged HV (n=10) and are only indicative.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
