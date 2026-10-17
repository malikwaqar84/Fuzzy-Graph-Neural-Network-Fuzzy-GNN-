"""Statistical validation (Section 4.3) and model comparison (Section 5.4)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, friedmanchisquare

from .alarms import evaluate_alarms

# Nemenyi critical values q_0.05 for k = 2..10 models (Demsar, 2006, Table 5a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def random_predictor_p(n_pred: int, n_seiz: int, fpr_h: float, sop_min: float) -> float:
    """Analytic random predictor (Schelter et al., 2006): P(at least n_pred of n_seiz by chance)."""
    P = 1 - np.exp(-fpr_h * sop_min / 60.0)                      # Eq. 22
    return float(binom.sf(n_pred - 1, n_seiz, P))


def surrogate_p(alarms, files, leads, cfg, n_sur=200, seed=0):
    """Seizure-time surrogates (Andrzejak et al., 2003): permute inter-seizure intervals."""
    rng = np.random.default_rng(seed)
    obs = evaluate_alarms(alarms, files, leads, cfg)["sensitivity"]
    if len(leads) < 2:
        return float("nan")
    t_min = min(f["t_end"][0] for f in files)
    t_max = max(f["t_end"][-1] for f in files)
    lead = np.array(sorted(leads))
    count = 0
    for _ in range(n_sur):
        ints = np.diff(np.concatenate([[t_min], lead]))
        rng.shuffle(ints)
        sur = t_min + np.cumsum(ints)
        shift = rng.uniform(0, max(0.0, t_max - sur[-1]))
        sur = sur + shift
        s = evaluate_alarms(alarms, files, list(sur), cfg)["sensitivity"]
        count += s >= obs
    return (count + 1) / (n_sur + 1)


def rank_models(sens: np.ndarray, fpr: np.ndarray) -> np.ndarray:
    """Per-patient ranks (1 = best) by sensitivity, ties broken by lower FPR/h. arrays [N, k]."""
    N, k = sens.shape
    ranks = np.zeros((N, k))
    for i in range(N):
        order = sorted(range(k), key=lambda j: (-np.nan_to_num(sens[i, j], nan=-1), np.nan_to_num(fpr[i, j], nan=1e9)))
        r = np.empty(k); r[order] = np.arange(1, k + 1)
        ranks[i] = r
    return ranks


def friedman_nemenyi(ranks: np.ndarray):
    N, k = ranks.shape
    stat, p = friedmanchisquare(*[ranks[:, j] for j in range(k)]) if k >= 3 else (np.nan, np.nan)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    return {"friedman_chi2": float(stat), "friedman_p": float(p), "cd": float(cd),
            "avg_ranks": ranks.mean(0).tolist(), "N": N, "k": k, "q05": Q05[k]}
