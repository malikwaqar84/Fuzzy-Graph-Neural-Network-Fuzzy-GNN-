"""Collect all LOSO runs into results/summary.json, results/summary.md and CSV tables.

    python scripts/summarize.py --results results

Every number is computed from the per-fold JSON files written by run_loso.py.
Runs with missing folds are flagged as incomplete in the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzygnn.stats import friedman_nemenyi, rank_models   # noqa: E402

BASELINE_ORDER = ["truong_cnn", "tsiouris_lstm", "eegnet", "eeg_conformer", "attention_wavelet_cnn",
                  "wtrpnet", "chen_gcn_gru", "tang_dcrnn", "dts_gan"]


def nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def mean_sd(v):
    v = np.array([x for x in v if not nan(x)], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")) if len(v) else (float("nan"),) * 2


def load_run(run_dir: Path):
    folds = [json.load(open(f)) for f in sorted((run_dir / "folds").glob("*.json"))]
    sel = json.load(open(run_dir / "selection.json")) if (run_dir / "selection.json").exists() else {}
    n_subj = len({p["case"] for p in sel.get("included", [])})
    pats = sorted([p for f in folds for p in f["patients"]], key=lambda r: r["case"])
    return {"folds": folds, "patients": pats, "complete": n_subj > 0 and len(pats) == n_subj,
            "n_expected": n_subj}


def summarise(run):
    P = run["patients"]
    if not P:
        return None
    alpha = 0.05
    s = {"n_cases": len(P), "complete": run["complete"]}
    ns, npred = sum(p["n_seizures"] for p in P), sum(p["n_predicted"] for p in P)
    nf, ih = sum(p["n_false"] for p in P), sum(p["interictal_h"] for p in P)
    s["seizures"], s["predicted"], s["false_alarms"], s["interictal_h"] = ns, npred, nf, round(ih, 2)
    s["sensitivity_pooled"] = npred / ns if ns else float("nan")
    s["fpr_h_pooled"] = nf / ih if ih else float("nan")
    for key in ("sensitivity", "fpr_h", "tiw", "mwt_min", "auc", "acc", "sn_window", "sp_window", "ppv", "npv", "f1"):
        s[key + "_mean"], s[key + "_sd"] = mean_sd([p.get(key) for p in P])
    TP, FP, TN, FN = (sum(p[k] for p in P) for k in ("TP", "FP", "TN", "FN"))
    s["window_counts_pooled"] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
                                 "ppv": TP / (TP + FP) if TP + FP else float("nan"),
                                 "npv": TN / (TN + FN) if TN + FN else float("nan"),
                                 "sn": TP / (TP + FN) if TP + FN else float("nan"),
                                 "sp": TN / (TN + FP) if TN + FP else float("nan")}
    s["n_significant_random"] = int(sum(1 for p in P if not nan(p.get("p_random")) and p["p_random"] < alpha))
    s["n_significant_surrogate"] = int(sum(1 for p in P if not nan(p.get("p_surrogate")) and p["p_surrogate"] < alpha))
    if "nofp_n_predicted" in P[0]:
        nf2 = sum(p["nofp_n_false"] for p in P)
        s["no_firing_power"] = {"sensitivity_pooled": sum(p["nofp_n_predicted"] for p in P) / ns if ns else float("nan"),
                                "fpr_h_pooled": nf2 / ih if ih else float("nan"), "false_alarms": nf2}
    rhos = [f["rho"] for f in run["folds"]]
    s["rho_per_fold"] = rhos
    mem = [f["membership"] for f in run["folds"] if f.get("membership")]
    if mem:
        s["theta_mean"], s["theta_sd"] = mean_sd([m["theta"] for m in mem])
        s["kappa_mean"], s["kappa_sd"] = mean_sd([m["kappa_or_sigma"] for m in mem])
    shap = [p["shap_mean_abs"] for p in P if p.get("shap_mean_abs")]
    if shap:
        s["shap_channels"] = next(p["shap_channels"] for p in P if p.get("shap_channels"))
        s["shap_mean_abs"] = np.mean(shap, 0).tolist()
    return s


def fmt(m, sd, d=2, pct=False):
    if nan(m):
        return "n/a"
    k = 100 if pct else 1
    return f"{m * k:.{d}f} ± {sd * k:.{d}f}" if not nan(sd) else f"{m * k:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    runs = {d.name: load_run(d) for d in sorted(root.iterdir()) if (d / "folds").is_dir()}
    if not runs:
        sys.exit(f"no runs under {root}")
    summary = {name: summarise(r) for name, r in runs.items()}
    summary = {k: v for k, v in summary.items() if v}

    # model comparison over the cases common to all complete runs (proposed + baselines)
    cmp_names = [n for n in ["fuzzygnn"] + BASELINE_ORDER if n in summary and summary[n]["complete"]]
    stats = None
    if len(cmp_names) >= 3:
        common = sorted(set.intersection(*[{p["case"] for p in runs[n]["patients"]} for n in cmp_names]))
        get = {n: {p["case"]: p for p in runs[n]["patients"]} for n in cmp_names}
        sens = np.array([[get[n][c]["sensitivity"] for n in cmp_names] for c in common], float)
        fpr = np.array([[get[n][c]["fpr_h"] for n in cmp_names] for c in common], float)
        stats = friedman_nemenyi(rank_models(sens, fpr))
        stats["models"] = cmp_names
    out = {"runs": summary, "model_comparison": stats}
    with open(root / "summary.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # per-case table of the proposed model (Table 3 and the counts table)
    if "fuzzygnn" in runs:
        P = runs["fuzzygnn"]["patients"]
        cols = ["case", "n_seizures", "n_predicted", "sensitivity", "n_false", "interictal_h", "fpr_h",
                "tiw", "mwt_min", "auc", "p_random", "TP", "FP", "TN", "FN", "n_pre_windows",
                "n_inter_windows", "acc", "sn_window", "sp_window", "ppv", "npv", "f1"]
        with open(root / "table_patients_fuzzygnn.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols)
            for p in P:
                w.writerow([p.get(c, "") for c in cols])

    lines = ["# Results summary", "", "Generated by scripts/summarize.py from the fold files.", ""]
    lines += ["| Run | Cases | Seizures predicted | Sn (pooled) | False alarms / interictal h | FPR/h (pooled) | "
              "Sn per case (%) | FPR/h per case | AUC | Significant (p<0.05) | Complete |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        lines.append(f"| {n} | {s['n_cases']} | {s['predicted']}/{s['seizures']} | {100 * s['sensitivity_pooled']:.1f}% | "
                     f"{s['false_alarms']} / {s['interictal_h']:.1f} | {s['fpr_h_pooled']:.3f} | "
                     f"{fmt(s['sensitivity_mean'], s['sensitivity_sd'], 1, True)} | {fmt(s['fpr_h_mean'], s['fpr_h_sd'], 3)} | "
                     f"{fmt(s['auc_mean'], s['auc_sd'], 3)} | {s['n_significant_random']}/{s['n_cases']} | "
                     f"{'yes' if s['complete'] else 'NO'} |")
    if stats:
        lines += ["", f"Friedman chi2 = {stats['friedman_chi2']:.2f}, p = {stats['friedman_p']:.4g}; "
                      f"Nemenyi CD = {stats['cd']:.2f} (N = {stats['N']}, k = {stats['k']})", "",
                  "| Model | Average rank |", "|---|---|"]
        lines += [f"| {m} | {r:.2f} |" for m, r in zip(stats["models"], stats["avg_ranks"])]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
