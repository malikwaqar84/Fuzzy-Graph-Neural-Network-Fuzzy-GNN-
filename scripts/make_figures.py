"""Figures from the LOSO outputs (run summarize.py first).

    python scripts/make_figures.py --results results --out figures

figure2_per_patient   sensitivity and FPR/h of the proposed model per case
figure3_comparison    proposed model vs baselines (pooled sensitivity vs pooled FPR/h)
figure4_ablation      ablation variants vs the full model
figure5_shap          mean |SHAP| per channel over test cases
figure6_trace_<case>  probability, firing power and alarms over time for one case
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--trace-case", default=None, help="case for the time-course figure (default: first)")
    args = ap.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = json.load(open(root / "summary.json"))["runs"]

    # figure 2
    pfile = root / "fuzzygnn" / "patients.csv"
    if pfile.exists():
        rows = list(csv.DictReader(open(pfile)))
        cases = [r["case"] for r in rows]
        sn = [100 * float(r["sensitivity"]) for r in rows]
        fpr = [float(r["fpr_h"]) for r in rows]
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
        ax[0].bar(cases, sn, color="#3B6EA8"); ax[0].set_ylabel("Sensitivity (%)"); ax[0].set_ylim(0, 105)
        ax[1].bar(cases, fpr, color="#C0504D"); ax[1].set_ylabel("FPR/h")
        ax[1].axhline(0.15, ls="--", lw=0.8, color="k")
        ax[1].tick_params(axis="x", rotation=90)
        save(fig, out, "figure2_per_patient", args.dpi)

    # figure 3
    base = [n for n in S if "__" not in n]
    if len(base) > 1:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        for n in base:
            s = S[n]
            ax.scatter(s["fpr_h_pooled"], 100 * s["sensitivity_pooled"], s=40,
                       color="#C0504D" if n == "fuzzygnn" else "#7F7F7F", zorder=3)
            ax.annotate(n, (s["fpr_h_pooled"], 100 * s["sensitivity_pooled"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("False positive rate (per hour)"); ax.set_ylabel("Sensitivity (%)")
        save(fig, out, "figure3_comparison", args.dpi)

    # figure 4
    abl = [n for n in S if n.startswith("fuzzygnn__")]
    if abl and "fuzzygnn" in S:
        names = ["fuzzygnn"] + abl
        lab = ["full"] + [n.split("__", 1)[1] for n in abl]
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
        ax[0].barh(lab, [100 * S[n]["sensitivity_pooled"] for n in names], color="#3B6EA8")
        ax[0].set_xlabel("Sensitivity (%)")
        ax[1].barh(lab, [S[n]["fpr_h_pooled"] for n in names], color="#C0504D")
        ax[1].set_xlabel("FPR/h"); ax[1].set_yticklabels([])
        if "no_firing_power" in S["fuzzygnn"]:
            nf = S["fuzzygnn"]["no_firing_power"]
            ax[0].barh(["no_firing_power"], [100 * nf["sensitivity_pooled"]], color="#9BB7D4")
            ax[1].barh(["no_firing_power"], [nf["fpr_h_pooled"]], color="#E6A3A1")
        save(fig, out, "figure4_ablation", args.dpi)

    # figure 5
    if "fuzzygnn" in S and "shap_mean_abs" in S["fuzzygnn"]:
        ch, v = S["fuzzygnn"]["shap_channels"], np.array(S["fuzzygnn"]["shap_mean_abs"])
        o = np.argsort(v)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.barh(np.array(ch)[o], v[o], color="#3B6EA8"); ax.set_xlabel("Mean |SHAP value|")
        save(fig, out, "figure5_shap", args.dpi)

    # figure 6
    wdir = root / "fuzzygnn" / "windows"
    files = sorted(wdir.glob("*.csv.gz")) if wdir.exists() else []
    if files:
        f = next((x for x in files if x.name.startswith(args.trace_case or "")), files[0])
        case = f.name.split(".")[0]
        t, y, p = [], [], []
        with gzip.open(f, "rt") as fh:
            for r in csv.DictReader(fh):
                t.append(float(r["t_end_s"]) / 3600); y.append(int(r["label"])); p.append(float(r["prob"]))
        t, y, p = map(np.array, (t, y, p))
        pat = next(r for r in csv.DictReader(open(root / "fuzzygnn" / "patients.csv")) if r["case"] == case)
        fold = next(json.load(open(j)) for j in (root / "fuzzygnn" / "folds").glob("*.json")
                    if case in json.load(open(j))["test_cases"])
        rec = next(r for r in fold["patients"] if r["case"] == case)
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(t, p, lw=0.4, color="#7F7F7F", label="window probability")
        for k, lab, col in ((1, "preictal", "#F2C14E"),):
            m = y == k
            ax.fill_between(t, 0, 1, where=m, color=col, alpha=0.35, step="mid", label=lab)
        for a in rec["alarm_times_s"]:
            ax.axvline(a / 3600, color="#C0504D", lw=0.8)
        ax.axhline(float(pat["rho"]), ls=":", color="k", lw=0.8)
        ax.set_xlabel("Time from start of recording (h)"); ax.set_ylabel("Probability")
        ax.set_title(case, fontsize=9); ax.legend(fontsize=7, loc="upper right", frameon=False)
        save(fig, out, f"figure6_trace_{case}", args.dpi)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
