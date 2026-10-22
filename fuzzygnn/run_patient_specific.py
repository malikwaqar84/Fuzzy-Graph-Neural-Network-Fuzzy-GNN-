"""Patient-specific evaluation: leave-one-seizure-out within each case.

    python scripts/run_patient_specific.py --model fuzzygnn
    python scripts/run_patient_specific.py --model fuzzygnn --cases chb01 chb05

For a case with N usable lead seizures, the interictal windows are split in time order into
N contiguous chunks. Fold k tests on lead seizure k (its preictal block) and interictal chunk k.
Seizure k-1 (cyclically) with chunk k-1 is the validation set for early stopping and for the
firing-power threshold rho. The model is trained on the remaining seizures and chunks of the
same case only. Every lead seizure and every interictal window is tested exactly once.

Output, in <out>/<model>_ps/ (same layout as run_loso.py, so summarize.py reads it):
    folds/<case>.json        per-seizure folds and the pooled per-case record
    windows/<case>.csv.gz    every test window with its fold, label and probability
    patients.csv             one row per case
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzygnn.alarms import alarm_times, evaluate_alarms                      # noqa: E402
from fuzzygnn.config import ABLATIONS, load_config, parse_override, with_overrides  # noqa: E402
from fuzzygnn.dataset import EXCL, INTER, PRE, Normalizer, PatientData, load_cases, select_patients  # noqa: E402
from fuzzygnn.evaluate import window_counts                                 # noqa: E402
from fuzzygnn.stats import random_predictor_p                               # noqa: E402
from fuzzygnn.train import build_model, pick_device, predict_patient, set_seed, static_coupling, train_model  # noqa: E402

# Defaults for this protocol (overridable with --set). Overlapping training sequences
# (stride 4 windows = 10 s) because one patient gives far fewer sequences than the LOSO pool.
PS_DEFAULTS = {"train.seq_stride": 4}

CSV_COLS = ["case", "n_seizures", "n_predicted", "sensitivity", "n_alarms", "n_false", "interictal_h", "fpr_h",
            "tiw", "mwt_min", "p_random", "TP", "FP", "TN", "FN", "n_pre_windows", "n_inter_windows", "acc",
            "sn_window", "sp_window", "ppv", "npv", "f1", "auc", "rho_mean", "nofp_n_predicted",
            "nofp_sensitivity", "nofp_n_false", "nofp_fpr_h"]


class View:
    """A patient restricted to some windows (others relabelled EXCL or cut out)."""

    def __init__(self, pd, files, leads):
        self.id, self.case, self.files, self._leads = pd.id, pd.case, files, leads

    def usable_leads(self, min_cover=0.5):
        return self._leads


def relabel(f, keep_mask):
    """Same file, windows outside keep_mask become EXCL (used for training data)."""
    g = dict(f)
    y = f["y"].copy()
    y[~keep_mask] = EXCL
    g["y"] = y
    return g


def segments(f, mask, y_override=None):
    """Cut the windows in mask into contiguous pseudo-files (used for test and validation)."""
    out = []
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1]); ends = np.concatenate([breaks + 1, [len(idx)]])
    y_all = f["y"] if y_override is None else y_override
    for a, b in zip(starts, ends):
        sl = slice(idx[a], idx[b - 1] + 1)
        g = {k: f[k] for k in ("name", "L", "S", "z_path", "z")}
        for k in ("t_end", "art", "feat_s", "feat_l", "corr", "act"):
            g[k] = f[k][sl]
        g["y"] = y_all[sl]
        g["offset"] = int(sl.start)
        out.append(g)
    return out


def fold_masks(pd, cfg):
    """Per lead seizure: preictal-block mask per file; interictal chunk masks per file."""
    p = cfg["protocol"]
    sph, sop = p["sph_min"] * 60, p["sop_min"] * 60
    lead_in = 10 * 60                                   # 10 min before the SOP so the firing-power buffer fills
    leads = pd.usable_leads()
    N = len(leads)
    # interictal windows are never part of a block (they belong to exactly one interictal chunk)
    blocks = [[(f["t_end"] >= on - sph - sop - lead_in) & (f["t_end"] <= on - sph) & (f["y"] != INTER) for f in pd.files]
              for on in leads]
    # interictal windows in time order, split into N contiguous chunks of equal size
    t = np.concatenate([f["t_end"][f["y"] == INTER] for f in pd.files])
    edges = np.quantile(t, np.linspace(0, 1, N + 1)) if len(t) else np.zeros(N + 1)
    chunks = []
    for k in range(N):
        lo, hi = edges[k], edges[k + 1]
        chunks.append([(f["y"] == INTER) & (f["t_end"] >= lo) & ((f["t_end"] < hi) if k < N - 1 else (f["t_end"] <= hi))
                       for f in pd.files])
    return leads, blocks, chunks


def eval_view(pd, file_masks, lead, cfg, own_pre_mask):
    """Pseudo-files for one held-out seizure block + interictal chunk. Preictal windows that
    belong to another seizure are relabelled EXCL so they count neither as hits nor as false alarms."""
    files = []
    for f, m, own in zip(pd.files, file_masks, own_pre_mask):
        y = f["y"].copy()
        y[(y == PRE) & ~own] = EXCL
        files += segments(f, m, y)
    files.sort(key=lambda g: g["t_end"][0])
    return View(pd, files, [lead])


def pick_rho(probs, view, cfg):
    best = None
    for rho in cfg["alarm"]["rho_grid"]:
        r = evaluate_alarms(alarm_times(probs, view.files, cfg, rho), view.files, view.usable_leads(), cfg)
        if r["fpr_h"] <= cfg["alarm"]["max_val_fpr"]:
            return float(rho)
        if best is None or r["fpr_h"] < best[1]:
            best = (float(rho), r["fpr_h"])
    return best[0]


def run_case(pd, args, cfg, device, run_dir, C):
    p = cfg["protocol"]
    sph, sop = p["sph_min"] * 60, p["sop_min"] * 60
    leads, blocks, chunks = fold_masks(pd, cfg)
    N = len(leads)
    pre_of = [[(f["t_end"] >= on - sph - sop) & (f["t_end"] < on - sph) for f in pd.files] for on in leads]
    folds, rows = [], []
    for k in range(N):
        v = (k - 1) % N
        set_seed(cfg["train"]["seed"] + k)
        test_m = [b | c for b, c in zip(blocks[k], chunks[k])]
        val_m = [b | c for b, c in zip(blocks[v], chunks[v])]
        held = [tm | vm for tm, vm in zip(test_m, val_m)]
        train_view = View(pd, [relabel(f, ~h) for f, h in zip(pd.files, held)], [l for i, l in enumerate(leads) if i not in (k, v)])
        val_view = eval_view(pd, val_m, leads[v], cfg, pre_of[v])
        test_view = eval_view(pd, test_m, leads[k], cfg, pre_of[k])

        norm = Normalizer().fit([train_view])
        model = build_model(args.model, cfg, C)
        if args.model == "fuzzygnn" and cfg["model"]["membership"] == "static":
            model.static_c.copy_(torch.from_numpy(static_coupling([train_view])))
        info = train_model(model, [train_view], [val_view], norm, cfg, device, log=lambda s: None)
        rho = pick_rho(predict_patient(model, val_view, norm, cfg, device), val_view, cfg)

        probs = predict_patient(model, test_view, norm, cfg, device)
        al = alarm_times(probs, test_view.files, cfg, rho)
        ev = evaluate_alarms(al, test_view.files, [leads[k]], cfg)
        al0 = alarm_times(probs, test_view.files, cfg, rho, use_fp=False)
        ev0 = evaluate_alarms(al0, test_view.files, [leads[k]], cfg)
        fold = {"seizure": k + 1, "onset_s": float(leads[k]), "val_seizure": v + 1, "rho": rho,
                "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"],
                "membership": model.membership_params() if hasattr(model, "membership_params") else None,
                "predicted": int(ev["n_predicted"]), "n_false": int(ev["n_false"]),
                "interictal_h": ev["interictal_h"], "warning_min": ev["mwt_min"],
                "alarm_times_s": [round(float(a), 1) for a in al],
                "nofp_predicted": int(ev0["n_predicted"]), "nofp_n_false": int(ev0["n_false"])}
        fold.update({k2: v2 for k2, v2 in window_counts(probs, test_view.files).items() if k2 in ("TP", "FP", "TN", "FN", "auc")})
        folds.append(fold)
        print(f"    seizure {k + 1}/{N}: {'PREDICTED' if fold['predicted'] else 'missed'}"
              f"  false alarms {fold['n_false']} in {fold['interictal_h']:.1f} h  rho {rho}  AUC {fold['auc']:.3f}"
              f"  ({info['epochs_run']} epochs)", flush=True)
        for g, pr in zip(test_view.files, probs):
            rows.append((k + 1, g, pr))
    # pooled per-case record: each test window once, each seizure once
    all_files = [g for _, g, _ in rows]
    all_probs = [pr for _, _, pr in rows]
    wc = window_counts(all_probs, all_files)
    ih = sum(f["interictal_h"] for f in folds)
    npred, nfalse = sum(f["predicted"] for f in folds), sum(f["n_false"] for f in folds)
    lab_t = sum(int((g["y"] >= 0).sum()) for g in all_files)
    rec = {"case": pd.id, "subject": pd.case.subject, "n_seizures": N, "n_predicted": npred,
           "sensitivity": npred / N, "n_alarms": sum(len(f["alarm_times_s"]) for f in folds), "n_false": nfalse,
           "interictal_h": ih, "fpr_h": nfalse / ih if ih else float("nan"),
           "mwt_min": float(np.nanmean([f["warning_min"] for f in folds])) if npred else float("nan"),
           "p_random": random_predictor_p(npred, N, nfalse / ih if ih else 0.0, p["sop_min"]),
           "p_surrogate": float("nan"), "rho_mean": float(np.mean([f["rho"] for f in folds])),
           "nofp_n_predicted": sum(f["nofp_predicted"] for f in folds),
           "nofp_sensitivity": sum(f["nofp_predicted"] for f in folds) / N,
           "nofp_n_false": sum(f["nofp_n_false"] for f in folds),
           "nofp_fpr_h": sum(f["nofp_n_false"] for f in folds) / ih if ih else float("nan")}
    rec["tiw"] = _tiw(rows, folds, cfg, lab_t)
    rec.update(wc)
    with gzip.open(run_dir / "windows" / f"{pd.id}.csv.gz", "wt") as fh:
        fh.write("case,fold,file,window,t_end_s,label,artefact,prob\n")
        for k, g, pr in rows:
            for i, (t, y, a, q) in enumerate(zip(g["t_end"], g["y"], g["art"], pr)):
                fh.write(f"{pd.id},{k},{g['name']},{g['offset'] + i},{t:.1f},{int(y)},{int(a)},{q:.6f}\n")
    mem = [f["membership"] for f in folds if f["membership"]]
    membership = {"theta": float(np.mean([m["theta"] for m in mem])),
                  "kappa_or_sigma": float(np.mean([m["kappa_or_sigma"] for m in mem]))} if mem else None
    return {"subject": pd.id, "test_cases": [pd.id], "protocol": "patient-specific leave-one-seizure-out",
            "rho": rec["rho_mean"], "membership": membership, "seizure_folds": folds, "patients": [rec]}


def _tiw(rows, folds, cfg, lab_t):
    p = cfg["protocol"]
    span = (p["sph_min"] + p["sop_min"]) * 60
    n = 0
    for k, g, _ in rows:
        al = np.array(folds[k - 1]["alarm_times_s"])
        t = g["t_end"][g["y"] >= 0]
        if len(al) and len(t):
            n += int(((t[:, None] >= al[None]) & (t[:, None] < al[None] + span)).any(1).sum())
    return n / lab_t if lab_t else float("nan")


def write_csv(run_dir):
    rows = []
    for fj in sorted((run_dir / "folds").glob("*.json")):
        rows += json.load(open(fj))["patients"]
    with open(run_dir / "patients.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["case"]):
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="fuzzygnn")
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="results")
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, {**PS_DEFAULTS, **dict(parse_override(s) for s in args.set)})
    if args.ablation != "full":
        cfg = with_overrides(cfg, ABLATIONS[args.ablation])
    run = f"{args.model}_ps" + ("" if args.ablation == "full" else f"__{args.ablation}")
    run_dir = Path(args.out) / run
    for sub in ("folds", "windows"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as fh:
        yaml.safe_dump({"protocol_type": "patient-specific leave-one-seizure-out", "model": args.model,
                        "ablation": args.ablation, "overrides": {**PS_DEFAULTS, **dict(parse_override(s) for s in args.set)}, **cfg}, fh, sort_keys=False)
    device = pick_device(cfg)
    cache = cfg["data"]["cache_dir"]
    t0 = time.time()
    patients = [PatientData(c, cache, cfg) for c in load_cases(cache)]
    inc, exc = select_patients(patients, cfg)
    print(f"run {run} on {device}: {len(inc)} cases loaded in {time.time() - t0:.0f}s")
    with open(run_dir / "selection.json", "w") as fh:
        json.dump({"included": [p.summary() for p in inc], "excluded": exc}, fh, indent=1)
    C = len(cfg["data"]["channels"])
    for i, pd in enumerate(inc):
        if args.cases and pd.id not in args.cases:
            continue
        run_case_file = run_dir / "folds" / f"{pd.id}.json"
        if run_case_file.exists():
            print(f"[{i + 1}/{len(inc)}] {pd.id}: done, skipping")
            continue
        print(f"[{i + 1}/{len(inc)}] {pd.id}: {len(pd.usable_leads())} lead seizures, {pd.interictal_hours():.1f} h interictal", flush=True)
        tc = time.time()
        res = run_case(pd, args, cfg, device, run_dir, C)
        res["seconds"] = round(time.time() - tc, 1)
        res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = run_case_file.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        tmp.replace(run_case_file)
        r = res["patients"][0]
        print(f"  {pd.id}: Sn {r['n_predicted']}/{r['n_seizures']}  false alarms {r['n_false']} in {r['interictal_h']:.1f} h "
              f"(FPR/h {r['fpr_h']:.3f})  AUC {r['auc']:.3f}  p {r['p_random']:.3g}", flush=True)
        write_csv(run_dir)
    rows = write_csv(run_dir)
    if rows:
        ns, npred = sum(r["n_seizures"] for r in rows), sum(r["n_predicted"] for r in rows)
        nf, ih = sum(r["n_false"] for r in rows), sum(r["interictal_h"] for r in rows)
        print(f"\n{run}: {len(rows)} cases, {npred}/{ns} seizures predicted, {nf} false alarms in {ih:.1f} h "
              f"(pooled FPR/h {nf / ih if ih else float('nan'):.3f})")


if __name__ == "__main__":
    main()
