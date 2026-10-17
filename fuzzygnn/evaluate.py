"""Per-patient metrics exported by run_loso.py (window counts, event metrics, significance)."""
from __future__ import annotations

import gzip

import numpy as np
from sklearn.metrics import roc_auc_score

from .alarms import alarm_times, evaluate_alarms
from .dataset import PRE
from .stats import random_predictor_p, surrogate_p


def _div(a, b):
    return float(a) / b if b else float("nan")


def window_counts(probs, files, thr: float = 0.5):
    """Confusion counts over valid (labelled, artefact-free) windows, and AUC from the scores."""
    y, p = [], []
    for prob, f in zip(probs, files):
        m = (f["y"] >= 0) & ~f["art"]
        y.append(f["y"][m] == PRE); p.append(prob[m])
    y = np.concatenate(y) if y else np.zeros(0, bool)
    p = np.concatenate(p) if p else np.zeros(0)
    pred = p >= thr
    TP = int((pred & y).sum()); FP = int((pred & ~y).sum())
    TN = int((~pred & ~y).sum()); FN = int((~pred & y).sum())
    auc = float(roc_auc_score(y, p)) if (y.any() and (~y).any()) else float("nan")
    sn, sp = _div(TP, TP + FN), _div(TN, TN + FP)
    ppv, npv = _div(TP, TP + FP), _div(TN, TN + FN)
    f1 = _div(2 * TP, 2 * TP + FP + FN)
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "n_pre_windows": TP + FN, "n_inter_windows": TN + FP,
            "acc": _div(TP + TN, TP + TN + FP + FN), "sn_window": sn, "sp_window": sp,
            "ppv": ppv, "npv": npv, "f1": f1, "auc": auc}


def event_metrics(probs, pd, cfg, rho, use_fp, n_sur, seed=0, prefix=""):
    leads = pd.usable_leads()
    al = alarm_times(probs, pd.files, cfg, rho, use_fp=use_fp)
    ev = evaluate_alarms(al, pd.files, leads, cfg)
    ev["p_random"] = random_predictor_p(ev["n_predicted"], ev["n_seizures"], ev["fpr_h"],
                                        cfg["protocol"]["sop_min"]) if ev["n_seizures"] else float("nan")
    ev["p_surrogate"] = surrogate_p(al, pd.files, leads, cfg, n_sur, seed) if n_sur > 0 else float("nan")
    ev["n_alarms"] = int(len(al))
    ev["alarm_times_s"] = [round(float(a), 1) for a in al]
    return {prefix + k: v for k, v in ev.items()}


def patient_record(probs, pd, cfg, rho, with_nofp=False, seed=0):
    n_sur = cfg["stats"]["n_surrogates"]
    rec = {"case": pd.id, "subject": pd.case.subject, "files": len(pd.files),
           "recorded_h": round(sum(f["hours"] for f in pd.files), 3),
           "lead_seizures_all": len(pd.leads), "rho": rho}
    rec.update(window_counts(probs, pd.files))
    rec.update(event_metrics(probs, pd, cfg, rho, cfg["alarm"].get("use_firing_power", True), n_sur, seed))
    if with_nofp:
        rec.update(event_metrics(probs, pd, cfg, rho, False, 0, seed, prefix="nofp_"))
    return rec


def write_window_csv(path, pd, probs):
    """Per-window predictions: case, file, window index, end time (s), label, artefact flag, probability."""
    with gzip.open(path, "wt") as fh:
        fh.write("case,file,window,t_end_s,label,artefact,prob\n")
        for f, p in zip(pd.files, probs):
            for i, (t, y, a, q) in enumerate(zip(f["t_end"], f["y"], f["art"], p)):
                fh.write(f"{pd.id},{f['name']},{i},{t:.1f},{int(y)},{int(a)},{q:.6f}\n")
