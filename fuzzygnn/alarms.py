"""Firing-power alarms (Eq. 20) and event-based evaluation under the SOP/SPH protocol."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, f1_score

from .dataset import INTER


def firing_power(prob: np.ndarray, n_fp: int) -> np.ndarray:
    """Eq. 20: fraction of the last n_fp windows with probability >= 0.5 (causal)."""
    hit = (prob >= 0.5).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(hit)])
    idx = np.arange(len(hit))
    start = np.maximum(0, idx - n_fp + 1)
    return (cs[idx + 1] - cs[start]) / n_fp


def alarm_times(files_prob, files, cfg, rho, use_fp=True):
    """Alarm times (s) over all files of a patient, with a refractory period of SPH + SOP."""
    p = cfg["protocol"]
    refractory = (p["sph_min"] + p["sop_min"]) * 60
    n_fp = cfg["alarm"]["n_fp"] if use_fp else 1
    thr = rho if use_fp else 1.0
    alarms, last = [], -np.inf
    for prob, f in zip(files_prob, files):
        fp = firing_power(prob, n_fp)
        for t, v in zip(f["t_end"], fp):
            if v >= thr - 1e-9 and t - last >= refractory:
                alarms.append(float(t)); last = t
    return np.array(alarms)


def evaluate_alarms(alarms, files, leads, cfg):
    """Sensitivity, FPR/h, time in warning and mean warning time for one patient."""
    p = cfg["protocol"]
    sph, sop = p["sph_min"] * 60, p["sop_min"] * 60
    stride = p["stride_s"]
    predicted, warn = [], []
    for on in leads:
        ok = alarms[(alarms >= on - sph - sop) & (alarms <= on - sph)]
        predicted.append(len(ok) > 0)
        if len(ok):
            warn.append((on - ok.min()) / 60)
    # an alarm is false if it is raised during an interictal window
    inter_t = np.concatenate([f["t_end"][f["y"] == INTER] for f in files]) if files else np.array([])
    inter_h = len(inter_t) * stride / 3600
    t_all = np.concatenate([f["t_end"] for f in files])
    y_all = np.concatenate([f["y"] for f in files])
    n_false = 0
    for a in alarms:
        j = np.searchsorted(t_all, a)
        j = min(j, len(t_all) - 1)
        if y_all[j] == INTER:
            n_false += 1
    # time in warning over labelled time
    lab_t = t_all[y_all >= 0]
    in_warn = np.zeros(len(lab_t), bool)
    for a in alarms:
        in_warn |= (lab_t >= a) & (lab_t < a + sph + sop)
    return {"n_seizures": len(leads), "n_predicted": int(sum(predicted)),
            "sensitivity": float(np.mean(predicted)) if leads else float("nan"),
            "fpr_h": n_false / inter_h if inter_h > 0 else float("nan"),
            "n_false": n_false, "interictal_h": inter_h,
            "tiw": float(in_warn.mean()) if len(lab_t) else float("nan"),
            "mwt_min": float(np.mean(warn)) if warn else float("nan")}


def window_metrics(files_prob, files):
    y, p = [], []
    for prob, f in zip(files_prob, files):
        m = (f["y"] >= 0) & ~f["art"]
        y.append(f["y"][m]); p.append(prob[m])
    y, p = np.concatenate(y), np.concatenate(p)
    out = {"auc": float("nan"), "ppv": float("nan"), "f1": float("nan")}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["ppv"] = float(precision_score(y, p >= 0.5, zero_division=0))
        out["f1"] = float(f1_score(y, p >= 0.5, zero_division=0))
    return out


def choose_rho(val_sets, cfg):
    """Smallest firing-power threshold whose mean validation FPR/h is <= max_val_fpr."""
    best = None
    for rho in cfg["alarm"]["rho_grid"]:
        fprs, sens = [], []
        for probs, pd in val_sets:
            al = alarm_times(probs, pd.files, cfg, rho)
            r = evaluate_alarms(al, pd.files, pd.usable_leads(), cfg)
            fprs.append(r["fpr_h"]); sens.append(r["sensitivity"])
        fpr = np.nanmean(fprs)
        if fpr <= cfg["alarm"]["max_val_fpr"]:
            return rho
        if best is None or fpr < best[1]:
            best = (rho, fpr)
    return best[0]
