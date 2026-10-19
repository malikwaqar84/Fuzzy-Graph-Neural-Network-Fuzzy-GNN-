"""Labelling with the SOP/SPH protocol, patient selection, normalisation and batching."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .chbmit import Case, lead_seizures
from .features import long_scale

PRE, INTER, EXCL = 1, 0, -1


def label_windows(t_end: np.ndarray, t_start: np.ndarray, seizures, leads, cfg) -> np.ndarray:
    """Label windows: 1 preictal, 0 interictal, -1 excluded (Section 4.2)."""
    p = cfg["protocol"]
    sph, sop = p["sph_min"] * 60, p["sop_min"] * 60
    gap = p["interictal_gap_h"] * 3600
    y = np.full(t_end.shape, EXCL, np.int8)
    inter = np.ones(t_end.shape, bool)
    for on, off in seizures:
        inter &= (t_end <= on - gap) | (t_start >= off + gap)
    y[inter] = INTER
    for on in leads:
        y[(t_end >= on - sph - sop) & (t_end < on - sph)] = PRE
    return y


class PatientData:
    """All windows of one CHB-MIT case, loaded from the cache written by prepare_data.py."""

    def __init__(self, case: Case, cache_dir: str, cfg: dict, load_raw: bool = False):
        self.case, self.cfg = case, cfg
        self.id = case.case_id
        p = cfg["protocol"]
        self.seizures = case.seizures_abs()
        self.leads = lead_seizures(case, p["lead_gap_min"] * 60)
        self.files = []
        if not case.has_times:          # cannot be placed on a timeline; excluded by select_patients
            return
        root = Path(cache_dir) / case.case_id
        fs = cfg["data"]["fs"]
        for f in case.files:
            npz = root / (Path(f.name).stem + ".npz")
            if not npz.exists():
                continue
            a = np.load(npz)
            L, S = int(a["L"]), int(a["S"])
            nW = a["feat_s"].shape[0]
            t0 = f.start_s + np.arange(nW) * S / fs
            t_end = t0 + L / fs
            y = label_windows(t_end, t0, self.seizures, self.leads, cfg)
            d = {"name": f.name, "t_end": t_end, "y": y, "art": a["art"],
                 "feat_s": a["feat_s"], "corr": a["corr"], "act": a["act"], "L": L, "S": S,
                 "hours": (f.end_s - f.start_s) / 3600}
            d["feat_l"] = long_scale(a["feat_s"], cfg["features"]["long_K"])
            zp = root / (Path(f.name).stem + "_z.npy")
            d["z_path"] = str(zp) if zp.exists() else None
            d["z"] = np.load(zp, mmap_mode="r") if (load_raw and zp.exists()) else None
            self.files.append(d)

    # ------------------------------------------------------------------
    def usable_leads(self, min_cover: float = 0.5):
        """Lead seizures whose preictal period is at least min_cover recorded."""
        p = self.cfg["protocol"]
        sph, sop = p["sph_min"] * 60, p["sop_min"] * 60
        stride = self.cfg["protocol"]["stride_s"]
        out = []
        for on in self.leads:
            n = sum(int(((f["t_end"] >= on - sph - sop) & (f["t_end"] < on - sph)).sum()) for f in self.files)
            if n * stride >= min_cover * sop:
                out.append(on)
        return out

    def interictal_hours(self):
        stride = self.cfg["protocol"]["stride_s"]
        return sum(int((f["y"] == INTER).sum()) for f in self.files) * stride / 3600

    def summary(self):
        ok = [(f["y"], ~f["art"]) for f in self.files]
        return {"case": self.id, "lead_seizures": len(self.usable_leads()),
                "all_seizures": len(self.seizures), "interictal_hours": round(self.interictal_hours(), 2),
                "files": len(self.files),
                "pre_windows": int(sum(((y == PRE) & a).sum() for y, a in ok)),
                "inter_windows": int(sum(((y == INTER) & a).sum() for y, a in ok))}


def select_patients(patients: list[PatientData], cfg: dict):
    p = cfg["protocol"]
    inc, exc = [], {}
    for pd in patients:
        if not pd.case.has_times:
            exc[pd.id] = "file start/end times missing from summary"
        elif len(pd.files) == 0:
            exc[pd.id] = "no readable files with all channels"
        elif len(pd.usable_leads()) < p["min_lead_seizures"]:
            exc[pd.id] = f"fewer than {p['min_lead_seizures']} lead seizures with recorded preictal period"
        elif pd.interictal_hours() < p["min_interictal_h"]:
            exc[pd.id] = f"less than {p['min_interictal_h']} h of interictal EEG"
        else:
            inc.append(pd)
    return inc, exc


# ----------------------------------------------------------------------
class Normalizer:
    """Feature standardisation and the interictal statistics m_j, s_j of Eq. 11."""

    def fit(self, patients: list[PatientData]):
        fs, fl, acts = [], [], []
        for pd in patients:
            for f in pd.files:
                m = (f["y"] >= 0) & ~f["art"]
                fs.append(f["feat_s"][m][::4]); fl.append(f["feat_l"][m][::4])
                mi = (f["y"] == INTER) & ~f["art"]
                acts.append(f["act"][mi][::4])
        fs, fl, acts = np.concatenate(fs), np.concatenate(fl), np.concatenate(acts)
        self.fs_mu, self.fs_sd = fs.mean((0, 1)), fs.std((0, 1)) + 1e-6
        self.fl_mu, self.fl_sd = fl.mean((0, 1)), fl.std((0, 1)) + 1e-6
        self.m, self.s = acts.mean(0), acts.std(0) + 1e-6
        return self

    def feats(self, f, idx):
        a = (f["feat_s"][idx] - self.fs_mu) / self.fs_sd
        b = (f["feat_l"][idx] - self.fl_mu) / self.fl_sd
        return np.concatenate([a, b], -1).astype(np.float32)

    def act(self, f, idx):
        return ((f["act"][idx] - self.m) / self.s).astype(np.float32)


def make_sequences(patients, seq_len):
    """Non-overlapping sequences of seq_len windows. Returns (pre, inter) lists of (patient, file, start)."""
    pre, inter = [], []
    for pi, pd in enumerate(patients):
        for fi, f in enumerate(pd.files):
            y, art = f["y"], f["art"]
            for s in range(0, len(y) - seq_len + 1, seq_len):
                ys, a = y[s:s + seq_len], art[s:s + seq_len]
                valid = (ys >= 0) & ~a
                if valid.sum() < seq_len // 2:
                    continue
                if (ys[valid] == PRE).any():
                    pre.append((pi, fi, s))
                elif (ys[valid] == INTER).all():
                    inter.append((pi, fi, s))
    return pre, inter


def make_batch(patients, norm, items, seq_len, need_raw=False, device="cpu"):
    feats, corrs, acts, ys, masks, raws = [], [], [], [], [], []
    for pi, fi, s in items:
        f = patients[pi].files[fi]
        idx = np.arange(s, s + seq_len)
        feats.append(norm.feats(f, idx)); acts.append(norm.act(f, idx))
        corrs.append(f["corr"][idx].astype(np.float32))
        y = f["y"][idx]
        ys.append(np.clip(y, 0, 1).astype(np.float32))
        masks.append(((y >= 0) & ~f["art"][idx]).astype(np.float32))
        if need_raw:
            raws.append(raw_windows(f, idx))
    b = {"feat": torch.from_numpy(np.stack(feats)), "corr": torch.from_numpy(np.stack(corrs)),
         "act": torch.from_numpy(np.stack(acts)), "y": torch.from_numpy(np.stack(ys)),
         "mask": torch.from_numpy(np.stack(masks))}
    if need_raw:
        b["raw"] = torch.from_numpy(np.stack(raws))
    return {k: v.to(device) for k, v in b.items()}


def raw_windows(f, idx):
    z = f["z"] if f["z"] is not None else np.load(f["z_path"], mmap_mode="r")
    L, S = f["L"], f["S"]
    return np.stack([np.asarray(z[:, i * S:i * S + L], dtype=np.float32) for i in idx])


def load_cases(cache_dir):
    with open(Path(cache_dir) / "cases.json") as fh:
        return [Case.from_dict(d) for d in json.load(fh)]
