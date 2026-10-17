"""Channel-level KernelSHAP (Section 3.8, Eq. 21)."""
from __future__ import annotations

import numpy as np
import shap
import torch

from .dataset import make_batch, PRE


def channel_shap(model, pd, norm, cfg, background_items, device, seed=0):
    """Mean |SHAP| per channel over preictal windows of one test patient.

    background_items: list of (PatientData, file_idx, start) interictal sequences from training patients.
    Each channel is one SHAP feature; a channel outside the coalition takes the values
    (features, activation, coupling row/column) of a background window.
    """
    sc = cfg["shap"]
    T = cfg["train"]["seq_len"]
    rng = np.random.default_rng(seed)
    model.eval()

    def batch_of(items_by_patient):
        bs = [make_batch([p], norm, [(0, fi, s)], T, model.needs_raw, device) for p, fi, s in items_by_patient]
        return {k: torch.cat([b[k] for b in bs]) for k in bs[0]}

    bg = batch_of(background_items[:sc["background"]])
    cand = [(fi, s) for fi, f in enumerate(pd.files) for s in range(0, len(f["y"]) - T + 1, T)
            if (f["y"][s:s + T] == PRE).all()]
    if not cand:
        return None
    pick = rng.choice(len(cand), min(sc["windows_per_patient"], len(cand)), replace=False)
    C = pd.files[0]["act"].shape[1]
    vals = []
    for k in pick:
        fi, s = cand[k]
        x = batch_of([(pd, fi, s)])

        def f(Z):
            outs = []
            for z in Z:
                keep = torch.tensor(z, dtype=torch.float32, device=device)            # [C]
                mix = {}
                kc = keep.view(1, 1, C, 1)
                mix["feat"] = x["feat"] * kc + bg["feat"] * (1 - kc)
                mix["act"] = x["act"] * keep.view(1, 1, C) + bg["act"] * (1 - keep.view(1, 1, C))
                pair = (keep.view(C, 1) * keep.view(1, C)).view(1, 1, C, C)
                mix["corr"] = x["corr"] * pair + bg["corr"] * (1 - pair)
                if "raw" in x:
                    kr = keep.view(1, 1, C, 1)
                    mix["raw"] = x["raw"] * kr + bg["raw"] * (1 - kr)
                mix["y"], mix["mask"] = bg["y"], bg["mask"]
                with torch.no_grad():
                    p = torch.sigmoid(model(mix))[:, -1].mean().item()   # prediction for the last window
                outs.append(p)
            return np.array(outs)

        expl = shap.KernelExplainer(f, np.zeros((1, C)))
        sv = expl.shap_values(np.ones((1, C)), nsamples=sc["nsamples"], silent=True)
        vals.append(np.abs(np.asarray(sv).reshape(-1)[:C]))
    return np.mean(vals, 0)
