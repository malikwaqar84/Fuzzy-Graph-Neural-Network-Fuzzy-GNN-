"""Model construction, training (Eqs. 17-19) and continuous inference on test recordings."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import BASELINES
from .dataset import INTER, PRE, make_sequences
from .model import FuzzyGNN


def pick_device(cfg) -> str:
    d = cfg["train"].get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_dims(cfg):
    d_short = len(cfg["features"]["bands"]) + 1          # band log-energies + spectral entropy
    return d_short, 2 * d_short                          # long scale = causal mean and slope


def build_model(name: str, cfg: dict, n_channels: int):
    d_short, d_long = feature_dims(cfg)
    L = int(round(cfg["protocol"]["window_s"] * cfg["data"]["fs"]))
    if name == "fuzzygnn":
        return FuzzyGNN(cfg, n_channels, d_short, d_long)
    if name not in BASELINES:
        raise ValueError(f"unknown model {name}; choose fuzzygnn or one of {sorted(BASELINES)}")
    return BASELINES[name](cfg, n_channels, d_short=d_short, L=L)


def static_coupling(patients, max_per_file: int = 200) -> np.ndarray:
    """Mean |c_ij| over interictal training windows (used by the 'static' ablation)."""
    acc, n = None, 0
    for pd in patients:
        for f in pd.files:
            idx = np.where((f["y"] == INTER) & ~f["art"])[0]
            if len(idx) == 0:
                continue
            idx = idx[:: max(1, len(idx) // max_per_file)]
            c = np.abs(f["corr"][idx].astype(np.float32)).sum(0)
            acc = c if acc is None else acc + c
            n += len(idx)
    return acc / max(n, 1)


# ----------------------------------------------------------------------
def batch_from_index(files, norm, items, need_raw, device):
    """items: list of (file_dict, index array). Works for any index array (padding allowed)."""
    from .dataset import raw_windows
    feats, corrs, acts, ys, masks, raws = [], [], [], [], [], []
    for f, idx in items:
        feats.append(norm.feats(f, idx)); acts.append(norm.act(f, idx))
        corrs.append(f["corr"][idx].astype(np.float32))
        y = f["y"][idx]
        ys.append(np.clip(y, 0, 1).astype(np.float32))
        masks.append(((y >= 0) & ~f["art"][idx]).astype(np.float32))
        if need_raw:
            raws.append(raw_windows(f, idx))
    b = {"feat": np.stack(feats), "corr": np.stack(corrs), "act": np.stack(acts),
         "y": np.stack(ys), "mask": np.stack(masks)}
    if need_raw:
        b["raw"] = np.stack(raws)
    return {k: torch.from_numpy(v).to(device) for k, v in b.items()}


def _items(patients, triples, T):
    return [(patients[pi].files[fi], np.arange(s, s + T)) for pi, fi, s in triples]


def loss_fn(logit, batch, pos_weight, lambda_temp):
    """Class-weighted BCE over valid windows (Eq. 17) + temporal consistency (Eq. 18)."""
    y, m = batch["y"], batch["mask"]
    w = m * torch.where(y > 0.5, torch.as_tensor(pos_weight, device=y.device), torch.ones_like(y))
    bce = F.binary_cross_entropy_with_logits(logit, y, weight=w, reduction="sum") / w.sum().clamp(min=1e-6)
    if lambda_temp <= 0 or logit.shape[1] < 2:
        return bce
    p = torch.sigmoid(logit)
    mm = m[:, 1:] * m[:, :-1]
    temp = (((p[:, 1:] - p[:, :-1]) ** 2) * mm).sum() / mm.sum().clamp(min=1e-6)
    return bce + lambda_temp * temp


def _sample(pre, inter, ratio, rng):
    n_int = min(len(inter), int(round(ratio * max(len(pre), 1))))
    pick = rng.choice(len(inter), n_int, replace=False) if n_int else np.array([], int)
    return list(pre) + [inter[i] for i in pick]


def _pos_weight(patients, items, T):
    n_pos = n_neg = 0
    for f, idx in _items(patients, items, T):
        v = (f["y"][idx] >= 0) & ~f["art"][idx]
        n_pos += int(((f["y"][idx] == PRE) & v).sum()); n_neg += int(((f["y"][idx] == INTER) & v).sum())
    return n_neg / max(n_pos, 1)


def train_model(model, train_pats, val_pats, norm, cfg, device, log=print):
    tc = cfg["train"]
    T, bs = tc["seq_len"], tc["batch_size"]
    rng = np.random.default_rng(tc["seed"])
    need_raw = getattr(model, "needs_raw", False)
    tr_pre, tr_int = make_sequences(train_pats, T)
    va_pre, va_int = make_sequences(val_pats, T)
    if not tr_pre:
        raise RuntimeError("no preictal training sequences")
    val_items = _sample(va_pre, va_int, tc["undersample_ratio"], np.random.default_rng(0))
    val_pw = _pos_weight(val_pats, val_items, T) if val_items else 1.0
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(tc["epochs"]):
        t0 = time.time()
        model.train()
        items = _sample(tr_pre, tr_int, tc["undersample_ratio"], rng)
        pw = _pos_weight(train_pats, items, T)
        order = rng.permutation(len(items))
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            sel = [items[i] for i in order[s:s + bs]]
            b = batch_from_index(None, norm, _items(train_pats, sel, T), need_raw, device)
            loss = loss_fn(model(b), b, pw, tc["lambda_temp"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        vl = evaluate_loss(model, val_pats, val_items, norm, cfg, device, val_pw) if val_items else tot / nb
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl, "sec": time.time() - t0})
        log(f"    epoch {ep + 1:3d}  train {tot / max(nb, 1):.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": float(best), "epochs_run": len(history), "history": history}


@torch.no_grad()
def evaluate_loss(model, pats, items, norm, cfg, device, pos_weight):
    model.eval()
    T, bs = cfg["train"]["seq_len"], cfg["train"]["batch_size"]
    need_raw = getattr(model, "needs_raw", False)
    tot, n = 0.0, 0
    for s in range(0, len(items), bs):
        b = batch_from_index(None, norm, _items(pats, items[s:s + bs], T), need_raw, device)
        tot += loss_fn(model(b), b, pos_weight, 0.0).item() * len(items[s:s + bs]); n += len(items[s:s + bs])
    return tot / max(n, 1)


@torch.no_grad()
def predict_patient(model, pd, norm, cfg, device, batch_size: int = 64):
    """Probability for every window of every file, processed in time order.

    Each file is cut into consecutive blocks of seq_len windows; the last block is
    right-aligned to the end of the file, and short files are padded by repeating the
    first window. Nothing from the future of a block is used.
    """
    model.eval()
    T = cfg["train"]["seq_len"]
    need_raw = getattr(model, "needs_raw", False)
    out = []
    for f in pd.files:
        nW = len(f["y"])
        prob = np.zeros(nW, np.float32)
        blocks = []
        if nW < T:
            idx = np.concatenate([np.zeros(T - nW, int), np.arange(nW)])
            blocks.append((idx, T - nW))
        else:
            starts = list(range(0, nW - T + 1, T))
            if starts[-1] + T < nW:
                starts.append(nW - T)
            prev_end = 0
            for s in starts:
                blocks.append((np.arange(s, s + T), max(0, prev_end - s)))
                prev_end = s + T
        for k in range(0, len(blocks), batch_size):
            chunk = blocks[k:k + batch_size]
            b = batch_from_index(None, norm, [(f, idx) for idx, _ in chunk], need_raw, device)
            p = torch.sigmoid(model(b)).cpu().numpy()
            for (idx, skip), pp in zip(chunk, p):
                prob[idx[skip:]] = pp[skip:]
        out.append(prob)
    return out
