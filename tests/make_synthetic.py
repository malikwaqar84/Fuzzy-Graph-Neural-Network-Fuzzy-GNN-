"""Create a tiny synthetic dataset through the real preprocessing and feature code.

    python tests/make_synthetic.py
    python scripts/run_loso.py --config tests/test_config.yaml --model fuzzygnn --shap --out tests/_synthetic/results

The signals are random noise with an artificial pre-seizure pattern. They exist only to
check that the software runs end to end. Never report numbers obtained from them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuzzygnn.chbmit import Case, EdfFile          # noqa: E402
from fuzzygnn.config import load_config            # noqa: E402
from fuzzygnn.pipeline import process_array, save_processed  # noqa: E402

N_CASES, N_FILES, FILE_MIN = 4, 4, 40
SEIZ_MIN = [20, 60, 100]          # onsets (min from start of the case), 30 s long


def synth_file(rng, t0_s, dur_s, fs, C, seizures_abs):
    n = int(dur_s * fs)
    t = t0_s + np.arange(n) / fs
    x = np.cumsum(rng.standard_normal((C, n)), axis=1)          # brown noise
    x -= np.convolve(x.mean(0), np.ones(fs) / fs, mode="same")  # crude detrending
    x = x - x.mean(1, keepdims=True)
    x = 30 * x / (x.std(1, keepdims=True) + 1e-6)
    x += 10 * rng.standard_normal((C, n))
    common = np.sin(2 * np.pi * 22 * t)
    for on, off in seizures_abs:
        ramp = np.clip((t - (on - 600)) / 600, 0, 1) * (t < on)          # 10 min pre-seizure build-up
        x[: C // 2] += 25 * ramp * common                               # synchronised beta in half the channels
        ict = (t >= on) & (t < off)
        x += 150 * ict * np.sin(2 * np.pi * 4 * t)                     # rhythmic ictal activity
    return x.astype(np.float32)


def main():
    cfg = load_config(str(ROOT / "tests" / "test_config.yaml"))
    cache = ROOT / cfg["data"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    fs, C = cfg["data"]["fs_in"], len(cfg["data"]["channels"])
    cases = []
    for k in range(1, N_CASES + 1):
        rng = np.random.default_rng(k)
        cid = f"syn{k:02d}"
        shift = rng.uniform(-3, 3)
        seiz = [((m + shift) * 60, (m + shift) * 60 + 30) for m in SEIZ_MIN]
        files = []
        for j in range(N_FILES):
            t0, t1 = j * FILE_MIN * 60, (j + 1) * FILE_MIN * 60
            rel = [(on - t0, off - t0) for on, off in seiz if t0 <= on < t1]
            name = f"{cid}_{j + 1:02d}.edf"
            files.append(EdfFile(name, float(t0), float(t1), rel))
            x = synth_file(rng, t0, t1 - t0, fs, C, seiz)
            feats, z = process_array(x, fs, cfg)
            save_processed(cache / cid, Path(name).stem, feats, z, save_raw=True)
        cases.append(Case(cid, files))
        print(f"{cid}: {N_FILES} files, seizures at {[round(s[0] / 60, 1) for s in seiz]} min")
    with open(cache / "cases.json", "w") as fh:
        json.dump([c.to_dict() for c in cases], fh, indent=1)
    print(f"synthetic cache written to {cache}")


if __name__ == "__main__":
    main()
