"""Preprocess every CHB-MIT recording once and cache features (Section 3.1-3.2).

    python scripts/prepare_data.py --jobs 4
    python scripts/prepare_data.py --config configs/default.yaml --cases chb01 chb02 --no-raw

Writes, under data.cache_dir:
    cases.json                 parsed summaries (file times, all seizures)
    <case>/<file>.npz          per-window features, coupling, activation, artefact flags
    <case>/<file>_z.npy        normalised signal (float16), needed only by raw-input baselines
    prepare_log.json           files that were skipped and why
    patients.json              per-case counts and the inclusion decision (Section 4.1)
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzygnn.chbmit import parse_summary                     # noqa: E402
from fuzzygnn.config import load_config, parse_override       # noqa: E402
from fuzzygnn.dataset import PatientData, select_patients     # noqa: E402
from fuzzygnn.pipeline import process_edf                     # noqa: E402


def _job(args):
    edf, out_dir, cfg, save_raw = args
    try:
        return str(edf), *process_edf(edf, out_dir, cfg, save_raw)
    except Exception as e:                                     # noqa: BLE001
        return str(edf), "error", repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.raw_dir=/mnt/chbmit")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--no-raw", action="store_true", help="do not store signals (only fuzzygnn and feature baselines can run)")
    args = ap.parse_args()
    cfg = load_config(args.config, dict(parse_override(s) for s in args.set))
    raw, cache = Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    summaries = sorted(raw.glob("chb*/chb*-summary.txt"))
    if args.cases:
        summaries = [s for s in summaries if s.name.split("-")[0] in args.cases]
    if not summaries:
        sys.exit(f"no chbXX-summary.txt files under {raw}")
    cases = [parse_summary(s) for s in summaries]
    with open(cache / "cases.json", "w") as fh:
        json.dump([c.to_dict() for c in cases], fh, indent=1)

    jobs = []
    for c in cases:
        for f in c.files:
            edf = raw / c.case_id / f.name
            if edf.exists():
                jobs.append((edf, cache / c.case_id, cfg, not args.no_raw))
            else:
                print(f"  missing file {edf}")
    log = {}
    print(f"{len(cases)} cases, {len(jobs)} EDF files")
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for k, fu in enumerate(as_completed(futs), 1):
                edf, st, msg = fu.result()
                log[edf] = [st, msg]
                print(f"[{k}/{len(jobs)}] {Path(edf).name}: {st} {msg}", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            edf, st, msg = _job(j)
            log[edf] = [st, msg]
            print(f"[{k}/{len(jobs)}] {Path(edf).name}: {st} {msg}", flush=True)
    with open(cache / "prepare_log.json", "w") as fh:
        json.dump(log, fh, indent=1)

    # per-case counts and inclusion decision
    pats = [PatientData(c, str(cache), cfg) for c in cases]
    inc, exc = select_patients(pats, cfg)
    rows = []
    for pd in pats:
        r = pd.summary() if pd.files else {"case": pd.id}
        r["included"] = pd in inc
        r["reason"] = exc.get(pd.id, "")
        rows.append(r)
    with open(cache / "patients.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nincluded {len(inc)} cases: {' '.join(p.id for p in inc)}")
    for k, v in exc.items():
        print(f"  excluded {k}: {v}")


if __name__ == "__main__":
    main()
