"""Leave-one-subject-out training and evaluation (Section 4).

    python scripts/run_loso.py --model fuzzygnn --shap
    python scripts/run_loso.py --model eegnet
    python scripts/run_loso.py --model fuzzygnn --ablation no_eai
    python scripts/run_loso.py --model fuzzygnn --set protocol.sop_min=60 --tag sop60

Output, in <out>/<run>/ (run = model, model__ablation, or with --tag appended):
    config_used.yaml, selection.json
    folds/<subject>.json          one per fold, written when the fold finishes (resume = skip existing)
    windows/<case>.csv.gz         every test window: label, artefact flag, predicted probability
    models/<subject>.pt           trained weights (use --no-save-models to skip)
    patients.csv                  one row per test case, rebuilt after every fold
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzygnn.alarms import choose_rho                                     # noqa: E402
from fuzzygnn.config import ABLATIONS, load_config, parse_override, with_overrides  # noqa: E402
from fuzzygnn.dataset import Normalizer, PatientData, load_cases, make_sequences, select_patients  # noqa: E402
from fuzzygnn.evaluate import patient_record, write_window_csv           # noqa: E402
from fuzzygnn.train import build_model, pick_device, predict_patient, set_seed, static_coupling, train_model  # noqa: E402

CSV_COLS = ["case", "subject", "files", "recorded_h", "lead_seizures_all", "n_seizures", "n_predicted",
            "sensitivity", "n_alarms", "n_false", "interictal_h", "fpr_h", "tiw", "mwt_min", "p_random",
            "p_surrogate", "TP", "FP", "TN", "FN", "n_pre_windows", "n_inter_windows", "acc", "sn_window",
            "sp_window", "ppv", "npv", "f1", "auc", "rho", "nofp_n_predicted", "nofp_sensitivity",
            "nofp_n_false", "nofp_fpr_h"]


def write_patients_csv(run_dir: Path):
    rows = []
    for fj in sorted((run_dir / "folds").glob("*.json")):
        rows += json.load(open(fj))["patients"]
    rows.sort(key=lambda r: r["case"])
    with open(run_dir / "patients.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="fuzzygnn")
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=5e-4")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shap", action="store_true", help="channel-level KernelSHAP for every test case")
    ap.add_argument("--folds", nargs="*", help="run only these test subjects, e.g. chb01 chb05")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, dict(parse_override(s) for s in args.set))
    if args.ablation != "full":
        if args.model != "fuzzygnn":
            sys.exit("ablations apply only to --model fuzzygnn")
        cfg = with_overrides(cfg, ABLATIONS[args.ablation])
    run = args.model + ("" if args.ablation == "full" else f"__{args.ablation}") + (f"__{args.tag}" if args.tag else "")
    run_dir = Path(args.out) / run
    for sub in ("folds", "windows", "models"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as fh:
        yaml.safe_dump({"model": args.model, "ablation": args.ablation, "cli_overrides": args.set, **cfg}, fh, sort_keys=False)
    device = pick_device(cfg)
    print(f"run {run} on {device}")

    # ---- data and patient selection (Section 4.1)
    cache = cfg["data"]["cache_dir"]
    need_raw = getattr(build_model(args.model, cfg, len(cfg["data"]["channels"])), "needs_raw", False)
    t0 = time.time()
    patients = [PatientData(c, cache, cfg) for c in load_cases(cache)]
    if need_raw:
        miss = [f["name"] for p in patients for f in p.files if f["z_path"] is None]
        if miss:
            sys.exit(f"{args.model} needs the stored signals but {len(miss)} files have none; rerun prepare_data.py without --no-raw")
    inc, exc = select_patients(patients, cfg)
    print(f"loaded {len(patients)} cases in {time.time() - t0:.0f}s; included {len(inc)}")
    sel = {"included": [p.summary() for p in inc], "excluded": exc}
    with open(run_dir / "selection.json", "w") as fh:
        json.dump(sel, fh, indent=1)
    subjects = sorted({p.case.subject for p in inc})
    by_subj = {s: [p for p in inc if p.case.subject == s] for s in subjects}
    n_val = cfg["train"]["val_patients"]
    C = len(cfg["data"]["channels"])

    for k, subj in enumerate(subjects):
        if args.folds and subj not in args.folds:
            continue
        fold_file = run_dir / "folds" / f"{subj}.json"
        if fold_file.exists():
            print(f"[{k + 1}/{len(subjects)}] {subj}: done, skipping")
            continue
        tf = time.time()
        test = by_subj[subj]
        pool = [s for s in subjects if s != subj]
        rng = np.random.default_rng(cfg["train"]["seed"] + k)
        val_subj = sorted(rng.choice(pool, size=min(n_val, len(pool) - 1), replace=False).tolist())
        val = [p for s in val_subj for p in by_subj[s]]
        train = [p for s in pool if s not in val_subj for p in by_subj[s]]
        print(f"[{k + 1}/{len(subjects)}] test {subj} ({', '.join(p.id for p in test)}), "
              f"val {', '.join(val_subj)}, {len(train)} training cases")

        set_seed(cfg["train"]["seed"] + k)
        norm = Normalizer().fit(train)
        model = build_model(args.model, cfg, C)
        if args.model == "fuzzygnn" and cfg["model"]["membership"] == "static":
            model.static_c.copy_(torch.from_numpy(static_coupling(train)))
        info = train_model(model, train, val, norm, cfg, device)

        # threshold rho on the validation subjects (never on the test subject)
        val_sets = [(predict_patient(model, p, norm, cfg, device), p) for p in val]
        rho = float(choose_rho(val_sets, cfg))

        recs = []
        for p in test:
            probs = predict_patient(model, p, norm, cfg, device)
            write_window_csv(run_dir / "windows" / f"{p.id}.csv.gz", p, probs)
            rec = patient_record(probs, p, cfg, rho, with_nofp=(run == "fuzzygnn"), seed=cfg["train"]["seed"])
            if args.shap and args.model == "fuzzygnn":
                from fuzzygnn.explain import channel_shap
                _, bg_int = make_sequences(train, cfg["train"]["seq_len"])
                pick = rng.choice(len(bg_int), min(cfg["shap"]["background"], len(bg_int)), replace=False)
                bg = [(train[bg_int[i][0]], bg_int[i][1], bg_int[i][2]) for i in pick]
                sv = channel_shap(model, p, norm, cfg, bg, device, seed=cfg["train"]["seed"])
                rec["shap_channels"] = cfg["data"]["channels"]
                rec["shap_mean_abs"] = None if sv is None else [float(v) for v in sv]
            recs.append(rec)
            print(f"    {p.id}: Sn {rec['n_predicted']}/{rec['n_seizures']}  false alarms {rec['n_false']} "
                  f"in {rec['interictal_h']:.1f} h (FPR/h {rec['fpr_h']:.3f})  AUC {rec['auc']:.3f}  "
                  f"TP {rec['TP']} FP {rec['FP']} TN {rec['TN']} FN {rec['FN']}")

        fold = {"subject": subj, "test_cases": [p.id for p in test], "val_subjects": val_subj,
                "train_cases": [p.id for p in train], "rho": rho, "device": device,
                "membership": model.membership_params() if hasattr(model, "membership_params") else None,
                "train": info, "seconds": round(time.time() - tf, 1),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "patients": recs}
        if not args.no_save_models:
            torch.save({"state_dict": model.state_dict(), "norm": norm.__dict__, "rho": rho}, run_dir / "models" / f"{subj}.pt")
        tmp = fold_file.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(fold, fh, indent=1, default=float)
        tmp.replace(fold_file)
        write_patients_csv(run_dir)

    rows = write_patients_csv(run_dir)
    if rows:
        ns, npred = sum(r["n_seizures"] for r in rows), sum(r["n_predicted"] for r in rows)
        nf, ih = sum(r["n_false"] for r in rows), sum(r["interictal_h"] for r in rows)
        print(f"\n{run}: {len(rows)} cases, {npred}/{ns} seizures predicted, {nf} false alarms in {ih:.1f} h "
              f"interictal (pooled FPR/h {nf / ih if ih else float('nan'):.3f})")


if __name__ == "__main__":
    main()
