# Fuzzy-GNN for EEG seizure prediction

This repository contains the code for the paper *Fuzzy Graph Neural Network with Attention and Explainable AI for Early Epileptic Seizure Prediction*.
It covers the full pipeline:

* reading and preprocessing the CHB-MIT recordings
* seizure-prediction labelling with a seizure occurrence period (SOP) and seizure prediction horizon (SPH)
* the Fuzzy-GNN model and its ablation variants
* nine re-implemented baseline methods
* leave-one-subject-out (LOSO) evaluation on continuous recordings
* statistical validation against a random predictor
* channel-level SHAP explanations
* the figures and tables of the paper

## Installation

```bash
git clone https://github.com/malikwaqar84/Fuzzy-Graph-Neural-Network-Fuzzy-GNN-.git
cd Fuzzy-Graph-Neural-Network-Fuzzy-GNN-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install the CUDA build of PyTorch first if you have a GPU
```

Tested with Python 3.11 and PyTorch 2.x.

## Data

Download the CHB-MIT Scalp EEG Database (version 1.0.0, about 43 GB) from PhysioNet:

```bash
python scripts/download_chbmit.py --out data/chbmit
# or
wget -r -N -c -np -nH --cut-dirs=3 -P data/chbmit https://physionet.org/files/chbmit/1.0.0/
```

The folder should contain `data/chbmit/chb01/chb01-summary.txt`, `data/chbmit/chb01/chb01_01.edf`, and so on.

## Running the experiments

```bash
bash scripts/run_all.sh                     # everything, in order
```

or step by step:

```bash
python scripts/prepare_data.py --jobs 4                        # preprocessing and features (cached)
python scripts/run_loso.py --model fuzzygnn --shap             # proposed model
python scripts/run_loso.py --model eegnet                      # one baseline
python scripts/run_loso.py --model fuzzygnn --ablation no_eai  # one ablation
python scripts/summarize.py                                    # results/summary.json + summary.md
python scripts/make_figures.py                                 # figures/figure2..6
```

Each LOSO fold is saved as soon as it finishes, so an interrupted run resumes where it stopped.
Config values can be overridden from the command line, e.g. `--set protocol.sop_min=60 train.lr=5e-4` (add `--tag sop60` so the run gets its own folder).

## Outputs

`prepare_data.py` writes `data/cache/patients.json`: for every case the number of lead seizures, interictal hours, preictal and interictal windows, and whether the case passes the inclusion rules (with the reason if not).

`run_loso.py` writes, for each run in `results/<run>/`:

| File | Content |
|---|---|
| `selection.json` | included cases with their counts, excluded cases with the reason |
| `folds/<subject>.json` | test, validation and training cases of the fold, chosen ρ, learned θ and κ, training history, and all metrics of each test case |
| `patients.csv` | one row per test case: seizures, predicted seizures, sensitivity, false alarms, interictal hours, FPR/h, TiW, MWT, random-predictor and surrogate p-values, window counts TP/FP/TN/FN, accuracy, window sensitivity and specificity, PPV, NPV, F1 and AUC (computed from the predicted probabilities) |
| `windows/<case>.csv.gz` | every test window with its time, label (1 preictal, 0 interictal, −1 excluded), artefact flag and predicted probability |
| `models/<subject>.pt` | trained weights, normalisation statistics and ρ |

Every reported number can be recomputed from `windows/*.csv.gz`. `summarize.py` pools the runs and writes `results/summary.json`, `results/summary.md` and `results/table_patients_fuzzygnn.csv`.

**Compute.** On a single modern GPU, preprocessing takes about 1 to 2 hours and one LOSO run takes a few hours. All 19 runs (the proposed model, 9 baselines and 8 ablations) take roughly one to two days. CPU-only runs are possible but slow.

## What the code does

| Paper | Code |
|---|---|
| Sec. 3.1, Eqs. 1–3: filtering, resampling, z-scoring, windows | `fuzzygnn/preprocess.py`, `fuzzygnn/features.py`, `fuzzygnn/pipeline.py` |
| Sec. 3.2, Eqs. 4–8: band energies, entropy, multi-scale features | `fuzzygnn/features.py` |
| Sec. 3.3, Eq. 10: sigmoidal fuzzy membership, learnable θ and κ | `FuzzyGNN.membership` in `fuzzygnn/model.py` |
| Sec. 3.4, Eqs. 11–12: Epileptogenic Activation Index | `FuzzyGNN.eai` |
| Sec. 3.5, Eqs. 13–14: fuzzy-attention message passing | `FuzzyAttentionLayer` |
| Sec. 3.6, Eqs. 15–19: readout and loss | `FuzzyGNN.forward`, `fuzzygnn/train.py` |
| Sec. 3.7, Eq. 20: firing-power alarms | `fuzzygnn/alarms.py` |
| Sec. 3.8, Eq. 21: channel-level KernelSHAP | `fuzzygnn/explain.py` |
| Sec. 4.1–4.2: patient selection, SOP/SPH labels, metrics | `fuzzygnn/dataset.py`, `fuzzygnn/alarms.py`, `fuzzygnn/evaluate.py` |
| Sec. 4: LOSO folds, threshold selection on validation subjects | `scripts/run_loso.py` |
| Sec. 4.3, Eq. 22: random predictor and surrogates | `fuzzygnn/stats.py` |
| Sec. 5.4: Friedman and Nemenyi tests | `fuzzygnn/stats.py`, `scripts/summarize.py` |

Protocol details:

* **Labels.** Windows between onset − (SPH + SOP) and onset − SPH are labelled preictal. Windows at least 4 h from any seizure are interictal, and all other windows are excluded from training and from the false-alarm count.
* **Lead seizures.** A seizure is a lead seizure if it starts at least SPH + SOP after the end of the previous seizure. Only lead seizures with at least half of their preictal period recorded are counted.
* **Inclusion.** A case is included if it has at least 3 such lead seizures and at least 4 h of interictal EEG, and its summary file gives file start and end times.
* **Subjects.** Cases chb01 and chb21 come from the same subject, so they are always kept in the same fold.
* **Validation and alarm threshold.** In every fold, two subjects are held out from the training subjects as validation data. They are used for early stopping and for choosing the firing-power threshold ρ: the smallest value in the grid whose validation FPR/h is at most 0.15.
* **Evaluation.** Test recordings are processed continuously and never resampled. A false alarm is an alarm raised during an interictal window, and FPR/h divides the false alarms by the interictal hours of that case. Window-level counts use the artefact-free labelled windows and a probability threshold of 0.5.

## Baselines

All baselines are our own compact re-implementations, written from the published descriptions (`fuzzygnn/baselines.py`). They share the inputs, folds, labels, loss, class weighting and alarm rule of the proposed model. The simplifications are:

| Baseline | Simplification |
|---|---|
| Truong et al. (2018), CNN | 2D convolutions with EEG channels as input planes instead of the original 3D first layer |
| Tsiouris et al. (2018), LSTM | uses our spectral features instead of the original feature set |
| EEGNet (Lawhern et al., 2018) | standard EEGNet-8,2 |
| EEG Conformer (Song et al., 2023) | 2 transformer layers, 4 heads |
| Attention wavelet CNN (Xin et al., 2022) | db4 DWT, one 1D-CNN branch per sub-band, squeeze-and-excitation attention |
| WTRPNet (Xin et al., 2021) | unthresholded recurrence plot of the level-3 db4 approximation |
| Chen et al. (2021), GCN–GRU | correlation-weighted GCN on our spectral features, GRU over time |
| Tang et al. (2022), DCRNN | top-3 correlation graph, 2-step diffusion, no self-supervised pre-training |
| DTS-GAN (Yan et al., 2025) | LSTM per node and one Gaussian-kernel graph attention layer |

If you have access to the authors' official code for any baseline, you can replace the class with it. Report in the paper which version you used.

## Repository layout

```
configs/default.yaml     all settings used in the paper
fuzzygnn/                the package (model, data, training, alarms, statistics, SHAP, baselines)
scripts/                 command-line entry points
tests/                   synthetic smoke test
```

## Testing

```bash
python tests/make_synthetic.py        # creates a tiny synthetic dataset through the real preprocessing code
python scripts/run_loso.py --config tests/test_config.yaml --model fuzzygnn --shap --out tests/_synthetic/results
```

The synthetic data exist only to check that the software runs. Never report numbers from them.

## Citation

If you use this code, please cite the paper (details will be added on publication).

## License

MIT
