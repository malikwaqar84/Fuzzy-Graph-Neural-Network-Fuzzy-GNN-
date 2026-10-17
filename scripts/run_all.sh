#!/usr/bin/env bash
# Full pipeline: download, preprocess, proposed model, 9 baselines, 8 ablations, summary, figures.
# Every LOSO run resumes per fold, so this script can simply be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/default.yaml}
JOBS=${JOBS:-4}

python scripts/download_chbmit.py --out data/chbmit
python scripts/prepare_data.py --config "$CONFIG" --jobs "$JOBS"

python scripts/run_loso.py --config "$CONFIG" --model fuzzygnn --shap

for m in truong_cnn tsiouris_lstm eegnet eeg_conformer attention_wavelet_cnn wtrpnet chen_gcn_gru tang_dcrnn dts_gan; do
  python scripts/run_loso.py --config "$CONFIG" --model "$m"
done

for a in hard_gcn gaussian fixed_membership static no_eai short_only no_attention no_temporal; do
  python scripts/run_loso.py --config "$CONFIG" --model fuzzygnn --ablation "$a"
done

python scripts/summarize.py --results results
python scripts/make_figures.py --results results --out figures
