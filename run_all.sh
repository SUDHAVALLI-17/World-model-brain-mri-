#!/usr/bin/env bash
# Full pipeline.
#
#   RAW=data/raw/ADNI XML_DIR=data/raw/ADNI MRI_KEY=All_Subjects_Key_MRI_10Nov2026.csv \
#   DXSUM=DXSUM.csv bash run_all.sh
#
# Metadata sources (give at least one that has labels):
#   META_CSV  IDA collection CSV          XML_DIR  IDA per-image metadata XMLs
#   MRI_KEY   All_Subjects_Key_MRI csv    DXSUM    Diagnostic Summary / ADNIMERGE (per-visit labels)
#
# OASIS instead of ADNI (OASIS-1 and/or OASIS-2 from oasis-brains.org):
#   DATASET=oasis OASIS2_DIR=data/raw/OASIS2 OASIS2_CSV=oasis_longitudinal.csv \
#   OASIS1_DIR=data/raw/OASIS1 OASIS1_CSV=oasis_cross-sectional.csv bash run_all.sh
#
# SMOKE=1 runs every stage for 1-3 epochs on whatever data you have (tiny subsets allowed,
# all subjects in every split) - to check the code runs, NOT to get results.
set -euo pipefail
DATASET=${DATASET:-adni}
OASIS1_DIR=${OASIS1_DIR:-}; OASIS1_CSV=${OASIS1_CSV:-}; OASIS2_DIR=${OASIS2_DIR:-}; OASIS2_CSV=${OASIS2_CSV:-}
if [ "$DATASET" = adni ]; then RAW=${RAW:?set RAW=folder that contains the ADNI/ image tree}; fi
META_CSV=${META_CSV:-}; XML_DIR=${XML_DIR:-}; MRI_KEY=${MRI_KEY:-}; DXSUM=${DXSUM:-}
TARGET=${TARGET:-96}        # 64 for CPU
FOLDS=${FOLDS:-5}
KEY=${KEY:-$HOME/.adni.key}
DEBIAS=${DEBIAS:-scanner}   # scanner (vendor + field strength) or site
# OASIS-1/2 used one scanner (Siemens 1.5T), so there is no scanner bias to remove
if [ "$DATASET" = oasis ]; then DEBIAS=${DEBIAS_OASIS:-none}; fi
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  EP_PRE=2; EP_CLS=3; SPLIT_FLAG=--smoke; QUICK="--bs 2 --freeze_epochs 1 --eps_ramp 1 --pgd_steps 2"
  EVAL_QUICK="--steps 2"; P1_QUICK="--steps 2 --eps 0.01 --age_years 5 --dt_years 1 --guard_n 4"
else
  EP_PRE=150; EP_CLS=60; SPLIT_FLAG=""; QUICK=""; EVAL_QUICK=""; P1_QUICK=""
fi
D=data; mkdir -p $D runs
[ -f "$KEY" ] || python src/secure_io.py genkey --key_file "$KEY"

if [ "$DATASET" = oasis ]; then
  python src/prepare_oasis.py ${OASIS1_DIR:+--oasis1_dir "$OASIS1_DIR" --oasis1_csv "$OASIS1_CSV"} \
    ${OASIS2_DIR:+--oasis2_dir "$OASIS2_DIR" --oasis2_csv "$OASIS2_CSV"} --out $D/index.csv
else
  python src/prepare_index.py --raw_dir "$RAW" --desc_regex "MPRAGE|MP-RAGE|SPGR" --keep_repeats \
    ${META_CSV:+--meta_csv "$META_CSV"} ${XML_DIR:+--meta_xml_dir "$XML_DIR"} \
    ${MRI_KEY:+--mri_key "$MRI_KEY"} ${DXSUM:+--dxsum "$DXSUM"} --out $D/index.csv
fi

python src/preprocess.py --index $D/index.csv --out_dir $D/proc \
  --target "$TARGET" --spacing 1.5 --skullstrip auto --qc --encrypt --key_file "$KEY"
python src/secure_io.py verify --manifest $D/proc/manifest.json
python src/metadata_guard.py sign --index $D/proc/index_proc.csv --key_file "$KEY"
python src/metadata_guard.py check --index $D/proc/index_proc.csv

python src/splits.py --index $D/proc/index_proc.csv --out $D/splits.json --folds "$FOLDS" $SPLIT_FLAG

# ---- DICOM header anomaly detector (rules + Isolation Forest + autoencoder), Paper 1
python src/header_anomaly.py extract --index $D/proc/index_proc.csv --out $D/headers.csv
python src/header_anomaly.py train --headers $D/headers.csv --splits $D/splits.json --out runs/header_anomaly
python src/header_anomaly.py evaluate --headers $D/headers.csv --splits $D/splits.json \
  --detector runs/header_anomaly/header_detector.joblib --out runs/header_anomaly
C="--index $D/proc/index_proc.csv --splits $D/splits.json --key_file $KEY"
NF=$(python -c "import json;print(len(json.load(open('$D/splits.json'))['folds']))")

# ---- world models: plain, and Paper 1 (monotonicity + Trajectory-TRADES)
python src/pretrain_world_model.py $C --out runs/pretrain --epochs $EP_PRE --w_mono 0
python src/pretrain_world_model.py $C --out runs/pretrain_p1 --epochs $EP_PRE --traj_eps 0.01

# ---- classifiers per fold
for ((f=0; f<NF; f++)); do
  F="$C --fold $f --epochs $EP_CLS $QUICK"
  python src/finetune_classify.py $F --pretrained runs/pretrain/last.pt --out runs/cls/fold$f
  python src/finetune_classify.py $F --pretrained runs/pretrain/last.pt --out runs/cls_adv/fold$f \
    --adv pgd --eps 0.01 --debias "$DEBIAS"
  python src/finetune_classify.py $F --pretrained runs/pretrain_p1/last.pt --out runs/p1_meta/fold$f --meta_dropout 0.2
  python src/finetune_classify.py $F --pretrained runs/pretrain_p1/last.pt --out runs/p1_metaadv/fold$f \
    --meta_dropout 0.2 --meta_adv 5
  python src/finetune_classify.py $F --pretrained runs/pretrain_p1/last.pt --out runs/p1_nometa/fold$f --no_meta
  # combined model: image + metadata adversarial training (joint PGD) + debiasing -> deployed
  python src/finetune_classify.py $F --pretrained runs/pretrain_p1/last.pt --out runs/p1_full/fold$f \
    --adv pgd --eps 0.01 --meta_dropout 0.2 --meta_adv 5 --debias "$DEBIAS"
done

# ---- evaluation
python src/evaluate.py $C --ckpts "runs/cls/fold*/best.pt" --out runs/test
python src/robustness_eval.py $C --run standard "runs/cls/fold*/best.pt" \
  --run adversarial "runs/cls_adv/fold*/best.pt" --run combined "runs/p1_full/fold*/best.pt" \
  --out runs/robustness $EVAL_QUICK
python src/paper1_eval.py $C --world_model runs/pretrain_p1/last.pt \
  --run img+meta "runs/p1_meta/fold*/best.pt" --run img+meta+metaAdv "runs/p1_metaadv/fold*/best.pt" \
  --run img_only "runs/p1_nometa/fold*/best.pt" --run combined "runs/p1_full/fold*/best.pt" \
  --out runs/paper1 $P1_QUICK
python src/paper1_eval.py $C --world_model runs/pretrain/last.pt \
  --out runs/paper1_nodefence $P1_QUICK   # ablation: no monotonicity / Trajectory-TRADES
HDET=$( [ -f runs/header_anomaly/header_detector.joblib ] && echo runs/header_anomaly/header_detector.joblib || true )
python deploy/package_models.py --classifiers "runs/p1_full/fold*/best.pt" \
  --world_model runs/pretrain_p1/last.pt --out models \
  ${HDET:+--header_detector "$HDET"} --trained_on "ADNI, $NF-fold, combined adversarial"
echo "done - models/ is ready for deploy/aws/setup_aws.sh. Results: see runs/test, runs/robustness, runs/paper1, runs/paper1_nodefence, runs/header_anomaly"
