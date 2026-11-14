#!/usr/bin/env bash
# Full pipeline. Default dataset: OASIS-2 (longitudinal), from oasis-brains.org.
#
#   OASIS2_DIR=data/raw/OASIS2 OASIS2_CSV=oasis_longitudinal_demographics.xlsx bash run_all.sh
#
# Optional extra single-visit data:  OASIS1_DIR=... OASIS1_CSV=oasis_cross-sectional.csv
#
# ADNI instead:
#   DATASET=adni RAW=data/raw/ADNI MRI_KEY=All_Subjects_Key_MRI.csv DXSUM=DXSUM.csv bash run_all.sh
#   (metadata sources: META_CSV = IDA collection CSV, XML_DIR = IDA XMLs, MRI_KEY, DXSUM)
#
# Where things go (point these at Google Drive on Colab so nothing is lost on a disconnect):
#   DATA=data  RUNS=runs  MODELS=models  KEY=~/.wm_data.key
# The script is resumable: finished stages are skipped, preprocessing skips finished scans and
# world-model pretraining continues from its last epoch. Just run the same command again.
#
# Size / speed knobs:
#   SMOKE=1        every stage for 1-3 epochs, to check the code runs (NOT results)
#   EP_PRE=150     world-model epochs          EP_CLS=60   classifier epochs
#   TRAIN_FOLDS=5  how many CV folds to train (1 = fold 0 only, quickest)
#   MODELSET=all   all | core   (core = cls, cls_adv, p1_nometa, p1_full: enough for the main tables)
#   TARGET=96      volume size (64 for CPU-only machines)
#   PREP_FLAGS=""  extra preprocessing options, e.g. "--no_neck_crop" or "--bet_f 0.5";
#                  check the result first with: python src/qc_mask.py --index $DATA/index.csv --compare_neck
set -euo pipefail
DATASET=${DATASET:-oasis}
OASIS1_DIR=${OASIS1_DIR:-}; OASIS1_CSV=${OASIS1_CSV:-}; OASIS2_DIR=${OASIS2_DIR:-}; OASIS2_CSV=${OASIS2_CSV:-}
META_CSV=${META_CSV:-}; XML_DIR=${XML_DIR:-}; MRI_KEY=${MRI_KEY:-}; DXSUM=${DXSUM:-}
if [ "$DATASET" = adni ]; then
  RAW=${RAW:?set RAW=folder that contains the ADNI/ image tree}
elif [ -z "$OASIS2_DIR$OASIS1_DIR" ]; then
  echo "set OASIS2_DIR and OASIS2_CSV (and/or OASIS1_DIR, OASIS1_CSV)"; exit 1
fi
TARGET=${TARGET:-96}
PREP_FLAGS=${PREP_FLAGS:-}
FOLDS=${FOLDS:-5}
DATA=${DATA:-data}; RUNS=${RUNS:-runs}; MODELS=${MODELS:-models}
KEY=${KEY:-$HOME/.wm_data.key}
MODELSET=${MODELSET:-all}
if [ "$DATASET" = oasis ]; then
  DEBIAS=${DEBIAS:-none}      # OASIS-1/2: one scanner (Siemens 1.5T), nothing to debias
  DSNAME="OASIS"
else
  DEBIAS=${DEBIAS:-scanner}   # scanner (vendor + field strength) or site
  DSNAME="ADNI"
fi
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  EP_PRE=${EP_PRE:-2}; EP_CLS=${EP_CLS:-3}; SPLIT_FLAG=--smoke
  QUICK="--bs 2 --freeze_epochs 1 --eps_ramp 1 --pgd_steps 2"
  EVAL_QUICK="--steps 2"; P1_QUICK="--steps 2 --eps 0.01 --age_years 5 --dt_years 1 --guard_n 4"
else
  EP_PRE=${EP_PRE:-150}; EP_CLS=${EP_CLS:-60}; SPLIT_FLAG=""; QUICK=""; EVAL_QUICK=""; P1_QUICK=""
fi
mkdir -p "$DATA" "$RUNS" "$DATA/.done" "$RUNS/.done"
[ -f "$KEY" ] || python src/secure_io.py genkey --key_file "$KEY"

# run a stage once: step <marker-dir> <name> <command...>  (re-running the script skips it)
step() {
  local dir=$1 name=$2; shift 2
  if [ -f "$dir/.done/$name" ]; then echo "== skip $name (done)"; return; fi
  echo "== $name"; "$@"; touch "$dir/.done/$name"
}

# ---- 1. index + preprocessing (digital image processing)
if [ "$DATASET" = oasis ]; then
  step "$DATA" index python src/prepare_oasis.py \
    ${OASIS1_DIR:+--oasis1_dir "$OASIS1_DIR" --oasis1_csv "$OASIS1_CSV"} \
    ${OASIS2_DIR:+--oasis2_dir "$OASIS2_DIR" --oasis2_csv "$OASIS2_CSV"} --out "$DATA/index.csv"
else
  step "$DATA" index python src/prepare_index.py --raw_dir "$RAW" --desc_regex "MPRAGE|MP-RAGE|SPGR" --keep_repeats \
    ${META_CSV:+--meta_csv "$META_CSV"} ${XML_DIR:+--meta_xml_dir "$XML_DIR"} \
    ${MRI_KEY:+--mri_key "$MRI_KEY"} ${DXSUM:+--dxsum "$DXSUM"} --out "$DATA/index.csv"
fi
step "$DATA" preprocess python src/preprocess.py --index "$DATA/index.csv" --out_dir "$DATA/proc" \
  --target "$TARGET" --spacing 1.5 --skullstrip auto --qc --encrypt --key_file "$KEY" --resume ${PREP_FLAGS:-}
python src/secure_io.py verify --manifest "$DATA/proc/manifest.json"
step "$DATA" sign python src/metadata_guard.py sign --index "$DATA/proc/index_proc.csv" --key_file "$KEY"
python src/metadata_guard.py check --index "$DATA/proc/index_proc.csv"
step "$DATA" splits python src/splits.py --index "$DATA/proc/index_proc.csv" --out "$DATA/splits.json" \
  --folds "$FOLDS" $SPLIT_FLAG

# ---- 2. DICOM header anomaly detector (Paper 1); skipped automatically without DICOM headers
step "$RUNS" header_anomaly bash -c "
  python src/header_anomaly.py extract --index '$DATA/proc/index_proc.csv' --out '$DATA/headers.csv' &&
  python src/header_anomaly.py train --headers '$DATA/headers.csv' --splits '$DATA/splits.json' --out '$RUNS/header_anomaly' &&
  python src/header_anomaly.py evaluate --headers '$DATA/headers.csv' --splits '$DATA/splits.json' \
    --detector '$RUNS/header_anomaly/header_detector.joblib' --out '$RUNS/header_anomaly'"

C="--index $DATA/proc/index_proc.csv --splits $DATA/splits.json --key_file $KEY"
NF=$(python -c "import json;print(len(json.load(open('$DATA/splits.json'))['folds']))")
TRAIN_FOLDS=${TRAIN_FOLDS:-$NF}
[ "$TRAIN_FOLDS" -gt "$NF" ] && TRAIN_FOLDS=$NF

# ---- 3. world models: plain, and Paper 1 (monotonicity + Trajectory-TRADES)
step "$RUNS" pretrain python src/pretrain_world_model.py $C --out "$RUNS/pretrain" --epochs "$EP_PRE" --w_mono 0 --resume
step "$RUNS" pretrain_p1 python src/pretrain_world_model.py $C --out "$RUNS/pretrain_p1" --epochs "$EP_PRE" --traj_eps 0.01 --resume

# ---- 4. classifiers per fold
cls() {  # cls <name> <fold> <pretrained> [extra args]
  local name=$1 f=$2 pre=$3; shift 3
  step "$RUNS" "${name}_fold$f" python src/finetune_classify.py $C --fold "$f" --epochs "$EP_CLS" $QUICK \
    --pretrained "$RUNS/$pre/last.pt" --out "$RUNS/$name/fold$f" "$@"
}
for ((f=0; f<TRAIN_FOLDS; f++)); do
  cls cls "$f" pretrain
  cls cls_adv "$f" pretrain --adv pgd --eps 0.01 --debias "$DEBIAS"
  cls p1_nometa "$f" pretrain_p1 --no_meta
  # combined model: image + metadata adversarial training (joint PGD) + debiasing -> deployed
  cls p1_full "$f" pretrain_p1 --adv pgd --eps 0.01 --meta_dropout 0.2 --meta_adv 5 --debias "$DEBIAS"
  if [ "$MODELSET" = all ]; then
    cls p1_meta "$f" pretrain_p1 --meta_dropout 0.2
    cls p1_metaadv "$f" pretrain_p1 --meta_dropout 0.2 --meta_adv 5
  fi
done

# ---- 5. evaluation
step "$RUNS" eval_test python src/evaluate.py $C --ckpts "$RUNS/cls/fold*/best.pt" --out "$RUNS/test"
step "$RUNS" eval_robustness python src/robustness_eval.py $C --run standard "$RUNS/cls/fold*/best.pt" \
  --run adversarial "$RUNS/cls_adv/fold*/best.pt" --run combined "$RUNS/p1_full/fold*/best.pt" \
  --out "$RUNS/robustness" $EVAL_QUICK
P1_RUNS=(--run img_only "$RUNS/p1_nometa/fold*/best.pt" --run combined "$RUNS/p1_full/fold*/best.pt")
if [ "$MODELSET" = all ]; then
  P1_RUNS+=(--run img+meta "$RUNS/p1_meta/fold*/best.pt" --run img+meta+metaAdv "$RUNS/p1_metaadv/fold*/best.pt")
fi
step "$RUNS" eval_paper1 python src/paper1_eval.py $C --world_model "$RUNS/pretrain_p1/last.pt" \
  "${P1_RUNS[@]}" --out "$RUNS/paper1" $P1_QUICK
step "$RUNS" eval_paper1_nodefence python src/paper1_eval.py $C --world_model "$RUNS/pretrain/last.pt" \
  --out "$RUNS/paper1_nodefence" $P1_QUICK   # ablation: no monotonicity / Trajectory-TRADES

# ---- 6. package the deployable model
HDET=$( [ -f "$RUNS/header_anomaly/header_detector.joblib" ] && echo "$RUNS/header_anomaly/header_detector.joblib" || true )
python deploy/package_models.py --classifiers "$RUNS/p1_full/fold*/best.pt" \
  --world_model "$RUNS/pretrain_p1/last.pt" --out "$MODELS" \
  ${HDET:+--header_detector "$HDET"} --trained_on "$DSNAME, $TRAIN_FOLDS of $NF folds, combined adversarial"
echo "done - $MODELS/ is ready for deploy/aws/setup_aws.sh. Results: $RUNS/test, $RUNS/robustness, $RUNS/paper1, $RUNS/paper1_nodefence"
