#!/usr/bin/env bash
# Controlled evaluation of the classifier: same folds, same settings, several seeds, one change at a
# time, then one pooled report with subject-level confidence intervals.
# Run AFTER preprocessing and world-model pretraining (run_all.sh stages 1 and 3).
#
#   TASK=cdr_binary bash run_cv.sh
#
# Knobs (environment variables):
#   TASK=cdr_binary   cdr_binary (CDR 0 vs >=0.5) | cn_vs_ad (CDR 0 vs >=1) | 3class (CDR 0 / 0.5 / >=1)
#   SEEDS="42 43 44"  one full cross-validation per seed
#   EXPS="scratch wm_ft wm_probe wm_ft_vol"   experiments to run (see the table below)
#   EP_CLS=40         fixed epoch budget (checkpoint = lowest class-balanced val loss)
#   PRETRAINED=$RUNS/pretrain/last.pt         world-model weights
#   INDEX_FIXED=      index_proc.csv of a --crop fixed preprocessing run; adds FIXED_EXPS
#   FIXED_EXPS="scratch_fixed wm_ft_fixed"   experiments run on INDEX_FIXED
#   AUG=0             0 = no augmentation, 1 = augmentation on (compare by running both into new RUNS_CV)
#   DATA=data RUNS=runs KEY=~/.wm_data.key RUNS_CV=$RUNS/cv
#
# Experiments (all: image-only, no label smoothing, no flip averaging, select by val loss):
#   scratch        encoder trained from random weights             -> the baseline
#   wm_ft          world-model encoder, fine-tuned                 -> does pretraining help?
#   wm_probe       world-model encoder frozen, only the head trains -> is the representation useful?
#   wm_ft_vol      wm_ft + brain volume / eTIV as an extra input   -> does whole-brain volume help?
#   scratch_mmbn   scratch with [mean, max] pooling + BatchNorm head -> fixes feature collapse?
#   wm_ft_mmbn     wm_ft   with [mean, max] pooling + BatchNorm head
#   *_fixed        same, on fixed-mm-crop volumes (needs INDEX_FIXED); wm_ft_fixed reuses a world model
#                  pretrained on brain-cropped volumes unless PRETRAINED points at one pretrained on them
# Note: the world model is pretrained on all dev subjects' images (no labels), including each
# fold's validation subjects; say so when reporting.
set -euo pipefail
TASK=${TASK:-cdr_binary}
SEEDS=${SEEDS:-"42 43 44"}
EXPS=${EXPS:-"scratch wm_ft wm_probe wm_ft_vol"}
EP_CLS=${EP_CLS:-40}
AUG=${AUG:-0}
DATA=${DATA:-data}; RUNS=${RUNS:-runs}; KEY=${KEY:-$HOME/.wm_data.key}
RUNS_CV=${RUNS_CV:-$RUNS/cv}
PRETRAINED=${PRETRAINED:-$RUNS/pretrain/last.pt}
INDEX=${INDEX:-$DATA/proc/index_proc.csv}
INDEX_FIXED=${INDEX_FIXED:-}
SPLITS=${SPLITS:-$DATA/splits.json}
OUT=$RUNS_CV/$TASK
mkdir -p "$OUT/.done"

NF=$(python -c "import json;print(len(json.load(open('$SPLITS'))['folds']))")
COMMON="--splits $SPLITS --key_file $KEY --task $TASK --epochs $EP_CLS --no_meta --label_smoothing 0 \
--no_tta --select loss --patience 1000"
[ "$AUG" = 0 ] && COMMON="$COMMON --no_aug"

step() {  # step <name> <command...>   (skipped if already done)
  local name=$1; shift
  if [ -f "$OUT/.done/$name" ]; then echo "== skip $name"; return; fi
  echo "== $name"; "$@"; touch "$OUT/.done/$name"
}

# 0. label audit + non-imaging baselines on the same folds (minutes, CPU)
step diagnose python src/diagnose_labels.py --index "$INDEX" --splits "$SPLITS" --out "$OUT/diagnose"

exp_args() {  # exp_args <exp> -> index + model flags
  case $1 in
    scratch)       echo "--index $INDEX" ;;
    wm_ft)         echo "--index $INDEX --pretrained $PRETRAINED" ;;
    wm_probe)      echo "--index $INDEX --pretrained $PRETRAINED --linear_probe" ;;
    wm_ft_vol)     echo "--index $INDEX --pretrained $PRETRAINED --extra_feats vol_ratio" ;;
    scratch_mmbn)  echo "--index $INDEX --pool meanmax --head_norm batch" ;;
    wm_ft_mmbn)    echo "--index $INDEX --pretrained $PRETRAINED --pool meanmax --head_norm batch" ;;
    scratch_fixed) echo "--index ${INDEX_FIXED:?set INDEX_FIXED} " ;;
    wm_ft_fixed)   echo "--index ${INDEX_FIXED:?set INDEX_FIXED} --pretrained $PRETRAINED" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}
FIXED_EXPS=${FIXED_EXPS:-"scratch_fixed wm_ft_fixed"}
[ -n "$INDEX_FIXED" ] && EXPS="$EXPS $FIXED_EXPS"

for e in $EXPS; do
  for s in $SEEDS; do
    for ((f=0; f<NF; f++)); do
      # shellcheck disable=SC2046
      step "${e}_seed${s}_fold${f}" python src/finetune_classify.py $COMMON $(exp_args "$e") \
        --seed "$s" --fold "$f" --out "$OUT/$e/seed$s/fold$f"
    done
  done
done

# pooled report: every experiment run so far for this task (also earlier calls), against scratch
REP=()
for d in "$OUT"/*/; do
  e=$(basename "$d")
  ls "$d"seed*/fold*/val_predictions.csv >/dev/null 2>&1 && REP+=(--exp "$e=$OUT/$e/seed*/fold*")
done
REF=""; [ -d "$OUT/scratch" ] && REF="--ref scratch"
python src/cv_report.py "${REP[@]}" $REF --out "$OUT/report"
echo "report: $OUT/report/cv_report.md   label audit: $OUT/diagnose/label_audit.md"
