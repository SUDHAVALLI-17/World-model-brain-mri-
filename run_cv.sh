#!/usr/bin/env bash
# Controlled evaluation of the classifier: same folds, same settings, several seeds, one change at a
# time, then one pooled report with subject-level confidence intervals.
# Run AFTER preprocessing (run_all.sh stage 1). Never touches the held-out test subjects.
#
#   TASK=cdr_binary bash run_cv.sh
#
# Order of work:
#   0  tissue volume (add_tissue_volume.py) + label audit and non-imaging baselines (diagnose_labels.py)
#   0b optional: one world model per fold, pretrained WITHOUT that fold's validation subjects and with the
#      anti-collapse loss (variance + covariance + feature queue)              FOLD_PRETRAIN=1
#   0c frozen-encoder probe (probe_cv.py): pooled features + logistic regression, no fine-tuning
#   1  sanity checks on fold 0: memorise OVERFIT_N scans; train on shuffled labels
#   2  fine-tuning experiments x seeds x folds, then cv_report.py
#
# Knobs (environment variables):
#   TASK=cdr_binary   cdr_binary (CDR 0 vs >=0.5) | cn_vs_ad (CDR 0 vs >=1) | 3class (CDR 0 / 0.5 / >=1)
#   SEEDS="42 43 44"  one full cross-validation per seed
#   EXPS="scratch_mmbn_noimag wm_probe_mmbn wm_ft_mmbn wm_ft_tissue"   fine-tuning experiments (table below)
#   EP_CLS=40         fixed epoch budget (checkpoint = lowest class-balanced val loss)
#   PRETRAINED=$RUNS/pretrain/last.pt   world-model weights; may contain {fold} (one per fold)
#   FOLD_PRETRAIN=0   1 = pretrain one world model per fold into $RUNS/pretrain_v2_fold{fold} (EP_PRE
#                     epochs each, new anti-collapse defaults) and use those as PRETRAINED
#   EP_PRE=150 PRE_FLAGS=""   pretraining epochs / extra flags for FOLD_PRETRAIN=1
#   PROBE=1           run the frozen-encoder probe (step 0c)
#   INDEX_FIXED=      index_proc.csv of a --crop fixed preprocessing run; adds FIXED_EXPS
#   FIXED_EXPS="scratch_fixed wm_ft_fixed"
#   AUG=0             0 = no augmentation, 1 = augmentation on
#   SANITY=1 SANITY_EXPS="scratch_mmbn_noimag wm_ft_mmbn" OVERFIT_N=64 OVERFIT_EP=60
#   DATA=data RUNS=runs KEY=~/.wm_data.key RUNS_CV=$RUNS/cv
#
# Experiments (all: image-only, no label smoothing, no flip averaging, select by val loss):
#   scratch / scratch_noimag / scratch_mmbn / scratch_mmbn_noimag
#                  encoder from random weights; _noimag = no imagined-future inputs (random, frozen
#                  dynamics in a scratch model); _mmbn = [mean, max] pooling + BatchNorm head
#   wm_ft          world-model encoder, fine-tuned (attention pool, LayerNorm head: collapsed on OASIS-2)
#   wm_probe       world-model encoder frozen, head only
#   wm_ft_mmbn     wm_ft with [mean, max] pooling + BatchNorm head
#   wm_probe_mmbn  frozen encoder + [mean, max] pooling + BatchNorm head (the probe, trained end to end)
#   wm_ft_vol      wm_ft + mask volume / eTIV input
#   wm_ft_tissue   wm_ft_mmbn + TISSUE volume / eTIV input (excludes CSF, where atrophy shows)
#   *_fixed        on fixed-mm-crop volumes (needs INDEX_FIXED)
set -euo pipefail
TASK=${TASK:-cdr_binary}
SEEDS=${SEEDS:-"42 43 44"}
EXPS=${EXPS:-"scratch_mmbn_noimag wm_probe_mmbn wm_ft_mmbn wm_ft_tissue"}
EP_CLS=${EP_CLS:-40}
AUG=${AUG:-0}
DATA=${DATA:-data}; RUNS=${RUNS:-runs}; KEY=${KEY:-$HOME/.wm_data.key}
RUNS_CV=${RUNS_CV:-$RUNS/cv}
FOLD_PRETRAIN=${FOLD_PRETRAIN:-0}; EP_PRE=${EP_PRE:-150}; PRE_FLAGS=${PRE_FLAGS:-}
PROBE=${PROBE:-1}
INDEX=${INDEX:-$DATA/proc/index_proc.csv}
INDEX_FIXED=${INDEX_FIXED:-}
SPLITS=${SPLITS:-$DATA/splits.json}
if [ "$FOLD_PRETRAIN" = 1 ]; then
  FOLD_DIR='pretrain_v2_fold{fold}'   # quoted: a literal } inside ${...:-...} would end it early
  PRETRAINED=${PRETRAINED:-$RUNS/$FOLD_DIR/last.pt}
else
  PRETRAINED=${PRETRAINED:-$RUNS/pretrain/last.pt}
fi
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
pre_for() { local pat='{fold}'; echo "${PRETRAINED//"$pat"/$1}"; }   # world-model path for fold $1

# 0. tissue volume + label audit + non-imaging baselines (minutes, CPU)
step tissue python src/add_tissue_volume.py --index "$INDEX" --key_file "$KEY"
step diagnose python src/diagnose_labels.py --index "$INDEX" --splits "$SPLITS" --out "$OUT/diagnose"

# 0b. optional fold-specific pretraining with the anti-collapse loss
if [ "$FOLD_PRETRAIN" = 1 ]; then
  for ((f=0; f<NF; f++)); do
    d=$(dirname "$(pre_for $f)")
    # shellcheck disable=SC2086
    step "pretrain_fold$f" python src/pretrain_world_model.py --index "$INDEX" --splits "$SPLITS" \
      --key_file "$KEY" --exclude_fold "$f" --epochs "$EP_PRE" --resume --out "$d" $PRE_FLAGS
  done
fi

# 0c. frozen-encoder probe: an honest number without fine-tuning
if [ "$PROBE" = 1 ]; then
  ADD="tissue_ratio"
  python - "$INDEX" <<'EOF' || ADD=""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(0 if "tissue_ml" in d and "etiv" in d and d.etiv.notna().any() else 1)
EOF
  step probe python src/probe_cv.py --index "$INDEX" --splits "$SPLITS" --key_file "$KEY" \
    --ckpt "world_model=$PRETRAINED" --task "$TASK" ${ADD:+--add $ADD} --out "$OUT/probe"
fi

exp_args() {  # exp_args <exp> <fold> -> index + model flags
  local P; P=$(pre_for "$2")
  case $1 in
    scratch)             echo "--index $INDEX" ;;
    scratch_noimag)      echo "--index $INDEX --imagine" ;;
    scratch_mmbn)        echo "--index $INDEX --pool meanmax --head_norm batch" ;;
    scratch_mmbn_noimag) echo "--index $INDEX --pool meanmax --head_norm batch --imagine" ;;
    wm_ft)               echo "--index $INDEX --pretrained $P" ;;
    wm_probe)            echo "--index $INDEX --pretrained $P --linear_probe" ;;
    wm_ft_mmbn)          echo "--index $INDEX --pretrained $P --pool meanmax --head_norm batch" ;;
    wm_probe_mmbn)       echo "--index $INDEX --pretrained $P --linear_probe --pool meanmax --head_norm batch" ;;
    wm_ft_vol)           echo "--index $INDEX --pretrained $P --extra_feats vol_ratio" ;;
    wm_ft_tissue)        echo "--index $INDEX --pretrained $P --pool meanmax --head_norm batch --extra_feats tissue_ratio" ;;
    scratch_fixed)       echo "--index ${INDEX_FIXED:?set INDEX_FIXED}" ;;
    wm_ft_fixed)         echo "--index ${INDEX_FIXED:?set INDEX_FIXED} --pretrained $P" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}
FIXED_EXPS=${FIXED_EXPS:-"scratch_fixed wm_ft_fixed"}
[ -n "$INDEX_FIXED" ] && EXPS="$EXPS $FIXED_EXPS"

# 1. sanity checks (fold 0, first seed), kept under sanity/ so they never enter the CV report
SANITY=${SANITY:-1}
SANITY_EXPS=${SANITY_EXPS:-"scratch_mmbn_noimag wm_ft_mmbn"}
OVERFIT_N=${OVERFIT_N:-64}; OVERFIT_EP=${OVERFIT_EP:-60}
S0=${SEEDS%% *}
if [ "$SANITY" = 1 ]; then
  for e in $SANITY_EXPS; do
    # shellcheck disable=SC2046
    step "sanity_overfit_$e" python src/finetune_classify.py $COMMON $(exp_args "$e" 0) --epochs "$OVERFIT_EP" \
      --overfit "$OVERFIT_N" --freeze_epochs 0 --seed "$S0" --fold 0 --out "$OUT/sanity/overfit_$e"
    # shellcheck disable=SC2046
    step "sanity_shuffled_$e" python src/finetune_classify.py $COMMON $(exp_args "$e" 0) --shuffle_labels \
      --seed "$S0" --fold 0 --out "$OUT/sanity/shuffled_$e"
  done
  python src/sanity_report.py --dir "$OUT/sanity"
fi

# 2. fine-tuning experiments x seeds x folds
for e in $EXPS; do
  for s in $SEEDS; do
    for ((f=0; f<NF; f++)); do
      # shellcheck disable=SC2046
      step "${e}_seed${s}_fold${f}" python src/finetune_classify.py $COMMON $(exp_args "$e" "$f") \
        --seed "$s" --fold "$f" --out "$OUT/$e/seed$s/fold$f"
    done
  done
done

# pooled report: every experiment run so far for this task, against the first scratch variant found
REP=()
for d in "$OUT"/*/; do
  e=$(basename "$d")
  ls "$d"seed*/fold*/val_predictions.csv >/dev/null 2>&1 && REP+=(--exp "$e=$OUT/$e/seed*/fold*")
done
REF=""
for r in scratch scratch_mmbn_noimag scratch_noimag scratch_mmbn; do
  [ -d "$OUT/$r" ] && { REF="--ref $r"; break; }
done
[ ${#REP[@]} -gt 0 ] && python src/cv_report.py "${REP[@]}" $REF --out "$OUT/report"
echo "probe: $OUT/probe/probe_cv.md   report: $OUT/report/cv_report.md   audit: $OUT/diagnose/label_audit.md"
