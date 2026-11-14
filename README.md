# Brain-MRI World Model: CN / MCI / AD classification (OASIS-2 or ADNI)

A self-supervised **world model** for T1 brain MRI. The default dataset is **OASIS-2**
(longitudinal, free from oasis-brains.org); ADNI is still supported with `DATASET=adni`.

## Quick start on OASIS-2

**Easiest: Google Colab.** Open `notebooks/colab_oasis2.ipynb` in Colab (File → Open notebook →
GitHub → this repository), choose a T4 GPU, paste the two OASIS-2 download links and run the
cells. Preprocessed data, checkpoints and results are kept on Google Drive, and every step can be
re-run after a disconnect: finished stages are skipped, preprocessing skips finished scans and the
world model continues from its last epoch.

**What OASIS-2 looks like and how it is handled**

| | OASIS-2 | What the code does |
|---|---|---|
| Download | `OAS2_RAW_PART1.tar.gz` (10 GB) + `PART2` (8 GB) + Demographic Data (.xlsx) | the notebook extracts only `mpr-1` of each session (one T1 is enough), about a quarter of the data |
| Layout | `OAS2_0001_MR1/RAW/mpr-1..4` as `.hdr` + `.img` pairs (Analyze-style) | `prepare_oasis.py` finds every session folder recursively |
| Scans | 150 people aged 60-96, 373 sessions, 2-5 visits at least a year apart; 3-4 repeat T1s per session, 256×256×128, 1×1×1.25 mm, sagittal | one scan per session; every visit pair becomes a world-model transition |
| Orientation | Analyze headers do not store it reliably | known OASIS geometry is applied before RAS reorientation (`geometry=oasis_raw`) |
| Labels | CDR per visit (0, 0.5, 1, 2); groups Nondemented / Demented / Converted | CDR 0 → CN, 0.5 → MCI (very mild), ≥ 1 → AD |
| Time and action | `MR Delay` = days since the first visit; `Age`, `M/F` per visit | Δt, age and sex for the action; visit order is exact |
| Scanner | one Siemens 1.5 T scanner | scanner debiasing switched off; DICOM header detector skipped (no DICOM headers) |
| Extra | `nWBV` (normalised whole-brain volume) per visit | can check the predicted atrophy against measured brain loss |

After preprocessing each scan is a 96³ volume of about 3.5 MB, so the whole training set is about
1.3 GB.

**Command line** (any machine with a GPU):

```bash
OASIS2_DIR=data/raw/OASIS2 OASIS2_CSV=oasis_longitudinal_demographics.xlsx SMOKE=1 bash run_all.sh   # 20-30 min check
OASIS2_DIR=data/raw/OASIS2 OASIS2_CSV=oasis_longitudinal_demographics.xlsx TRAIN_FOLDS=1 MODELSET=core bash run_all.sh
```

`TRAIN_FOLDS=1 MODELSET=core` is the quickest real run (fold 0; standard, adversarial,
image-only and combined models). `TRAIN_FOLDS=5 MODELSET=all` is the full paper setup.
`DATA=`, `RUNS=` and `MODELS=` choose where outputs go. Single-visit OASIS-1 can be added with
`OASIS1_DIR=` and `OASIS1_CSV=` for extra pretraining data.

---

The rest of this README describes the method. It was written for ADNI; everything applies to
OASIS-2 except the ADNI-specific download and table steps.

The model was designed for a small subset.
It learns brain anatomy without labels (3D JEPA) and learns how a brain changes
over time (an **action-conditioned latent dynamics model**). A classifier then uses
the current latent state *plus imagined future states* to predict CN, MCI or AD.

```
             ┌─────────────────────────── DIP preprocessing ───────────────────────────┐
 ADNI IDA →  │ DICOM/NIfTI → RAS reorient → N4 bias correction → 1.5 mm resample →     │
 subset      │ anisotropic-diffusion denoise → skull strip (SynthStrip/HD-BET/Otsu+     │
             │ morphology) → crop/pad/resize 96³ → intensity window [0,1] → AES encrypt  │
             └──────────────────────────────────────────────────────────────────────────┘
                                             │  subject-level splits (no leakage)
                                             ▼
   ┌──────────────────── Stage A: world-model pretraining (no labels) ───────────────────┐
   │  x_t ──mask 3D blocks──► Encoder (3D ResNet) ─► context tokens ─► JEPA predictor ─┐  │
   │  x_t ─────────────────► EMA target encoder ─► target tokens ◄──── L_jepa ─────────┘  │
   │  x_t ─► Encoder ─► z_t ─┐                                                            │
   │  action a=[Δt, age, sex]┴► Dynamics (FiLM + transformer) ─► ẑ_{t+Δt}, μ, σ²          │
   │  x_{t+Δt} ─► EMA target encoder ─► z_{t+Δt}  ◄──── L_dyn + Gaussian NLL ─────────────│
   └─────────────────────────────────────────────────────────────────────────────────────┘
                                             ▼
   ┌──────────────── Stage B: classification (k-fold CV, then held-out test) ────────────┐
   │  features = [ pool(z_t) , μ(t+1y) − g_t , μ(t+2y) − g_t , age, sex ] → MLP → CN/MCI/AD │
   │  frozen-encoder warm-up → fine-tune, class weights, label smoothing, flip TTA        │
   └─────────────────────────────────────────────────────────────────────────────────────┘
                                             ▼
        test metrics + bootstrap CI · fold ensemble · 3D Grad-CAM · latent drift & uncertainty
```

## Why this design for a *subset*

| Problem with small ADNI subsets | What the pipeline does |
|---|---|
| Few labelled scans → 3D CNNs overfit | Self-supervised JEPA pretraining uses every scan; supervised head is small |
| Same subject in train and test inflates accuracy (a common bug in ADNI papers) | `splits.py` splits **by subject** with StratifiedGroupKFold |
| Scanner/site intensity differences | N4 bias correction + percentile windowing + bias-field/gamma augmentation |
| MCI is the hardest class (it sits between CN and AD) | Dynamics adds "how this brain is expected to change" features |
| Class imbalance | Class-weighted loss or `--balanced_sampler`, balanced accuracy for model selection |
| Single split is noisy | 5-fold CV + held-out test + subject-level bootstrap 95% CI |

The **actions** in this world model are `[Δt (years), age, sex]`: "advance this brain by
Δt years". The dynamics can be rolled forward (`BrainWorldModel.rollout`) to imagine a
trajectory, and its predicted variance gives an uncertainty for each prediction.
Extra actions (e.g. treatment, APOE4 status) can be added by extending `encode_action`
in `common.py` and `action_dim`.

## Files

| File | Stage |
|---|---|
| `src/plan_download.py` | Pick a balanced, longitudinal subject list from the MRI key (+ DXSUM) |
| `src/adni_tables.py` / `src/ida_xml.py` | Readers for the MRI key, DXSUM/ADNIMERGE and IDA metadata XMLs |
| `src/prepare_index.py` | Find images in the IDA download, join with the IDA CSV (+ optional ADNIMERGE per-visit diagnosis) |
| `src/preprocess.py` | DIP pipeline (SimpleITK) → `.npy` / encrypted `.npy.enc`, QC PNGs |
| `src/splits.py` | Subject-level test split + CV folds |
| `src/data.py` | Datasets (single scans, longitudinal transitions) + 3D augmentation |
| `src/models.py` | Encoder, JEPA predictor, action dynamics, world model, classifier |
| `src/pretrain_world_model.py` | Stage A |
| `src/finetune_classify.py` | Stage B (one fold per run) |
| `src/evaluate.py` | Test set: ensemble, metrics, CI, Grad-CAM, imagined latent drift |
| `src/adversarial.py` | FGSM / PGD / TRADES attacks, eps schedule, gradient-reversal debiasing adversary |
| `src/trajectory_attacks.py` | Paper 1: attacks on the forecast (pixel hijack, fake/hide decline, Δt and age action attacks) |
| `src/metadata_attacks.py` | Paper 1: age / sex / missing / inconsistent / joint metadata attacks |
| `src/metadata_guard.py` | Paper 1: HMAC signing + plausibility rules for header metadata |
| `src/header_anomaly.py` | Paper 1: DICOM header anomaly detection (rules + Isolation Forest + autoencoder) and 12 header attacks |
| `src/paper1_eval.py` | Paper 1: one report per research question |
| `src/robustness_eval.py` | Standard vs adversarial: attack curves, MRI corruptions, site probe, attack figure |
| `src/secure_io.py` | AES encryption at rest, SHA-256 integrity manifest, key handling |
| `src/make_synthetic.py` | Fake ADNI-shaped data for testing the pipeline end to end |
| `run_all.sh` | Every stage in order |

## Setup

```bash
pip install -r requirements.txt
# skull stripping is auto-detected in this order:
#   1. SynthStrip (FreeSurfer: mri_synthstrip)   - best, recommended for the paper
#   2. HD-BET (pip install hd-bet)               - downloads model weights on first use
#   3. BET-style surface fit (pip install brainextractor) - no weights; default fallback
#   4. Otsu threshold + morphology               - last resort, often keeps skull on real ADNI
# Before any of them, the neck is cropped (top 170 mm kept, like FSL robustfov) because ADNI
# sagittal MPRAGEs include a lot of neck. Series shorter than 100 mm are rejected as
# truncated, and masks outside 900-2000 ml are flagged in the log (column mask_ml).
```

GPU recommended (any 8 GB+ card). CPU works for testing with `--target 64`.

## Planning the download (ADNI study-data tables)

`All_Subjects_Key_MRI_<date>.csv` lists every ADNI image (subject, visit code, date, scanner)
but has no diagnosis. Add the **Diagnostic Summary (DXSUM)** table or `ADNIMERGE.csv` and
`plan_download.py` picks a class-balanced set of subjects with enough follow-up visits:

```bash
python src/plan_download.py --mri_key All_Subjects_Key_MRI_10Nov2026.csv --dxsum DXSUM.csv \
    --per_class 40 --min_visits 3 --max_visits 4 --out data/plan
# -> data/plan/download_subjects.txt : paste into IDA Advanced Search (Subject ID), T1 only
```

MCI subjects are split between converters (diagnosis changes during follow-up) and stable
ones, because converters are what the world model's forecasts matter for. Use `--field 3.0`
or `--phases ADNIGO/ADNI2 ADNI3` to limit scanner variety.

When indexing, pass the same files so every scan gets its visit code, scanner and the
diagnosis **at that visit**:

```bash
python src/prepare_index.py --raw_dir data/raw/ADNI --meta_xml_dir data/raw/ADNI \
    --mri_key All_Subjects_Key_MRI_10Nov2026.csv --dxsum DXSUM.csv --desc_regex "MPRAGE|MP-RAGE|SPGR"
```

Labels are joined on subject + visit code, and fall back to the nearest exam date
(within 180 days) when codes differ.

## Getting your subset ready

1. From IDA, download T1 scans (e.g. MPRAGE / IR-SPGR) **as NIfTI or DICOM** together
   with the **collection CSV** (Download → CSV). Keep the IDA folder layout
   (`ADNI/<subject>/<description>/<date>/I<imageID>/…`); the code reads the image ID from the path.
   If you have the per-image **metadata XMLs** instead of (or as well as) the CSV, pass
   `--meta_xml_dir` to `prepare_index.py`: it reads research group, age at scan, sex, APOE,
   MMSE/CDR/FAQ and scanner protocol from them (`src/ida_xml.py`). Only full XMLs carry
   labels; the small stubs inside image folders hold IDs only.
2. Optional but recommended: `ADNIMERGE.csv` (from the ADNI study data). The IDA "Group"
   is the diagnosis at **screening**; ADNIMERGE gives the diagnosis at **each visit**,
   which is the correct label for follow-up scans of MCI converters.
3. To train the dynamics, include **2+ visits for as many subjects as you can**
   (e.g. screening, m12, m24). Single-visit subjects still help the JEPA part.
4. If you downloaded pre-processed images (`MT1; GradWarp; N3m`), add `--no_n4`.

Then:

```bash
RAW=data/raw/ADNI XML_DIR=data/raw/ADNI MRI_KEY=All_Subjects_Key_MRI_10Nov2026.csv \
DXSUM=DXSUM.csv bash run_all.sh
```

**Check first with `SMOKE=1`.** It runs every stage for 1–3 epochs on whatever you have,
including a 2-subject subset (every subject goes into train, val and test, so the numbers
are meaningless). It was tested end to end on real ADNI DICOMs (GE 1.5T + Philips 3T):

```bash
SMOKE=1 RAW=data/raw/ADNI XML_DIR=data/raw/ADNI MRI_KEY=All_Subjects_Key_MRI_10Nov2026.csv bash run_all.sh
```

**Compute.** Measured on CPU with the full 96³ model (13.5M parameters): about 2 s per scan
per pretraining epoch, about 3.5 s with Trajectory-TRADES. For ~480 scans that is ~25 min per epoch,
so use a GPU (Colab/Kaggle T4 or a college GPU) for the real runs.

## Recommended settings by subset size

| Scans in subset | `--target` | pretrain `--epochs` | Notes |
|---|---|---|---|
| < 100 | 64 | 200 | Use `--widths 16,32,64,128 --embed_dim 128`; expect wide CIs |
| 100 – 500 | 96 | 150 | Default config |
| 500+ | 96 or 128 | 100 | Raise `--bs`, consider `--blocks` deeper |

Always compare against a **baseline from scratch** (`finetune_classify.py` without
`--pretrained`, and with `--imagine` empty) so the world-model gain is measured, not assumed.
That comparison is the core ablation table for your paper:

| Model | Pretraining | Imagined futures |
|---|---|---|
| 3D ResNet baseline | none | no |
| + JEPA | JEPA only | no |
| + World model | JEPA + dynamics | no |
| Full | JEPA + dynamics | yes (1y, 2y) |

## Adversarial training (robustness + debiasing)

Two kinds of adversarial training, both in `src/adversarial.py`, switched on with flags in
`finetune_classify.py`:

**1. Robustness to adversarial perturbations.** During training, each batch is attacked and the
model learns from the attacked copy too.

| Flag | Method | Cost per step |
|---|---|---|
| `--adv fgsm` | one signed-gradient step (Goodfellow 2015) | ~2× |
| `--adv pgd --pgd_steps 5` | projected gradient descent, random start (Madry 2018) | ~6× |
| `--adv trades --trades_beta 6` | clean loss + KL(attacked ‖ clean) (Zhang 2019) | ~6× |

* `--eps` is the L∞ budget in [0,1] intensity units (0.01 ≈ 1% of the intensity range, invisible to the eye).
* Perturbations are applied **only to brain voxels**, so the background stays black.
* `--eps_ramp N` grows eps from 0 over N epochs (full eps from the start destabilises training);
  attacks start after the frozen-encoder warm-up.
* Checkpoint selection uses the mean of clean and PGD-robust validation balanced accuracy.

**2. Debiasing with an adversary (DANN, Ganin 2016).** `--debias site` adds a gradient-reversal
adversary that tries to predict the ADNI acquisition site (the first part of the subject ID,
`002` in `002_S_0295`) from the classifier's features. The encoder is trained to defeat it, so
predictions can't lean on scanner/site shortcuts. `--debias sex` does the same for sex.
Sites with fewer than 4 training scans are pooled into "other".

```bash
python src/finetune_classify.py --fold 0 --pretrained runs/pretrain/last.pt \
    --out runs/cls_adv/fold0 --adv pgd --eps 0.01 --pgd_steps 5 --debias site
```

**Evaluating it.** `robustness_eval.py` compares runs on the held-out test set:

```bash
python src/robustness_eval.py --run standard "runs/cls/fold*/best.pt" \
    --run adversarial "runs/cls_adv/fold*/best.pt" --out runs/robustness
```

It reports balanced accuracy under FGSM / PGD-10 at several eps (`robustness_curve.png`), under
realistic MRI corruptions (noise, bias field, contrast change, blur, rotation), a **site probe**
(how well a linear model recovers the site from the features; closer to chance = less bias),
and an example attacked scan (`adv_example_*.png`).

Expect the usual **robustness–accuracy trade-off**: the adversarial model loses some clean
accuracy and keeps much more accuracy under attack. Report both in the paper, e.g.:

| Model | Clean | PGD ε=0.005 | PGD ε=0.01 | PGD ε=0.02 | Site probe |
|---|---|---|---|---|---|
| Standard | | | | | |
| + PGD adversarial training | | | | | |
| + PGD + site debiasing | | | | | |

## Paper 1: pixel and metadata attacks on the world model

*"Small Change, Different Future: Pixel and Metadata Attacks on Action-Conditioned
World Models for Alzheimer's Progression."* The world model's action is
`[Δt, age, sex]`, taken from DICOM header fields: study/acquisition date
(0008,0020/0022) → Δt, PatientAge (0010,1010) / PatientBirthDate (0010,0030), and
PatientSex (0010,0040). Tampering with the header therefore attacks the action directly.

### What was added

| Piece | File | Purpose |
|---|---|---|
| Brain-volume head (BPF readout) | `models.py`, `data.py` | Brain parenchymal fraction = tissue voxels / intracranial voxels (DIP thresholding). `v(mu)` gives **predicted future atrophy** |
| Atrophy monotonicity loss | `models.py` (`--w_mono`) | Predicted volume must not grow with Δt. Also a defence: bounds what a Δt attack can do |
| Brain-age head | `models.py` (`--w_age`) | Age predicted from the image; used to check stated age |
| Trajectory-TRADES | `pretrain_world_model.py` (`--traj_eps 0.01`) | Predicted future of an attacked scan must match the clean one |
| Trajectory attacks | `trajectory_attacks.py` | Pixel *hijack*, *targeted* fake/hide decline, **Δt and age attacks on the action**, amplification A(h) |
| Metadata attacks | `metadata_attacks.py` | Age PGD, sex flip, missing, inconsistent, joint pixel+age, minimal flipping budget |
| Metadata guard | `metadata_guard.py` | HMAC signatures over header fields + plausibility rules (ranges, age–date consistency across visits) |
| Classifier flags | `finetune_classify.py` | `--no_meta` (image-only), `--meta_dropout 0.2`, `--meta_adv 5` (worst-case ±5-year age) |
| Evaluation | `paper1_eval.py` | One report per research question |

### Research question → output

| RQ | Question | Where in `runs/paper1/` |
|---|---|---|
| RQ1 | Does an invisible pixel change alter the predicted 1–3-year future? Does the error grow with horizon? Can it fake or hide decline? | `paper1_report.json` → `RQ1_and_RQ-M2_trajectory` (hijack shift, amplification A(h), targeted change); `paper1_figures.png` left |
| RQ-M1 | Smallest metadata change vs smallest pixel change that flips a diagnosis; accuracy under each metadata attack; does metadata make pixel attacks cheaper? | `classification_under_attack.csv`, `RQ-M1_minimal_flip_budget`, rows "pixel only / age only / joint" |
| RQ-M2 | How far do Δt and age tampering move the forecast vs pixel attacks? | `action_vs_pixel_channel.csv`, `paper1_figures.png` right, forecast error on real follow-ups |
| RQ-M3 | Does metadata help or hurt robustness? | compare the `img+meta` and `img_only` columns |
| RQ-M4 | Which layer catches which attack? | `metadata_guard_catch_rates.csv` (plausibility vs HMAC), brain-age AUROC, uncertainty-detector AUROC, monotonicity violation rate |

### Commands

```bash
# 1. world model with all Paper 1 losses (+ Trajectory-TRADES)
python src/pretrain_world_model.py --out runs/pretrain_p1 --traj_eps 0.01
# ablation without the defence terms:
python src/pretrain_world_model.py --out runs/pretrain_p1_nodef --w_mono 0 --traj_eps 0

# 2. classifiers per fold (RQ-M3), plus the combined model that gets deployed
python src/finetune_classify.py --fold $f --pretrained runs/pretrain_p1/last.pt --out runs/p1_meta/fold$f --meta_dropout 0.2
python src/finetune_classify.py --fold $f --pretrained runs/pretrain_p1/last.pt --out runs/p1_metaadv/fold$f --meta_dropout 0.2 --meta_adv 5
python src/finetune_classify.py --fold $f --pretrained runs/pretrain_p1/last.pt --out runs/p1_nometa/fold$f --no_meta
python src/finetune_classify.py --fold $f --pretrained runs/pretrain_p1/last.pt --out runs/p1_full/fold$f \
    --adv pgd --eps 0.01 --meta_dropout 0.2 --meta_adv 5 --debias scanner     # combined

# 3. sign metadata (HMAC) once, before anything else touches it
python src/metadata_guard.py sign  --index data/proc/index_proc.csv --key_file ~/.adni.key
python src/metadata_guard.py check --index data/proc/index_proc.csv   # plausibility report

# 4. full report
python src/paper1_eval.py --world_model runs/pretrain_p1/last.pt \
    --run img+meta "runs/p1_meta/fold*/best.pt" \
    --run img+meta+metaAdv "runs/p1_metaadv/fold*/best.pt" \
    --run img_only "runs/p1_nometa/fold*/best.pt" \
    --run combined "runs/p1_full/fold*/best.pt" --out runs/paper1
```

**Combined model (`p1_full`).** When `--adv pgd` (or `fgsm`) and `--meta_adv` are both set,
each batch is attacked by **one joint PGD** that moves the pixels (inside the brain mask,
L∞ ≤ eps) and the age (± `meta_adv` years) together, after random sex flips on 25 % of
samples. The loss mixes clean and jointly attacked cross-entropy (`--adv_ratio`). Scanner
debiasing (DANN) runs at the same time. During the first epochs, while eps is still ramping
up from 0, only the metadata attack is used. Checkpoints are selected on the mean of clean
and **joint-attack** balanced accuracy on the dev split. This is the model `run_all.sh`
packages for deployment. Compared with the single-defence runs (`cls_adv`, `p1_metaadv`) it
answers the question: do the two defences add up, or does one cost the other?

Run `paper1_eval.py` once for `pretrain_p1` and once for `pretrain_p1_nodef`. The difference
between the two is the evidence that monotonicity and Trajectory-TRADES defend the forecast.

**Data caution:** the IDA CSV "Age" may be the age at screening rather than at each visit.
Compute per-visit age from dates (or take it from ADNIMERGE) before running, otherwise the
clean metadata already fails the age–date consistency rule and the guard results are muddied.

### Header anomaly detection (tampering without a trusted hash)

HMAC signing catches every change to signed metadata, but only when a trusted signature
exists. `src/header_anomaly.py` judges a scan's DICOM header on its own, for legacy or
external scans and for tampering that happened before signing:

| Detector | What it uses |
|---|---|
| `rules` | Larmor frequency vs field strength (42.58 MHz/T), TE < TR, plausible geometry/flip/weight, study vs acquisition date, header vs form (age, sex, date), age 40–100, vendor/model pairs seen in training |
| `iforest` | Isolation Forest on ~40 numeric + one-hot header features (exact count depends on vendors seen) |
| `ae` | small denoising autoencoder, reconstruction error |
| `hybrid` | rules OR iforest OR ae; `hybrid+hash` adds the trusted-hash baseline |

The learned detectors are fitted on clean training headers. Their thresholds are set on clean
validation headers at a 5 % target false-positive rate, so they never see an attack before
the test (unlike a supervised detector trained on the same attacks it is scored on).
`evaluate` applies 12 controlled attacks to held-out test headers: age shifts (5 y, 20 y,
implausible), sex flips (form only / form + header), date tampering (inconsistent /
consistently shifted), field-strength and vendor swaps, protocol and geometry tampering, and
missing fields. Three of them are *stealthy*: they keep every field consistent with the rest.
It then reports recall, precision, F1, FPR and AUROC per attack and detector.

```bash
python src/header_anomaly.py extract  --index data/proc/index_proc.csv --out data/headers.csv
python src/header_anomaly.py train    --headers data/headers.csv --splits data/splits.json --out runs/header_anomaly
python src/header_anomaly.py evaluate --headers data/headers.csv --splits data/splits.json \
    --detector runs/header_anomaly/header_detector.joblib --out runs/header_anomaly
# -> header_summary.md, header_results.csv, header_recall_heatmap.png, header_flags_clean.csv
```

`run_all.sh` runs all three steps and packages the detector with the model. The web app then
adds a "Header anomaly detector" check for DICOM uploads. NIfTI files have no DICOM header,
so the step is skipped for them. Paper angle: compare hash, rules and learned detectors,
show which attacks only the image-based brain-age check can catch, and report false
positives on rare but legitimate scanners.

## Security (networks & cybersecurity component)

* `--encrypt` stores every preprocessed volume AES-encrypted (Fernet: AES-CBC + HMAC-SHA256);
  decryption happens only in memory inside the DataLoader.
* `manifest.json` stores SHA-256 hashes; `secure_io.py verify` detects tampered or corrupted
  files (a defence against training-data poisoning).
* Keys are kept in a `0600` file or the `ADNI_FERNET_KEY` env variable, never in git.
* Natural extensions for the networks part: federated training across sites over mutually
  authenticated TLS (e.g. Flower), differential privacy with Opacus, and adversarial-robustness
  tests of the classifier.

## Web app and AWS deployment

`deploy/` holds a login-protected HTTPS web app. You upload a T1 scan (NIfTI or zipped DICOM)
and get CN/MCI/AD probabilities, Grad-CAM, brain age, a 1–3 year atrophy forecast and the
Paper 1 metadata checks, as an HMAC-signed report.

* `deploy/app/` – FastAPI app (`engine.py` inference, `security.py` auth/signing, `static/` UI)
* `deploy/package_models.py` – bundles trained checkpoints with a SHA-256 manifest
* `deploy/Dockerfile`, `docker-compose*.yml`, `Caddyfile` – hardened container + automatic HTTPS
* `deploy/aws/` – `setup_aws.sh` (one-command AWS build), `update_app.sh`, `power.sh`, `teardown.sh`
* `deploy/tests/test_app.py` – security and end-to-end tests

Step-by-step guide, costs and the security mapping: **`deploy/DEPLOY_AWS.md`**.

## Outputs

* `runs/pretrain/last.pt` – world model
* `runs/cls/fold*/best.pt`, `val_predictions.csv` – per-fold classifiers
* `runs/test/test_report.json` – accuracy, balanced accuracy, macro-F1, macro AUC, CN-vs-AD AUC,
  95% CI, confusion matrix
* `runs/test/test_predictions.csv` – per-scan probabilities + predicted 1/2/3-year latent drift and uncertainty
* `runs/test/gradcam/*.png` – 3D Grad-CAM overlays (check they highlight hippocampus / ventricles,
  not the skull or background)

## ADNI data use

Images are covered by the ADNI Data Use Agreement: do not commit them (or the `data/` folder)
to git, and acknowledge ADNI in any publication as the DUA requires.
