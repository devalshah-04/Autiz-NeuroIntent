# Autiz (NeuroIntent) — Project Status

_This is the single living status document. Last updated: 2026-10-06, Phase 0 committed (`62910a9`), Phase 1 in progress._
_Everything below was checked against the repository on that date. Items that could not be checked from
the repository are listed in [section 8](#8-known-limitations) or marked **unverified**. Results from the
v2 training run do not exist yet and are deliberately absent._

---

## 1. What this project is

Autiz is a research prototype that tries to score **what a speaker says** separately from **how they say it**.
It takes a short spoken answer (for example an interview answer), writes down the words, measures the
delivery (pitch, loudness, pauses, speaking rate), and keeps those two things apart so that a flat or slow
delivery does not lower a score that is meant to reflect only the content.

Terms used throughout:

- **Prosody** — the "music" of speech: pitch, loudness, rhythm, pauses.
- **ASR** (automatic speech recognition) — software that turns audio into text. We use **Whisper**, an open-source ASR model.
- **GeMAPS** — a standard list of acoustic measurements of a voice. We extract it with **openSMILE**, a free audio-feature tool. The version we use (GeMAPSv01b, "Functionals") gives **62 numbers per clip**. (Earlier documents said 88; that was wrong for this feature set and has been corrected.)
- **RoBERTa** — a pretrained language model. We use it to turn a transcript into a list of 1,024 numbers called the **CLS embedding** (a numerical summary of the sentence).
- **Embedding / representation** — a list of numbers that summarizes something so a model can work with it.
- **Modal** — a cloud service where the backend is deployed.

## 2. Problem statement

Automated hiring tools are tuned on typical ("neurotypical") speaking styles. Published work (cited in the
README) reports that such tools can weight delivery heavily compared with content. A person whose delivery
is systematically different — for instance flatter pitch or longer pauses — can then be scored lower for
how they spoke rather than what they said. This project treats that as a **measurement problem**: build a
content score that does not depend on delivery, and test that claim with statistics.

**This project makes no claim about detecting or diagnosing autism or any other condition.**

## 3. Scope

### In scope (current milestone: working demo + measurable evaluation)

- A **content scorer** trained on ChaLearn First Impressions V2 transcripts.
- A **v2 two-stage model**: a frozen content path **C**, and a prosody path **D** trained to reconstruct the voice measurements while being statistically independent of C.
- **Independence tests**: distance correlation and cross-validated ridge R² in both directions (explained in section 4).
- A **prosody-only baseline** — an illustrative comparison model, not an evaluator or vendor tool.
- **Evaluation** on 500 synthetic pairs and on **LOSO** (leave-one-speaker-out: hold one real speaker out, train on the rest, test on the held-out speaker; repeated for each of the 8 speakers) over 355 real clips from speakers `S001..S008` (called `speaker_01..08` in the notebook).
- A **WER** spot check (word error rate: the share of words the ASR gets wrong) and an optional Vaani prosody add-on.
- Backend serving v2, the rehearsal frontend working end to end, Modal deploy, docs kept current.

### Out of scope (future work, see section 11)

Intent classification, Llama 3 LoRA reasoning, HuBERT SLU branch, emotion2vec, Whisper fine-tuning,
the live overlay and `/stream`, the evaluator portal as a working product, NDAR/ADOS data, any
autism-detection claim.

## 4. Architecture v2 in plain words

```
transcript --RoBERTa--> CLS embedding [1024] --ContentBranch--> C [256] --content_head--> content score
voice audio --openSMILE--> GeMAPS [62] --(z-scored)--> ProsodyBranch --> D [256] --recon_head--> GeMAPS again
```

- **C** is the 256-number output of the `ContentBranch` network (decision 1): the layer just before the final score in the content scorer. The chain is RoBERTa CLS (1024) → `ContentBranch` (256 = **C**) → `content_head` `Linear(256, 1)` → content score. C is **frozen** (never changed) once Stage 1 ends.
- **Stage 1** trains the content scorer on ChaLearn transcripts and their labels.
- **Stage 2** trains **D**, a 256-number summary of the voice, with two goals: (a) it must be able to rebuild the original voice measurements (so it keeps real delivery information), and (b) it must be **statistically independent of C**, enforced by a penalty computed over each training batch. The `recon_head` network that does the rebuilding is used for training only.
- **Checkpoint** `autiz_v2.pt` (planned): a file with the weights of `content_branch`, `content_head`, `prosody_branch`, `recon_head`, plus a `config`.
- The **z-score scaler** for the voice measurements is fit on original (unmodified) LibriSpeech clips from the training split only.

**Important honest caveat.** The content score is computed from the transcript only; it never sees the
audio. So it **cannot** change when delivery changes — that is true by construction, not a result of training.
A "100 % gap reduction" on synthetic pairs would therefore prove nothing. The informative results are:

1. **Independence tests** between C and D:
   - **Distance correlation** — a statistic that is zero only if two sets of numbers are completely unrelated, including in non-linear ways. We use a bias-corrected version.
   - **Cross-validated ridge R²** — try to predict D from C (and C from D) with a simple regularized linear model on held-out data. An R² near or below zero means no linear predictability.
2. **Does D still carry real delivery information** (reconstruction quality, ability to tell original from flattened clips)?
3. **The real-speaker (LOSO) evaluation**, which uses real recordings the model was never trained on.

### Exact v2 design (as written in `notebooks/autiz_v2.ipynb`; not yet run)

**Stage 1 — content scorer (C).** Frozen RoBERTa-large (no fine-tuning) turns each transcript into a 1,024-number CLS embedding, computed once per text set and cached as `.npy`. Only `ContentBranch` (1024 → 256 = **C**) and `content_head` (`Linear(256, 1)`) are trained, on the 6,000 ChaLearn training transcripts against the `interview` label only, with weight decay and early stopping. The split is by **YouTube video id** (80/20, fixed seed, zero id overlap asserted), never by clip; early stopping uses a further 10 % of the training-side videos, so the reported Pearson r on the 20 % held-out videos is untouched. C is frozen afterwards. If `CONTENT_LABEL_SOURCE = "proxy"`, Stage 1 is skipped and every output is stamped "PROXY, not a trained scorer".

**Stage 2 — delivery representation (D).** `ProsodyBranch` (62 z-scored GeMAPS → 256 = **D**) is trained with (i) reconstruction MSE of the 62 z-scored features from D through a `recon_head`, plus (ii) a **batch-level independence penalty** between D and the frozen C: the mean squared cross-correlation of their dimensions (distance correlation is available as an alternative). C for a synthetic pair comes from the pair's ground-truth LibriSpeech transcript. A weight λ on the penalty is swept over {0, 0.1, 1, 10, 100}; λ = 0 is a no-penalty ablation and is never selected. Removed from v1: the wrong-sign prosody loss `-cosine(D_orig, D_flat)`, the always-zero content-MSE term, and the intent head with its labels.

**Data and splits.** 500 synthetic pairs (original + pitch-flattened, 15 % slower copy) are regenerated from LibriSpeech test-clean, split by **speaker** 80/20 (a further 20 % of the training speakers is the validation set used for λ selection and early stopping). The GeMAPS scaler is fit on original clips of the training split only and saved to `gemaps_scaler.json`.

**Independence tests (held-out speakers).** Bias-corrected distance correlation between C and D, and grouped cross-validated ridge R² in both directions (C → D and D → C). Each is compared with a baseline made by shuffling the C–D pairing 200 times, and the notebook reports where the real value falls among the 200. Plain-language fields in `eval_results_independence.json` explain each number.

**Prosody-only baseline.** Ridge regression from the z-scored features to a hand-built arousal proxy; labelled "illustrative comparison model, not an evaluator or vendor tool". Synthetic evaluation compares its gap between original and flattened copies with our content gap, which is **zero by construction** (stated in the JSON).

**Real speakers (355 clips, 8 folds).** Each clip is transcribed with faster-whisper (`base`, same as `src/api/pipeline.py`), embedded with frozen RoBERTa and passed through the frozen `ContentBranch`; the content path never trains on real clips. With `LOSO_RETRAIN=True` the `ProsodyBranch` is retrained per held-out speaker on LibriSpeech-train plus the other 7 speakers. Per fold: clip count, prosody distance (z-scored against the LibriSpeech scaler) and within-speaker variance; independence metrics only for folds with at least 10 clips (S007 has 3 and is reported as null, "n < 10"). A **pooled out-of-fold** test collects (C, D) for all clips, each D from the model that did not train on that clip's speaker. Aggregates are given unweighted (mean, std across folds) and clip-weighted, because S001 and S008 make up 62 % of the clips.

**SHAP** explains the prosody-only baseline only. If SHAP fails, a labelled coefficient-magnitude proxy is recorded instead and is never called SHAP.

The model code lives in `src/api/autiz_model.py` (single source of truth); the notebook holds an identical copy in a cell marked `KEEP IN SYNC`.

## 5. Phase checklist

| Phase | What | Exit criteria | Status | Updated |
|---|---|---|---|---|
| 0 | Docs baseline, remove fake-result fallback, CORS, small API/frontend fixes | Docs match the repo; no random scores anywhere in the rehearsal app; `npm run build` passes for all three apps | **Done** — committed as `62910a9`; `src/api/test.wav` is untracked (the file stays on disk and is git-ignored) | 2026-10-06 |
| 1 | Notebook v2 edits (not executed) | `notebooks/autiz_v2.ipynb` and `src/api/autiz_model.py` written; JSON valid and every code cell parses; statistics and loaders checked on fake data; nothing executed on Kaggle | **In progress** — notebook written, awaiting review before Phase 2 | 2026-10-06 |
| 2 | Run the notebook once on Kaggle; ingest and validate artifacts | All artifacts present and validated; real numbers in these docs | Not started | — |
| 3 | WER spot check; Vaani prosody add-on (optional) | WER reported with sample size; add-on optional | Not started | — |
| 4 | Backend serves v2 | Loads `autiz_v2.pt`; honest response schema; tests pass; temp-file cleanup in `finally`; `/stream` disabled | Not started | — |
| 5 | Rehearsal frontend end to end | Real error states; contract matches backend; placeholder delete/label UI removed; evaluator and overlay labelled prototype | Not started | — |
| 6 | Modal redeploy and live test | One account; unified `.env`; end-to-end run recorded; demo script | Not started | — |
| 7 | Final docs | Results, limitations, future work, reproduction steps | Not started | — |

## 6. What is real vs stub today

Checked 2026-10-05 by reading the code (tests were not run).

| Part | Real or stub | Detail |
|---|---|---|
| `POST /analyze`, `POST /score` | Real inference, with weak parts | Run Whisper **base**, openSMILE GeMAPS, RoBERTa, and the **v1** fusion checkpoint. |
| Content score in responses | **Proxy, not a trained scorer** | `pipeline.py` uses the RoBERTa CLS vector's length divided by 40. |
| `score_without_system` | **Invented formula** | Content score times a fixed penalty (0.35 or 0.12). Not measured from any evaluator. |
| `interpretation` text | **Template string** | Not model output. |
| `intent_label` | **Meaningless** | The v1 model was never trained with intent labels, so its classifier head is untrained. |
| `audit_trail` | **Proxy** | Size of vectors, not SHAP (an explanation method for model outputs). |
| `mode` | **No effect** | The field is read as a URL query parameter, not from the form body, and `pipeline.run` ignores it. |
| `POST /stream` (WebSocket) | **Stub** | Still imports `mock_pipeline.py`, which returns constants; also loads a speech-detector model at import time (network call). |
| `run_llama_reasoning` in `modal_app.py` | **Stub** | Returns a hardcoded dict; never called. |
| v1 checkpoint (`fusion_layer_trained_v1.pt`) | Exists | Trained with a loss that does not give orthogonal vectors (commit `2f15a3d`, notebook comments). Not re-loaded for this document. |
| `notebooks/autiz_v2.ipynb` | Written, **never executed** | Replaces `training_v1.ipynb` (renamed with `git mv`; the v1 version, with its stale outputs, stays in git history). No saved outputs. First run is Phase 2. |
| `src/api/autiz_model.py` | New, not yet used by the backend | Model classes for v2; `pipeline.py` still loads v1 until Phase 4. |
| Rehearsal app | Real after Phase 0 | Failures show errors and a retry button; never invented scores. Still has the self-label step and a "Delete my data" button that does not delete anything (fixed in Phase 5). |
| Evaluator app | Prototype | Posts JSON to `/score`, which expects an audio upload; clip audio is `null`; password hardcoded in the page code. |
| Overlay app | Prototype | Never sends audio to `/stream`. |
| `mock-server.js` | Dev mock | Random values; labelled. |
| `scripts/Preprocessing_Pipeline.py` | Skeleton | Every functional part raises `NotImplementedError`; nothing uses it. |
| `src/layer1`, `layer2`, `layer3`, `models/adapters`, `data`, `tests` (repo root) | Empty directories | |
| Backend tests (`src/api/tests/test_api.py`) | 8 tests, **not run** | They call the real pipeline, so they would load large models. |
| CORS | Added in Phase 0 | `src/api/main.py`; origins from `CORS_ALLOWED_ORIGINS` (default: the three local Vite dev ports). |

## 7. Datasets

Access status for requests is **as written in the original README and unverified** from the repository.

| Dataset | Role | Status |
|---|---|---|
| LibriSpeech test-clean | Source of clean speech; used to make **synthetic pairs** (the same clip with pitch flattened and slowed using Praat/parselmouth); its `*.trans.txt` ground-truth transcripts supply the pair transcripts | **Used.** Taken from `<chalearn dataset>/LibriSpeech/test-clean` if present, otherwise downloaded. |
| Real clips, speakers S001–S008 (355 clips) | Real-speaker evaluation | **Used** via the private Kaggle dataset `autiz-real-asd-clips` (not in this repo). Layout verified by the project lead on Kaggle: `wav/S001 … wav/S008`, `.wav` only; counts S001 110, S002 22, S003 39, S004 35, S005 15, S006 21, S007 3, S008 110. S001 and S008 are 62 % of the clips and S007 has 3, so the folds are very uneven. Per the project lead the speakers are from YouTube; provenance and licensing are not documented here. |
| ChaLearn First Impressions V2 | Training the content scorer (transcripts + human-impression labels) | **Used in Stage 1** via the private Kaggle dataset `autiz-chalearn-fi-v2`. Facts verified by the project lead on Kaggle (not re-checked from this repo): only `train-annotation/annotation_training.pkl` (dict of dicts with keys extraversion, neuroticism, agreeableness, conscientiousness, interview, openness; each maps a clip name such as `J4GQm9j0JZ0.003.mp4` to a float in [0, 1]; 6,000 entries) and `train-transcription/transcription_training.pkl` (the same 6,000 names → transcript) are used, and only `interview` is the label. The YouTube video id is the part of the name before the first dot; several clips share a video id, so splits are by video id. The dataset also holds stale folders from an earlier notebook (`LibriSpeech/`, `test-clean/LibriSpeech/`, `synthetic_pairs/`, `test_clip*.wav`, `__huggingface_repos__.json`); `synthetic_pairs/` came from the flawed v1 pipeline and is ignored, all 500 pairs are regenerated. Licence not documented here. |
| Vaani | Optional prosody add-on | **Next (optional)**, not started |
| Mozilla Common Voice, MSP-Podcast, MuSe, CMU-MOSI/MOSEI, IEMOCAP, SEMAINE | Original plan | **Parked** (README lists access requests for MSP-Podcast and IEMOCAP as submitted) |
| NDAR (NIH) | Adult ASD speech | **Parked** (README: application submitted); out of scope |

## 8. Known limitations

- **Only 8 real speakers**, all from YouTube. Any real-speaker result is anecdotal-scale and may not generalize. Whether those recordings may be used for this purpose is **unverified**.
- **No intent labels** exist anywhere in the project, so no intent model can be trained or evaluated.
- **ChaLearn labels are human impressions of video.** They will be used as a stand-in for "content quality of the words alone". That is a weak proxy: the raters saw faces and heard voices, and the labels are not about answer quality.
- **Whisper base**, not the large-v3 model the original specification names, is what the code runs; no fine-tuning has been done.
- Synthetic "ASD-style" pairs are produced by an automatic pitch-flatten and slow-down. They are a stress test, not real data, and LibriSpeech speakers reading different books may link text and voice in ways the independence tests can pick up.
- The content score is blind to audio by construction (section 4), so its invariance is not evidence of training success.
- The v1 model and its numbers should not be cited as results.
- Backend is deployed on a third party's Modal account; its CORS behaviour and weights-download behaviour are **unverified**.

### Assumptions made while writing the v2 notebook (unverified until the Phase 2 run)

The first one was flagged by the project lead; the rest are mine and are listed so they can be challenged.

1. **RoBERTa-large is a frozen feature extractor** in this milestone (no fine-tuning); CLS embeddings are precomputed once and cached. *(Flagged open assumption.)*
2. Stage 1 early stopping uses an extra early-stopping split (10 % of the training-side videos) inside the 80 % train side, so the reported r on the 20 % held-out videos is not used for model selection. Hyperparameters (AdamW, lr 1e-3, weight decay 1e-2, batch 128, patience 20, MSE, raw un-normalised CLS input) are untuned defaults.
3. Synthetic speakers are split 80/20, and 20 % of the training speakers form a validation set for λ selection and early stopping. "Training split" for the scaler, baseline and LOSO retraining means train + validation speakers (the 80 %). The selection score `(1 − validation reconstruction R²) + validation distance correlation` is my own arbitrary choice.
4. The scaler also clips z-scores at ±10 (stored in `gemaps_scaler.json` as `z_clip`) and floors tiny standard deviations at 1; decision 4 only specified mean and std. **The backend must apply the same clip in Phase 4.**
5. LibriSpeech ground-truth transcripts are lower-cased then sentence-cased; ChaLearn transcripts are used as they are and real-clip transcripts are raw Whisper output, so the three text sources differ in style. Candidate LibriSpeech clips are shorter than 10 s and shuffled with the fixed seed; failed renders are replaced so exactly 500 pairs exist.
6. Real clips are always converted to 16 kHz mono with ffmpeg before feature extraction and transcription, even though they are `.wav`. Whisper `base`, float16, beam size 3 is hard-coded to match `src/api/pipeline.py` (not imported).
7. Real clips with an empty transcript are excluded from evaluation and counted per speaker in the JSON.
8. Independence is tested separately for original and flattened synthetic clips (they share the same C). "Verified" means all six permutation p-values are at least 0.05; failing to detect a link is not proof of independence.
9. Per-fold ridge uses train-to-held-out transfer (a single speaker cannot be grouped). The pooled out-of-fold test mixes D from eight separately trained models (same initial seed, similar but not identical spaces), which can bias pooled ridge R² towards zero; this is stated in the JSON.
10. LOSO folds train for a fixed `max(10, main best epoch)` epochs because there is no validation set inside a fold.
11. The arousal proxy for the baseline is the mean of three z-scored GeMAPS features (pitch variability, loudness, voiced segments per second), so a ridge on all 62 features recovers it easily; SHAP on it is an illustration only.
12. Extra output `eval_results_content.json` (Stage 1 metrics) is not in the original artifact list. `transcripts_real.csv` contains speech content of real speakers and must **not** be committed; clip ids are anonymised (`speaker_01_clip_000`, in sorted-filename order).
13. The preflight requires the ChaLearn dataset even if `CONTENT_LABEL_SOURCE = "proxy"`; the proxy mode keeps a randomly initialised, frozen ContentBranch and stamps outputs. It also fails if any per-speaker clip count differs from the numbers above.
14. The ChaLearn pickles are loaded with `pickle`, which is only safe for trusted files (these are your private dataset).

## 9. How to run and reproduce

Commands below were derived from the repository files; in this session only `npm run build` (three apps) and the CORS behaviour were actually executed.

**Backend (local).** From `src/api/`:
```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```
It loads large models on the first request. The `/stream` router downloads a speech-detector model when the app starts (removed in Phase 4). Optional: set `CORS_ALLOWED_ORIGINS` to a comma-separated list of origins; the default is `http://localhost:5173,5174,5175`.

**Frontends.** Each of `src/frontend/rehearsal`, `evaluator`, `overlay`:
```bash
npm install
npm run dev
```
They read `VITE_BACKEND_URL` from a `.env` file at the **repository root** (git-ignored; there is no `.env.example` yet — added in Phase 6). Without it they fall back to `http://localhost:8000`.

**Production build check.** `npm run build` in each app (passed on 2026-10-05).

**Modal deploy.** `modal deploy src/api/modal_app.py` (needs a Modal account; will be redone in Phase 6).

**Backend tests.** `pytest` from `src/api/` — currently loads real models and needs network access (fixed in Phase 4).

**Training.** `notebooks/autiz_v2.ipynb` is written but has not been run. It is meant to run once, top to bottom, on a Kaggle GPU (Internet on) with the two private datasets attached; its first cells are a preflight and a self-test that stop the run early if anything is wrong. Full reproduction steps will be written in Phase 7.

**Testing the error state in the rehearsal app.** Point the app at a backend that is not running, then record and submit answers:
```bash
cd src/frontend/rehearsal
VITE_BACKEND_URL=http://localhost:9 npm run dev
```
Expected: each answer shows a red "Error" row with a reason, and "Retry failed answers". Choosing to view the report shows an error card per failed answer with a "Retry this answer" button; no scores appear. (A process environment variable overrides the `.env` file in Vite.) To test a real backend failure instead, run the backend and stop it between answers.

## 10. How to update this file after every phase

1. Change the phase's row in section 5: status, exit criteria met or not, and date.
2. Re-check section 6 against the code; anything that changed from stub to real (or the reverse) must be edited.
3. Add new numbers only from files in `src/eval/results/`; quote the file name.
4. Move anything newly discovered that is wrong or unverified into section 8.
5. Update the "Last updated" line at the top.
6. Do not describe something as done unless it was run or read in that phase.

## 11. Future work

- **Gap-data collection.** The rehearsal app's speaker self-labels and the evaluator portal's labels are the intended source of the "ASD Intent Gap Dataset" (`configs/Dataset_Schema.json`): what the speaker meant versus how evaluators read it. In this milestone the backend **stores nothing** (no database, no stored audio, no persisted labels). Collecting this needs consent handling, storage and deletion design first.
- Intent classification (needs labelled data), Llama 3 LoRA reasoning, HuBERT SLU branch, emotion2vec, Whisper fine-tuning, live overlay and streaming, a real evaluator portal with proper login, NDAR/ADOS-based adaptation, SHAP-based audit trails, fairness audit (statistical parity), choosing a licence.
