# Autiz (NeuroIntent) — Project Status

_This is the single living status document. Last updated: 2026-10-07. Phase 1, 1b and 1c done; smoke run passed at commit `887a06a`; Phase 4 done (backend); Phase 5 in progress (rehearsal frontend). The full Kaggle run (Phase 2) is still pending._
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
| 1 | Notebook v2 edits | `notebooks/autiz_v2.ipynb` and `src/api/autiz_model.py` written; JSON valid and every code cell parses; statistics and loaders checked on fake data | **Done** (commit `e33a1df`) | 2026-10-07 |
| 1b | Notebook smoke mode and static review | `SMOKE = True` runs every code path cheaply (20 pairs, ~300 ChaLearn clips from ≥100 videos, 3 real clips per speaker, 2 epochs, 2 λ values, 5 permutations, all 8 LOSO folds, real Whisper and RoBERTa) and writes only to `/kaggle/working/smoke_outputs/`, every file stamped "SMOKE RUN, NOT RESULTS"; `SMOKE = False` behaves as before | **Done** (commit `887a06a`); the first smoke run hit the Whisper/PyAV problem (issue 11) | 2026-10-07 |
| 1c | Fix Whisper/PyAV incompatibility and re-validate | Cell 2 monkeypatches `av.open` to ignore `metadata_errors` (an upgrade of `av` was not possible on Kaggle); smoke run passes | **Done** (commit `45862cc`); smoke run passed on tiny data. The full run is still pending (Phase 2) | 2026-10-07 |
| 2 | Run the notebook once on Kaggle; ingest and validate artifacts | All artifacts present and validated; real numbers in these docs | Not started | — |
| 3 | WER spot check; Vaani prosody add-on (optional) | WER reported with sample size; add-on optional | Not started | — |
| 4 | Backend serves v2 | Loads `autiz_v2.pt`, `gemaps_scaler.json`, `prosody_baseline.joblib` (lazily, with clear errors; smoke-stamped artifacts refused unless `ALLOW_SMOKE_ARTIFACTS=1`); honest response schema (section 6a); tests pass offline without the checkpoint; temp-file cleanup in `finally`; `/stream` removed; no storage | **Done** (not committed yet) — backend code and offline tests finished: 57 passed, 2 skipped with torch; 31 passed, 1 skipped without it. Hand tests 7 and 8 (real run against a smoke checkpoint, and then the full checkpoint) are still pending the Kaggle full run, which is Phase 2 | 2026-10-07 |
| 5 | Rehearsal frontend end to end | Real error states; contract matches backend; placeholder delete/label UI removed; evaluator and overlay labelled prototype | **In progress** — see the Frontend status table in section 6. The success screen cannot be seen against a real model until a checkpoint exists | 2026-10-07 |
| 6 | Modal redeploy and live test | One account; unified `.env`; end-to-end run recorded; demo script | Not started | — |
| 7 | Final docs | Results, limitations, future work, reproduction steps | Not started | — |

## 6. What is real vs stub today

Backend rows re-checked 2026-10-07 (Phase 4) by reading the code and running the backend tests; other rows were checked 2026-10-05 by reading the code.

| Part | Real or stub | Detail |
|---|---|---|
| `POST /analyze`, `POST /score` | Real code, **never run against a trained checkpoint** | Whisper **base**, openSMILE GeMAPS, RoBERTa-large, then the v2 content head and the prosody-only baseline from `models/checkpoints/` (or `AUTIZ_MODELS_DIR`). The checkpoint does not exist until Phase 2, so until then these endpoints return 503 (tested). Tested with stubbed encoders and small fixture checkpoints only. Response: section 6a. |
| Content score in responses | From the trained content head once a checkpoint exists | `content_score` (clipped to 0..1) and `content_score_raw`; labelled `"proxy"` with the stamp "PROXY, not a trained scorer" if the checkpoint says Stage 1 was skipped. The old CLS-length formula is gone. |
| `score_without_system` | **Removed** | Was an invented formula (fixed penalty 0.35 or 0.12). |
| `interpretation` text | **Template string** | Not model output; the response says so (`interpretation_source: "template"`). |
| `intent_label` | **Always `null`** | No intent labels exist; `intent_status` is `"not_trained"`. The v1 classifier was removed. |
| `explanation` (was `audit_trail`) | **Labelled proxy** | Coefficient × z-score of the prosody-only baseline; explains the baseline only. SHAP is computed offline in the notebook, not per request. |
| `mode` | **No effect** | Now a required form field, echoed back with `mode_effect: "none in this milestone"`; `pipeline.run` ignores it. |
| `POST /stream` (WebSocket) | **Removed** | Phase 4 deleted the router and its import-time `torch.hub` download. The overlay app has nothing to connect to. |
| `run_llama_reasoning` in `modal_app.py` | **Stub** | Returns a hardcoded dict; never called. |
| v1 checkpoint (`fusion_layer_trained_v1.pt`) | Exists, **no longer loaded** | Trained with a loss that does not give orthogonal vectors (commit `2f15a3d`, notebook comments). The backend does not use it. |
| `notebooks/autiz_v2.ipynb` | Written, **never executed** | Replaces `training_v1.ipynb` (renamed with `git mv`; the v1 version, with its stale outputs, stays in git history). No saved outputs. First run is Phase 2. |
| `src/api/autiz_model.py` | Used by the backend (Phase 4) | `pipeline.py` imports `load_autiz_v2`, `load_scaler`, `apply_scaler` from it; the shared block was not edited. |
| Frontends | See "Frontend status" below | |
| `mock-server.js` | Dev mock, **does not match v2** | Emits the old `/stream` payload with random values; `/stream` no longer exists. Not edited (issue 28). |
| `scripts/Preprocessing_Pipeline.py` | Skeleton | Every functional part raises `NotImplementedError`; nothing uses it. |
| `src/layer1`, `layer2`, `layer3`, `models/adapters`, `data`, `tests` (repo root) | Empty directories | |
| Backend tests (`src/api/tests/`) | Offline; see the Phase 4 result in the report | `pipeline.run` is stubbed (no RoBERTa/Whisper); artifact tests build tiny checkpoints inside the test and are skipped when torch is missing. The test with a real smoke checkpoint is skipped until files are placed in `models/smoke/`. `mock_pipeline.py` now lives in `src/api/tests/` as a stub. |
| CORS | Added in Phase 0 | `src/api/main.py`; origins from `CORS_ALLOWED_ORIGINS` (default: the three local Vite dev ports). |

### Frontend status

| App | Status | What it does and what it does not |
|---|---|---|
| Rehearsal (`src/frontend/rehearsal`) | **Working prototype, Phase 5 in progress.** Updated to the v2 contract; the success screen has not been seen against a real model | Records five answers, sends each to `POST /analyze` (audio and `mode` as form fields) through `src/api.js`, and shows per-answer results or an error card with retry. Nothing is stored; there is no self-label step and no delete button. Not exercised in a browser yet (hand tests below in the Phase 5 report) |
| Evaluator (`src/frontend/evaluator`) | **Prototype, not connected to live data** | Labelled with a banner. Clip audio is `null`, the password is hard-coded in the page, and labels are not sent anywhere (the old JSON call to `/score` was removed) |
| Overlay (`src/frontend/overlay`) | **Prototype, not connected to live data** | Labelled with a banner. It opens a WebSocket to `/stream`, which no longer exists, so it can only show a connection error |

## 6a. API contract (Phase 4)

The backend stores nothing: no database, no stored audio, no persisted labels. Uploaded audio is written to a
temporary file with a generated name and deleted in a `finally` block whether the request succeeds or fails.
There is no delete endpoint because there is nothing to delete.

The machine-readable copy is the Pydantic models in `src/api/schemas.py`; a real-shaped example is
`docs/api_example_response.json`.

**Configuration (environment variables).**

| Variable | Default | Meaning |
|---|---|---|
| `AUTIZ_MODELS_DIR` | `models/checkpoints` (or `/root/models/checkpoints` on Modal) | Folder holding `autiz_v2.pt`, `gemaps_scaler.json`, `prosody_baseline.joblib` |
| `AUTIZ_RESULTS_DIR` | `src/eval/results` | Folder holding `eval_results_independence.json` (read only by `GET /about`) |
| `ALLOW_SMOKE_ARTIFACTS` | unset | Set to `1` to allow loading files stamped "SMOKE RUN, NOT RESULTS". Without it the backend refuses them |
| `CORS_ALLOWED_ORIGINS` | the three local Vite ports | Comma-separated browser origins |

**Smoke stamp (as the notebook writes it).** Every JSON (including `gemaps_scaler.json`) has a top-level key
`run_stamp`; the checkpoint has `config["run_stamp"]`; the joblib file is a dict with a `run_stamp` key; CSV files
start with the line `# SMOKE RUN, NOT RESULTS`. The value is `"SMOKE RUN, NOT RESULTS"`. Full-run files have no
`run_stamp` key at all. The backend treats the whole artifact set as smoke if any file it loads carries a stamp.
When smoke artifacts are allowed, every response carries `"smoke_artifacts": true`.

### `POST /analyze` (multipart form)

Form fields: `audio` (file, required) and `mode` (text, required: `universal_fairness` or `speaker_declared`;
anything else is HTTP 400). `mode` is a form field, not a query parameter. There is **no** `audio_storage_consent`.

| Field | Meaning in plain language |
|---|---|
| `content_score` | The content score, **clipped to the range 0 to 1**. Computed from the transcript only; the audio never enters it. |
| `content_score_raw` | The same score **before clipping**. The scoring layer is an unbounded linear layer, so it can fall below 0 or above 1; this shows when clipping changed the number. |
| `content_score_source` | `"trained_head"` when the content scorer was trained on ChaLearn labels; `"proxy"` when the checkpoint says Stage 1 was skipped. |
| `content_scorer_stamp` | `null`, or the text `"PROXY, not a trained scorer"` copied from the checkpoint when `content_score_source` is `"proxy"`. |
| `content_label_source` | Where the content scorer's training labels came from, as recorded in the checkpoint (`"chalearn"` or `"proxy"`). ChaLearn labels are human impressions of video, not ratings of answer quality. |
| `intent_label` | Always `null`. No intent model exists. |
| `intent_status` | Always `"not_trained"`. |
| `prosody_only_baseline_score` | Output of a simple linear model that looks at the voice measurements only. It predicts a hand-built "arousal proxy" in standard-deviation units of the training data (it is not limited to 0 to 1). |
| `prosody_only_baseline_label` | Always "illustrative comparison model, not an evaluator or vendor tool". It is not any real hiring system. |
| `transcript` | What Whisper heard. If no speech is found the request fails with HTTP 422 instead of scoring an invented transcript. |
| `acoustic_observations` | Measured values from the audio, no judgement: `pitch_variation_stddev_norm` (normalised pitch spread, unitless), `mean_unvoiced_segment_sec`, `voiced_segments_per_sec`. Any can be `null`. |
| `delivery_pattern` | The same measured values written as short sentences. No thresholds or verdicts. |
| `explanation` | Why the baseline gave its number: `method` is `"shap"` or `"proxy"` (always `"proxy"` today: coefficient × z-score, which for a linear model is the linear SHAP value relative to the training mean, but SHAP itself is not run per request); `explains` is `"prosody_only_baseline_score"` (the content score has **no** explanation); `top_features` lists the 5 features with the largest absolute contribution; `note` says this in words. |
| `interpretation` | One short paragraph. |
| `interpretation_source` | Always `"template"`: the paragraph is filled in from a template, it is not model output. |
| `mode` | The `mode` you sent, echoed back. |
| `mode_effect` | Always `"none in this milestone"`: both modes run the identical pipeline. |
| `session` | `speaker_id` (random 12-character id made fresh for this request, not linked to a person), `system_stage` (`"research_pilot"`), `candidate_disclosure_required` (`true`). |
| `smoke_artifacts` | `true` only when the loaded files are smoke files and `ALLOW_SMOKE_ARTIFACTS=1`; otherwise `false`. Smoke numbers mean nothing. |

Removed from the old response (they were invented or meaningless): `score_without_system`, `score_with_system`,
`content_quality_score`, `intent_confidence`, `confidence_bound`, `prosody_decoupling_applied`,
`audit_trail` (including the per-request `decoupling_verified`), and `session.audio_storage_consent`.

### `POST /score` (multipart form)

Same form fields. `mode=speaker_declared` is refused with HTTP 403. Returns `content_score`, `content_score_raw`,
`content_score_source`, `content_scorer_stamp`, `mode` (always `"universal_fairness"`), `mode_effect`, `system_stage`,
`candidate_disclosure_required` and `smoke_artifacts`, with the meanings above.

### `GET /about`

Describes the model, not a request. Fields: `name`, `system_stage`, `stores_nothing` (`true`),
`content_label_source` (from the independence results file; `null` when that file is absent),
`content_label_note` (why ChaLearn labels are a weak proxy), `results_available`, `independence_results`
(the contents of `eval_results_independence.json`, or `null`), `results_note` (why results are missing, if they are),
`limitations` (list of plain sentences) and `smoke_artifacts`. A missing results file is **not** an error:
`results_available` is `false`. A results file carrying the smoke stamp is treated like a missing file (with the
reason in `results_note`) unless `ALLOW_SMOKE_ARTIFACTS=1`.

### `GET /health`

`status` (`"ok"`), `system_stage` and `smoke_artifacts`. Does not load any model.

### Errors

| Status | When |
|---|---|
| 400 | `mode` is not one of the two allowed values |
| 403 | `/score` with `mode=speaker_declared` |
| 422 | Missing form field, audio that cannot be decoded, or audio with no detectable speech |
| 503 | A required artifact file is missing or unreadable, or the artifacts are smoke-stamped and `ALLOW_SMOKE_ARTIFACTS` is not `1` (the message says which) |

`/stream` no longer exists; the live overlay is future work.

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

### Known issues

| ID | Phase found | What | Where | Impact | Status |
|---|---|---|---|---|---|
| 1 | 1 | Notebook torch cells untested until the Kaggle run | notebook cells 11-42 | Notebook cells 11-42 smoke-tested on Kaggle at commit 887a06a on tiny data; full run still pending | open (Phase 2) |
| 2 | 1 | Pooled out-of-fold ridge R² mixes D from 8 separately trained models, which biases it toward 0 | notebook section J; `eval_results_loso.json` | Pooled independence number can look better than it is | open; read per-fold results alongside it in Phase 2 |
| 3 | 1 | Independence is tested on LibriSpeech text and Whisper transcripts, while the content path was trained on ChaLearn text | notebook sections G and J | Independence may not carry over to the text the content scorer was trained on | open |
| 4 | 1 | Serving must apply the scaler z-score clip and std floor | `src/api/pipeline.py` | Un-clipped features would not match training | fixed in Phase 4; verified by `test_serving_applies_the_scaler_z_clip` (a raw z of 1000 is clipped to 10 and the baseline score equals the clipped sum). The std floor is already baked into the saved std; serving uses the shared `apply_scaler` |
| 5 | 1 | ChaLearn licence for redistributing trained weights is unverified | `models/checkpoints/autiz_v2.pt` (future) | Phase 6 ships weights to Modal; may not be allowed | open; check before Phase 6 |
| 6 | 1 | RoBERTa frozen is an unverified assumption | notebook cell 20; assumption 1 | Fine-tuning might do better; the checkpoint config records `roberta_frozen: true` | open |
| 7 | 1 | `transcripts_real.csv` must never be committed (speech of real people) | `src/eval/results/` | Privacy | workaround in Phase 4: `transcripts_real.csv` added to `.gitignore`; verified with `git check-ignore -v src/eval/results/transcripts_real.csv` |
| 8 | 1 | The ChaLearn Kaggle dataset still holds ~918 MB of stale folders | Kaggle dataset `autiz-chalearn-fi-v2` | Wasted space only | open (optional cleanup) |
| 9 | 0 | CORS only tested with stubbed routers; the retry UI is only built, not exercised in a browser | `src/api/main.py`; rehearsal app | A real browser may still hit a CORS or error-state problem | open. Phase 4 added a preflight test with a stubbed pipeline. Phase 5 added a typed `Network` error (covers CORS) and verified the error mapping with a throwaway Node script against a fake `fetch` (17 checks passed; the script is not committed). Not yet seen in a real browser (the in-app browser could not open localhost, and recording needs a microphone): closes with hand tests (a) to (c) in the Phase 5 report |
| 10 | 0 | Pre-existing lint errors in the rehearsal app: unused `consentData` prop, unused `err` (two files), a `useEffect` dependency warning; Phase 5 also found an unused `useEffect` import in `RecordingScreen.jsx` | `src/frontend/rehearsal` | Lint noise | fixed in Phase 5 for rehearsal; verified by `npm run lint` in `src/frontend/rehearsal` exiting clean (baseline before: 4 errors, 1 warning) and `npm run build` passing. The evaluator's one lint error (`err` unused) went away when its dead `/score` call was removed (lint clean, build passes). The overlay's error is a different one, issue 30 |
| 11 | 1b | Kaggle PyAV 19.0.1 lacks the `metadata_errors` argument in `av.open`, which faster-whisper passes | notebook cell 2 | Notebook monkeypatches it; the serving image may hit the same incompatibility | workaround (monkeypatch in cell 2; smoke run passed). Phase 6 must pin `av` and `faster-whisper` versions in the Modal image and test a cold start |
| 12 | 1b | The per-fold independence branch (folds with at least 10 clips) never ran in smoke, because every smoke fold had 3 clips | notebook section J | First real execution is the full run | open (Phase 2) |
| 13 | 1b | In smoke, `decoupling_verified` was True with only 4 test pairs | `eval_results_independence.json` | The boolean is not meaningful at small n | open; Phase 2 must judge from n, the permutation null and per-fold agreement. `/about` passes the boolean through from the file, so read it with the caveats in that file |
| 14 | 1b | Smoke baseline R² was 0.967; the arousal proxy is the mean of 3 of the 62 features (assumption 11), so a ridge on all 62 recovers it trivially (circular) | notebook section H | The baseline R² says nothing about real arousal | open; Phase 2 to check the proxy definition. The API labels the baseline an illustrative comparison model |
| 15 | 1c | Docs disagree on the code-cell count (28 vs 29) | `PHASE_1_HANDOFF.md`, `PHASE_1_SUMMARY.md`, notebook | Confusing | open; align with the notebook |
| 16 | 4 | The DECISIONS block named in the Phase 4 prompt was not in the message or the repo | Phase 4 prompt | Decision 1 was taken from section 4 of this file (C is the ContentBranch output; checkpoint keys `content_branch`, `content_head`, `prosody_branch`, `recon_head`, `config`) | workaround; project lead to confirm no other decision changes serving |
| 17 | 4 | Phase 6 must include `eval_results_independence.json` in the Modal image (or set `AUTIZ_RESULTS_DIR` to a copy) | `src/api/modal_app.py` | Modal ships only `src/api` and `models/`, so `/about` would report `results_available: false` | open (Phase 6) |
| 18 | 4 | `.gitignore` had a corrupted last line (UTF-16 bytes from a shell append), so `models/smoke/` was not actually ignored | `.gitignore` | Smoke artifacts could have been committed | fixed in Phase 4; verified with `git check-ignore -v models/smoke/autiz_v2.pt` and `file .gitignore` (UTF-8) |
| 19 | 4 | The prosody branch D is loaded and checked but no response field uses it | `src/api/pipeline.py` | Serving does not exercise independence; that claim rests on the offline results | open by design; `/about` is the only place independence appears |
| 20 | 4 | Per-request SHAP is not computed; `explanation.method` is always `proxy` (coefficient x z-score for the baseline) | `src/api/pipeline.py` | Explanation covers the baseline only; the content score has none | open (future work) |
| 21 | 4 | No file-size or duration limit on uploads | `src/api/routers/` | A very large upload could exhaust disk or time | open |
| 22 | 4 | `/health` and `/about` detect smoke artifacts from the scaler stamp only (no torch); `/analyze` and `/score` use all three files | `pipeline.smoke_artifacts_flag` | A set where only the checkpoint or joblib is stamped shows `smoke_artifacts: false` on `/health` until the first `/analyze` (which would refuse it anyway) | open; low risk, the notebook stamps all files together |
| 23 | 4 | The joblib baseline is a pickle, so loading it runs code and needs the same scikit-learn version as training | `src/api/pipeline.py`; `requirements.txt` | Only load files you made; a version mismatch can fail or warn | open; Phase 6 must pin scikit-learn to the Kaggle version. `modal_app.py` has its own package list that lacks scikit-learn and joblib |
| 24 | 4 | No-speech audio now returns HTTP 422 (before, a placeholder transcript was scored). Input is always converted to 16 kHz mono as in training (before, `.wav` was used as is) | `src/api/pipeline.py` | Behaviour change for the frontends | fixed in Phase 4 by design; the no-speech path is tested (`test_no_speech_is_422_and_temp_wav_is_deleted`); the ffmpeg conversion is patched out in tests, so test it by hand |
| 25 | 4 | The rehearsal and overlay apps still use the old contract (`content_quality_score`, `score_without_system`, `confidence_bound`, `/stream`; the rehearsal app already sent `mode` as a form field, so the "query parameter" wording earlier was wrong for it) | `src/frontend/rehearsal/src/ReportScreen.jsx`; `src/frontend/overlay/src/OverlayCard.jsx` | The rehearsal app would show errors or blanks against the v2 backend | rehearsal: fixed in Phase 5 (reads `content_score`, `explanation`, `prosody_only_baseline_*`, `interpretation_source`, `mode_effect`, `smoke_artifacts`; every optional field is null-safe); verified by lint, build and the scripted `api.js` check, not yet in a browser (see issue 33). overlay: open by design, it still opens a WebSocket to the removed `/stream`; a banner says it is a prototype |
| 26 | 4 | `requirements.txt` and `modal_app.py` still list `silero-vad`, `websockets` and `torchaudio`, now unused | `src/api/requirements.txt`; `src/api/modal_app.py` | Larger image | open (Phase 6) |
| 27 | 4 | The real Whisper, RoBERTa, openSMILE and ffmpeg path in `pipeline.run` has not been executed; tests use fake encoders | `src/api/pipeline.py` | Possible integration bugs | open; first real run is the hand tests in the Phase 4 report |
| 28 | 5 | `mock-server.js` still emits the old `/stream` payload (random `content_quality_score`, `misread_flag`), which no longer matches anything in v2 | `mock-server.js` | Misleading dev tool; the overlay's only data source | open; not edited, as instructed |
| 29 | 5 | The consent wording was changed to match decision 1 (the old text said features are "stored for research" and that ASD consent gives "personalized interpretation") | `src/frontend/rehearsal/src/ConsentScreen.jsx` | Consent text is the project lead's call and may need review | open; project lead to review the wording |
| 30 | 5 | The overlay has a lint error: `setStatus('Off')` inside an effect (`react-hooks/set-state-in-effect`) | `src/frontend/overlay/src/OverlayCard.jsx:36` | `npm run lint` fails for the overlay; the build passes | open; not fixed, because the overlay is banner-only in Phase 5 (confirmed by `npm run lint`, same error before and after) |
| 31 | 5 | Evaluator and overlay still hold unused scaffold files (`App.css`, `src/assets/*`, `public/icons.svg`); the evaluator's "All done" screen still says labels were given "Thank you", the password is hard-coded, and its labels now go nowhere (the dead JSON call to `/score` was removed) | `src/frontend/evaluator`, `src/frontend/overlay` | Clutter; the evaluator is a labelled prototype | open; left alone, since only a banner was allowed there. The evaluator's submit button now reads "Not stored (prototype)" instead of "Saved!" |
| 32 | 5 | No frontend test runner exists, so the error mapping in `api.js` has no committed tests | `src/frontend/rehearsal` | Regressions in error handling would go unnoticed | open; no framework added, as instructed. Verified once by a throwaway Node script (17 checks) that stubs `fetch`; proposal: add vitest later |
| 33 | 5 | Recording (microphone), the success screen, and the withdraw and abort paths have not been exercised in a browser | rehearsal app | Possible runtime bugs the build and lint cannot catch | open; hand tests in the Phase 5 report. The success screen needs a real checkpoint (or a stub-backed dev run, proposed in the report) |

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
It loads the artifacts and the large models on the first `/analyze` or `/score` request (503 with a clear message until `autiz_v2.pt`, `gemaps_scaler.json` and `prosody_baseline.joblib` exist in `models/checkpoints/`, or in `AUTIZ_MODELS_DIR`). Smoke files are refused unless `ALLOW_SMOKE_ARTIFACTS=1`. Optional: set `CORS_ALLOWED_ORIGINS` to a comma-separated list of origins; the default is `http://localhost:5173,5174,5175`.

**Frontends.** Each of `src/frontend/rehearsal`, `evaluator`, `overlay`:
```bash
npm install
npm run dev
```
They read `VITE_BACKEND_URL` from a `.env` file at the **repository root** (git-ignored; there is no `.env.example` yet — added in Phase 6). Without it they fall back to `http://localhost:8000`.

**Production build check.** `npm run build` in each app (passed on 2026-10-05).

**Modal deploy.** `modal deploy src/api/modal_app.py` (needs a Modal account; will be redone in Phase 6).

**Backend tests.** `pytest` from `src/api/`. They run offline and load no RoBERTa/Whisper; with torch, scikit-learn and joblib installed the artifact tests also run, otherwise they are skipped.

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
