# Autiz (NeuroIntent) — Project Status

_This is the single living status document. Last updated: 2026-10-05, after Phase 0._
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

## 5. Phase checklist

| Phase | What | Exit criteria | Status | Updated |
|---|---|---|---|---|
| 0 | Docs baseline, remove fake-result fallback, CORS, small API/frontend fixes | Docs match the repo; no random scores anywhere in the rehearsal app; `npm run build` passes for all three apps | **Done**, except untracking `src/api/test.wav` (needs the maintainer to run `git rm --cached src/api/test.wav`) | 2026-10-05 |
| 1 | Notebook v2 edits (not executed) | `notebooks/autiz_v2.ipynb` and `src/api/autiz_model.py` written; code paths checked on fabricated data | Not started | — |
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
| `notebooks/training_v1.ipynb` | Exists, partly stale | Saved output shows 100 pairs and cos(C,D) ≈ −0.999; the code afterwards was edited to 500 pairs and the corrected loss but those cells were never re-run. Cells after training have no saved output. |
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
| LibriSpeech test-clean | Source of clean speech; used to make **synthetic pairs** (the same clip with pitch flattened and slowed using Praat/parselmouth) | **Used** (public, downloaded inside the notebook) |
| Real clips, speakers S001–S008 (355 clips) | Real-speaker evaluation | **Used** via a Kaggle dataset `autiz-real-asd-clips` (not in this repo). Per the project lead the speakers are from YouTube. Provenance and licensing are not documented here. |
| ChaLearn First Impressions V2 | Training the content scorer (transcripts + human-impression labels) | **Next.** Availability of transcripts, file layout and licence **unverified**. |
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

**Training.** Only the v1 notebook exists (`notebooks/training_v1.ipynb`, run on Kaggle with a T4 GPU). Reproduction steps for v2 will be written in Phase 7.

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
