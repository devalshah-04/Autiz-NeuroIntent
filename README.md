# Autiz-Intent-Recognition-for-ASD-Candidates-in-Hiring

**Communication-style-invariant intent interpretation for ASD speakers in automated hiring pipelines.**

[![Status](https://img.shields.io/badge/status-research%20prototype-blue)](PROJECT_STATUS.md)
[![Python](https://img.shields.io/badge/python-3.11-blue)](.)
[![Phase](https://img.shields.io/badge/phase-v2%20in%20progress-orange)](PROJECT_STATUS.md)

---

## Problem Statement

Automated hiring systems — video interview platforms, ATS tools, AI scoring pipelines — are calibrated on neurotypical communication norms. Research from MIT Media Lab demonstrated empirically that these systems weight prosodic delivery (pitch variation, speaking rate, pause frequency) at disproportionately high levels relative to content quality when scoring candidates. For individuals with Autism Spectrum Disorder (ASD), whose communication patterns diverge systematically from neurotypical norms not due to competency gaps but due to neurological difference, this constitutes a documented and correctable algorithmic bias.

A candidate who delivers a technically precise, well-reasoned answer with flat prosody, extended processing pauses, or hyper-literal phrasing is penalized not for what they said, but for how they said it. NeuroIntent addresses this as a technical calibration problem, not a welfare intervention.

> **The system does not normalize ASD speakers toward neurotypical communication. It corrects the measurement instrument.**

---

## Approach

NeuroIntent is a three-layer multimodal interpretation pipeline that decouples speech content from speech delivery, then reasons about speaker intent independently of prosodic style.

```
Raw Speech
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — Speech Decomposition                         │
│  Whisper large-v3 · wav2vec 2.0 / HuBERT · GeMAPS 62   │
│  Parallel: Direct audio-to-intent branch (SLU)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2 — Cross-Modal Fusion and Decoupling            │
│  Cross-modal transformer · Cosine similarity            │
│  decoupling loss · RoBERTa content scorer               │
│  emotion2vec (ASD-adapted) · Llama 3 8B + LoRA         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3 — Output Surfaces                              │
│  Real-time evaluator overlay (~200ms latency)           │
│  Speaker reflection report · Platform API (JSON)        │
└─────────────────────────────────────────────────────────┘
```

> **Implementation status:** the diagram above is the long-term *target*. The current milestone builds only a content scorer, a prosody branch kept statistically independent of it, and the serving/evaluation around them. Everything else in the diagram (Llama 3, SLU branch, emotion2vec, real-time overlay, platform API) is **future work** — see [PROJECT_STATUS.md](PROJECT_STATUS.md).

The **novel contribution** is the cross-modal fusion and decoupling layer. Content representation **C** and delivery representation **D** are trained to be orthogonal via a cosine similarity penalty in the loss function. Same answer, different prosody → identical content score. This is the property that no existing hiring evaluation system has.

See [Architecture Specification.md](Architecture%20Specification.md) for the full target architecture, and [PROJECT_STATUS.md](PROJECT_STATUS.md) for what is actually implemented today.

---

## Novel Dataset Contribution

Existing ASD speech datasets share a fundamental limitation: they are annotated for diagnostic severity (ADOS-2 clinical scores), not for communicative intent. A recording labeled "ASD Level 2" provides no signal about whether the speaker was expressing confidence, uncertainty, enthusiasm, or something else entirely.

We are constructing the **ASD Intent Gap Dataset** — the first dataset (to our knowledge) that captures:

- ASD speaker self-labeled intent, recorded immediately after utterance
- Independent neurotypical evaluator interpretations of the same utterance
- The computed gap between speaker intent and evaluator interpretation

This gap vector is the primary training signal for the decoupling module and the primary evaluation metric for the system. A model that reduces this gap by predicting speaker intent correctly — against the systematic misinterpretation baseline established by neurotypical evaluators — is a model that works.

Dataset schema: [`configs/Dataset_Schema.json`](configs/Dataset_Schema.json)

---

## Tech Stack

| Component | Tool | Role | Status |
|---|---|---|---|
| ASR | Whisper (base model in use; large-v3 planned) | Transcription | Built (base, not fine-tuned) |
| Audio embeddings | wav2vec 2.0 / HuBERT / WavLM | Frame-level speech representations | Future work |
| Prosody features | openSMILE GeMAPSv01b (Functionals) | 62 interpretable acoustic features | Built |
| Direct SLU branch | HuBERT fine-tuned | Parallel audio-to-intent without ASR | Future work |
| Content scoring | RoBERTa-large embedding + small scorer network | Text-only content quality regression | In progress (v2) |
| Emotion encoding | emotion2vec (ASD-adapted) | Audio-side affective representation | Future work |
| Intent reasoning | Llama 3 8B + LoRA | Structured intent interpretation | Future work (stub only) |
| Explainability | SHAP (KernelExplainer) | Feature attribution for audit trail | Future work (magnitude proxy only) |
| Training framework | PyTorch + HuggingFace Transformers | | Built |
| Fine-tuning | PEFT / bitsandbytes / TRL | LoRA, 4-bit quantization | Future work |
| Serving | FastAPI + Modal.com | Inference API | Built (v1 model; v2 in Phase 4) |
| Real-time overlay | WebSocket `/stream` + overlay app | Evaluator-facing live card | Prototype, not connected |
| Evaluator portal | Evaluator app | Label collection | Prototype, not connected |

---

## Datasets

| Dataset | Role | Access |
|---|---|---|
| Mozilla Common Voice | Whisper ASR fine-tuning | Public |
| MSP-Podcast | Neurotypical prosody baseline | Request (UT Dallas) |
| CMU-MOSI / MOSEI | Content-prosody decoupling training | Public (SDK) |
| IEMOCAP | Multimodal architecture validation | Request (USC SAIL) |
| SEMAINE | Gap annotation schema reference | Public |
| MuSe Challenge | Prosody encoder pretraining | Public (registration) |
| NDAR (NIH) | Adult ASD speech | Application submitted |
| ASD Intent Gap Dataset | **Novel — intended primary training signal** | **Future work.** The rehearsal tool does not store anything in the current milestone. |

---

## Current Status

**Phase: research prototype — v2 model under construction.** Full, dated detail lives in [`PROJECT_STATUS.md`](PROJECT_STATUS.md); this section is only a summary.

What exists in the repository today:
- A FastAPI backend (`src/api/`) with `/analyze`, `/score` and `/health`, deployable on Modal.
- A rehearsal web app (`src/frontend/rehearsal/`) that records answers, sends them to `/analyze` and shows a per-answer report. If the backend fails, the app now shows an error, never placeholder scores.
- A Kaggle training notebook (`notebooks/`) and a v1 fusion-layer checkpoint (`models/checkpoints/fusion_layer_trained_v1.pt`). v1 was trained with a loss that does not produce orthogonal representations and is being **replaced by v2**; do not treat v1 numbers as results.
- Prototype evaluator and overlay apps that are **not connected** to a working backend path.

Not built (future work): intent classification, Llama 3 LoRA reasoning, HuBERT SLU branch, emotion2vec adaptation, Whisper fine-tuning, live overlay streaming, the gap-annotation data collection, NDAR/ADOS data. Nothing in this project makes or supports any autism-detection claim.

**No evaluation results have been recorded in this repository yet.** They will be added to `PROJECT_STATUS.md` and the results docs after the v2 training run.

### Phase checklist

| Phase | What | Status |
|---|---|---|
| 0 | Docs baseline, remove fake-result fallback, CORS, small fixes | Done (commit `62910a9`) |
| 1 | Notebook v2 edits (not executed) | In progress (notebook written, not yet run) |
| 1b | Notebook smoke mode and static review (not executed) | Done (smoke run hit Whisper/PyAV issue) |
| 1c | Fix Whisper/PyAV incompatibility and re-validate notebook | In progress |
| 2 | Run notebook on Kaggle; ingest and validate artifacts | Not started |
| 3 | WER spot check; Vaani prosody add-on (optional) | Not started |
| 4 | Backend serves v2 | Not started |
| 5 | Rehearsal frontend end to end | Not started |
| 6 | Modal redeploy, live test, demo script | Not started |
| 7 | Final docs | Not started |

Exit criteria and dates are in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Roadmap

> The dates below are the original proposal and are stale. The current milestone plan is in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

```
Q1 2025 ─── Architecture specification              ✓ Complete
         ─── Dataset schema and pipeline design      ✓ Complete
         ─── Access requests (MSP, IEMOCAP, NDAR)   ✓ Submitted

Q2 2025 ─── Data preprocessing and manifests        ← Current
         ─── Synthetic ASD pair generation
         ─── Annotation tool (rehearsal frontend)
         ─── Whisper fine-tuning
         ─── RoBERTa content scorer

Q3 2025 ─── Fusion layer implementation
         ─── Decoupling loss training
         ─── emotion2vec ASD adaptation
         ─── Alpha: rehearsal tool live, data collection begins

Q4 2025 ─── Llama 3 LoRA fine-tuning
         ─── End-to-end system evaluation
         ─── Gap reduction metric results
         ─── FastAPI serving layer
         ─── Beta: evaluator overlay + platform API

Q1 2026 ─── Speaker personalization (adapter layers)
         ─── Paper submission (Interspeech / ICASSP)
         ─── Gap dataset paper (LREC-COLING)
```

---

## Ethical Framework

- The system interprets speakers as they are. It never suggests they should communicate differently.
- All audio collected through the rehearsal tool requires explicit informed consent.
- Consented audio is anonymized before any model training use.
- Every system output includes a SHAP-derived audit trail explaining which features drove the prediction.
- Candidate-facing disclosure is required for any platform API integration.

---

## Team

| Member | Role |
|---|---|
| Deval | AI Architecture Lead — model design, training, decoupling loss, evaluation |
| Krishiv | Engineering Lead — backend, API serving, deployment infrastructure |
| Dev | Product & Data Lead — frontend, annotation tooling, data pipelines |

---

## References

- Fusaroli et al. (2017). *Is voice a marker for autism spectrum disorder?* Research in Autism Spectrum Disorders.
- Tsai et al. (2019). *Multimodal Transformer for Unaligned Multimodal Language Sequences.* NeurIPS. [[paper]](https://arxiv.org/abs/1906.00295)
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. [[paper]](https://arxiv.org/abs/2106.09685)
- Conneau et al. (2017). *Supervised Learning of Universal Sentence Representations.* [[paper]](https://arxiv.org/abs/1705.02364)
- Amir et al. (2020). *Automatic Recognition of Emotions in ASD.* Interspeech.
- MIT Media Lab. *Automated Interview Scoring Dataset.* [[lab]](https://affect.media.mit.edu/)

---

**License:** not yet chosen.

*B.Tech Major Project · 2025–2026*
