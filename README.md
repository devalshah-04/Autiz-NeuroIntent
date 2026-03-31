# Autiz-Intent-Recognition-for-ASD-Candidates-in-Hiring

**Communication-style-invariant intent interpretation for ASD speakers in automated hiring pipelines.**

[![Status](https://img.shields.io/badge/status-architecture%20%26%20data%20preparation-blue)](.)
[![Python](https://img.shields.io/badge/python-3.11-blue)](.)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-pre--training-orange)](.)

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
│  Whisper large-v3 · wav2vec 2.0 / HuBERT · GeMAPS 88   │
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

The **novel contribution** is the cross-modal fusion and decoupling layer. Content representation **C** and delivery representation **D** are trained to be orthogonal via a cosine similarity penalty in the loss function. Same answer, different prosody → identical content score. This is the property that no existing hiring evaluation system has.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical specification.

---

## Novel Dataset Contribution

Existing ASD speech datasets share a fundamental limitation: they are annotated for diagnostic severity (ADOS-2 clinical scores), not for communicative intent. A recording labeled "ASD Level 2" provides no signal about whether the speaker was expressing confidence, uncertainty, enthusiasm, or something else entirely.

We are constructing the **ASD Intent Gap Dataset** — the first dataset (to our knowledge) that captures:

- ASD speaker self-labeled intent, recorded immediately after utterance
- Independent neurotypical evaluator interpretations of the same utterance
- The computed gap between speaker intent and evaluator interpretation

This gap vector is the primary training signal for the decoupling module and the primary evaluation metric for the system. A model that reduces this gap by predicting speaker intent correctly — against the systematic misinterpretation baseline established by neurotypical evaluators — is a model that works.

Dataset schema: [`dataset_schema.json`](dataset_schema.json)

---

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| ASR | Whisper large-v3 | Transcription + word-level timestamps |
| Audio embeddings | wav2vec 2.0 / HuBERT / WavLM | Frame-level speech representations |
| Prosody features | openSMILE GeMAPSv01b | 88 interpretable acoustic features |
| Direct SLU branch | HuBERT fine-tuned | Parallel audio-to-intent without ASR |
| Content scoring | RoBERTa-large | Text-only content quality regression |
| Emotion encoding | emotion2vec (ASD-adapted) | Audio-side affective representation |
| Intent reasoning | Llama 3 8B + LoRA | Structured intent interpretation |
| Explainability | SHAP (KernelExplainer) | Feature attribution for audit trail |
| Training framework | PyTorch + HuggingFace Transformers | |
| Fine-tuning | PEFT / bitsandbytes / TRL | LoRA, 4-bit quantization |
| Serving (planned) | FastAPI + vLLM + Modal.com | Real-time inference |

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
| ASD Intent Gap Dataset | **Novel — primary training signal** | Collected via rehearsal tool |

---

## Current Status

**Phase: Architecture and data preparation**

The model architecture is fully specified. The data schema, preprocessing pipeline, and annotation tooling are in active development. No training runs have been executed yet.

Completed:
- [x] Full architecture specification and design decisions
- [x] Three-level data schema (raw audio record, processed features, gap annotation)
- [x] Dataset access requests submitted (MSP-Podcast, IEMOCAP, NDAR)
- [x] Preprocessing pipeline skeleton
- [x] Annotation schema for gap dataset
- [x] Synthetic prosody perturbation strategy defined (parselmouth-based)

In progress:
- [ ] Common Voice preprocessing and manifest generation
- [ ] openSMILE GeMAPS extraction pipeline
- [ ] wav2vec / HuBERT benchmark experiment design
- [ ] Rehearsal tool (annotation collection frontend)
- [ ] Synthetic ASD pair generation from CMU-MOSI

Upcoming:
- [ ] Whisper large-v3 fine-tuning on atypical speech
- [ ] RoBERTa content scorer training
- [ ] Cross-modal fusion layer implementation
- [ ] Decoupling loss training and ablation study
- [ ] Llama 3 LoRA fine-tuning on gap annotations
- [ ] End-to-end evaluation: gap reduction metric

---

## Roadmap

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

*B.Tech Major Project · 2025–2026*
