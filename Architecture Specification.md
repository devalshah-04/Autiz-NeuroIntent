# Autiz — Architecture Specification

> **Scope note (read first).** This document describes the full *target* architecture. The current
> milestone implements only a small part of it: a content scorer, a prosody branch trained to be
> statistically independent of the content representation, independence tests, and a serving
> pipeline. Intent classification, Llama 3 LoRA, the HuBERT SLU branch, emotion2vec, Whisper
> fine-tuning, the live overlay (`/stream`) and the evaluator portal are **future work**.
> See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for what is actually built, and for the v2 design
> decisions that differ from the sections below (frozen content path C trained on ChaLearn
> First Impressions V2 transcripts; ProsodyBranch D trained with GeMAPS reconstruction plus a
> batch-level independence penalty instead of the cosine penalty in section 2.2).

---

## Overview

Autiz is a three-layer multimodal pipeline for communication-style-invariant intent interpretation. The central design principle is **decoupling**: content quality and prosodic delivery are modeled as orthogonal representations, trained to be separable, and evaluated independently before fusion. This is the property that allows the system to give equivalent content scores to equivalent answers regardless of the speaker's prosodic style.

The architecture extends the Multimodal Transformer (MulT) framework (Tsai et al., NeurIPS 2019) with a novel decoupling objective and ASD-specific adaptation strategy.

---

## Implementation status

Status of this document as of 2026-10-05. The sections below describe the full target design; each
section carries a **Status** line saying what exists. The living, dated record is
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

| Part | Status |
|---|---|
| Whisper transcription | Built with the **base** model, not fine-tuned (section 1.1) |
| wav2vec / HuBERT / WavLM audio embeddings | **Planned** |
| GeMAPS prosody features | **Built** — 62 features (GeMAPSv01b Functionals) |
| Temporal alignment with pause tokens | **Planned** |
| SLU branch | **Planned** |
| Cross-modal transformer | **Planned** — v1 and v2 use small MLP branches instead |
| Decoupling objective | v1 **built with a flawed loss**; v2 uses a different objective (see "v2 design" below) |
| Classifier head / intent | v1 built but **untrained** (no intent labels exist); dropped in v2; intent is **future work** |
| RoBERTa content scorer | v2 trains it on ChaLearn First Impressions V2 transcripts (not the 500 annotated clips described in 2.4) |
| emotion2vec, Llama 3 LoRA | **Planned**; Llama exists only as a hardcoded stub |
| Evaluator overlay / `/stream` | **Prototype, not connected**; to be disabled in this milestone |
| Speaker reflection report | Partly built in the rehearsal app, without Llama text |
| Platform API | `/analyze` and `/score` exist; their response differs from section 3.3 |
| SHAP audit trail | **Planned**; the current audit trail is a vector-size proxy |

## v2 design (current milestone)

C is the 256-dim output of `ContentBranch` — the layer just before the final score in the content
scorer. Chain: RoBERTa-large CLS (1024) → `ContentBranch` (256 = **C**) → `content_head` `Linear(256, 1)`
= content score. C is frozen after Stage 1 (trained on ChaLearn First Impressions V2 transcripts).
Stage 2 trains `ProsodyBranch` (62 z-scored GeMAPS → 256 = **D**) with:

- a **reconstruction loss** — a small `recon_head` must rebuild the z-scored GeMAPS input from D, so D keeps real delivery information;
- a **batch-level independence penalty** between D and the frozen C, replacing the cosine penalty of section 2.2.

Independence is *tested*, not assumed: distance correlation (bias-corrected, with a permutation test) and
cross-validated ridge R² predicting D from C and C from D. A content path that is blind to audio makes the
content score invariant to delivery **by construction**, so those invariance numbers are not evidence of
training success; the independence tests and the evaluation on real held-out speakers are.
Serving checkpoint: `models/checkpoints/autiz_v2.pt` (`content_branch`, `content_head`,
`prosody_branch`, `recon_head`, `config`); the GeMAPS z-score scaler is fit on original LibriSpeech
clips of the training split only (`models/checkpoints/gemaps_scaler.json`).

---

## Full Architecture Diagram

```
                        ┌──────────────────────┐
                        │    Raw Speech Audio   │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              │                    │                      │
              ▼                    ▼                      ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │  Whisper         │  │  wav2vec 2.0    │  │  openSMILE       │
   │  large-v3        │  │  or HuBERT      │  │  GeMAPSv01b      │
   │  (fine-tuned)    │  │  (benchmarked)  │  │  62 features     │
   │                  │  │                 │  │                  │
   │  → transcript    │  │  → frame-level  │  │  → pitch, rate,  │
   │  → word stamps   │  │    embeddings   │  │    energy, pause │
   │    [T_word, d]   │  │    [T_20ms,1024]│  │    [62]          │
   └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘
            │                     │                     │
            │         ┌───────────┴─────────────────────┤
            │         │   TEMPORAL ALIGNMENT             │
            │         │   Word-level pooling             │
            │         │   → unified sequence             │
            │         │   [N_words, d_text+d_audio+62]   │
            │         └───────────┬─────────────────────┘
            │                     │
            │    ┌────────────────┘
            │    │
            │    │         ┌──────────────────────────┐
            │    │         │  PARALLEL BRANCH (SLU)    │
            │    │         │  Direct audio → intent    │
            │    │         │  HuBERT fine-tuned        │
            │    │         │  No ASR intermediate step  │
            │    │         └──────────────┬────────────┘
            │    │                        │
            ▼    ▼                        │
   ┌──────────────────────────────────────┼────────────────────┐
   │  LAYER 2 — Cross-Modal Fusion and Decoupling              │
   │                                      │                    │
   │  ┌─────────────────┐  ←cross-attn→  ┌──────────────────┐ │
   │  │  Content Branch │                 │  Prosody Branch  │ │
   │  │  (RoBERTa init) │  ←cross-attn→  │  (wav2vec init)  │ │
   │  │  text only      │                 │  audio + GeMAPS  │ │
   │  │  → C [d_c]      │                 │  → D [d_d]       │ │
   │  └────────┬────────┘                 └────────┬─────────┘ │
   │           │                                   │           │
   │           └──────────────┬────────────────────┘           │
   │                          │                                 │
   │              ┌───────────┴──────────┐                     │
   │              │  DECOUPLING PENALTY  │                     │
   │              │  L_decouple =        │                     │
   │              │  cos_sim(C, D)       │                     │
   │              │  (minimized)         │                     │
   │              └───────────┬──────────┘                     │
   │                          │                                 │
   │              ┌───────────┴──────────┐                     │
   │              │  CLASSIFIER HEAD     │                     │
   │              │  [C; D; C-D; C⊙D]   │                     │
   │              │  → MLP (3 layers)    │                     │
   │              │  → intent vector     │                     │
   │              └───────────┬──────────┘                     │
   │                          │                    SLU branch   │
   │                          │ ◄──────────────────────────────┘
   │              ┌───────────┴──────────┐                     │
   │              │  Llama 3 8B + LoRA   │                     │
   │              │  Intent reasoning    │                     │
   │              │  (structured prompt) │                     │
   │              └───────────┬──────────┘                     │
   └──────────────────────────┼────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Evaluator   │  │  Speaker     │  │  Platform    │
   │  Overlay     │  │  Reflection  │  │  API (JSON)  │
   │  ~200ms      │  │  Report      │  │  + audit     │
   │  real-time   │  │  (offline)   │  │    trail     │
   └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Layer 1 — Speech Decomposition

### 1.1 Automatic Speech Recognition: Whisper large-v3

**Status:** built with Whisper **base** (via faster-whisper); no fine-tuning, no word-timestamp use yet.

**Model:** `openai/whisper-large-v3` (1.5B parameters)
**HuggingFace:** https://huggingface.co/openai/whisper-large-v3

Whisper provides two outputs critical to the pipeline:
1. Transcript text — passed to the content branch
2. Word-level timestamps — used for temporal alignment of audio embeddings

**Fine-tuning strategy:**
- Freeze encoder layers 0–15 (low-level phoneme representations, already accurate)
- Fine-tune encoder layers 16–23 and full decoder
- Training data: Mozilla Common Voice (filtered, quality-verified) + any available ASD-specific speech
- Objective: minimize CTC loss on atypical speech
- Target: reduce WER from baseline ~30% to below 15% on held-out atypical speech

**Why Whisper over alternatives:**
Word-level timestamp output is not available from all ASR systems. Timestamps are load-bearing for temporal alignment — without them, the audio embedding pooling step cannot be performed accurately.

---

### 1.2 Audio Embeddings: wav2vec 2.0 / HuBERT / WavLM

**Status:** planned; not implemented.

Three models are benchmarked on a prosody classification task (flat vs. expressive pitch, fast vs. slow rate, long vs. short pauses) before committing to one.

| Model | HuggingFace | Key property |
|---|---|---|
| wav2vec 2.0 large | `facebook/wav2vec2-large-960h` | Strong prosody representations, small data fine-tuning |
| HuBERT large | `facebook/hubert-large-ls960-ft` | Offline clustering, often outperforms wav2vec on downstream tasks |
| WavLM large | `microsoft/wavlm-large` | Denoising pretraining, best for Zoom/Teams audio conditions |

**Output:** Frame-level embeddings `[T, 1024]` where T = number of 20ms frames.

**Fine-tuning:** Top 4 transformer layers updated on MSP-Podcast + MuSe for prosody-aware representations. Lower layers frozen.

---

### 1.3 Prosody Features: openSMILE GeMAPSv01b

**Status:** built; clip-level only. Normalization in v2 uses a LibriSpeech training-split scaler, not MSP-Podcast + MuSe.

**Library:** https://audeering.github.io/opensmile-python

Extracts 62 interpretable acoustic features per clip (the GeMAPSv01b *Functionals* set; verified by running openSMILE in `notebooks/`) including:
- F0 (pitch): mean, range, variance, percentiles
- Energy: mean, variance, rising/falling slopes
- Spectral features: spectral flux, centroid
- Temporal: speaking rate, pause rate, pause duration statistics
- Voice quality: jitter, shimmer, HNR

These features are **not** used as the primary neural input but serve two specific roles:
1. Auxiliary supervision signal during fusion layer training
2. SHAP attribution input for Layer 3 explainability — they are human-readable, which audio embeddings are not

**Normalization:** Per-feature z-score normalization computed on the combined MSP-Podcast + MuSe neurotypical baseline. ASD speech deviations from this normalization represent the delivery signal the system learns to interpret rather than penalize.

---

### 1.4 Temporal Alignment

**Status:** planned; not implemented.

The three streams operate at different temporal resolutions and must be synchronized before fusion.

**Procedure:**
1. Use Whisper word timestamps as the master clock
2. For each word spanning `[t_start, t_end]`:
   - Pool all wav2vec frames where `frame_center ∈ [t_start, t_end]` → mean pooling → single 1024-dim vector
   - Average all GeMAPS windows overlapping with `[t_start, t_end]` → single 62-dim vector
3. Insert explicit `[PAUSE:Xs]` tokens between words where silence duration exceeds 200ms
4. Output: unified sequence `[N_words + N_pauses, d_text + 1024 + 62]`

Pauses are first-class tokens. They carry prosodic information (duration, position relative to sentence structure) that is systematically informative for ASD communication patterns.

---

### 1.5 Parallel Branch: Direct Audio-to-Intent (SLU)

**Status:** planned; out of scope for the current milestone.

Motivation: The ASR-mediated pipeline discards signal at the transcription step. End-to-end spoken language understanding (SLU) models avoid this by operating directly on audio representations.

**Architecture:** HuBERT large, top 6 layers fine-tuned for intent classification without generating transcripts. Trained on SLURP dataset (Bastianelli et al., 2020) then adapted on available ASD intent-labeled audio.

**Role in the system:** The SLU branch outputs a parallel intent probability distribution that is concatenated into the fusion layer alongside the ASR-derived representations. The two branches are complementary — the SLU branch preserves signal the ASR step may discard; the ASR branch enables explainability through content analysis.

**Reference:** Qian et al. (2021), *Speech-BERT: A Joint Sequence Modelling Framework for Spoken Language Understanding*. Interspeech.

---

## Layer 2 — Cross-Modal Fusion and Decoupling

### 2.1 Cross-Modal Transformer

**Status:** planned. v1 and v2 use two-layer MLP branches on a single RoBERTa CLS vector and clip-level GeMAPS, with no cross-attention.

Based on Tsai et al. (2019), *Multimodal Transformer for Unaligned Multimodal Language Sequences* (MulT), extended with an explicit decoupling objective.

**Content branch:**
- 4-layer transformer encoder
- Initialized from RoBERTa-large weights
- Input: word embeddings only (text tokens)
- Cross-attention keys/values sourced from prosody branch
- Output: content representation **C** `[d_c]`

**Prosody branch:**
- 4-layer transformer encoder
- Initialized from selected audio encoder weights (wav2vec / HuBERT / WavLM)
- Input: audio embeddings + GeMAPS features concatenated
- Cross-attention keys/values sourced from content branch
- Output: delivery representation **D** `[d_d]`

Cross-modal attention is bidirectional: content attends to prosody context, prosody attends to content context. This allows the model to learn, for example, that a pause following a complex technical question carries different meaning than a pause following a simple yes/no question.

---

### 2.2 Decoupling Loss — The Novel Objective

> **Correction (2026-10-05).** The formulation below is flawed and is **not** what v2 uses. `CosineEmbeddingLoss(C, D, target=-1)` is minimized when cos(C, D) = **−1**, i.e. it drives the two vectors **anti-parallel** — which is maximal dependence, not orthogonality. The v1 run shows exactly this (cos(C,D) ≈ −0.999). Minimizing the *squared* cosine would target 0, but a cosine between two different learned spaces is also a weak test of independence. v2 therefore uses a batch-level independence penalty plus distance-correlation and ridge-R² tests (see "v2 design"). The "verification criterion" below should read |cos| < 0.20 if cosine is used at all.

**Motivation:** Without an explicit separation constraint, the two branches converge toward encoding similar information (the path of least resistance during gradient descent). The decoupling loss actively penalizes this convergence.

**Formulation:**

```
L_total = L_intent 
        + λ₁ · CosineEmbeddingLoss(C, D, target=-1)
        + λ₂ · L_content_auxiliary 
        + λ₃ · L_prosody_auxiliary
```

Where:
- `L_intent` — cross-entropy over intent classes
- `λ₁ · cos_sim(C, D)` — penalizes cosine similarity between C and D (as written this does **not** force orthogonality; see the correction above)
- `L_content_auxiliary` — MSE between content branch output and RoBERTa content score (direct supervision)
- `L_prosody_auxiliary` — MSE between prosody branch output and GeMAPS-derived prosody labels (direct supervision)

**Starting hyperparameters:** λ₁=0.5, λ₂=0.3, λ₃=0.2 — swept during training.

**Verification criterion (as originally written):** After training, mean cosine similarity between C and D on held-out test set must be below 0.20. Note that a value near −1 satisfies "below 0.20" while being the worst case; use |cos|. If above, increase λ₁ and retrain.

**Contrastive pair test:** Same transcript, perturbed prosody (flattened pitch, extended pauses). Content scores from both conditions must differ by less than 0.05. This is the empirical proof that decoupling succeeded.

---

### 2.3 Classifier Head

**Status:** v1 has this head but it was never trained (no intent labels). v2 drops it. Intent classification is future work.

Concatenation pattern from Conneau et al. (2017) — proven more expressive than raw concatenation:

```python
# Combined representation
z = torch.cat([C, D, C - D, C * D], dim=-1)

# MLP classifier
intent_logits = mlp(z)  # 3-layer MLP with GELU activations and dropout
```

The difference vector `C - D` explicitly encodes the gap between content and delivery. The elementwise product `C ⊙ D` captures interactions between the two. This four-part combination is the input to the intent classifier.

---

### 2.4 RoBERTa Content Scorer

**Status:** in progress. v2 trains the scorer on ChaLearn First Impressions V2 transcripts (human impression labels), not on 500 manually annotated clips; the current serving code uses a vector-size proxy instead of a trained scorer.

**Model:** `roberta-large` (355M parameters)
**HuggingFace:** https://huggingface.co/roberta-large

Fine-tuned as a regression head on 500 manually annotated interview transcript clips. Scoring criteria (text only, no audio):
- Relevance to the question (0–5)
- Clarity of explanation (0–5)
- Depth of reasoning (0–5)

Composite score normalized to [0, 1]. Pearson correlation target: r > 0.75 against held-out human annotations.

This model is audio-blind by design. It reads only the Whisper transcript. Any influence of prosody on its output represents a failure mode, not intended behavior.

---

### 2.5 emotion2vec (ASD-Adapted)

**Status:** planned; out of scope for the current milestone.

**Model:** `emotion2vec` (ddlBoJack, 2023)
**GitHub:** https://github.com/ddlBoJack/emotion2vec

Base model pretrained on neurotypical emotion datasets. **Not deployed raw.** Fine-tuned on ASD speech where speakers self-labeled their emotional state via the rehearsal tool.

The fine-tuning teaches the model that flat prosody ≠ emotional absence for ASD speakers. Without this adaptation, emotion2vec encodes exactly the neurotypical norm that produces the bias being corrected.

---

### 2.6 Llama 3 8B + LoRA Intent Reasoning

**Status:** planned; only a hardcoded stub function exists. Out of scope for the current milestone.

**Model:** `meta-llama/Meta-Llama-3-8B`
**HuggingFace:** https://huggingface.co/meta-llama/Meta-Llama-3-8B

**Configuration:**
- 4-bit NF4 quantization via bitsandbytes (inference-time memory: ~5GB)
- LoRA rank=16, alpha=32, target modules: `q_proj`, `v_proj`
- Trainable parameters: <0.5% of total

**Input prompt structure:**
```
System: You are an ASD-aware communication interpreter. Your task is to
identify what a speaker actually meant. Flat prosody does not indicate
low confidence. Extended pauses indicate processing, not hesitation.
Hyper-literal phrasing means exactly what was said.

Input:
- Transcript: "{transcript}"
- Content quality score: {content_score:.2f}
- Prosody pattern: {prosody_summary}
- Pause classification: {pause_classification}
- ASD communication flags: {flags}
- SLU branch intent probability: {slu_distribution}

Output (JSON):
{
  "intent": "...",
  "confidence_level": "high|medium|low",
  "communication_style_note": "...",
  "evaluator_note": "..."
}
```

**Deployment note:** Llama 3 runs asynchronously. The real-time evaluator overlay uses only the fast MLP classifier output from the fusion layer (~200ms latency). Llama 3 generates enriched interpretations for the speaker reflection report and platform audit trail.

---

## Layer 3 — Output Surfaces

### 3.1 Real-Time Evaluator Overlay

**Status:** prototype UI exists but is not connected (it never sends audio; `/stream` is a stub). To be disabled in the current milestone.

**Latency target:** <250ms end-to-end from sentence completion

**Pipeline (per sentence):**
```
Silero VAD detects sentence end
        │
        ├──→ wav2vec extraction (async, ~80ms)
        ├──→ GeMAPS extraction (~5ms, CPU)
        └──→ RoBERTa transcript scoring (async, ~60ms)
                │
                ▼
        Fusion layer inference (~120ms)
                │
                ▼
        Overlay card update (~200ms total)
```

**Output card fields:**
- Intent label (confident / uncertain / enthusiastic / explaining / requesting)
- Communication style note (one sentence, plain language)
- Content quality indicator
- Bias correction flag (boolean — was a style-based downgrade detected and corrected?)

---

### 3.2 Speaker Reflection Report

**Status:** partly built in the rehearsal app (per-answer scores and delivery notes), without Llama-generated text.

Asynchronous. Generated after session completion using Llama 3 output.

**Report sections:**
- Per-answer content quality score vs estimated standard-evaluator score (gap visualization)
- Delivery pattern summary (pitch profile, pause frequency, speaking rate)
- Timestamp-linked interpretation: what the system detected and why at each moment
- Communication style profile (persistent patterns across the session)

---

### 3.3 Platform API

**Status:** `/analyze` and `/score` exist but return a different schema from the one below; `candidate_disclosure_required` is hardcoded `true` as specified.

REST endpoint. Designed for silent integration into hiring platforms.

**Response schema:**
```json
{
  "clip_id": "string",
  "content_score": 0.87,
  "delivery_pattern": "flat_prosody_extended_pauses",
  "intent_label": "confident",
  "confidence_level": "high",
  "bias_correction_applied": true,
  "standard_evaluator_estimated_score": 0.52,
  "corrected_score": 0.87,
  "audit_trail": {
    "top_content_features": ["..."],
    "top_prosody_features": ["..."],
    "shap_content_attribution_share": 0.71,
    "decoupling_verified": true,
    "cos_sim_C_D": 0.14
  },
  "communication_style_note": "...",
  "candidate_disclosure_required": true
}
```

**Requirement:** `candidate_disclosure_required` is hardcoded `true`. Platform integrations must disclose to candidates that ASD-aware interpretation was applied.

---

## Training Sequence

**Status:** only step 3 was attempted (v1, 100 synthetic pairs, flawed loss). The other steps are planned. v2 replaces steps 2–3 with ChaLearn content-scorer training and the two-stage procedure above.

| Step | Component | Data | Objective | Target metric |
|---|---|---|---|---|
| 1 | Audio encoder (top 4 layers) | MSP-Podcast + MuSe | Prosody classification | Accuracy >85% |
| 2 | RoBERTa content scorer | 500 annotated transcripts | Content quality regression | Pearson r >0.75 |
| 3 | Fusion layer | 5,000 synthetic pairs (MOSI-perturbed) | Decoupling + intent | cos(C,D) <0.20 |
| 4 | Full pipeline end-to-end | 200+ real ASD clips (re-annotated) | Intent classification | Accuracy >78% |
| 5 | Llama 3 LoRA | 50–200 gap dataset examples | Intent generation | Human rating >4.0/5.0 |
| 6 | Bias correction pass | MIT Interview + FairCVtest | Fairness audit | SPD <0.05 |

---

## Evaluation Framework

**Status:** planned. The current milestone reports independence tests, reconstruction quality and leave-one-speaker-out evaluation instead of gap reduction against evaluators, because no evaluator labels exist.

### Primary Metric: Gap Reduction

```
gap_reduction = (distance_without_system - distance_with_system) 
                / distance_without_system
```

Where distance is computed between evaluator interpretation and speaker self-labeled intent across the held-out gap dataset. Target: >40% gap reduction.

### Supporting Metrics

| Metric | Target |
|---|---|
| WER on atypical speech (post fine-tune) | <15% |
| cos(C, D) after decoupling training | <0.20 |
| Content score variance on contrastive pairs | <0.05 |
| Intent accuracy vs speaker self-labels | >78% |
| Statistical Parity Difference (ASD vs NT) | <0.05 |
| SHAP content attribution share | >65% |

---

## References

1. Tsai et al. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. *NeurIPS*. https://arxiv.org/abs/1906.00295
2. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. https://arxiv.org/abs/2106.09685
3. Conneau et al. (2017). Supervised Learning of Universal Sentence Representations from Natural Language Inference Data. https://arxiv.org/abs/1705.02364
4. Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.
5. Hsu et al. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *IEEE/ACM TASLP*.
6. Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE JSTSP*.
7. Bastianelli et al. (2020). SLURP: A Spoken Language Understanding Resource Package. *EMNLP*.
8. Fusaroli et al. (2017). Is voice a marker for autism spectrum disorder? *Research in Autism Spectrum Disorders*.
