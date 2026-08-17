"""
neurointent/preprocessing/pipeline.py

Preprocessing pipeline for the NeuroIntent ASD Intent Gap Dataset.

Transforms raw audio files into the unified representation consumed by
the cross-modal fusion layer. Produces three artefacts per clip:
  1. Level 1 JSON record (AudioRecord) — identity, transcript, labels
  2. GeMAPS feature vector — 62 floats, .npy
  3. wav2vec frame embeddings — [T, 1024] floats, .npy

The pipeline is designed to be run dataset-by-dataset in the sequence
documented in ARCHITECTURE.md. Output format is identical regardless of
source dataset — this is enforced by the Pydantic schema at write time.

Usage:
    python -m neurointent.preprocessing.pipeline \
        --dataset common_voice \
        --input_dir data/raw/common_voice \
        --output_dir data/processed/common_voice \
        --split train

Authors: Deval (AI Lead), Krishiv (Engineering), Dev (Data)
Status: In development — TODOs mark incomplete sections
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("neurointent.preprocessing")


# ─────────────────────────────────────────────────────────────────────────────
# Schema definitions (mirror of schema.py — kept here for pipeline self-
# containment; source of truth is neurointent/schema.py)
# ─────────────────────────────────────────────────────────────────────────────

class AudioRecord(BaseModel):
    """Level 1 record. One per audio clip. Serialised to JSON."""

    clip_id: str
    source_dataset: str
    speaker_id: str

    audio_path: str
    sample_rate: int
    duration_sec: float
    channels: int

    transcript: str
    transcript_source: str
    word_timestamps: Optional[list] = None

    is_asd: bool
    intent_label: Optional[str] = None
    emotion_label: Optional[str] = None
    consent_given: bool

    split: str
    age_group: str = "unknown"
    language: str = "en"
    preprocessing_status: str = "raw"

    @validator("sample_rate")
    def must_be_16k(cls, v):
        if v != 16000:
            raise ValueError(f"sample_rate must be 16000, got {v}")
        return v

    @validator("channels")
    def must_be_mono(cls, v):
        if v != 1:
            raise ValueError(f"channels must be 1 (mono), got {v}")
        return v

    @validator("duration_sec")
    def valid_duration(cls, v):
        if v < 2.0 or v > 60.0:
            raise ValueError(
                f"duration {v:.2f}s outside valid range [2.0, 60.0]"
            )
        return v

    @validator("consent_given")
    def consent_required(cls, v):
        if not v:
            raise ValueError(
                "Cannot create record without consent_given=True. "
                "Audio must never be stored without explicit informed consent."
            )
        return v


@dataclass
class PipelineConfig:
    """
    Runtime configuration for the preprocessing pipeline.
    All paths are relative to the project root.
    """
    dataset: str
    input_dir: Path
    output_dir: Path
    split: str

    # Audio preprocessing
    target_sample_rate: int = 16000
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0

    # Whisper configuration
    whisper_model_size: str = "large-v3"
    whisper_model_path: Optional[str] = None  # None → download from HuggingFace
    whisper_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_batch_size: int = 8

    # Audio encoder configuration
    # Set after benchmarking wav2vec2 vs HuBERT vs WavLM — see ARCHITECTURE.md §1.2
    audio_encoder_name: str = "facebook/wav2vec2-large-960h"  # TODO: update after benchmark
    audio_encoder_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    audio_encoder_batch_size: int = 4

    # GeMAPS configuration
    gemaps_config: str = "GeMAPSv01b"

    # Normalization stats (computed on MSP-Podcast + MuSe, loaded at runtime)
    normalization_stats_path: Optional[Path] = None

    # Output control
    overwrite_existing: bool = False
    save_frame_embeddings: bool = True  # frame-level [T, 1024] .npy files (large)

    # Logging
    log_every_n: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Audio loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate_audio(
    file_path: Path,
    config: PipelineConfig,
) -> tuple[np.ndarray, int]:
    """
    Load audio from file and validate against pipeline requirements.

    Returns:
        audio: np.ndarray, shape [n_samples], dtype float32
        sample_rate: int (always config.target_sample_rate after resampling)

    Raises:
        ValueError: if file fails validation and cannot be salvaged
        FileNotFoundError: if file does not exist
    """
    # TODO: implement audio loading
    # Recommended approach:
    #   import soundfile as sf
    #   import librosa
    #
    #   audio, sr = sf.read(str(file_path), dtype='float32')
    #   if audio.ndim > 1:
    #       audio = audio.mean(axis=1)  # mix down stereo to mono
    #   if sr != config.target_sample_rate:
    #       audio = librosa.resample(audio, orig_sr=sr, target_sr=config.target_sample_rate)
    #       sr = config.target_sample_rate
    #
    # Validate duration:
    #   duration = len(audio) / sr
    #   if duration < config.min_duration_sec or duration > config.max_duration_sec:
    #       raise ValueError(f"Duration {duration:.2f}s out of range")

    raise NotImplementedError(
        "load_and_validate_audio: TODO — see implementation notes above"
    )


def resample_file_inplace(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
) -> None:
    """
    Convert audio file to target sample rate and mono WAV using ffmpeg.
    Faster than librosa for bulk conversion.

    ffmpeg command:
        ffmpeg -i <input> -ar 16000 -ac 1 <output> -y -loglevel error
    """
    # TODO: implement ffmpeg subprocess call
    # import subprocess
    # subprocess.run(
    #     ["ffmpeg", "-i", str(input_path), "-ar", str(target_sr),
    #      "-ac", "1", str(output_path), "-y", "-loglevel", "error"],
    #     check=True
    # )
    raise NotImplementedError("resample_file_inplace: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Whisper transcription and word timestamp extraction
# ─────────────────────────────────────────────────────────────────────────────

class WhisperTranscriber:
    """
    Wrapper around fine-tuned Whisper for transcript + word-timestamp extraction.

    The word timestamps are load-bearing for temporal alignment (§ ARCHITECTURE.md §1.4).
    Do not use a Whisper variant that does not support word-level timestamps.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load(self) -> "WhisperTranscriber":
        """
        Load Whisper model and processor.
        Uses fine-tuned checkpoint if config.whisper_model_path is set,
        otherwise loads base model from HuggingFace.
        """
        # TODO: implement model loading
        # from transformers import WhisperForConditionalGeneration, WhisperProcessor
        #
        # model_id = self.config.whisper_model_path or f"openai/whisper-{self.config.whisper_model_size}"
        # self.processor = WhisperProcessor.from_pretrained(model_id)
        # self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        # self.model = self.model.to(self.config.whisper_device)
        # self.model.eval()
        # logger.info(f"Whisper loaded: {model_id} on {self.config.whisper_device}")
        raise NotImplementedError("WhisperTranscriber.load: TODO")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
    ) -> dict:
        """
        Run Whisper inference on a single audio clip.

        Returns dict with keys:
            text: str — full transcript
            words: list of {word: str, start: float, end: float}
                   Word-level timestamps. Required for temporal alignment.
        """
        # TODO: implement Whisper inference with word timestamps
        #
        # input_features = self.processor(
        #     audio, sampling_rate=16000, return_tensors="pt"
        # ).input_features.to(self.config.whisper_device)
        #
        # with torch.no_grad():
        #     # return_timestamps="word" enables word-level timestamps
        #     result = self.model.generate(
        #         input_features,
        #         return_timestamps=True,  # sentence-level
        #         return_token_timestamps=True,  # word-level
        #         language=language,
        #     )
        #
        # decoded = self.processor.batch_decode(result, skip_special_tokens=True)
        # # Extract word timestamps from result.token_timestamps
        # # See HuggingFace Whisper docs for token_timestamps format
        raise NotImplementedError("WhisperTranscriber.transcribe: TODO")

    def transcribe_batch(
        self,
        audio_paths: list[Path],
    ) -> list[dict]:
        """
        Batch transcription. More efficient than per-file inference.
        """
        # TODO: implement batched inference
        # Batch size controlled by config.whisper_batch_size
        raise NotImplementedError("WhisperTranscriber.transcribe_batch: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Audio embedding extraction (wav2vec 2.0 / HuBERT / WavLM)
# ─────────────────────────────────────────────────────────────────────────────

class AudioEmbeddingExtractor:
    """
    Extracts frame-level audio embeddings using the selected audio encoder.

    The audio encoder is selected by benchmarking wav2vec 2.0, HuBERT, and WavLM
    on a prosody classification task before committing — see ARCHITECTURE.md §1.2.
    The winning model name is set in PipelineConfig.audio_encoder_name.

    Output shape: [T, 1024] where T = number of 20ms frames.
    Pooled output: [1024] (mean across time).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load(self) -> "AudioEmbeddingExtractor":
        """Load selected audio encoder from HuggingFace."""
        # TODO: implement encoder loading
        # from transformers import AutoModel, AutoProcessor
        #
        # self.processor = AutoProcessor.from_pretrained(self.config.audio_encoder_name)
        # self.model = AutoModel.from_pretrained(self.config.audio_encoder_name)
        # self.model = self.model.to(self.config.audio_encoder_device)
        # self.model.eval()
        #
        # # Freeze all layers — we use this as a feature extractor during preprocessing.
        # # Fine-tuning of top 4 layers happens during training, not here.
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # logger.info(f"Audio encoder loaded: {self.config.audio_encoder_name}")
        raise NotImplementedError("AudioEmbeddingExtractor.load: TODO")

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract frame-level and pooled embeddings from one audio clip.

        Returns:
            frame_embeddings: np.ndarray, shape [T, 1024]
            pooled_embedding: np.ndarray, shape [1024]
        """
        # TODO: implement embedding extraction
        # inputs = self.processor(
        #     audio, sampling_rate=sample_rate, return_tensors="pt"
        # ).to(self.config.audio_encoder_device)
        #
        # with torch.no_grad():
        #     outputs = self.model(**inputs, output_hidden_states=False)
        #
        # # Last hidden state: [1, T, 1024]
        # frame_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        # pooled_embedding = frame_embeddings.mean(axis=0)
        # return frame_embeddings, pooled_embedding
        raise NotImplementedError("AudioEmbeddingExtractor.extract: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — openSMILE GeMAPS feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class GeMAPSExtractor:
    """
    Extracts the 62 GeMAPSv01b acoustic features using openSMILE.

    These features are human-interpretable (pitch, energy, rate, pauses)
    and serve two roles:
      1. Auxiliary prosody supervision signal during fusion layer training
      2. SHAP attribution inputs for Layer 3 explainability

    The canonical feature order is fixed by openSMILE's GeMAPSv01b config.
    Do not reorder features — the normalization stats and SHAP feature names
    depend on consistent ordering.
    """

    # GeMAPSv01b produces exactly 62 features. If this changes, the schema
    # and normalization pipeline must be updated to match.
    N_FEATURES = 62

    def __init__(self):
        self.smile = None

    def load(self) -> "GeMAPSExtractor":
        """Initialize openSMILE with GeMAPSv01b config."""
        # TODO: implement openSMILE initialization
        # import opensmile
        # self.smile = opensmile.Smile(
        #     feature_set=opensmile.FeatureSet.GeMAPSv01b,
        #     feature_level=opensmile.FeatureLevel.Functionals,
        # )
        # logger.info(f"openSMILE initialized: GeMAPSv01b ({self.N_FEATURES} features)")
        raise NotImplementedError("GeMAPSExtractor.load: TODO")

    def extract(self, wav_path: Path) -> np.ndarray:
        """
        Extract 62 GeMAPS features from a WAV file.

        Returns:
            features: np.ndarray, shape [62], dtype float32

        Note: openSMILE requires a file path, not an in-memory array.
        Ensure the WAV file exists at wav_path before calling.
        """
        # TODO: implement feature extraction
        # features = self.smile.process_file(str(wav_path))
        # values = features.values[0].astype(np.float32)
        # assert values.shape == (self.N_FEATURES,), (
        #     f"Expected {self.N_FEATURES} features, got {values.shape[0]}. "
        #     "Check openSMILE config version."
        # )
        # return values
        raise NotImplementedError("GeMAPSExtractor.extract: TODO")


def compute_normalization_stats(
    gemaps_array: np.ndarray,
    output_path: Path,
) -> dict:
    """
    Compute per-feature mean and standard deviation from a large collection
    of GeMAPS feature vectors. Called once on the combined MSP-Podcast + MuSe
    neurotypical baseline before any other normalization is performed.

    Args:
        gemaps_array: np.ndarray, shape [N_clips, 62]
        output_path: path to save stats JSON

    Returns:
        stats: dict with keys 'mean' ([62] floats) and 'std' ([62] floats)
    """
    # TODO: implement stats computation
    # mean = gemaps_array.mean(axis=0)
    # std = gemaps_array.std(axis=0)
    # std = np.where(std == 0, 1.0, std)  # avoid division by zero
    # stats = {"mean": mean.tolist(), "std": std.tolist()}
    # with open(output_path, "w") as f:
    #     json.dump(stats, f, indent=2)
    # logger.info(f"Normalization stats saved to {output_path}")
    # return stats
    raise NotImplementedError("compute_normalization_stats: TODO")


def normalize_gemaps(
    features: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """
    Apply z-score normalization to GeMAPS features using precomputed stats.

    Args:
        features: np.ndarray, shape [62]
        stats: dict with keys 'mean' and 'std', each a list of 62 floats

    Returns:
        normalized: np.ndarray, shape [62]
    """
    # TODO: implement normalization
    # mean = np.array(stats["mean"], dtype=np.float32)
    # std = np.array(stats["std"], dtype=np.float32)
    # return (features - mean) / std
    raise NotImplementedError("normalize_gemaps: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Temporal alignment: unified word-level sequence construction
# ─────────────────────────────────────────────────────────────────────────────

def build_word_sequence(
    word_timestamps: list[dict],
    frame_embeddings: np.ndarray,
    gemaps_features: np.ndarray,
    frame_duration_ms: float = 20.0,
    pause_threshold_sec: float = 0.2,
) -> list[dict]:
    """
    Construct the unified word-aligned sequence that feeds the fusion layer.

    For each word in word_timestamps:
      - Compute which wav2vec frames fall within [word.start, word.end]
      - Mean-pool those frames → single 1024-dim vector
      - Associate the GeMAPS features (clip-level for now; windowed GeMAPS
        is a planned enhancement — see TODO below)
      - Insert PAUSE tokens for silences exceeding pause_threshold_sec

    Returns:
        sequence: list of dicts, each with keys:
            token: str — the word or "[PAUSE:Xs]"
            token_type: "word" | "pause"
            start_sec: float
            end_sec: float
            wav2vec_pooled: list[float] — 1024-dim
            gemaps: list[float] — 62-dim (clip-level placeholder until windowed extraction)

    Args:
        word_timestamps: from WhisperTranscriber.transcribe()
        frame_embeddings: [T, 1024] from AudioEmbeddingExtractor.extract()
        gemaps_features: [62] from GeMAPSExtractor.extract()
        frame_duration_ms: duration of each wav2vec frame in milliseconds
        pause_threshold_sec: minimum silence duration to insert PAUSE token
    """
    # TODO: implement temporal alignment
    #
    # sequence = []
    # prev_end = 0.0
    #
    # for word_entry in word_timestamps:
    #     word = word_entry["word"]
    #     t_start = word_entry["start"]
    #     t_end = word_entry["end"]
    #
    #     # Insert PAUSE token if gap exceeds threshold
    #     gap = t_start - prev_end
    #     if gap >= pause_threshold_sec:
    #         sequence.append({
    #             "token": f"[PAUSE:{gap:.2f}s]",
    #             "token_type": "pause",
    #             "start_sec": prev_end,
    #             "end_sec": t_start,
    #             "pause_duration_sec": gap,
    #             "wav2vec_pooled": np.zeros(1024, dtype=np.float32).tolist(),
    #             "gemaps": gemaps_features.tolist(),
    #         })
    #
    #     # Pool wav2vec frames within [t_start, t_end]
    #     frame_rate = 1.0 / (frame_duration_ms / 1000.0)
    #     start_frame = int(t_start * frame_rate)
    #     end_frame = int(t_end * frame_rate)
    #     end_frame = min(end_frame, len(frame_embeddings) - 1)
    #
    #     if start_frame <= end_frame:
    #         word_frames = frame_embeddings[start_frame:end_frame + 1]
    #         pooled = word_frames.mean(axis=0)
    #     else:
    #         pooled = np.zeros(1024, dtype=np.float32)
    #
    #     sequence.append({
    #         "token": word,
    #         "token_type": "word",
    #         "start_sec": t_start,
    #         "end_sec": t_end,
    #         "wav2vec_pooled": pooled.tolist(),
    #         "gemaps": gemaps_features.tolist(),  # TODO: replace with windowed GeMAPS
    #     })
    #
    #     prev_end = t_end
    #
    # return sequence
    #
    # --- PLANNED ENHANCEMENT ---
    # Windowed GeMAPS: instead of using clip-level GeMAPS for all tokens,
    # extract GeMAPS over the specific time window [t_start, t_end] for each
    # word. Requires passing the raw audio to this function and calling
    # openSMILE with a custom window. This will significantly improve the
    # prosody branch's per-token resolution.
    raise NotImplementedError("build_word_sequence: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Persistence: save Level 1 record + feature arrays
# ─────────────────────────────────────────────────────────────────────────────

def save_record(record: AudioRecord, output_dir: Path) -> Path:
    """
    Serialise an AudioRecord to JSON. Validates schema before writing.
    Returns path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{record.clip_id}.json"
    with open(json_path, "w") as f:
        f.write(record.json(indent=2))
    return json_path


def save_features(
    clip_id: str,
    output_dir: Path,
    gemaps_raw: np.ndarray,
    gemaps_normalized: Optional[np.ndarray],
    wav2vec_pooled: np.ndarray,
    wav2vec_frames: Optional[np.ndarray],
    word_sequence: Optional[list],
) -> dict[str, Path]:
    """
    Save feature arrays as .npy files and return path mapping.
    Frame-level embeddings are large — only saved if config.save_frame_embeddings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    np.save(output_dir / f"{clip_id}_gemaps_raw.npy", gemaps_raw)
    paths["gemaps_raw"] = output_dir / f"{clip_id}_gemaps_raw.npy"

    if gemaps_normalized is not None:
        np.save(output_dir / f"{clip_id}_gemaps_norm.npy", gemaps_normalized)
        paths["gemaps_normalized"] = output_dir / f"{clip_id}_gemaps_norm.npy"

    np.save(output_dir / f"{clip_id}_wav2vec_pooled.npy", wav2vec_pooled)
    paths["wav2vec_pooled"] = output_dir / f"{clip_id}_wav2vec_pooled.npy"

    if wav2vec_frames is not None:
        np.save(output_dir / f"{clip_id}_wav2vec_frames.npy", wav2vec_frames)
        paths["wav2vec_frames"] = output_dir / f"{clip_id}_wav2vec_frames.npy"

    if word_sequence is not None:
        seq_path = output_dir / f"{clip_id}_word_sequence.json"
        with open(seq_path, "w") as f:
            json.dump(word_sequence, f)
        paths["word_sequence"] = seq_path

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Main pipeline: orchestrate all steps per dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> None:
    """
    Run the full preprocessing pipeline on one dataset.

    Iterates over all audio files in config.input_dir, applies steps 1-6,
    and writes outputs to config.output_dir. Skips files where output already
    exists unless config.overwrite_existing is True.

    Dataset-specific manifest loading (Common Voice TSV, MOSI SDK output, etc.)
    is handled by dataset adapters — see neurointent/preprocessing/adapters/.
    """
    logger.info(f"Starting pipeline | dataset={config.dataset} | split={config.split}")

    # TODO: implement pipeline orchestration
    #
    # 1. Load dataset manifest using the appropriate adapter:
    #    from neurointent.preprocessing.adapters import get_adapter
    #    adapter = get_adapter(config.dataset)
    #    clips = adapter.load_manifest(config.input_dir, config.split)
    #
    # 2. Load models (once, not per-clip):
    #    transcriber = WhisperTranscriber(config).load()
    #    embedder = AudioEmbeddingExtractor(config).load()
    #    gemaps = GeMAPSExtractor().load()
    #
    # 3. Load normalization stats if available:
    #    norm_stats = None
    #    if config.normalization_stats_path and config.normalization_stats_path.exists():
    #        with open(config.normalization_stats_path) as f:
    #            norm_stats = json.load(f)
    #
    # 4. Per-clip processing loop:
    #    for i, clip_meta in enumerate(clips):
    #        clip_id = clip_meta["clip_id"]
    #        wav_path = Path(clip_meta["wav_path"])
    #
    #        if not config.overwrite_existing:
    #            if (config.output_dir / f"{clip_id}.json").exists():
    #                continue
    #
    #        try:
    #            audio, sr = load_and_validate_audio(wav_path, config)
    #            result = transcriber.transcribe(audio)
    #            frames, pooled = embedder.extract(audio)
    #            raw_gemaps = gemaps.extract(wav_path)
    #            norm_gemaps = normalize_gemaps(raw_gemaps, norm_stats) if norm_stats else None
    #            word_seq = build_word_sequence(result["words"], frames, raw_gemaps)
    #
    #            record = AudioRecord(
    #                clip_id=clip_id,
    #                source_dataset=config.dataset,
    #                speaker_id=clip_meta["speaker_id"],
    #                audio_path=str(wav_path),
    #                sample_rate=sr,
    #                duration_sec=len(audio) / sr,
    #                channels=1,
    #                transcript=result["text"],
    #                transcript_source="whisper_auto",
    #                word_timestamps=result["words"],
    #                is_asd=clip_meta.get("is_asd", False),
    #                intent_label=clip_meta.get("intent_label"),
    #                consent_given=clip_meta.get("consent_given", True),
    #                split=config.split,
    #                preprocessing_status="complete",
    #            )
    #
    #            save_record(record, config.output_dir)
    #            save_features(
    #                clip_id, config.output_dir,
    #                gemaps_raw=raw_gemaps,
    #                gemaps_normalized=norm_gemaps,
    #                wav2vec_pooled=pooled,
    #                wav2vec_frames=frames if config.save_frame_embeddings else None,
    #                word_sequence=word_seq,
    #            )
    #
    #        except Exception as e:
    #            logger.warning(f"Skipping {clip_id}: {e}")
    #            continue
    #
    #        if (i + 1) % config.log_every_n == 0:
    #            logger.info(f"Processed {i + 1}/{len(clips)} clips")
    #
    # 5. Write final manifest JSON listing all successfully processed clip_ids:
    #    manifest = [f.stem for f in config.output_dir.glob("*.json")]
    #    manifest_path = config.output_dir / f"manifest_{config.split}.json"
    #    with open(manifest_path, "w") as f:
    #        json.dump(manifest, f, indent=2)
    #    logger.info(f"Manifest written: {manifest_path} ({len(manifest)} clips)")

    raise NotImplementedError("run_pipeline: TODO — see implementation notes above")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific synthetic pair generation (CMU-MOSI)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_asd_pair(
    wav_path: Path,
    output_path: Path,
    target_pitch_hz: float = 120.0,
    pause_stretch_factor: float = 2.5,
    energy_smoothing_window: int = 5,
) -> None:
    """
    Generate a synthetic ASD-style audio clip from a neurotypical source clip.

    Applies three prosody perturbations:
      1. Pitch flattening — reduce pitch variation to simulate flat prosody
      2. Pause extension — multiply silence gap durations by pause_stretch_factor
      3. Energy normalization — reduce amplitude variance to simulate monotone delivery

    The original and perturbed clips form a contrastive pair:
    same transcript, same content score, different delivery.
    This is the primary training signal for the decoupling loss.

    Args:
        wav_path: source WAV file (neurotypical speech)
        output_path: destination for perturbed WAV
        target_pitch_hz: target fundamental frequency for pitch flattening
        pause_stretch_factor: multiplier for silence durations
        energy_smoothing_window: window size for energy envelope smoothing
    """
    # TODO: implement prosody perturbation
    #
    # import parselmouth
    # from parselmouth.praat import call
    #
    # PITCH FLATTENING (parselmouth / Praat):
    # sound = parselmouth.Sound(str(wav_path))
    # manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    # pitch_tier = call(manipulation, "Extract pitch tier")
    # call(pitch_tier, "Remove points between", 0, sound.duration)
    # call(pitch_tier, "Add point", sound.duration / 2, target_pitch_hz)
    # call([pitch_tier, manipulation], "Replace pitch tier")
    # result = call(manipulation, "Get resynthesis (overlap-add)")
    #
    # PAUSE EXTENSION:
    # duration_tier = call(manipulation, "Extract duration tier")
    # # Identify silence regions and add duration points with stretch_factor
    # # Implementation requires detecting silence boundaries first using VAD
    # # then inserting duration multiplier points in silence regions
    #
    # ENERGY SMOOTHING:
    # import numpy as np
    # audio_array = np.array(result.values[0])
    # # Apply rolling mean to amplitude envelope
    # # This reduces the loud/quiet variation that signals emphasis in NT speech
    #
    # result.save(str(output_path), "WAV")
    raise NotImplementedError("generate_synthetic_asd_pair: TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuroIntent preprocessing pipeline")
    parser.add_argument("--dataset", required=True,
                        choices=["common_voice", "msp_podcast", "muse",
                                 "iemocap", "mosi", "semaine", "rehearsal"],
                        help="Dataset to process")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--whisper_model_path", default=None,
                        help="Path to fine-tuned Whisper checkpoint. "
                             "If not set, loads base model from HuggingFace.")
    parser.add_argument("--audio_encoder", default="facebook/wav2vec2-large-960h",
                        help="HuggingFace model ID for audio encoder. "
                             "Set after wav2vec2 vs HuBERT vs WavLM benchmark.")
    parser.add_argument("--normalization_stats", default=None, type=Path,
                        help="Path to normalization_stats.json (MSP+MuSe baseline). "
                             "Required for is_asd=True datasets.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    parser.add_argument("--no_frame_embeddings", action="store_true",
                        help="Skip saving frame-level wav2vec embeddings (saves ~10x storage).")

    args = parser.parse_args()

    config = PipelineConfig(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split=args.split,
        whisper_model_path=args.whisper_model_path,
        audio_encoder_name=args.audio_encoder,
        normalization_stats_path=args.normalization_stats,
        overwrite_existing=args.overwrite,
        save_frame_embeddings=not args.no_frame_embeddings,
    )

    run_pipeline(config)
