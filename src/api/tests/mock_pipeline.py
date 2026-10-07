"""Test stub standing in for pipeline.run so API tests load no models and run offline. Returns fixed values."""


def stub_result(**overrides) -> dict:
    result = {
        "content_score": 0.63,
        "content_score_raw": 0.63,
        "content_score_source": "trained_head",
        "content_scorer_stamp": None,
        "content_label_source": "chalearn",
        "intent_label": None,
        "intent_status": "not_trained",
        "prosody_only_baseline_score": -0.21,
        "prosody_only_baseline_label": "illustrative comparison model, not an evaluator or vendor tool",
        "transcript": "stub transcript",
        "acoustic_observations": {
            "pitch_variation_stddev_norm": 0.21,
            "mean_unvoiced_segment_sec": 0.18,
            "voiced_segments_per_sec": 2.9,
        },
        "delivery_pattern": ["Pitch variation (normalised std dev, unitless): 0.21"],
        "explanation": {
            "method": "proxy",
            "explains": "prosody_only_baseline_score",
            "top_features": [{"feature": "loudness_sma3_amean", "contribution": 0.4}],
            "note": "stub",
        },
        "interpretation": "stub interpretation",
        "interpretation_source": "template",
        "smoke_artifacts": False,
    }
    result.update(overrides)
    return result


def run(audio_path: str, mode: str) -> dict:
    return stub_result()
