# This is a mock version of Deval's pipeline.run() function
# Replace this with the real import once Deval's pipeline is ready
# It accepts audio_path and mode, and returns a hardcoded response dict

def run(audio_path: str, mode: str) -> dict:

    # Log what was received — useful for debugging during development
    print(f"[mock pipeline] called with audio_path={audio_path}, mode={mode}")

    # Return hardcoded response matching the exact NeuroIntent schema
    return {
        "content_quality_score": 0.87,
        "prosody_decoupling_applied": True,
        "confidence_bound": "medium",

        # Hardcoded — never changes, marks this as a research system
        "system_stage": "research_pilot",

        # Hardcoded — always true, required by project spec
        "candidate_disclosure_required": True,

        # Acoustic observations block — describes prosody without affecting score
        "acoustic_observations": {
            "flat_pitch_detected": True,
            "processing_pause_sec": 1.2,
            "content_score_unaffected": True
        },

        # Audit trail — explains what drove the content score
        "audit_trail": {
            "top_features": ["vocabulary_richness", "semantic_coherence", "argument_structure"],
            "content_attribution_share": 0.72,
            "decoupling_verified": True
        }
    }
