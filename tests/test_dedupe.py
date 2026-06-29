from undertone_audio.dedupe import hamming_distance_hex, text_signature_for_segments
from undertone_audio.schema import Segment


def test_text_simhash_tolerates_small_asr_drift():
    base = [
        Segment(
            segment_id="a",
            speaker_id="S1",
            start_ms=0,
            end_ms=1000,
            text="same meeting agenda decisions blockers and next steps",
        )
    ]
    drifted = [
        Segment(
            segment_id="a",
            speaker_id="S1",
            start_ms=0,
            end_ms=1000,
            text="same meeting agenda decision blockers next steps",
        )
    ]

    left = text_signature_for_segments(base)
    right = text_signature_for_segments(drifted)

    assert left is not None
    assert right is not None
    unrelated = text_signature_for_segments(
        [
            Segment(
                segment_id="a",
                speaker_id="S1",
                start_ms=0,
                end_ms=1000,
                text="cooking recipes grocery shopping weekend dinner plans",
            )
        ]
    )

    assert unrelated is not None
    assert left.algorithm == "simhash64-token-word4-v1"
    assert hamming_distance_hex(left.value, right.value) <= 20
    assert hamming_distance_hex(left.value, unrelated.value) > 20
