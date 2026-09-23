import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import data_url, make_wav, voiced_like  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from monica.server import app

    with TestClient(app) as c:
        yield c


QUESTIONS = {
    "feel": {"type": "noul", "instructions": "Does the speaker sound angry?"},
    "threat": {
        "type": "choice",
        "instructions": "How threatening is the tone overall?",
        "criteria": {
            "calm": "calm, no threat",
            "uneasy": "slightly uneasy",
            "tense": "tense",
            "threatening": "actively threatening",
        },
    },
    "tension": {
        "type": "score",
        "instructions": "Rate vocal tension.",
        "criteria": ["completely relaxed", "slightly tense", "noticeably tense",
                     "very tense", "extremely tense"],
    },
}


def _body(dur=1.5, seed=1):
    return {
        "model": "monica/gemma4-e2b-systemone-v1",
        "state": {"audio": data_url(make_wav(voiced_like(dur, seed=seed)))},
        "questions": QUESTIONS,
    }


def test_missing_audio_rejected(client):
    r = client.post("/v1/systemone", json={"questions": QUESTIONS})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_audio"


def test_bad_data_url_rejected(client):
    r = client.post("/v1/systemone", json={"state": {"audio": "nonsense"}, "questions": QUESTIONS})
    assert r.status_code == 400
    assert r.json()["error"]["code"] in ("malformed_data_url", "decode_error")


def test_non_wav_mime_rejected(client):
    r = client.post("/v1/systemone",
                    json={"state": {"audio": "data:audio/mpeg;base64,AAAA"}, "questions": QUESTIONS})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_valid_request(client):
    r = client.post("/v1/systemone", json=_body())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model"]
    assert set(data["answers"]) == {"feel", "threat", "tension"}
    assert abs(sum(data["answers"]["threat"]["probabilities"].values()) - 1.0) < 1e-4
    assert abs(sum(data["answers"]["tension"]["probabilities"].values()) - 1.0) < 1e-4
    assert 0.0 <= data["answers"]["feel"]["noul"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert "audio" in data
    assert data["usage"]["audio_encoder_calls"] == 1
    assert data["usage"]["generated_tokens"] == 0
    assert data["audio"]["channels"] == 1
    assert data["model"]


def test_empty_questions(client):
    r = client.post("/v1/systemone", json={"state": {"audio": data_url(make_wav(voiced_like(1.0)))},
                                            "questions": {}})
    assert r.status_code == 200
    assert r.json()["answers"] == {}
    assert r.json()["usage"]["audio_encoder_calls"] == 0


def test_audio_format_variants_agree(client):
    import io as _io

    import soxr
    import soundfile as sf

    x = voiced_like(1.2, seed=2)
    mono16 = _io.BytesIO()
    sf.write(mono16, x, 16000, format="WAV", subtype="PCM_16")
    stereo = _io.BytesIO()
    sf.write(stereo, np.stack([x, x * 0.8], axis=1), 16000, format="WAV")
    s48 = _io.BytesIO()
    sf.write(s48, soxr.resample(x, 16000, 48000), 48000, format="WAV")

    r = client.post("/v1/systemone", json={"state": {"audio": data_url(mono16.getvalue())},
                                            "questions": QUESTIONS})
    assert r.status_code == 200
    base = np.array(list(r.json()["answers"]["threat"]["probabilities"].values()))

    for name, raw in [("stereo", stereo.getvalue()), ("sr48", s48.getvalue())]:
        rr = client.post("/v1/systemone", json={"state": {"audio": data_url(raw)},
                                                "questions": QUESTIONS})
        assert rr.status_code == 200, rr.text
        assert rr.json()["audio"]["channels"] == (2 if name == "stereo" else 1)
        p = np.array(list(rr.json()["answers"]["threat"]["probabilities"].values()))
        # JS divergence between formats. MPS matmul is not bit-deterministic
        # across processes, and resampled containers round-trip through the
        # frontend's internal 16k resampler; near-tie decisions can wobble.
        m = (base + p) / 2
        js = float((base * np.log((base + 1e-9) / m) + p * np.log((p + 1e-9) / m)).sum() / 2)
        tol = 0.06 if name in ("sr48", "sr8k") else 0.02
        assert js < tol, f"{name} JS={js:.4f} >= {tol}: {p} vs {base}"


def test_music_input_low_information(client):
    from conftest import music_like

    r = client.post("/v1/systemone",
                    json={"state": {"audio": data_url(make_wav(music_like(2.0)))},
                          "questions": QUESTIONS})
    assert r.status_code == 200
    data = r.json()
    assert data["audio"]["music_probable"] is True
    assert data["audio"]["low_information"] is True
    assert data["usage"]["low_information"] is True
    assert data["answers"]["threat"]["confidence"] == 0.0


def test_unknown_question_type(client):
    r = client.post("/v1/systemone", json={"state": {"audio": data_url(make_wav(voiced_like(1.0)))},
                                             "questions": {"weird": {"type": "weird", "instructions": "?"}}})
    assert r.status_code == 422


def test_missing_options(client):
    r = client.post("/v1/systemone", json={"state": {"audio": data_url(make_wav(voiced_like(1.0)))},
                                             "questions": {"choice": {"type": "choice", "instructions": "?"}}})
    assert r.status_code == 422


def test_audio_too_long_rejected_not_truncated(client):
    long = voiced_like(125.0, seed=3)
    r = client.post("/v1/systemone", json={"state": {"audio": data_url(make_wav(long))},
                                             "questions": QUESTIONS})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "audio_too_long"
