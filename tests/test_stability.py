import numpy as np
import pytest

from conftest import voiced_like, make_wav, data_url


def test_identical_input_identical_output(client):
    from conftest import voiced_like, make_wav, data_url
    raw = make_wav(voiced_like(2.0, seed=4))
    Q = {"feel": {"type": "noul", "instructions": "Does the speaker sound angry?"},
         "threat": {"type": "choice", "instructions": "How threatening?",
                    "criteria": {"calm": "calm", "uneasy": "uneasy", "tense": "tense",
                                 "threatening": "threatening"}}}
    body = {"state": {"audio": data_url(raw)}, "questions": Q}
    r1 = client.post("/v1/systemone", json=body).json()
    r2 = client.post("/v1/systemone", json=body).json()
    assert r1["answers"] == r2["answers"]
    assert abs(r1["confidence"] - r2["confidence"]) < 1e-9
