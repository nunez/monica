import numpy as np
import pytest

from conftest import make_wav, data_url, voiced_like, music_like

from monica.audioio import (AudioInputError, MAX_AUDIO_SECONDS, analyze_quality,
                              decode_audio, load_wav_bytes, normalize_intensity,
                              parse_data_url)


def test_parse_valid():
    raw = make_wav(voiced_like(0.5), 16000)
    mime, b = parse_data_url(data_url(raw))
    assert mime == "audio/wav"
    assert b == raw


def test_rejects_non_data_url():
    with pytest.raises(AudioInputError):
        parse_data_url("not a data url")


def test_rejects_non_wav_mime():
    with pytest.raises(AudioInputError):
        parse_data_url("data:audio/mpeg;base64,AAAA")


def test_rejects_bad_base64():
    with pytest.raises(AudioInputError):
        parse_data_url("data:audio/wav;base64,!!!not base64!!!")


def test_empty_payload():
    with pytest.raises(AudioInputError) as e:
        parse_data_url("data:audio/wav;base64,")
    assert e.value.code == "empty_audio"


def test_zero_length_audio():
    raw = make_wav(np.zeros(0, dtype=np.float32))
    with pytest.raises(AudioInputError):
        decode_audio(raw)


def test_decode_bit_depths():
    for sub in ["PCM_16", "PCM_24", "FLOAT"]:
        x = voiced_like(0.3)
        raw = make_wav(x, 16000, subtype=sub)
        audio, info = decode_audio(raw)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert abs(len(audio) / 16000 - 0.3) < 0.01


def test_multichannel_downmix():
    x = voiced_like(0.4)
    stereo = np.stack([x, x * -0.5], axis=1)
    raw = make_wav(stereo, 16000)
    audio, info = decode_audio(raw)
    assert audio.ndim == 1
    assert info["orig_channels"] == 2
    expected = normalize_intensity(((x + -0.5 * x) / 2.0).astype(np.float32))
    assert np.allclose(audio[: len(expected)], expected, atol=1e-4)


def test_resample_from_48k():
    x = voiced_like(0.5)  # 16 kHz reference
    x48 = np.repeat(x, 3)  # crude 48 kHz stand-in
    raw = make_wav(x48, 48000)
    audio, info = decode_audio(raw)
    assert info["orig_sr"] == 48000
    assert abs(len(audio) - len(x)) < 160  # within 10 ms


def test_long_audio_rejected_not_truncated():
    x = voiced_like(MAX_AUDIO_SECONDS + 1.0)
    with pytest.raises(AudioInputError) as e:
        decode_audio(make_wav(x))
    assert e.value.code == "audio_too_long"


def test_nan_rejected():
    x = voiced_like(0.5)
    x[10] = np.nan
    with pytest.raises(AudioInputError) as e:
        decode_audio(make_wav(x.astype(np.float32), subtype="FLOAT"))
    assert e.value.code == "corrupted_audio"


def test_quality_analysis_speech_like():
    q = analyze_quality(voiced_like(2.0))
    assert q.speech_ratio > 0.5
    assert not q.music_probable
    assert q.low_information is False


def test_quality_analysis_music_like():
    q = analyze_quality(music_like(2.0))
    assert q.music_probable is True
    assert q.low_information is True


def test_quality_analysis_silence():
    q = analyze_quality(np.zeros(16000, dtype=np.float32))
    assert q.low_information is True
