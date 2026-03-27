# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the playground TTS client (playground/tts/api_client.py).

These tests exercise payload building, SSE parsing, audio decoding, and
stream buffering logic without requiring a running server.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import wave
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest

# Make the playground/tts package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "playground" / "tts"))

from api_client import (
    _audio_bytes_to_numpy,
    _decode_sse_audio,
    build_payload,
    synthesize_speech_stream,
)


def _import_app():
    """Import app.py with a mock ``gradio`` module so tests run without it."""
    import importlib
    import types

    # Build a minimal gr stub with Warning as a no-op.
    gr_stub = types.ModuleType("gradio")
    gr_stub.Warning = lambda *a, **kw: None  # type: ignore[attr-defined]
    gr_stub.State = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules.setdefault("gradio", gr_stub)

    import app as _app

    importlib.reload(_app)  # ensure it picks up the stub
    return _app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Encode int16 samples into a minimal WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def _make_pcm_bytes(samples: np.ndarray) -> bytes:
    """Encode int16 samples as raw PCM (no header)."""
    return samples.astype(np.int16).tobytes()


def _make_sse_line(
    audio_bytes: bytes,
    fmt: str = "wav",
    sample_rate: int = 44100,
    index: int = 0,
) -> str:
    """Build a single SSE ``data:`` line with an audio envelope."""
    mime = {"wav": "audio/wav", "pcm": "audio/L16"}.get(fmt, f"audio/{fmt}")
    payload = {
        "id": "speech-test",
        "object": "audio.speech.chunk",
        "index": index,
        "audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": fmt,
            "mime_type": mime,
            "sample_rate": sample_rate,
        },
        "finish_reason": None,
    }
    return f"data: {json.dumps(payload)}"


def _make_finish_line(index: int = 1) -> str:
    payload = {
        "id": "speech-test",
        "object": "audio.speech.chunk",
        "index": index,
        "audio": None,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
    }
    return f"data: {json.dumps(payload)}"


def _make_mock_stream_yields(
    sr: int, chunk_size: int, n_chunks: int
) -> list[tuple[int, np.ndarray]]:
    """Build the list of (sample_rate, numpy) tuples that
    synthesize_speech_stream would yield."""
    return [
        (sr, np.zeros(chunk_size, dtype=np.float32)) for _ in range(n_chunks)
    ]


# ---------------------------------------------------------------------------
# build_payload tests
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_basic_fields(self):
        p = build_payload("hello", None, "")
        assert p["input"] == "hello"
        assert p["voice"] == "default"
        assert p["response_format"] == "wav"
        assert p["stream"] is False
        assert "ref_audio" not in p
        assert "ref_text" not in p

    def test_ref_audio_included(self):
        p = build_payload("hi", "/tmp/ref.wav", "transcript text")
        assert p["ref_audio"] == "/tmp/ref.wav"
        assert p["ref_text"] == "transcript text"

    def test_ref_text_stripped(self):
        p = build_payload("hi", "/tmp/ref.wav", "  padded  ")
        assert p["ref_text"] == "padded"

    def test_ref_text_empty_not_included(self):
        p = build_payload("hi", "/tmp/ref.wav", "   ")
        assert "ref_text" not in p

    def test_stream_forces_pcm(self):
        p = build_payload("hello", None, "", stream=True)
        assert p["response_format"] == "pcm"
        assert p["stream"] is True

    def test_generation_params(self):
        p = build_payload(
            "hello",
            None,
            "",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_new_tokens=512,
        )
        assert p["temperature"] == 0.5
        assert p["top_p"] == 0.9
        assert p["top_k"] == 50
        assert p["max_new_tokens"] == 512


# ---------------------------------------------------------------------------
# _decode_sse_audio tests
# ---------------------------------------------------------------------------


class TestDecodeSSEAudio:
    def test_wav_envelope(self):
        samples = np.array([0, 1000, -1000, 500], dtype=np.int16)
        wav_bytes = _make_wav_bytes(samples, sample_rate=24000)
        line = _make_sse_line(wav_bytes, fmt="wav", sample_rate=24000)
        data_str = line[len("data: "):]

        result = _decode_sse_audio(data_str)
        assert result is not None
        sr, arr = result
        assert sr == 24000
        assert len(arr) == 4
        assert arr.dtype == np.float32

    def test_pcm_envelope(self):
        samples = np.array([100, -200, 300], dtype=np.int16)
        pcm_bytes = _make_pcm_bytes(samples)
        line = _make_sse_line(pcm_bytes, fmt="pcm", sample_rate=44100)
        data_str = line[len("data: "):]

        result = _decode_sse_audio(data_str)
        assert result is not None
        sr, arr = result
        assert sr == 44100
        assert len(arr) == 3
        assert arr.dtype == np.float32

    def test_no_audio_returns_none(self):
        """Finish-reason-only events have audio=null."""
        data_str = json.dumps(
            {"audio": None, "finish_reason": "stop", "usage": {}}
        )
        assert _decode_sse_audio(data_str) is None

    def test_empty_data_returns_none(self):
        data_str = json.dumps({"audio": {"data": "", "format": "wav"}})
        assert _decode_sse_audio(data_str) is None

    def test_malformed_json_returns_none(self):
        assert _decode_sse_audio("not json {{{") is None

    def test_sample_rate_defaults_to_24000(self):
        """If sample_rate is missing from the envelope, default to 24000."""
        samples = np.array([10, 20], dtype=np.int16)
        payload = {
            "audio": {
                "data": base64.b64encode(_make_pcm_bytes(samples)).decode(),
                "format": "pcm",
            }
        }
        result = _decode_sse_audio(json.dumps(payload))
        assert result is not None
        sr, _ = result
        assert sr == 24000


# ---------------------------------------------------------------------------
# _audio_bytes_to_numpy tests
# ---------------------------------------------------------------------------


class TestAudioBytesToNumpy:
    def test_pcm_int16(self):
        raw = np.array([32767, -32768, 0], dtype=np.int16).tobytes()
        arr = _audio_bytes_to_numpy(raw, "pcm")
        assert arr.dtype == np.float32
        assert len(arr) == 3
        np.testing.assert_allclose(arr[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(arr[1], -1.0, atol=1e-4)
        np.testing.assert_allclose(arr[2], 0.0, atol=1e-4)

    def test_wav_16bit(self):
        samples = np.array([100, -100, 0], dtype=np.int16)
        wav = _make_wav_bytes(samples, sample_rate=16000)
        arr = _audio_bytes_to_numpy(wav, "wav")
        assert len(arr) == 3
        assert arr.dtype == np.float32

    def test_wav_32bit(self):
        buf = io.BytesIO()
        samples = np.array([2147483647, -2147483648, 0], dtype=np.int32)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)
            wf.setframerate(44100)
            wf.writeframes(samples.tobytes())
        arr = _audio_bytes_to_numpy(buf.getvalue(), "wav")
        assert len(arr) == 3
        np.testing.assert_allclose(arr[0], 1.0, atol=1e-4)

    def test_unknown_format_without_soundfile_raises(self):
        """When neither wave nor soundfile can decode, ValueError is raised."""
        with pytest.raises(ValueError, match="Cannot decode audio"):
            _audio_bytes_to_numpy(b"not audio data", "opus")


# ---------------------------------------------------------------------------
# synthesize_speech_stream buffering tests
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Minimal mock for httpx streaming response context manager."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeClient:
    """Minimal mock for httpx.Client context manager."""

    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestSynthesizeSpeechStream:
    def _run_stream(
        self,
        sse_lines: list[str],
        chunk_seconds: float = 0.5,
        prebuffer_chunks: int = 2,
    ) -> list[tuple[int, np.ndarray]]:
        fake_resp = _FakeStreamResponse(sse_lines)
        fake_client = _FakeClient(fake_resp)

        with patch("api_client.httpx.Client", return_value=fake_client):
            return list(
                synthesize_speech_stream(
                    "http://fake:8000",
                    {"input": "hello", "stream": True, "response_format": "pcm"},
                    chunk_seconds=chunk_seconds,
                    prebuffer_chunks=prebuffer_chunks,
                )
            )

    def test_prebuffering(self):
        """First yield should combine prebuffer_chunks=2 chunks."""
        sr = 44100
        # Each chunk: 22050 samples = 0.5s at 44100 Hz.
        chunk_samples = 22050
        samples = np.zeros(chunk_samples, dtype=np.int16)
        pcm = _make_pcm_bytes(samples)

        lines = [
            _make_sse_line(pcm, fmt="pcm", sample_rate=sr, index=0),
            _make_sse_line(pcm, fmt="pcm", sample_rate=sr, index=1),
            _make_sse_line(pcm, fmt="pcm", sample_rate=sr, index=2),
            _make_finish_line(index=3),
            "data: [DONE]",
        ]

        results = self._run_stream(lines, chunk_seconds=0.5, prebuffer_chunks=2)

        # First yield: prebuffer of 2 chunks (44100 samples).
        # Second yield: third chunk (22050 samples).
        assert len(results) == 2
        assert results[0][0] == sr
        assert len(results[0][1]) == chunk_samples * 2
        assert len(results[1][1]) == chunk_samples

    def test_sample_rate_propagated(self):
        """Each yielded tuple must carry the sample_rate from the SSE envelope."""
        sr = 22050
        samples = np.zeros(sr, dtype=np.int16)  # 1s of audio
        pcm = _make_pcm_bytes(samples)

        lines = [
            _make_sse_line(pcm, fmt="pcm", sample_rate=sr, index=0),
            _make_finish_line(index=1),
            "data: [DONE]",
        ]

        results = self._run_stream(lines, chunk_seconds=0.1, prebuffer_chunks=1)
        assert len(results) >= 1
        for returned_sr, _ in results:
            assert returned_sr == sr

    def test_flush_remaining(self):
        """Remaining samples below chunk threshold are flushed at the end."""
        sr = 44100
        # Very small chunk: 100 samples, well below 0.5s threshold.
        samples = np.ones(100, dtype=np.int16)
        pcm = _make_pcm_bytes(samples)

        lines = [
            _make_sse_line(pcm, fmt="pcm", sample_rate=sr, index=0),
            _make_finish_line(index=1),
            "data: [DONE]",
        ]

        results = self._run_stream(lines, chunk_seconds=0.5, prebuffer_chunks=2)
        # Only one yield (the flush), since we never hit the prebuffer count
        # with enough samples.
        assert len(results) == 1
        assert len(results[0][1]) == 100

    def test_empty_stream(self):
        """A stream with no audio chunks yields nothing."""
        lines = [
            _make_finish_line(index=0),
            "data: [DONE]",
        ]
        results = self._run_stream(lines)
        assert results == []

    def test_wav_format_chunks(self):
        """WAV-formatted SSE chunks are decoded correctly."""
        sr = 24000
        samples = np.array([500, -500, 0] * 8000, dtype=np.int16)
        wav = _make_wav_bytes(samples, sample_rate=sr)

        lines = [
            _make_sse_line(wav, fmt="wav", sample_rate=sr, index=0),
            _make_finish_line(index=1),
            "data: [DONE]",
        ]

        results = self._run_stream(lines, chunk_seconds=0.1, prebuffer_chunks=1)
        assert len(results) >= 1
        total = sum(len(r[1]) for r in results)
        assert total == 24000
        for returned_sr, _ in results:
            assert returned_sr == sr


# ---------------------------------------------------------------------------
# Three-step lifecycle tests: _prepare → _synthesize → _finalize
# ---------------------------------------------------------------------------


class TestPrepare:
    """Tests for the _prepare step."""

    @pytest.fixture(autouse=True)
    def _load_app(self):
        self.app = _import_app()

    def test_empty_text_returns_none_state(self):
        """Empty text should return unchanged history and None state."""
        history, state = self.app._prepare(
            False, "   ", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert history == []
        assert state is None

    def test_streaming_placeholder(self):
        """Streaming mode should show 'Generating (streaming)...' placeholder."""
        history, state = self.app._prepare(
            True, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert state is not None
        assert state["stream_on"] is True
        assert len(history) == 2
        assert "streaming" in history[1]["content"].lower()

    def test_non_streaming_placeholder(self):
        """Non-streaming mode should show 'Generating (non-streaming)...'."""
        history, state = self.app._prepare(
            False, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert state is not None
        assert state["stream_on"] is False
        assert "non-streaming" in history[1]["content"].lower()

    def test_ref_audio_not_embedded(self):
        """Reference audio should NOT be embedded as an inline player."""
        history, state = self.app._prepare(
            True, "hello", "/tmp/ref.wav", "transcript", 0.8, 0.8, 30, 2048, []
        )
        user_content = history[0]["content"]
        # User content should include text but no audio file path dict.
        assert any(isinstance(item, str) and "hello" in item for item in user_content)
        assert not any(
            isinstance(item, dict) and "path" in item for item in user_content
        )

    def test_state_has_payload(self):
        """State dict must contain the API payload."""
        _, state = self.app._prepare(
            True, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert "payload" in state
        assert state["payload"]["input"] == "hello"
        assert state["payload"]["stream"] is True

    def test_state_has_accumulators(self):
        """State dict must have mutable accumulators for _synthesize."""
        _, state = self.app._prepare(
            True, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert state["samples"] == []
        assert state["chunk_count"] == 0
        assert state["error"] is None


class TestSynthesize:
    """Tests for the _synthesize step."""

    @pytest.fixture(autouse=True)
    def _load_app(self):
        self.app = _import_app()

    def test_none_state_returns_nothing(self):
        """When state is None (invalid input), nothing is yielded."""
        yields = list(self.app._synthesize(None, api_base="http://fake"))
        assert yields == []

    def test_streaming_yields_audio_chunks(self):
        """Streaming mode yields (sr, numpy) tuples from the stream."""
        sr = 44100
        chunks = _make_mock_stream_yields(sr, 1000, 3)

        state = {
            "payload": {"input": "hi", "stream": True, "response_format": "pcm"},
            "stream_on": True,
            "t0": time.perf_counter(),
            "samples": [],
            "sr": 24000,
            "chunk_count": 0,
            "wav_bytes": None,
            "error": None,
        }

        with patch(
            "app.synthesize_speech_stream", return_value=iter(chunks)
        ):
            yields = list(self.app._synthesize(state, api_base="http://fake"))

        assert len(yields) == 3
        for yielded_sr, arr in yields:
            assert yielded_sr == sr
            assert isinstance(arr, np.ndarray)

        # State should be mutated in-place with accumulated data.
        assert len(state["samples"]) == 3
        assert state["sr"] == sr
        assert state["chunk_count"] == 3

    def test_non_streaming_yields_once(self):
        """Non-streaming mode yields the full audio exactly once."""
        sr = 24000
        samples = np.array([100, -100, 0], dtype=np.int16)
        wav_bytes = _make_wav_bytes(samples, sample_rate=sr)

        state = {
            "payload": {"input": "hi", "stream": False, "response_format": "wav"},
            "stream_on": False,
            "t0": time.perf_counter(),
            "samples": [],
            "sr": 24000,
            "chunk_count": 0,
            "wav_bytes": None,
            "error": None,
        }

        with patch("app.synthesize_speech", return_value=wav_bytes):
            yields = list(self.app._synthesize(state, api_base="http://fake"))

        assert len(yields) == 1
        yielded_sr, arr = yields[0]
        assert yielded_sr == sr
        assert len(arr) == 3
        assert state["wav_bytes"] == wav_bytes

    def test_error_stored_in_state(self):
        """When synthesis fails, the error is stored in state (not raised)."""
        state = {
            "payload": {"input": "hi", "stream": False, "response_format": "wav"},
            "stream_on": False,
            "t0": time.perf_counter(),
            "samples": [],
            "sr": 24000,
            "chunk_count": 0,
            "wav_bytes": None,
            "error": None,
        }

        with patch(
            "app.synthesize_speech",
            side_effect=ConnectionError("no server"),
        ):
            yields = list(self.app._synthesize(state, api_base="http://fake"))

        assert yields == []
        assert "no server" in state["error"]


class TestFinalize:
    """Tests for the _finalize step."""

    @pytest.fixture(autouse=True)
    def _load_app(self):
        self.app = _import_app()

    def test_none_state_returns_history_unchanged(self):
        """When state is None, history is returned as-is."""
        history = [{"role": "user", "content": "hi"}]
        result = self.app._finalize(history, None)
        assert result is history

    def test_error_in_state(self):
        """When state has an error, history shows the error message."""
        history = [
            {"role": "user", "content": ["hi"]},
            {"role": "assistant", "content": "Generating..."},
        ]
        state = {
            "t0": time.perf_counter(),
            "stream_on": False,
            "samples": [],
            "sr": 24000,
            "chunk_count": 0,
            "wav_bytes": None,
            "error": "no server",
        }
        result = self.app._finalize(history, state)
        assert "Error" in result[-1]["content"]
        assert "no server" in result[-1]["content"]

    def test_streaming_finalize_has_wav_and_stats(self):
        """After streaming, the history must contain a WAV path and stats."""
        sr = 44100
        state = {
            "t0": time.perf_counter() - 1.0,  # 1 second ago
            "stream_on": True,
            "samples": [np.zeros(1000, dtype=np.float32)],
            "sr": sr,
            "chunk_count": 3,
            "wav_bytes": None,
            "error": None,
        }
        history = [
            {"role": "user", "content": ["hi"]},
            {"role": "assistant", "content": "Generating (streaming)..."},
        ]
        result = self.app._finalize(history, state)
        content = result[-1]["content"]

        assert isinstance(content, list)
        assert any(
            isinstance(item, dict) and item.get("mime_type") == "audio/wav"
            for item in content
        )
        assert any(isinstance(item, str) and "chunks" in item for item in content)

    def test_non_streaming_finalize_has_wav_and_stats(self):
        """After non-streaming, the history must contain a WAV path and stats."""
        sr = 24000
        samples = np.array([100, -100, 0], dtype=np.int16)
        wav_bytes = _make_wav_bytes(samples, sample_rate=sr)

        state = {
            "t0": time.perf_counter() - 0.5,
            "stream_on": False,
            "samples": [np.zeros(3, dtype=np.float32)],
            "sr": sr,
            "chunk_count": 0,
            "wav_bytes": wav_bytes,
            "error": None,
        }
        history = [
            {"role": "user", "content": ["hi"]},
            {"role": "assistant", "content": "Generating (non-streaming)..."},
        ]
        result = self.app._finalize(history, state)
        content = result[-1]["content"]

        assert isinstance(content, list)
        assert any(
            isinstance(item, dict) and item.get("mime_type") == "audio/wav"
            for item in content
        )
        assert any(isinstance(item, str) and "KB" in item for item in content)


class TestFullLifecycle:
    """End-to-end tests: _prepare → _synthesize → _finalize."""

    @pytest.fixture(autouse=True)
    def _load_app(self):
        self.app = _import_app()

    def test_streaming_lifecycle(self):
        """Full streaming lifecycle produces correct history and audio."""
        sr = 44100
        chunks = _make_mock_stream_yields(sr, 1000, 2)

        # Step 1: prepare
        history, state = self.app._prepare(
            True, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert state is not None
        assert len(history) == 2
        assert "streaming" in history[1]["content"].lower()

        # Step 2: synthesize
        with patch(
            "app.synthesize_speech_stream", return_value=iter(chunks)
        ):
            audio_yields = list(
                self.app._synthesize(state, api_base="http://fake")
            )

        # Audio chunks are yielded directly (no history tuples).
        assert len(audio_yields) == 2
        for yielded_sr, arr in audio_yields:
            assert yielded_sr == sr

        # Step 3: finalize
        history = self.app._finalize(history, state)
        content = history[-1]["content"]
        assert isinstance(content, list)
        assert any(
            isinstance(item, dict) and item.get("mime_type") == "audio/wav"
            for item in content
        )

    def test_non_streaming_lifecycle(self):
        """Full non-streaming lifecycle produces correct history and audio."""
        sr = 24000
        samples = np.array([100, -100, 0], dtype=np.int16)
        wav_bytes = _make_wav_bytes(samples, sample_rate=sr)

        # Step 1: prepare
        history, state = self.app._prepare(
            False, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )
        assert state is not None

        # Step 2: synthesize
        with patch("app.synthesize_speech", return_value=wav_bytes):
            audio_yields = list(
                self.app._synthesize(state, api_base="http://fake")
            )

        assert len(audio_yields) == 1
        assert audio_yields[0][0] == sr

        # Step 3: finalize
        history = self.app._finalize(history, state)
        content = history[-1]["content"]
        assert isinstance(content, list)
        assert any(
            isinstance(item, dict) and item.get("mime_type") == "audio/wav"
            for item in content
        )

    def test_error_lifecycle(self):
        """When synthesis fails, the error is shown in history."""
        # Step 1: prepare
        history, state = self.app._prepare(
            False, "hello", None, "", 0.8, 0.8, 30, 2048, []
        )

        # Step 2: synthesize (fails)
        with patch(
            "app.synthesize_speech",
            side_effect=ConnectionError("no server"),
        ):
            audio_yields = list(
                self.app._synthesize(state, api_base="http://fake")
            )

        assert audio_yields == []

        # Step 3: finalize
        history = self.app._finalize(history, state)
        assert "Error" in history[-1]["content"]
        assert "no server" in history[-1]["content"]
