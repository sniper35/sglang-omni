# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the sglang-omni /v1/audio/speech endpoint.

Supports both non-streaming (full WAV response) and streaming (SSE with
incremental audio chunks) modes.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import wave
from typing import Any, Generator

import httpx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 300.0


def build_payload(
    text: str,
    ref_audio: str | None,
    ref_text: str,
    *,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    max_new_tokens: int = 2048,
    stream: bool = False,
) -> dict[str, Any]:
    """Build the ``/v1/audio/speech`` request payload.

    When *stream* is True the format is set to ``"pcm"`` — it is the
    lightest encoding (no headers, no codec overhead) and trivially
    decoded on the client side.
    """
    payload: dict[str, Any] = {
        "input": text,
        "voice": "default",
        "response_format": "pcm" if stream else "wav",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text.strip()
    return payload


def synthesize_speech(api_base: str, payload: dict[str, Any]) -> bytes:
    """Non-streaming TTS. Returns raw audio bytes (format per payload).

    Raises:
        httpx.HTTPStatusError: on non-2xx responses.
        httpx.ConnectError: if the server is unreachable.
    """
    resp = httpx.post(
        f"{api_base}/v1/audio/speech",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.content


def synthesize_speech_stream(
    api_base: str,
    payload: dict[str, Any],
    *,
    chunk_seconds: float = 0.5,
    prebuffer_chunks: int = 2,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Streaming TTS via SSE. Yields ``(sample_rate, float32_array)`` tuples.

    Each SSE event carries a self-describing audio envelope with per-chunk
    ``format``, ``mime_type``, and ``sample_rate``.  This function decodes
    by the advertised format, buffers small fragments, and yields pieces
    suitable for ``gr.Audio(streaming=True)``.
    """
    payload = {**payload, "stream": True}

    pending: list[np.ndarray] = []
    pending_len = 0
    chunks_yielded = 0
    prebuffer: list[np.ndarray] = []

    # sample_rate is read from the first SSE envelope; updated per chunk.
    sample_rate: int | None = None
    min_chunk_samples: int | None = None

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{api_base}/v1/audio/speech",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break

                result = _decode_sse_audio(data_str)
                if result is None:
                    continue
                chunk_sr, audio_samples = result

                # Track sample rate from the envelope.
                if sample_rate is None or chunk_sr != sample_rate:
                    sample_rate = chunk_sr
                    min_chunk_samples = int(sample_rate * chunk_seconds)

                pending.append(audio_samples)
                pending_len += len(audio_samples)

                if min_chunk_samples and pending_len >= min_chunk_samples:
                    audio_chunk = np.concatenate(pending)
                    pending.clear()
                    pending_len = 0

                    if chunks_yielded < prebuffer_chunks:
                        prebuffer.append(audio_chunk)
                        chunks_yielded += 1
                        if chunks_yielded == prebuffer_chunks:
                            yield (sample_rate, np.concatenate(prebuffer))
                            prebuffer.clear()
                    else:
                        yield (sample_rate, audio_chunk)

    # Flush remaining samples.
    remaining = prebuffer + pending
    if remaining and sample_rate is not None:
        yield (sample_rate, np.concatenate(remaining))


def _decode_sse_audio(data_str: str) -> tuple[int, np.ndarray] | None:
    """Decode a single SSE ``data:`` payload.

    Returns ``(sample_rate, float32_numpy)`` or ``None`` when the event
    carries no audio (e.g. the final finish-reason-only event).
    """
    try:
        event = json.loads(data_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse SSE event: %s", data_str[:120])
        return None

    audio_obj = event.get("audio")
    if not audio_obj or not isinstance(audio_obj, dict):
        return None
    b64_data = audio_obj.get("data")
    if not b64_data:
        return None

    sample_rate = int(audio_obj.get("sample_rate", 24000))
    fmt = audio_obj.get("format", "wav")

    raw = base64.b64decode(b64_data)
    audio_np = _audio_bytes_to_numpy(raw, fmt)
    return sample_rate, audio_np


def _audio_bytes_to_numpy(raw: bytes, fmt: str) -> np.ndarray:
    """Convert encoded audio bytes to a float32 numpy array in [-1, 1]."""
    if fmt == "pcm":
        samples = np.frombuffer(raw, dtype=np.int16).copy()
        return samples.astype(np.float32) / 32767.0

    # WAV — use the stdlib wave module (no extra deps).
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
            sw = wf.getsampwidth()
        if sw == 2:
            samples = np.frombuffer(pcm, dtype=np.int16).copy()
            return samples.astype(np.float32) / 32767.0
        elif sw == 4:
            samples = np.frombuffer(pcm, dtype=np.int32).copy()
            return samples.astype(np.float32) / 2147483647.0
    except Exception:
        pass

    # Last resort: try soundfile if available (handles mp3, flac, etc.).
    try:
        import soundfile as sf

        arr, _ = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr
    except Exception as exc:
        raise ValueError(f"Cannot decode audio (format={fmt})") from exc
