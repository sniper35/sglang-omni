# CLAUDE.md — S2-Pro Web Demo Streaming & Refactoring

> **Issue**: sgl-project/sglang-omni#205
> **Title**: Add Streaming Support to S2 Pro Web Demo + Demo Code Refactoring

## Overview

The Gradio-based TTS playground (`playground/tts/`) currently only supports non-streaming audio generation. Users must wait for the entire audio to be generated before hearing anything. This task adds real-time streaming playback and refactors the demo code for maintainability.

The backend **already supports streaming** via SSE on `POST /v1/audio/speech` with `"stream": true`. The work is entirely on the Gradio frontend side. Backend changes are out of scope unless end-to-end verification reveals a bug.

---

## Architecture Context

### S2-Pro 3-Stage Pipeline

```
Text Input
  |
  v
[Stage 1: Preprocessing] (CPU)
  Text tokenization + reference audio encoding to VQ codes
  |
  v
[Stage 2: Dual-AR Engine] (GPU)
  Slow AR (Qwen3-based, 4B params): generates semantic tokens autoregressively
  Fast AR (400M params): predicts 9 residual codebook codes per semantic token
  |  (streaming: incremental vocoding via _maybe_build_incremental_audio_chunk)
  v
[Stage 3: Vocoder] (GPU/CPU)
  DAC codec: 10 codebook codes -> audio waveform
  |
  v
Audio Output (WAV/PCM/MP3/FLAC)
```

### Streaming Protocol (SSE)

When `"stream": true` is sent to `/v1/audio/speech`, the server returns SSE events. Each audio chunk is a self-describing envelope with **per-chunk format, mime_type, and sample_rate** — the client must not assume any particular format or rate.

```
data: {"id": "speech-xxx", "object": "audio.speech.chunk", "index": 0,
       "audio": {"data": "<base64>", "format": "wav", "mime_type": "audio/wav", "sample_rate": 44100},
       "finish_reason": null}

data: {"id": "speech-xxx", "object": "audio.speech.chunk", "index": N,
       "audio": null, "finish_reason": "stop", "usage": {...}}

data: [DONE]
```

The `format` field reflects what was actually encoded — it may differ from the requested `response_format` if a codec is unavailable (e.g. requested mp3, server fell back to wav). The `sample_rate` comes from the model pipeline and can vary by model (S2-Pro uses 44100, other models use 24000). See `openai_api.py:501-516` and `test_openai_api.py:278-320`.

**Design choice for the streaming client**: request `response_format: "pcm"` for the streaming path (like vllm-omni does at `gradio_demo.py:55`). PCM is the lightest format — no headers, no codec overhead — and trivially decoded. For non-streaming, keep `"wav"` since it's a single file download. In both cases, read `sample_rate` from the SSE envelope rather than hardcoding.

### Key Files

| File | Role |
|------|------|
| `playground/tts/app.py` | **Gradio web UI** — the file being modified |
| `playground/tts/start.sh` | Launch script (backend + UI) — no changes needed |
| `sglang_omni/serve/openai_api.py` | API server — `/v1/audio/speech` endpoint (already supports streaming) |
| `sglang_omni/serve/protocol.py` | `CreateSpeechRequest` model — has `stream: bool` field |
| `sglang_omni/client/audio.py` | Audio utilities (encode/decode helpers) |
| `sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py` | Pipeline stages including streaming vocoder |

### Reference Implementation

The vllm-omni project has a working Gradio streaming demo at:
- `vllm-omni/examples/online_serving/fish_speech/gradio_demo.py`

Key patterns from that implementation:
- Uses `httpx.Client.stream()` for HTTP streaming
- Prebuffers 2 initial chunks for smooth playback
- Uses `gr.Audio(streaming=True, autoplay=True)` for real-time output
- **Locks format to PCM when streaming is enabled** (`gradio_demo.py:55`)
- Uses `chunk_seconds=0.5` for buffering threshold

**Important difference**: vllm-omni streams raw PCM bytes directly, but sglang-omni streams **SSE events with JSON payloads** containing base64-encoded audio chunks. The sglang-omni client must parse SSE lines, extract base64 data, and decode per the advertised format.

---

## Implementation Plan

### Step 1: Verify backend streaming end-to-end

Before writing any code, confirm the SSE endpoint works:
```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "stream": true, "response_format": "pcm"}'
```
Keep backend changes out unless this verification fails.

### Step 2: Extract API client module (`playground/tts/api_client.py`)

Separate transport/decoding from UI:

- `build_payload()` — constructs request JSON; sets `response_format: "pcm"` when `stream=True`
- `synthesize_speech()` — non-streaming: single POST, returns WAV bytes
- `synthesize_speech_stream()` — streaming: opens HTTP stream, parses SSE lines, decodes audio per the advertised `format` and `sample_rate` in each chunk envelope, buffers and yields `(sample_rate, float32_numpy)` tuples
- `_decode_sse_audio()` — parses one SSE `data:` line, returns `(sample_rate, float32_array)` or None. Reads `format` and `sample_rate` from the envelope rather than assuming WAV/44100.
- `_audio_bytes_to_numpy()` — dispatches on format: pcm (trivial int16 decode), wav (stdlib wave module), other (soundfile fallback)

### Step 3: Refactor `app.py` UI

Keep `app.py` focused on Gradio layout and event wiring:
- Import from `api_client.py`
- Single `_dispatch()` generator that routes to streaming or non-streaming path
- `gr.Audio(streaming=True, autoplay=True)` output works for both modes
- Stream toggle checkbox, chat history for both modes
- Propagate sample_rate from chunks (not hardcoded)

### Step 4: Add unit tests (`tests/test_playground_tts.py`)

Cover the new failure-prone logic without needing a running server:

- **`test_build_payload_*`** — basic fields, ref_audio inclusion, stream forces pcm
- **`test_decode_sse_audio_wav`** — WAV envelope → correct numpy + sample_rate
- **`test_decode_sse_audio_pcm`** — PCM envelope → correct numpy + sample_rate
- **`test_decode_sse_audio_no_audio`** — finish-reason-only event → None
- **`test_decode_sse_audio_malformed`** — bad JSON → None (no crash)
- **`test_audio_bytes_to_numpy_*`** — pcm, wav16, wav32, soundfile fallback
- **`test_synthesize_speech_stream_buffering`** — mock HTTP response with SSE lines, verify prebuffering (first yield after 2 chunks), chunk accumulation, flush of remaining samples
- **`test_synthesize_speech_stream_sample_rate_propagation`** — verify sample_rate from SSE envelope is passed through, not hardcoded

### Step 5: Manual E2E verification

With a running server, test both modes via the Gradio UI (see Testing section).

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `playground/tts/api_client.py` | **CREATE** | HTTP client with format-aware SSE streaming |
| `playground/tts/app.py` | **REWRITE** | Refactored UI with streaming toggle, uses api_client |
| `tests/test_playground_tts.py` | **CREATE** | Unit tests for payload, SSE parsing, buffering |
| `playground/tts/start.sh` | **NO CHANGE** | Already works |

---

## Technical Constraints

1. **Sample Rate**: Read from per-chunk SSE envelope (`audio.sample_rate`), do not hardcode. S2-Pro uses 44100, but other models may use 24000.
2. **Streaming Format**: Request `response_format: "pcm"` for streaming (lightweight, no header overhead). Server may fall back to WAV if PCM encoding fails — client must handle both.
3. **SSE Envelope**: Each chunk's `audio.format` is the *actual* format used, which may differ from the requested format. Always decode by advertised format.
4. **Gradio Streaming**: `gr.Audio(streaming=True)` expects `(sample_rate, numpy_array)` tuples from a generator.
5. **Prebuffering**: First 2 chunks should be buffered before playback starts to avoid choppy audio.
6. **Byte Alignment**: When handling PCM data, ensure 2-byte alignment (int16 samples).
7. **Chat History**: Gradio Chatbot with multimodal content uses `{"path": ..., "mime_type": ...}` format.

## Testing

### Unit Tests (no server needed)

```bash
python -m pytest tests/test_playground_tts.py -v
```

Covers: payload building, SSE parsing (wav/pcm/malformed), audio decoding, stream buffering, sample-rate propagation.

### Manual E2E Tests

```bash
# Start server
sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000

# Verify streaming endpoint
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?", "stream": true, "response_format": "pcm"}'

# Launch Gradio UI
python playground/tts/app.py --api-base http://localhost:8000
```

| Test Case | Steps | Expected |
|---|---|---|
| Non-streaming | Enter text, uncheck "Stream output", click Synthesize | Full audio plays after generation; chat history shows WAV + timing |
| Streaming | Enter text, check "Stream output", click Synthesize | Audio plays progressively; chat shows duration/chunk stats |
| Voice cloning (both modes) | Upload reference audio + transcript | Cloned voice in output |
| Empty text | Leave text blank, click Synthesize | Warning toast, no crash |
| Server down | Stop the server, click Synthesize | Error shown in chat history |
| Toggle switch | Flip between modes repeatedly | Both work, audio component resets |

---

## Build & Run

```bash
# Install dependencies
uv pip install -e ".[s2pro]"

# Download model
hf download fishaudio/s2-pro

# Launch full playground (backend + UI)
./playground/tts/start.sh

# Or launch components separately:
sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000
python playground/tts/app.py --api-base http://localhost:8000 --port 7899
```

## Code Style

- Apache-2.0 license header on all new files
- Type hints on all function signatures
- Use `from __future__ import annotations` for modern type syntax
- Use `httpx` (already a dependency) for HTTP client
- Use `numpy` for audio array manipulation
- Docstrings on public functions
- No unnecessary abstractions — keep it simple for a demo
