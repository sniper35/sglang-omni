# Testing the S2-Pro TTS Streaming Demo

## Network topology

```
Mac (browser) → SSH → box (host) → Docker container (code + GPU)
```

## Ports

| Port | Service |
|------|---------|
| 8000 | Backend API (sglang-omni) |
| 7899 | Gradio UI |

## 1. Unit tests (inside Docker, no server needed)

```bash
cd /sgl-workspace/sglang/sglang-omni
python -m pytest tests/test_playground_tts.py -v
```

30 tests covering: payload building, SSE parsing (wav/pcm/malformed), audio decoding, stream buffering, sample-rate propagation, and Gradio generator/history lifecycle.

## 2. Launch the server (inside Docker)

### Option A: All-in-one

```bash
cd /sgl-workspace/sglang/sglang-omni
./playground/tts/start.sh
```

### Option B: Manual (two terminals)

```bash
# Terminal 1: backend
cd /sgl-workspace/sglang/sglang-omni
sgl-omni serve --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml --port 8000

# Terminal 2: Gradio UI (wait for backend health check first)
cd /sgl-workspace/sglang/sglang-omni
python playground/tts/app.py --api-base http://localhost:8000 --port 7899
```

## 3. curl sanity check (inside Docker)

```bash
# Non-streaming — should save a WAV file
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output /tmp/test.wav

# Streaming SSE — should print data: lines with audio chunks
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?", "stream": true, "response_format": "pcm"}'
```

## 4. SSH tunnel from your Mac

Forward port 7899 through box into the Docker container:

```bash
# On your Mac:
ssh -L 7899:localhost:7899 <user>@box
```

If Docker doesn't use `--network host`, check the mapped port on box first:

```bash
# On box (outside Docker):
docker port <container_id> 7899
```

Then open **http://localhost:7899** in your Mac browser.

## 5. Browser tests

| Test | Steps | Expected |
|------|-------|----------|
| Non-streaming | Uncheck "Stream output", enter text, click Synthesize | Full audio plays after generation; chat shows WAV + timing |
| Streaming | Check "Stream output", enter text, click Synthesize | Audio plays progressively; no duplication at the end; chat shows WAV path + stats |
| Voice cloning (non-streaming) | Upload reference audio + transcript, uncheck streaming | Cloned voice in output |
| Voice cloning (streaming) | Upload reference audio + transcript, check streaming | Progressive cloned-voice playback |
| Empty text | Leave text blank, click Synthesize | Warning toast, no crash |
| Server down | Stop the backend, click Synthesize | Error shown in chat history |
| Toggle switch | Flip between streaming/non-streaming repeatedly | Both modes work, audio resets cleanly |
