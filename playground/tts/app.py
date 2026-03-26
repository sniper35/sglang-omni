# SPDX-License-Identifier: Apache-2.0
"""Gradio TTS playground for S2-Pro — text-to-speech with voice cloning.

Supports both non-streaming (full audio) and streaming (progressive playback)
modes via the ``/v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import argparse
import tempfile
import time

import gradio as gr
import numpy as np

from api_client import (
    DEFAULT_API_BASE,
    build_payload,
    synthesize_speech,
    synthesize_speech_stream,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _dispatch(
    stream_on: bool,
    text: str,
    ref_audio: str | None,
    ref_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    history: list[dict],
    api_base: str,
):
    """Unified generator: yields ``(history, audio_output)`` tuples.

    * Non-streaming — one yield with full audio after synthesis completes.
    * Streaming — multiple yields with progressive audio chunks.

    The audio output is always ``(sample_rate, float32_numpy)`` which works
    with ``gr.Audio(streaming=True)``.
    """
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        yield history, None
        return

    payload = build_payload(
        text,
        ref_audio,
        ref_text,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stream=stream_on,
    )

    user_content: list = [text]
    if ref_audio is not None:
        user_content.append({"path": ref_audio, "mime_type": "audio/wav"})

    if stream_on:
        yield from _streaming_path(api_base, payload, history, user_content)
    else:
        yield from _non_streaming_path(api_base, payload, history, user_content)


def _non_streaming_path(api_base, payload, history, user_content):
    """Synthesize fully, then yield once with the result."""
    t0 = time.perf_counter()
    try:
        wav_bytes = synthesize_speech(api_base, payload)
    except Exception as exc:
        history = history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"Error: {exc}"},
        ]
        yield history, None
        return

    elapsed = time.perf_counter() - t0

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    history = history + [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [
                {"path": tmp.name, "mime_type": "audio/wav"},
                f"{elapsed:.1f}s | {len(wav_bytes) / 1024:.0f} KB",
            ],
        },
    ]
    # Yield as numpy for the streaming-capable audio component.
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    audio_np = np.frombuffer(pcm, dtype=np.int16).copy().astype(np.float32) / 32767.0
    yield history, (sr, audio_np)


def _streaming_path(api_base, payload, history, user_content):
    """Stream audio chunks, updating history at the end.

    After streaming completes the accumulated audio is saved to a temp WAV
    file and embedded in chat history so the result is replayable — matching
    the non-streaming path.
    """
    import io
    import wave

    # Immediately show the user's message with a "generating..." placeholder
    # *before* any audio arrives (avoids perceived-latency dead-time while
    # the prebuffer fills).
    pending_history = history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Generating (streaming)..."},
    ]
    yield pending_history, None

    t0 = time.perf_counter()
    chunk_count = 0
    all_samples: list[np.ndarray] = []
    last_sr = 24000  # updated from each chunk's envelope
    try:
        for sr, audio_chunk in synthesize_speech_stream(api_base, payload):
            chunk_count += 1
            all_samples.append(audio_chunk)
            last_sr = sr
            yield pending_history, (sr, audio_chunk)
    except Exception as exc:
        pending_history[-1] = {
            "role": "assistant",
            "content": f"Streaming error: {exc}",
        }
        yield pending_history, None
        return

    elapsed = time.perf_counter() - t0
    total_samples = sum(len(s) for s in all_samples)
    duration = total_samples / last_sr if total_samples else 0

    # Save the full audio to a temp WAV so it appears in chat history.
    tmp_path: str | None = None
    if all_samples:
        full_audio = np.concatenate(all_samples)
        pcm_int16 = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(last_sr)
            wf.writeframes(pcm_int16.tobytes())
        tmp.close()
        tmp_path = tmp.name

    # Finalize the history entry with the replayable audio file.
    assistant_content: list = []
    if tmp_path:
        assistant_content.append({"path": tmp_path, "mime_type": "audio/wav"})
    assistant_content.append(
        f"Streamed {duration:.1f}s audio in {elapsed:.1f}s "
        f"({chunk_count} chunks)"
    )
    pending_history[-1] = {
        "role": "assistant",
        "content": assistant_content,
    }

    # Final yield updates history only.  Do NOT re-send the full audio —
    # gr.Audio(streaming=True) accumulates all prior chunk yields internally,
    # so sending the concatenated waveform again would duplicate the playback.
    yield pending_history, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def create_demo(api_base: str) -> gr.Blocks:
    with gr.Blocks(title="S2-Pro TTS Playground") as demo:
        gr.Markdown("## S2-Pro Text-to-Speech")
        gr.Markdown(
            "*First request may take 10-20s due to warmup. "
            "Subsequent requests are much faster thanks to KV cache reuse.*",
            elem_classes=["note"],
        )

        with gr.Row():
            # Left column: input controls
            with gr.Column(scale=1, min_width=320):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=4,
                )

                gr.Markdown("#### Voice Cloning (optional)")
                ref_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="Reference Text",
                    placeholder="Transcript of the reference audio",
                    lines=2,
                )

                with gr.Accordion("Generation Parameters", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.05,
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                    )
                    top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        value=30,
                        step=1,
                    )
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=128,
                        maximum=4096,
                        value=2048,
                        step=128,
                    )

                stream_toggle = gr.Checkbox(
                    label="Stream output",
                    value=False,
                    info="Progressive audio playback during generation",
                )
                synth_btn = gr.Button("Synthesize", variant="primary")

            # Right column: outputs
            with gr.Column(scale=2, min_width=480):
                chatbot = gr.Chatbot(label="History", height=480)
                audio_output = gr.Audio(
                    label="Audio Output",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                )
                clear_btn = gr.Button("Clear History")

        # -- Event wiring --

        all_inputs = [
            stream_toggle,
            text_input,
            ref_audio,
            ref_text,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            chatbot,
        ]

        def handler(*args):
            yield from _dispatch(*args, api_base=api_base)

        synth_btn.click(
            fn=handler,
            inputs=all_inputs,
            outputs=[chatbot, audio_output],
        )
        text_input.submit(
            fn=handler,
            inputs=all_inputs,
            outputs=[chatbot, audio_output],
        )
        clear_btn.click(
            fn=lambda: ([], None),
            outputs=[chatbot, audio_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="S2-Pro TTS Gradio playground")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.api_base)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
