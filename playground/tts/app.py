# SPDX-License-Identifier: Apache-2.0
"""Gradio TTS playground for S2-Pro — text-to-speech with voice cloning.

Supports both non-streaming (full audio) and streaming (progressive playback)
modes via the ``/v1/audio/speech`` endpoint.

The event flow uses ``.then()`` chaining so that audio streaming yields only
to ``gr.Audio(streaming=True)`` — never alongside a chatbot update in the
same generator.  This avoids component-reset issues in Gradio where a
dual-output generator causes the streaming audio component to lose its
internal accumulation state on each chatbot re-render.
"""

from __future__ import annotations

import argparse
import io
import tempfile
import time
import wave

import gradio as gr
import numpy as np

from api_client import (
    DEFAULT_API_BASE,
    build_payload,
    synthesize_speech,
    synthesize_speech_stream,
)

# ---------------------------------------------------------------------------
# Step 1: Prepare — validate input, build payload, update chatbot
# ---------------------------------------------------------------------------


def _prepare(
    stream_on: bool,
    text: str,
    ref_audio: str | None,
    ref_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    history: list[dict],
) -> tuple[list[dict], dict | None]:
    """Validate input, build the API payload, and show a chatbot placeholder.

    Returns ``(updated_history, state_dict)`` where *state_dict* is ``None``
    when the input is invalid (empty text).
    """
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return history, None

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

    # Show the target text in the chatbot.  Do NOT embed the reference audio
    # file — an inline audio player in the chat history creates the false
    # impression that the reference recording is being played back, when in
    # fact the audio output is the synthesised voice-cloned speech.
    user_content: list = [text]
    if ref_audio is not None:
        user_content.append("(voice clone reference provided)")

    mode = "streaming" if stream_on else "non-streaming"
    new_history = history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"Generating ({mode})\u2026"},
    ]

    state = {
        "payload": payload,
        "stream_on": stream_on,
        "t0": time.perf_counter(),
        # Mutable accumulators — updated in-place by _synthesize.
        "samples": [],
        "sr": 24000,
        "chunk_count": 0,
        "wav_bytes": None,
        "error": None,
    }
    return new_history, state


# ---------------------------------------------------------------------------
# Step 2: Synthesize — generator that yields ONLY to the audio output
# ---------------------------------------------------------------------------


def _synthesize(state: dict | None, api_base: str):
    """Yield ``(sample_rate, float32_numpy)`` tuples to ``gr.Audio``.

    For streaming requests the generator yields multiple times (progressive
    playback).  For non-streaming it yields once with the full waveform.

    Accumulated samples and metadata are stored in *state* (mutated in-place)
    so that the follow-up :func:`_finalize` handler can build the chat-history
    entry without needing the audio component's internal buffer.
    """
    if state is None:
        return

    try:
        if state["stream_on"]:
            for sr, chunk in synthesize_speech_stream(api_base, state["payload"]):
                state["samples"].append(chunk)
                state["sr"] = sr
                state["chunk_count"] += 1
                yield (sr, chunk)
        else:
            wav_bytes = synthesize_speech(api_base, state["payload"])
            state["wav_bytes"] = wav_bytes
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
            audio_np = (
                np.frombuffer(pcm, dtype=np.int16).copy().astype(np.float32)
                / 32767.0
            )
            state["samples"].append(audio_np)
            state["sr"] = sr
            yield (sr, audio_np)
    except Exception as exc:
        state["error"] = str(exc)


# ---------------------------------------------------------------------------
# Step 3: Finalize — update chatbot with WAV file + stats
# ---------------------------------------------------------------------------


def _finalize(history: list[dict], state: dict | None) -> list[dict]:
    """Replace the chatbot placeholder with a replayable WAV and timing info."""
    if state is None or not history:
        return history

    elapsed = time.perf_counter() - state["t0"]

    if state["error"]:
        history[-1] = {
            "role": "assistant",
            "content": f"Error: {state['error']}",
        }
        return history

    samples = state["samples"]
    sr = state["sr"]

    # Save audio to a temp WAV so it appears in the chat history.
    tmp_path: str | None = None
    if state["stream_on"] and samples:
        full_audio = np.concatenate(samples)
        pcm_int16 = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_int16.tobytes())
        tmp.close()
        tmp_path = tmp.name
    elif state["wav_bytes"]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(state["wav_bytes"])
        tmp.close()
        tmp_path = tmp.name

    total = sum(len(s) for s in samples)
    duration = total / sr if total else 0

    assistant_content: list = []
    if tmp_path:
        assistant_content.append({"path": tmp_path, "mime_type": "audio/wav"})
    if state["stream_on"]:
        assistant_content.append(
            f"Streamed {duration:.1f}s audio in {elapsed:.1f}s "
            f"({state['chunk_count']} chunks)"
        )
    else:
        kb = len(state["wav_bytes"] or b"") / 1024
        assistant_content.append(f"{elapsed:.1f}s | {kb:.0f} KB")

    history[-1] = {"role": "assistant", "content": assistant_content}
    return history


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

        # Hidden state passed between the prepare → synthesize → finalize chain.
        synth_state = gr.State(None)

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

        # -- Event wiring via .then() chaining --
        #
        # The chain is:
        #   1. _prepare  → updates chatbot + synth_state  (non-generator)
        #   2. _synthesize → yields audio chunks           (generator, audio_output only)
        #   3. _finalize → updates chatbot with result     (non-generator)
        #
        # Crucially, the audio generator (_synthesize) targets ONLY audio_output.
        # This prevents the gr.Audio(streaming=True) component from resetting its
        # internal accumulation buffer on each chatbot re-render, which was the
        # root cause of audio degrading to noise after the first prebuffered chunk.

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

        def prepare(*args):
            return _prepare(*args)

        def synthesize(state):
            yield from _synthesize(state, api_base=api_base)

        def finalize(history, state):
            return _finalize(history, state)

        def _wire_chain(trigger):
            trigger.then(
                fn=synthesize,
                inputs=[synth_state],
                outputs=[audio_output],
            ).then(
                fn=finalize,
                inputs=[chatbot, synth_state],
                outputs=[chatbot],
            )

        _wire_chain(
            synth_btn.click(
                fn=prepare,
                inputs=all_inputs,
                outputs=[chatbot, synth_state],
            )
        )
        _wire_chain(
            text_input.submit(
                fn=prepare,
                inputs=all_inputs,
                outputs=[chatbot, synth_state],
            )
        )

        clear_btn.click(
            fn=lambda: ([], None, None),
            outputs=[chatbot, audio_output, synth_state],
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
