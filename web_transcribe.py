#!/usr/bin/env python3
"""Gradio UI wrapper for the LAN transcriber pipeline."""

import os
import sys
import types
import asyncio
import tempfile
from pathlib import Path

try:
    import httpx  # noqa: F401 - used in LLM client
except ModuleNotFoundError:  # pragma: no cover - CI stub
    sys.modules.setdefault("httpx", types.ModuleType("httpx"))

# When running in CI we stub heavy dependencies so the module imports.
if os.getenv("CI") == "true":  # pragma: no cover - CI stub

    class _Dummy:
        def __getattr__(self, _name):
            return _Dummy()

        def __call__(self, *a, **k):
            return _Dummy()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Stub(types.ModuleType):
        def __getattr__(self, _name):
            return _Dummy()

    def _fake(mod: str) -> None:
        sys.modules[mod] = _Stub(mod)

    for mod in (
        "torch",
        "torchvision",
        "torchaudio",
        "faster_whisper",
        "pyannote",
        "pyannote.audio",
        "pyannote.pipeline",
        "gradio",
        "numpy",
    ):
        _fake(mod)

from lan_transcriber import llm_client, pipeline

import gradio as gr  # type: ignore
from pyannote.audio import Pipeline  # type: ignore
import torch  # type: ignore


DEVICE = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"


def transcribe(audio_path: str):
    """Run the pipeline and adapt the result for the UI."""
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization@3.2").to(DEVICE)
    cfg = pipeline.Settings()
    result = asyncio.run(
        pipeline.run_pipeline(Path(audio_path), cfg, llm_client.LLMClient(), diar)
    )
    return (
        f"### Summary  \n{result.summary}\n\n---\n\n",
        f"### Friendly-score: **{result.friendly}**",
        result.body,
        result.summary_path,
        result.body_path,
        "\n".join(str(p) for p in result.unknown_chunks) or "—",
    )


async def transcribe_and_summarize(text: str) -> tuple[Path, Path]:
    """Simpler helper used in tests."""
    sys_prompt = (
        "You are an assistant who writes concise 5-8 bullet summaries of any audio transcript. "
        "Return only the list without extra explanation."
    )
    msg = await llm_client.generate(system_prompt=sys_prompt, user_prompt=text)
    summary = msg.get("content", "") if isinstance(msg, dict) else str(msg)
    tmp = Path(tempfile.mkdtemp(prefix="trs_"))
    sum_path = tmp / "summary.md"
    full_path = tmp / "full.md"
    sum_path.write_text(summary, encoding="utf-8")
    full_path.write_text(text, encoding="utf-8")
    return sum_path, full_path


def enroll_speaker(voice_path: str, name: str):
    """Dummy implementation kept for UI compatibility."""
    if not voice_path or not name:
        return "⚠️ Upload voice sample AND type the name first."
    cfg = pipeline.Settings()
    cfg.voices_dir.mkdir(exist_ok=True)
    import shutil

    shutil.copy(voice_path, cfg.voices_dir / f"{name}.wav")
    return f"✅ Speaker **{name}** added. You can re-run transcription."


with gr.Blocks(
    title="LAN Recording-Transcriber",
    css=".scroll {max-height: 65vh; overflow-y: auto;}",
) as demo:  # pragma: no cover - UI glue
    gr.Markdown(
        "## LAN Recording-Transcriber  \n_Offline: WhisperX · pyannote · external LLM_"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Drop WAV / MP3 here")
            btn_proc = gr.Button("Process", variant="primary")
            btn_clear = gr.Button("Clear")
        with gr.Column(scale=2):
            out_md = gr.Markdown(label="Summary + friendly-score")
            out_full = gr.Markdown(label="Full transcript", elem_classes="scroll")
            file_sum = gr.File(label="Download summary.md")
            file_md = gr.File(label="Download full.md")
            unknown = gr.Markdown(label="New voices saved")
            out_voice = gr.Markdown(label="Unknown speaker chunks")

    with gr.Accordion("Add speaker to database", open=False):
        with gr.Row():
            new_voice = gr.Audio(type="filepath", label="Voice sample (~5 sec)")
            new_name = gr.Textbox(label="Person name")
        add_btn = gr.Button("Add")
        add_out = gr.Markdown()

    btn_proc.click(
        transcribe,
        audio_in,
        outputs=[out_md, out_full, file_sum, file_md, unknown, out_voice],
    )
    btn_clear.click(
        lambda: (None,) * 5, None, [audio_in, out_md, out_full, file_sum, file_md]
    )
    add_btn.click(enroll_speaker, inputs=[new_voice, new_name], outputs=add_out)

    demo.load(lambda: None)

    demo.launch()
