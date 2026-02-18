#!/usr/bin/env python3
"""Compatibility entrypoint for the Gradio UI app."""

from lan_app.ui import app, demo, enroll_speaker, transcribe, transcribe_and_summarize

__all__ = [
    "app",
    "demo",
    "transcribe",
    "transcribe_and_summarize",
    "enroll_speaker",
]


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
