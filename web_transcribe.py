#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────
# LAN Recording-Transcriber
#  * faster-whisper large-v3  (ASR)
#  * pyannote.audio 3.3       (diarization + speaker-ID)
#  * sentiment RoBERTa        (friendly score)
#  * Llama-3 via Ollama       (summary)
#  * Gradio 5.x UI
# -----------------------------------------------------------------------

import os, json, tempfile, itertools, shutil, subprocess, math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

try:
    import torch
except ModuleNotFoundError:                 # CI runner has no torch wheels
    import types, sys
    torch = types.ModuleType("torch")
    torch.__dict__.update(
        __version__="0.0.0-stub",
        device=lambda *a, **k: None,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = torch
try:
    import gradio as gr
except ModuleNotFoundError:                   # CI runner: stub gradio
    import types, sys
    gr = types.ModuleType("gradio")
    gr.Blocks = object             # minimal dummies so code parses
    gr.Interface = lambda *a, **k: None
    gr.Markdown  = lambda *a, **k: None
    gr.Audio     = lambda *a, **k: None
    gr.Button    = lambda *a, **k: None
    gr.File      = lambda *a, **k: None
    gr.launch    = lambda *a, **k: None
    sys.modules["gradio"] = gr
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model
from pyannote.audio.utils.signal import Binarize
from transformers import pipeline
import ollama


# ─── 0. Константы ──────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE  = "float16" if DEVICE == "cuda" else "int8"
DB_PATH       = Path.home() / "speakers.json"    # база эталонных голосов
VOICES_DIR    = Path.home() / "voices"           # сюда кладём эталоны вручную
NEW_VOICE_DIR = Path.home() / "unknown_voices"   # сюда скрипт пишет фрагменты
LLM_MODEL     = "llama3:8b"                      # тег в ollama

# пороги
EMB_THRESHOLD = 0.65    # косинус-distance  (меньше → тот же спикер)
MERGE_SIMILAR = 0.90    # если две соседние фразы совпадают >90 % → сливаем

# ─── 1. Загрузка длинных моделей (один раз) ───────────────────────────
print("⏳  Whisper large-v3")
asr = WhisperModel("large-v3", device=DEVICE, compute_type=COMPUTE_TYPE)

print("⏳  Pyannote diarization")
diar = Pipeline.from_pretrained("pyannote/speaker-diarization@3.2").to(DEVICE)

print("⏳  Speaker-embedding model")
embedder = Model.from_pretrained("pyannote/embedding").to(DEVICE)

print("⏳  Sentiment pipeline")
sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if DEVICE == "cuda" else -1,
)

print("⏳  Ollama client")
ollama_client = ollama.Client(host="http://127.0.0.1:11434")

# ─── 2. Загружаем / создаём базу спикеров ─────────────────────────────
def load_speaker_db() -> dict:
    if DB_PATH.exists():
        return json.load(open(DB_PATH))
    VOICES_DIR.mkdir(exist_ok=True)
    return {}

def save_speaker_db(db: dict):
    json.dump(db, open(DB_PATH, "w"))

SPEAKER_DB = load_speaker_db()


def embed_audio(wave: torch.Tensor, sr: int) -> np.ndarray:
    if sr != 16000:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sr, 16000)
    with torch.inference_mode():
        emb = embedder(wave.to(embedder.device)).mean(0).cpu().numpy()
    return emb / np.linalg.norm(emb)


def identify_speaker(embedding: np.ndarray) -> Tuple[str, float]:
    """Вернёт (имя, distance). Если база пуста → ('Speaker ?', inf)."""
    if not SPEAKER_DB:
        return "Speaker ?", math.inf
    names, embs = zip(*[(k, np.array(v)) for k, v in SPEAKER_DB.items()])
    dists = 1 - np.dot(embs, embedding)
    idx = int(np.argmin(dists))
    return (names[idx], float(dists[idx]))


# ─── 3. Анти-дубликатор для последовательных реплик ───────────────────
def merge_similar(lines: List[str]) -> List[str]:
    out = []
    for line in lines:
        if not out:
            out.append(line); continue
        prev = out[-1]
        # наивная метрика: доля одинаковых символов
        sim = sum(a == b for a, b in itertools.zip_longest(prev, line)) / max(len(prev), len(line))
        if sim >= MERGE_SIMILAR:
            continue
        out.append(line)
    return out


# ─── 4. Основная функция обработки ────────────────────────────────────
def transcribe(audio_path: str):
    fname = Path(audio_path).name
    stem  = Path(audio_path).stem
    print("→ ASR", fname)

    # 4-a. whisperX
    segments, info = asr.transcribe(audio_path, vad_filter=True, language="auto")
    # segments — generator; превращаем в список
    segments = list(segments)

    # 4-b. диаризация (сначала VAD→сегменты, потом assign speakers)
    diar_ann = diar(audio_path)
    diar_tracks = list(diar_ann.itertracks(yield_label=False))

    # 4-c. строим список строк + собираем unknowns
    md_lines, unknown_chunks = [], []
    import torchaudio
    full_wave, full_sr = torchaudio.load(audio_path)

    for seg, _ in diar_tracks:
        # текст кусочка
        text = whisperx.utils.get_segments({"segments": segments}, seg.start, seg.end).strip()
        if not text:
            continue
        # вычисляем эмбеддинг фрагмента
        start_smpl = int(seg.start * full_sr)
        end_smpl   = int(seg.end   * full_sr)
        emb = embed_audio(full_wave[:, start_smpl:end_smpl], full_sr)

        name, dist = identify_speaker(emb)
        if dist > EMB_THRESHOLD:
            # новый человек
            name = f"Speaker ?"
            # сохраним 3-сек фрагмент для удобства
            NEW_VOICE_DIR.mkdir(exist_ok=True)
            frag_path = NEW_VOICE_DIR / f"{stem}_{seg.start:.2f}.wav"
            torchaudio.save(frag_path, full_wave[:, start_smpl:end_smpl], full_sr)
            unknown_chunks.append(frag_path)

        md_lines.append(f"[{seg.start:0.02f}–{seg.end:0.02f}] **{name}:** {text}")

    # 4-d. пост-обработка
    md_lines = merge_similar(md_lines)
    md_transcript = "\n".join(md_lines)

    # 4-e. дружелюбность
    sent = sentiment(md_transcript[:4000])[0]  # длинные тексты обрезаем
    if sent["label"] == "positive":
        friendly = +sent["score"]
    elif sent["label"] == "negative":
        friendly = -sent["score"]
    else:
        friendly = 0.0

    # 4-f. summary через Ollama
    sys_prompt = (
        "You are an assistant who writes concise 5-8 bullet summaries of any audio transcript. "
        "Return only the list without extra explanation."
    )
    msg = (
        f"{sys_prompt}\n\nTRANSCRIPT:\n{md_transcript}\n\n"
        "SUMMARY:"
    )
    summary_resp = ollama_client.chat(model=LLM_MODEL, messages=[
        {"role": "user", "content": msg}
    ])
    summary = summary_resp["message"]["content"]

    # 4-g. сохраняем во временную папку
    tmp = Path(tempfile.mkdtemp(prefix="trs_"))
    (tmp / f"{stem}.md").write_text(md_transcript, encoding="utf-8")
    (tmp / f"{stem}_summary.md").write_text(summary, encoding="utf-8")

    return (
        f"### Summary  \n{summary}\n\n---\n\n"
        f"### Friendly-score: **{friendly:+.2f}**",
        md_transcript,
        tmp / f"{stem}_summary.md",
        tmp / f"{stem}.md",
        "\n".join(str(p) for p in unknown_chunks) or "—"
    )


# ─── 5. Обработчик добавления нового спикера ──────────────────────────
def enroll_speaker(voice_path: str, name: str):
    if not voice_path or not name:
        return gr.Info("Upload voice sample AND type the name first.")
    import torchaudio
    w, sr = torchaudio.load(voice_path)
    emb = embed_audio(w, sr)
    SPEAKER_DB[name.strip()] = emb.tolist()
    save_speaker_db(SPEAKER_DB)
    # копируем файл в voices/ (для человека)
    VOICES_DIR.mkdir(exist_ok=True)
    shutil.copy(voice_path, VOICES_DIR / f"{name}.wav")
    return gr.Success(f"Speaker **{name}** added. You can re-run transcription.")


# ─── 6. Gradio интерфейс ──────────────────────────────────────────────
with gr.Blocks(title="LAN Recording-Transcriber") as demo:
    gr.Markdown("## LAN Recording-Transcriber  \n"
                "_Offline: WhisperX · pyannote · Ollama_")

    with gr.Row():
        # левая колонка
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Drop WAV / MP3 here")
            btn_proc = gr.Button("Process", variant="primary")
            btn_clear = gr.Button("Clear")

        # правая колонка
        with gr.Column(scale=2):
            out_md   = gr.Markdown(label="Summary + friendly-score")
            out_full = gr.Markdown(label="Full transcript", elem_classes="scroll")
            file_sum = gr.File(label="Download summary.md")
            file_md  = gr.File(label="Download full.md")
            unknown  = gr.Markdown(label="New voices saved")

    # вкладка добавления спикера
    with gr.Accordion("Add speaker to database", open=False):
        with gr.Row():
            new_voice = gr.Audio(type="filepath", label="Voice sample (~5 sec)")
            new_name  = gr.Textbox(label="Person name")
        add_btn   = gr.Button("Add")
        add_out   = gr.Markdown()

    # связи
    btn_proc.click(transcribe, audio_in,
                   outputs=[out_md, out_full, file_sum, file_md, unknown])
    btn_clear.click(lambda: (None,)*5,
                    None, [audio_in, out_md, out_full, file_sum, file_md])

    add_btn.click(enroll_speaker, inputs=[new_voice, new_name],
                  outputs=add_out)

# немного CSS, чтобы transcript был прокручиваемым
demo.load(lambda: None, js="",
          css=".scroll {max-height: 65vh; overflow-y: auto;}")

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)