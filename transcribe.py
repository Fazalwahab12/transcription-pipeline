"""
Transcription pipeline: audio (WAV/MP3/...) -> text with per-segment timestamps.

Engineering choices:
- Open-source model via faster-whisper (CTranslate2 backend) -> fast, local, no API cost.
  Swap-in OpenAI Whisper API is trivial (see transcribe_api()).
- ffmpeg does format normalization, so we accept anything ffmpeg can decode.
- Whisper emits segment-level timestamps natively; we also support word-level.
- Long audio is handled by Whisper's internal 30s windowing + optional VAD to skip silence.

Run:
    pip install faster-whisper
    # ffmpeg must be installed on the system (brew install ffmpeg / apt install ffmpeg)
    python transcribe.py path/to/audio.mp3 --model base --vad
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict


@dataclass
class Segment:
    start: float   # seconds
    end: float     # seconds
    text: str


def to_wav_16k_mono(input_path: str) -> str:
    """Normalize ANY ffmpeg-decodable format to 16 kHz mono PCM WAV.
    This is why we don't care about the input container/codec (mp3, wav, m4a, ogg...)."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-f", "wav", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def transcribe(input_path: str, model_size: str = "base", use_vad: bool = True,
               word_timestamps: bool = False) -> dict:
    """Transcribe an audio file and return text + per-segment timestamps."""
    from faster_whisper import WhisperModel

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    wav = to_wav_16k_mono(input_path)
    try:
        # CPU int8 is a good default; use device="cuda" where a GPU is available.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        seg_iter, info = model.transcribe(
            wav,
            vad_filter=use_vad,          # skip long silences -> speed + accuracy on long files
            word_timestamps=word_timestamps,
            beam_size=5,
        )
        segments = [Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in seg_iter]
    finally:
        os.remove(wav)  # clean up temp file

    return {
        "language": info.language,
        "duration_sec": round(info.duration, 2),
        "text": " ".join(s.text for s in segments).strip(),
        "segments": [asdict(s) for s in segments],
    }


def transcribe_api(input_path: str) -> dict:
    """Alternative: OpenAI Whisper API (verbose_json gives segment timestamps).
    Shown to illustrate the pluggable 'engine' boundary - same output contract."""
    from openai import OpenAI
    client = OpenAI()
    with open(input_path, "rb") as f:
        r = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return {
        "language": r.language,
        "duration_sec": r.duration,
        "text": r.text,
        "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in r.segments],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audio -> text with timestamps")
    p.add_argument("audio", help="Path to audio file (wav, mp3, m4a, ...)")
    p.add_argument("--model", default="base", help="tiny|base|small|medium|large-v3")
    p.add_argument("--vad", action="store_true", help="Enable voice-activity detection")
    p.add_argument("--words", action="store_true", help="Also emit word-level timestamps")
    a = p.parse_args()
    result = transcribe(a.audio, a.model, a.vad, a.words)
    print(json.dumps(result, indent=2, ensure_ascii=False))
