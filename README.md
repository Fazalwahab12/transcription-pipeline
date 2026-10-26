# Transcription Pipeline

A simple, production-minded pipeline that converts audio (WAV, MP3, M4A, OGG, …) into
text **with per-segment timestamps**, ready for downstream use (subtitles, search, syncing).

The focus here is on **engineering decisions**, not training a model from scratch.

---

## Quick start

```bash
# 1. ffmpeg must be installed on the system
brew install ffmpeg        # macOS
# sudo apt install ffmpeg  # Debian/Ubuntu

# 2. Python deps
pip install faster-whisper

# 3. Run
python transcribe.py path/to/audio.mp3 --model base --vad
```

Output is JSON on stdout:

```json
{
  "language": "en",
  "duration_sec": 12.34,
  "text": "full transcript ...",
  "segments": [
    { "start": 0.0,  "end": 3.2,  "text": "first segment" },
    { "start": 3.2,  "end": 7.8,  "text": "second segment" }
  ]
}
```

---

## Part 1 — Transcription pipeline (design decisions)

### 1. Accepting an audio file
A CLI/function entrypoint takes an audio path (an HTTP endpoint would take an upload).
I **don't special-case codecs**: `ffmpeg` decodes the input and normalizes it to
**16 kHz mono PCM WAV** — the format Whisper expects — so WAV, MP3, M4A, OGG, etc. all
work through one path. The file is validated up front and we **fail fast with a clear
error** on missing/corrupt input, then clean up the temp WAV afterward.

### 2. Transcribing speech to text
I use an open-source **Whisper** model via **faster-whisper** (CTranslate2 backend),
running locally on CPU (`int8`) or GPU — no API cost, data stays private. Model size is
a speed/accuracy trade-off (`tiny`/`base` for speed, `large-v3` for accuracy). The engine
sits behind **one function**, so swapping in the OpenAI Whisper API (same output contract,
see `transcribe_api()`) is a one-line change.

### 3. Timestamps per segment
faster-whisper emits **segment-level timestamps natively**. Each segment is collected as
`{start, end, text}` (seconds, rounded), plus the full joined text, and returned as the
JSON above. `word_timestamps=True` is available when word-level granularity is needed.
These timestamps make the output usable for subtitles/captions (SRT/VTT), search, and
syncing text to audio.

### 4. Different audio formats
Delegated to ffmpeg (see §1) — anything ffmpeg can decode is supported, which avoids
per-format bugs. `pydub`/`soundfile` are reasonable fallbacks if ffmpeg is unavailable.
Duration / sample-rate / channels are validated, failing fast on unsupported files.

### 5. Long audio files
Several layers:
1. Whisper processes audio in **30-second windows internally**, so long files work
   without loading everything into memory.
2. A **VAD (voice-activity-detection) filter** skips long silences — faster and more accurate.
3. For very long / streaming input: **chunk on silence boundaries** (so words aren't cut
   mid-sentence), transcribe chunks in parallel, and **offset each chunk's timestamps** by
   its start time before merging.
4. Stream results segment-by-segment and run it as a **queue-backed background job**
   (e.g. Redis/BullMQ) with progress updates instead of a blocking request.

---

## Part 2 — System design (scaling it up)

### How would you handle concurrent uploads?
Accept uploads **asynchronously**: the API saves the raw file to object storage (S3/GCS)
and **enqueues a job**, returning a `job_id` immediately — the HTTP request never blocks on
transcription. Transcription **workers** pull from a queue (Redis/BullMQ, SQS, or Celery)
and scale horizontally, so N concurrent uploads just mean more queued jobs, not contention.
Each upload gets a **UUID key** to avoid filename collisions, uploads are **streamed**
(not fully buffered in memory), plus per-user rate limits and a max file size. A concurrency
cap per worker (GPU/CPU bound) protects the model from overload while the queue absorbs bursts.

### How would you store audio and transcripts?
- **Audio → object storage** (S3/GCS), not the DB — cheap, durable, scales to large files.
  Stored by UUID key with a lifecycle rule to auto-expire raw audio after processing if not needed.
- **Transcripts + metadata → relational DB (Postgres).** A `jobs` table
  (`id, user_id, audio_key, status, language, duration, created_at, error`) and a `segments`
  table (`job_id, start, end, text`) — or segments as JSONB on the job for simpler reads.

This keeps audio bytes out of the DB, makes transcripts queryable (full-text search on
`text`), and lets us regenerate output formats (SRT/VTT/JSON) on demand.

### How do you retry or recover failed transcriptions?
Make transcription **idempotent and retryable**. Each job has a status
(`queued → processing → done/failed`) and an **attempt counter**. On failure the worker
catches the error, logs it, and the queue **retries with exponential backoff** up to a max
(e.g. 3 attempts); exhausted jobs go to a **dead-letter queue** for inspection/manual replay.
Because audio is persisted in object storage, a retry just re-reads the same key — no data
loss. Workers use **visibility timeouts/heartbeats** so a crashed worker's job is auto-requeued
rather than lost. Status + error are surfaced via the API, with alerting on the dead-letter queue.

### How would you expose this as an API?
A small **REST API**:
- `POST /transcriptions` — accepts the audio (multipart or a pre-signed upload URL),
  creates a job, returns `202 {job_id, status: "queued"}`.
- `GET /transcriptions/{id}` — returns status and, when done, the transcript
  `{text, segments, language, duration}`, with a `format` param for `json`/`srt`/`vtt`.
- Optional **webhook callback** or **SSE/websocket** to push completion instead of polling.

Add auth (API key/JWT), request validation, rate limiting, and pagination for listing.
Documented with **OpenAPI**. I'd build it with **FastAPI** — typed models + auto docs, and it
pairs naturally with the Python transcription code.

---

## Architecture at a glance

```
         upload                 enqueue                 pull job
 client ───────▶ API ──┬──────▶ object storage (S3/GCS) ──────┐
    ▲                  │                                       ▼
    │                  └──────▶ queue (Redis/SQS) ──────▶ transcription worker
    │   poll / webhook                                         │ (ffmpeg → faster-whisper)
    └──────────────────────────  Postgres (jobs, segments) ◀──┘
```

## Tech choices (summary)
| Concern            | Choice                          | Why |
|--------------------|---------------------------------|-----|
| Decoding/normalize | ffmpeg → 16 kHz mono WAV        | One path for every format |
| Model              | faster-whisper (CTranslate2)    | Fast, local, no API cost, pluggable |
| Timestamps         | Segment-level (native)          | Subtitles/search/sync |
| Long audio         | 30s windows + VAD + chunking    | Memory-safe, accurate |
| Storage            | S3/GCS + Postgres               | Bytes out of DB, queryable transcripts |
| Scale              | Queue + horizontal workers      | Absorbs bursts, no blocking |
| Reliability        | Retries + dead-letter queue     | Idempotent recovery |
| API                | FastAPI REST + OpenAPI          | Typed, documented, async-friendly |

## Files
- `transcribe.py` — the working pipeline (ffmpeg normalize → faster-whisper → JSON with timestamps).
- `README.md` — this document.
