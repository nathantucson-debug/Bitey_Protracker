# ProTracker Module

This folder contains the Atari/ProTracker-style music generator as an isolated module.

## Input model

- Source is an uploaded WAV file (`source_wav` form field).
- Engine analyzes source key, tempo movement, and section energy.
- Output keeps source timeline length so it can be layered with the original in sync.

## Entry points

- Integrated app: `/atari-tracker` on the main Flask service (`app/main.py`)
- Standalone app: `protracker/main.py`

## Run standalone

```bash
python3 protracker/main.py
```

Then open: `http://localhost:8080/atari-tracker`

## API

- `POST /api/atari/session` (multipart form with `source_wav`) create remix session
- `GET /api/atari/session/<session_id>/mix.wav` stream rendered mix
- `GET /api/atari/session/<session_id>/stem/<stem_name>.wav` stream stem WAV
- `POST /api/atari/session/<session_id>/export` export ZIP (mix + stems + session metadata)
- `POST /api/atari/export` export by `session_id`

## Files

- `protracker/__init__.py` blueprint, WAV analysis engine, API routes
- `protracker/templates/atari_tracker.html` mixer UI
- `protracker/main.py` standalone Flask launcher
