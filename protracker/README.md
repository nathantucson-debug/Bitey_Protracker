# ProTracker Module

This folder contains the Atari/ProTracker-style music generator as an isolated module.

## Entry points

- Integrated app: `/atari-tracker` on the main Flask service (`app/main.py`)
- Standalone app: `protracker/main.py`

## Run standalone

```bash
python3 protracker/main.py
```

Then open: `http://localhost:8080/atari-tracker`

## API

- `POST /api/atari/session` create full-length remix session
- `GET /api/atari/session/<session_id>/mix.wav` stream rendered mix
- `GET /api/atari/session/<session_id>/stem/<stem_name>.wav` stream stem WAV
- `POST /api/atari/session/<session_id>/export` export ZIP (mix + stems + session metadata)
- `GET /api/atari/preview` quick WAV preview
- `POST /api/atari/export` direct export path (supports optional `session_id`)

## Files

- `protracker/__init__.py` blueprint, synthesis engine, API routes
- `protracker/templates/atari_tracker.html` mixer UI
- `protracker/main.py` standalone Flask launcher
