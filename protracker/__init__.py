import hashlib
import io
import json
import math
import random
import re
import threading
import time
import uuid
import wave
import zipfile
from array import array
from datetime import datetime, timezone
from urllib import parse, request

from flask import Blueprint, Response, jsonify, render_template, request as flask_request

protracker_bp = Blueprint("protracker", __name__, template_folder="templates")

SPOTIFY_TRACK_RE = re.compile(r"^[A-Za-z0-9]{22}$")
ATARI_TRACK_NAMES = ["kick", "snare", "hat", "bass", "arp", "chord", "lead", "fx"]
ATARI_NOTE_POOL = [36, 38, 41, 43, 45, 48, 50, 53, 55, 57, 60, 62, 65, 67, 69, 72]
ATARI_DURATION_RE = re.compile(r'"duration_ms"\s*:\s*(\d{4,8})')
ATARI_SESSION_TTL_SECONDS = 3600
ATARI_SESSION_CACHE: dict[str, dict] = {}
ATARI_SESSION_LOCK = threading.Lock()


def _midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def _extract_spotify_track_id(spotify_url: str) -> str:
    value = (spotify_url or "").strip()
    if not value:
        return ""
    if SPOTIFY_TRACK_RE.match(value):
        return value
    parsed = parse.urlparse(value)
    if parsed.netloc not in {"open.spotify.com", "spotify.com"}:
        return ""
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] == "track" and SPOTIFY_TRACK_RE.match(path_parts[1]):
        return path_parts[1]
    if parsed.fragment:
        frag = parsed.fragment.strip()
        if frag.startswith("track:"):
            candidate = frag.split(":", 1)[1]
            if SPOTIFY_TRACK_RE.match(candidate):
                return candidate
    return ""


def _fetch_spotify_duration_seconds(track_id: str) -> int | None:
    track_url = f"https://open.spotify.com/track/{track_id}"
    req = request.Request(
        track_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    matches = ATARI_DURATION_RE.findall(body)
    if not matches:
        return None
    try:
        duration_ms = int(matches[0])
    except ValueError:
        return None
    if duration_ms < 20000:
        return None
    return max(20, min(1200, int(round(duration_ms / 1000.0))))


def _cleanup_atari_sessions() -> None:
    cutoff = time.time() - ATARI_SESSION_TTL_SECONDS
    with ATARI_SESSION_LOCK:
        stale_ids = [sid for sid, payload in ATARI_SESSION_CACHE.items() if payload.get("created_at", 0.0) < cutoff]
        for sid in stale_ids:
            ATARI_SESSION_CACHE.pop(sid, None)


def _create_atari_session(remix: dict, source_url: str, duration_input: int | None) -> str:
    _cleanup_atari_sessions()
    session_id = uuid.uuid4().hex
    payload = {
        "id": session_id,
        "created_at": time.time(),
        "source_url": source_url,
        "duration_input": duration_input,
        "track_id": remix["track_id"],
        "bpm": remix["bpm"],
        "sample_rate": remix["sample_rate"],
        "duration_seconds": remix["duration_seconds"],
        "duration_source": remix["duration_source"],
        "mix_wav": remix["mix_wav"],
        "stems_wav": remix["stems_wav"],
    }
    with ATARI_SESSION_LOCK:
        ATARI_SESSION_CACHE[session_id] = payload
    return session_id


def _get_atari_session(session_id: str) -> dict | None:
    _cleanup_atari_sessions()
    with ATARI_SESSION_LOCK:
        return ATARI_SESSION_CACHE.get(session_id)


def _simple_lowpass(samples: list[float], alpha: float = 0.22) -> list[float]:
    if not samples:
        return samples
    out = [0.0] * len(samples)
    prev = 0.0
    for i, sample in enumerate(samples):
        prev = prev + alpha * (sample - prev)
        out[i] = prev
    return out


def _bitcrush(samples: list[float], levels: int = 40, hold: int = 2) -> list[float]:
    if not samples:
        return samples
    out = [0.0] * len(samples)
    held = 0.0
    for i, sample in enumerate(samples):
        if i % max(1, hold) == 0:
            held = round(sample * levels) / levels
        out[i] = held
    return out


def _normalize(samples: list[float], ceiling: float = 0.92) -> list[float]:
    peak = max(max(samples, default=0.0), abs(min(samples, default=0.0)))
    if peak <= 0:
        return samples
    if peak > ceiling:
        scale = ceiling / peak
        return [s * scale for s in samples]
    return samples


def _pcm16_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    clipped = [max(-1.0, min(1.0, x)) for x in samples]
    int_samples = array("h", (int(s * 32767.0) for s in clipped))
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())
        return buf.getvalue()


def _add_tone(
    buffer: list[float],
    start_sample: int,
    length_samples: int,
    sample_rate: int,
    frequency: float,
    amp: float,
    wave_name: str,
    rng: random.Random,
) -> None:
    attack = int(sample_rate * 0.008)
    release = int(sample_rate * 0.04)
    sustain = max(0, length_samples - attack - release)
    phase = 0.0
    phase_inc = frequency / sample_rate
    flutter_rate = 0.18 + rng.random() * 0.3
    for i in range(length_samples):
        idx = start_sample + i
        if idx < 0 or idx >= len(buffer):
            break
        if i < attack:
            env = i / max(1, attack)
        elif i < attack + sustain:
            env = 1.0
        else:
            rel_pos = i - attack - sustain
            env = max(0.0, 1.0 - rel_pos / max(1, release))
        flutter = 1.0 + 0.003 * math.sin((2 * math.pi * flutter_rate * i) / sample_rate)
        phase += phase_inc * flutter
        phase_mod = phase % 1.0
        if wave_name == "square":
            sample = 1.0 if phase_mod < 0.5 else -1.0
        elif wave_name == "triangle":
            sample = 1.0 - 4.0 * abs(phase_mod - 0.5)
        elif wave_name == "saw":
            sample = 2.0 * phase_mod - 1.0
        else:
            sample = math.sin(2 * math.pi * phase_mod)
        buffer[idx] += sample * amp * env


def _add_noise_hit(
    buffer: list[float], start_sample: int, length_samples: int, amp: float, rng: random.Random, decay: float
) -> None:
    for i in range(length_samples):
        idx = start_sample + i
        if idx < 0 or idx >= len(buffer):
            break
        env = math.exp(-decay * i / max(1, length_samples))
        buffer[idx] += (rng.random() * 2.0 - 1.0) * amp * env


def _add_kick(buffer: list[float], start_sample: int, sample_rate: int, amp: float) -> None:
    length_samples = int(sample_rate * 0.22)
    phase = 0.0
    for i in range(length_samples):
        idx = start_sample + i
        if idx < 0 or idx >= len(buffer):
            break
        t = i / sample_rate
        freq = 118.0 * math.exp(-10.5 * t) + 42.0
        phase += freq / sample_rate
        env = math.exp(-14.0 * t)
        click = 0.35 * math.exp(-90.0 * t)
        buffer[idx] += (math.sin(2 * math.pi * phase) + click) * amp * env


def _build_atari_remix(spotify_url: str, duration_seconds: int | None = None) -> dict:
    track_id = _extract_spotify_track_id(spotify_url)
    if not track_id:
        raise ValueError("Invalid Spotify track URL. Use a link like https://open.spotify.com/track/...")

    seed_int = int(hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed_int)

    sample_rate = 16000
    bpm = rng.randint(90, 136)
    step_samples = int((60.0 / bpm) / 4.0 * sample_rate)
    if duration_seconds is not None:
        seconds_target = max(20, min(1200, int(duration_seconds)))
        duration_source = "user"
    else:
        spotify_duration = _fetch_spotify_duration_seconds(track_id)
        if spotify_duration is not None:
            seconds_target = spotify_duration
            duration_source = "spotify"
        else:
            seconds_target = 180
            duration_source = "fallback"

    total_steps = max(64, int(math.ceil((seconds_target * sample_rate) / max(1, step_samples))))
    total_samples = total_steps * step_samples
    duration_seconds_rendered = int(round(total_samples / sample_rate))
    cell_steps = min(total_steps, 16 * 16)
    cell_samples = cell_steps * step_samples

    root_idx = seed_int % len(ATARI_NOTE_POOL)
    root_note = ATARI_NOTE_POOL[root_idx]
    progression = [0, 5, 7, 3, 0, 8, 5, 7]
    stems_cell = {name: [0.0] * cell_samples for name in ATARI_TRACK_NAMES}

    for step in range(cell_steps):
        step_in_bar = step % 16
        bar_idx = step // 16
        section_idx = (bar_idx // 8) % 4
        chord_root = root_note + progression[bar_idx % len(progression)]
        start = step * step_samples

        is_intro = section_idx == 0
        is_break = section_idx == 2

        if step_in_bar in (0, 8) or (step_in_bar == 12 and rng.random() < 0.26):
            _add_kick(stems_cell["kick"], start, sample_rate, 0.85)
        if not is_intro and step_in_bar in (4,) and rng.random() < 0.3:
            _add_kick(stems_cell["kick"], start, sample_rate, 0.45)

        if step_in_bar in (4, 12) or (step_in_bar == 15 and rng.random() < 0.15):
            _add_noise_hit(stems_cell["snare"], start, int(sample_rate * 0.18), 0.55, rng, decay=5.2)
            _add_tone(
                stems_cell["snare"],
                start,
                int(sample_rate * 0.14),
                sample_rate,
                190.0,
                0.18,
                "triangle",
                rng,
            )

        if step_in_bar % 2 == 0 and not (is_break and step_in_bar in (0, 8)):
            hat_amp = 0.22 if step_in_bar not in (14,) else 0.3
            _add_noise_hit(stems_cell["hat"], start, int(sample_rate * 0.05), hat_amp, rng, decay=7.5)

        if step_in_bar in (0, 3, 8, 11) and not (is_intro and step_in_bar == 3):
            bass_note = chord_root - 12 + (0 if step_in_bar in (0, 8) else 7)
            _add_tone(
                stems_cell["bass"],
                start,
                int(step_samples * 1.6),
                sample_rate,
                _midi_to_hz(bass_note),
                0.35,
                "square",
                rng,
            )

        if step_in_bar % 2 == 0 and not is_break:
            arp_offsets = [0, 7, 12, 7]
            arp_note = chord_root + arp_offsets[(step_in_bar // 2) % len(arp_offsets)]
            _add_tone(
                stems_cell["arp"],
                start,
                int(step_samples * 0.9),
                sample_rate,
                _midi_to_hz(arp_note + 12),
                0.17,
                "square",
                rng,
            )

        if step_in_bar in (0, 8) and not (is_intro and step_in_bar == 8):
            chord_stack = [0, 4, 7]
            for n in chord_stack:
                _add_tone(
                    stems_cell["chord"],
                    start,
                    int(step_samples * 7.2),
                    sample_rate,
                    _midi_to_hz(chord_root + n + 12),
                    0.09,
                    "triangle",
                    rng,
                )

        if step_in_bar in (2, 6, 10, 14) and not is_intro:
            melody_offsets = [12, 10, 7, 14, 15, 12, 19, 17]
            note = chord_root + melody_offsets[(bar_idx + step_in_bar) % len(melody_offsets)]
            _add_tone(
                stems_cell["lead"],
                start,
                int(step_samples * 1.2),
                sample_rate,
                _midi_to_hz(note),
                0.18,
                "saw",
                rng,
            )

        if step_in_bar == 15 and bar_idx % 4 == 3:
            _add_noise_hit(stems_cell["fx"], start, int(sample_rate * 0.22), 0.27, rng, decay=2.0)

    processed_cell = {}
    for name, data in stems_cell.items():
        low = _simple_lowpass(data, alpha=0.17 if name in {"lead", "arp", "hat"} else 0.24)
        crushed = _bitcrush(low, levels=36 if name in {"lead", "arp", "bass"} else 42, hold=2)
        processed_cell[name] = _normalize(crushed, ceiling=0.88)

    processed_stems = {}
    for name, cell_data in processed_cell.items():
        repeats = int(math.ceil(total_samples / max(1, len(cell_data))))
        tiled = (cell_data * repeats)[:total_samples]
        processed_stems[name] = tiled

    mix = [0.0] * total_samples
    for name in ATARI_TRACK_NAMES:
        weight = 1.0
        if name == "hat":
            weight = 0.8
        if name == "fx":
            weight = 0.75
        for i, sample in enumerate(processed_stems[name]):
            mix[i] += sample * weight

    mix = _simple_lowpass(mix, alpha=0.21)
    mix = [math.tanh(sample * 1.15) for sample in mix]
    mix = _normalize(mix, ceiling=0.9)
    stems_wav = {name: _pcm16_wav_bytes(processed_stems[name], sample_rate) for name in ATARI_TRACK_NAMES}

    return {
        "track_id": track_id,
        "sample_rate": sample_rate,
        "bpm": bpm,
        "steps": total_steps,
        "duration_seconds": duration_seconds_rendered,
        "duration_source": duration_source,
        "stems_float": processed_stems,
        "stems_wav": stems_wav,
        "mix_float": mix,
        "mix_wav": _pcm16_wav_bytes(mix, sample_rate),
    }



def atari_tracker():
    return render_template("atari_tracker.html")


@protracker_bp.get("/atari-tracker")
def atari_tracker():
    return render_template("atari_tracker.html")


@protracker_bp.post("/api/atari/session")
def atari_session_create():
    payload = flask_request.get_json(silent=True) or {}
    spotify_url = (payload.get("spotify_url") or flask_request.form.get("spotify_url") or "").strip()
    duration_value = payload.get("duration_seconds") or flask_request.form.get("duration_seconds") or ""
    duration_seconds = None
    if str(duration_value).strip():
        try:
            duration_seconds = int(duration_value)
        except ValueError:
            return jsonify({"error": "duration_seconds must be an integer"}), 400

    try:
        remix = _build_atari_remix(spotify_url, duration_seconds=duration_seconds)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    session_id = _create_atari_session(remix, spotify_url, duration_seconds)
    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "track_id": remix["track_id"],
            "bpm": remix["bpm"],
            "duration_seconds": remix["duration_seconds"],
            "duration_source": remix["duration_source"],
            "tracks": ATARI_TRACK_NAMES,
            "mix_url": f"/api/atari/session/{session_id}/mix.wav",
            "stem_urls": {name: f"/api/atari/session/{session_id}/stem/{name}.wav" for name in ATARI_TRACK_NAMES},
            "export_url": f"/api/atari/session/{session_id}/export",
        }
    )


@protracker_bp.get("/api/atari/session/<session_id>/mix.wav")
def atari_session_mix(session_id: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404
    filename = f"atari-mix-{session['track_id']}.wav"
    return Response(
        session["mix_wav"],
        mimetype="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{filename}"', "X-Atari-BPM": str(session["bpm"])},
    )


@protracker_bp.get("/api/atari/session/<session_id>/stem/<stem_name>.wav")
def atari_session_stem(session_id: str, stem_name: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404
    if stem_name not in ATARI_TRACK_NAMES:
        return jsonify({"error": "unknown stem"}), 404
    filename = f"atari-stem-{stem_name}-{session['track_id']}.wav"
    return Response(
        session["stems_wav"][stem_name],
        mimetype="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@protracker_bp.post("/api/atari/session/<session_id>/export")
def atari_session_export(session_id: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_prefix = f"atari8track-{session['track_id']}-{stamp}"
    with io.BytesIO() as zip_buf:
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{file_prefix}/mix.wav", session["mix_wav"])
            manifest = {
                "source": session["source_url"],
                "track_id": session["track_id"],
                "engine": "Deterministic Atari-inspired procedural renderer",
                "bpm": session["bpm"],
                "duration_seconds": session["duration_seconds"],
                "duration_source": session["duration_source"],
                "sample_rate": session["sample_rate"],
                "tracks": ATARI_TRACK_NAMES,
            }
            zf.writestr(f"{file_prefix}/session.json", json.dumps(manifest, indent=2))
            for stem_name in ATARI_TRACK_NAMES:
                zf.writestr(f"{file_prefix}/stems/{stem_name}.wav", session["stems_wav"][stem_name])
        zip_bytes = zip_buf.getvalue()
    return Response(
        zip_bytes,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_prefix}.zip"'},
    )


@protracker_bp.get("/api/atari/preview")
def atari_preview():
    spotify_url = (flask_request.args.get("spotify_url") or "").strip()
    duration_value = flask_request.args.get("duration_seconds", "")
    duration_seconds = None
    if duration_value.strip():
        try:
            duration_seconds = int(duration_value)
        except ValueError:
            return jsonify({"error": "duration_seconds must be an integer"}), 400

    try:
        remix = _build_atari_remix(spotify_url, duration_seconds=duration_seconds)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    filename = f"atari-preview-{remix['track_id']}.wav"
    return Response(
        remix["mix_wav"],
        mimetype="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Atari-BPM": str(remix["bpm"]),
            "X-Atari-Duration-Source": remix["duration_source"],
            "X-Atari-Duration-Seconds": str(remix["duration_seconds"]),
        },
    )


@protracker_bp.post("/api/atari/export")
def atari_export():
    payload = flask_request.get_json(silent=True) or {}
    spotify_url = (payload.get("spotify_url") or flask_request.form.get("spotify_url") or "").strip()
    session_id = (payload.get("session_id") or flask_request.form.get("session_id") or "").strip()
    if session_id:
        session = _get_atari_session(session_id)
        if not session:
            return jsonify({"error": "session not found or expired"}), 404
        return atari_session_export(session_id)

    duration_value = payload.get("duration_seconds") or flask_request.form.get("duration_seconds") or ""
    duration_seconds = None
    if str(duration_value).strip():
        try:
            duration_seconds = int(duration_value)
        except ValueError:
            return jsonify({"error": "duration_seconds must be an integer"}), 400

    try:
        remix = _build_atari_remix(spotify_url, duration_seconds=duration_seconds)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_prefix = f"atari8track-{remix['track_id']}-{stamp}"
    with io.BytesIO() as zip_buf:
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{file_prefix}/mix.wav", remix["mix_wav"])
            manifest = {
                "source": spotify_url,
                "track_id": remix["track_id"],
                "engine": "Deterministic Atari-inspired procedural renderer",
                "bpm": remix["bpm"],
                "duration_seconds": remix["duration_seconds"],
                "duration_source": remix["duration_source"],
                "sample_rate": remix["sample_rate"],
                "tracks": ATARI_TRACK_NAMES,
            }
            zf.writestr(f"{file_prefix}/session.json", json.dumps(manifest, indent=2))
            for stem_name in ATARI_TRACK_NAMES:
                wav_data = remix["stems_wav"][stem_name]
                zf.writestr(f"{file_prefix}/stems/{stem_name}.wav", wav_data)

        zip_bytes = zip_buf.getvalue()

    return Response(
        zip_bytes,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_prefix}.zip"'},
    )

