import hashlib
import io
import json
import math
import os
import random
import threading
import time
import uuid
import wave
import zipfile
from array import array
from datetime import datetime, timezone

from flask import Blueprint, Response, jsonify, render_template, request as flask_request, send_file

protracker_bp = Blueprint("protracker", __name__, template_folder="templates")

ATARI_TRACK_NAMES = ["kick", "snare", "hat", "bass", "arp", "chord", "lead", "fx"]
ATARI_SESSION_TTL_SECONDS = 3600
ATARI_SESSION_CACHE: dict[str, dict] = {}
ATARI_SESSION_LOCK = threading.Lock()
ATARI_SESSION_DIR = "/tmp/atari_sessions"
ATARI_JOB_TTL_SECONDS = 3600
ATARI_JOB_CACHE: dict[str, dict] = {}
ATARI_JOB_LOCK = threading.Lock()


def _midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def _cleanup_atari_sessions() -> None:
    cutoff = time.time() - ATARI_SESSION_TTL_SECONDS
    with ATARI_SESSION_LOCK:
        stale_ids = [sid for sid, payload in ATARI_SESSION_CACHE.items() if payload.get("created_at", 0.0) < cutoff]
        for sid in stale_ids:
            session = ATARI_SESSION_CACHE.get(sid) or {}
            session_dir = session.get("session_dir")
            if session_dir and os.path.isdir(session_dir):
                for root, dirs, files in os.walk(session_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except OSError:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except OSError:
                            pass
                try:
                    os.rmdir(session_dir)
                except OSError:
                    pass
            ATARI_SESSION_CACHE.pop(sid, None)


def _cleanup_atari_jobs() -> None:
    cutoff = time.time() - ATARI_JOB_TTL_SECONDS
    with ATARI_JOB_LOCK:
        stale_ids = [jid for jid, payload in ATARI_JOB_CACHE.items() if payload.get("created_at", 0.0) < cutoff]
        for jid in stale_ids:
            ATARI_JOB_CACHE.pop(jid, None)


def _set_job_status(job_id: str, payload: dict) -> None:
    with ATARI_JOB_LOCK:
        if job_id in ATARI_JOB_CACHE:
            ATARI_JOB_CACHE[job_id].update(payload)


def _start_atari_job(file_bytes: bytes, source_name: str) -> str:
    _cleanup_atari_jobs()
    job_id = uuid.uuid4().hex
    with ATARI_JOB_LOCK:
        ATARI_JOB_CACHE[job_id] = {
            "created_at": time.time(),
            "status": "queued",
            "source_name": source_name,
        }

    def worker() -> None:
        _set_job_status(job_id, {"status": "processing"})
        try:
            remix = _build_atari_remix_from_wav(file_bytes, source_name)
            session_id = _create_atari_session(remix, source_name)
            result = {
                "ok": True,
                "session_id": session_id,
                "source_id": remix["source_id"],
                "source_name": source_name,
                "bpm": remix["bpm"],
                "duration_seconds": remix["duration_seconds"],
                "estimated_key": remix["estimated_key"],
                "tempo_map_bpm": remix["tempo_map_bpm"],
                "tracks": ATARI_TRACK_NAMES,
                "mix_url": f"/api/atari/session/{session_id}/mix.wav",
                "stem_urls": {name: f"/api/atari/session/{session_id}/stem/{name}.wav" for name in ATARI_TRACK_NAMES},
                "export_url": f"/api/atari/session/{session_id}/export",
            }
            _set_job_status(job_id, {"status": "ready", "result": result})
        except ValueError as exc:
            _set_job_status(job_id, {"status": "failed", "error": str(exc)})
        except Exception:
            _set_job_status(job_id, {"status": "failed", "error": "Build failed on server. Try a shorter PCM WAV file."})

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _create_atari_session(remix: dict, source_name: str) -> str:
    _cleanup_atari_sessions()
    session_id = uuid.uuid4().hex
    os.makedirs(ATARI_SESSION_DIR, exist_ok=True)
    session_dir = os.path.join(ATARI_SESSION_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    stems_dir = os.path.join(session_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    mix_path = os.path.join(session_dir, "mix.wav")
    with open(mix_path, "wb") as f:
        f.write(remix["mix_wav"])

    stem_paths: dict[str, str] = {}
    for stem_name, stem_bytes in remix["stems_wav"].items():
        path = os.path.join(stems_dir, f"{stem_name}.wav")
        with open(path, "wb") as f:
            f.write(stem_bytes)
        stem_paths[stem_name] = path

    payload = {
        "id": session_id,
        "created_at": time.time(),
        "session_dir": session_dir,
        "source_name": source_name,
        "source_id": remix["source_id"],
        "bpm": remix["bpm"],
        "sample_rate": remix["sample_rate"],
        "duration_seconds": remix["duration_seconds"],
        "estimated_key": remix["estimated_key"],
        "tempo_map_bpm": remix["tempo_map_bpm"],
        "mix_path": mix_path,
        "stem_paths": stem_paths,
    }
    with ATARI_SESSION_LOCK:
        ATARI_SESSION_CACHE[session_id] = payload
    return session_id


def _get_atari_session(session_id: str) -> dict | None:
    _cleanup_atari_sessions()
    with ATARI_SESSION_LOCK:
        return ATARI_SESSION_CACHE.get(session_id)


def _simple_lowpass(samples: list[float], alpha: float = 0.22) -> array:
    if not samples:
        return array("f")
    out = array("f", [0.0]) * len(samples)
    prev = 0.0
    for i, sample in enumerate(samples):
        prev = prev + alpha * (sample - prev)
        out[i] = prev
    return out


def _bitcrush(samples: list[float], levels: int = 40, hold: int = 2) -> array:
    if not samples:
        return array("f")
    out = array("f", [0.0]) * len(samples)
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
        for i in range(len(samples)):
            samples[i] = samples[i] * scale
    return samples


def _pcm16_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    int_samples = array("h")
    int_samples.extend(int(max(-1.0, min(1.0, s)) * 32767.0) for s in samples)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())
        return buf.getvalue()


def _resample_linear(samples: list[float], src_rate: int, dst_rate: int) -> list[float]:
    if not samples or src_rate <= 0 or dst_rate <= 0:
        return samples
    if src_rate == dst_rate:
        return samples[:]
    ratio = src_rate / dst_rate
    dst_len = max(1, int(len(samples) * (dst_rate / src_rate)))
    out = [0.0] * dst_len
    for i in range(dst_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        if idx >= len(samples) - 1:
            out[i] = samples[-1]
        else:
            out[i] = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
    return out


def _read_wav_mono(file_bytes: bytes) -> tuple[list[float], int]:
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if channels < 1 or sample_rate <= 0:
                raise ValueError("Invalid WAV format")
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV file: {exc}") from exc

    if sample_width == 2:
        data = array("h")
        data.frombytes(raw)
        scale = 32768.0
        if channels == 1:
            mono = [max(-1.0, min(1.0, x / scale)) for x in data]
        else:
            mono = []
            for i in range(0, len(data), channels):
                frame = data[i : i + channels]
                mono.append(max(-1.0, min(1.0, (sum(frame) / len(frame)) / scale)))
    elif sample_width == 1:
        if channels == 1:
            mono = [((x - 128) / 128.0) for x in raw]
        else:
            mono = []
            for i in range(0, len(raw), channels):
                frame = raw[i : i + channels]
                avg = (sum(frame) / len(frame) - 128.0) / 128.0
                mono.append(max(-1.0, min(1.0, avg)))
    else:
        raise ValueError("Only 8-bit and 16-bit PCM WAV files are supported")

    if not mono:
        raise ValueError("WAV file has no audio samples")

    return mono, sample_rate


def _rms(block: list[float]) -> float:
    if not block:
        return 0.0
    return math.sqrt(sum(x * x for x in block) / len(block))


def _estimate_pitch_class_hist(samples: list[float], sample_rate: int) -> list[float]:
    frame = 2048
    hop = 1024
    lag_min = max(16, int(sample_rate / 500))
    lag_max = min(frame - 2, int(sample_rate / 70))
    hist = [0.0] * 12
    if len(samples) < frame:
        return hist

    for start in range(0, len(samples) - frame, hop):
        block = samples[start : start + frame]
        energy = _rms(block)
        if energy < 0.02:
            continue

        zc = 0
        prev = block[0]
        for x in block[1:]:
            if (prev <= 0 < x) or (prev >= 0 > x):
                zc += 1
            prev = x
        zcr = zc / len(block)
        if zcr > 0.22:
            continue

        best_lag = 0
        best_corr = -1e9
        for lag in range(lag_min, lag_max):
            corr = 0.0
            for i in range(frame - lag):
                corr += block[i] * block[i + lag]
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        if best_lag <= 0:
            continue

        freq = sample_rate / best_lag
        if freq < 70 or freq > 500:
            continue

        midi = int(round(69 + 12 * math.log2(freq / 440.0)))
        hist[midi % 12] += energy

    return hist


def _estimate_key(samples: list[float], sample_rate: int) -> tuple[str, int]:
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    hist = _estimate_pitch_class_hist(samples, sample_rate)
    if max(hist, default=0.0) <= 0:
        return "A minor", 57

    root_class = max(range(12), key=lambda i: hist[i])
    major_third = hist[(root_class + 4) % 12]
    minor_third = hist[(root_class + 3) % 12]
    is_major = major_third >= minor_third
    mode = "major" if is_major else "minor"

    root_midi = 48 + root_class
    while root_midi < 40:
        root_midi += 12
    while root_midi > 64:
        root_midi -= 12

    return f"{pitch_names[root_class]} {mode}", root_midi


def _energy_and_onsets(samples: list[float], sample_rate: int) -> tuple[list[float], list[float], float]:
    frame = 1024
    hop = 512
    if len(samples) < frame:
        e = [_rms(samples)]
        return e, [0.0], sample_rate / hop

    energies = []
    for start in range(0, len(samples) - frame, hop):
        energies.append(_rms(samples[start : start + frame]))

    smooth = _simple_lowpass(energies, alpha=0.35)
    onsets = [0.0] * len(smooth)
    for i in range(1, len(smooth)):
        d = smooth[i] - smooth[i - 1]
        onsets[i] = d if d > 0 else 0.0

    frames_per_second = sample_rate / hop
    return smooth, onsets, frames_per_second


def _estimate_tempo_map(onsets: list[float], fps: float, duration_seconds: int) -> tuple[list[float], float]:
    if not onsets or fps <= 0:
        return [120.0], 120.0

    seg_seconds = 8
    seg_frames = max(8, int(seg_seconds * fps))
    bpm_map = []
    min_lag = max(2, int((60.0 / 180.0) * fps))
    max_lag = min(int((60.0 / 70.0) * fps), max(3, seg_frames - 1))

    for seg_start in range(0, len(onsets), seg_frames):
        seg = onsets[seg_start : seg_start + seg_frames]
        if len(seg) < min_lag + 2:
            continue
        seg_mean = sum(seg) / max(1, len(seg))
        seg = [x - seg_mean for x in seg]

        best_lag = 0
        best_score = -1e9
        for lag in range(min_lag, max_lag):
            score = 0.0
            limit = len(seg) - lag
            if limit <= 0:
                break
            for i in range(limit):
                score += seg[i] * seg[i + lag]
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag > 0:
            bpm_map.append(60.0 * fps / best_lag)

    if not bpm_map:
        bpm_map = [120.0]

    bpm_map = [max(70.0, min(180.0, b)) for b in bpm_map]
    avg_bpm = sum(bpm_map) / len(bpm_map)

    target_segments = max(1, int(math.ceil(duration_seconds / 8.0)))
    if len(bpm_map) < target_segments:
        last = bpm_map[-1]
        bpm_map.extend([last] * (target_segments - len(bpm_map)))
    elif len(bpm_map) > target_segments:
        bpm_map = bpm_map[:target_segments]

    return bpm_map, avg_bpm


def _build_step_starts(total_samples: int, sample_rate: int, bpm_map: list[float]) -> list[int]:
    starts = []
    t = 0
    segments = max(1, len(bpm_map))
    while t < total_samples:
        starts.append(t)
        pos = t / max(1, total_samples)
        seg_idx = min(segments - 1, int(pos * segments))
        bpm = max(70.0, min(180.0, bpm_map[seg_idx]))
        step = int((60.0 / bpm) / 4.0 * sample_rate)
        t += max(8, step)
        if len(starts) > 10000:
            break
    return starts


def _sample_envelope(envelope: list[float], sample_index: int, total_samples: int) -> float:
    if not envelope:
        return 0.5
    pos = sample_index / max(1, total_samples)
    idx = min(len(envelope) - 1, int(pos * len(envelope)))
    v = envelope[idx]
    return max(0.0, min(1.0, v))


def _compress_range(values: list[float]) -> list[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


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


def _build_atari_remix_from_wav(file_bytes: bytes, source_name: str) -> dict:
    src_samples, src_rate = _read_wav_mono(file_bytes)
    duration_seconds = max(1, int(round(len(src_samples) / src_rate)))
    duration_seconds = max(10, min(240, duration_seconds))

    analysis_rate = 11025
    analysis_samples = _resample_linear(src_samples, src_rate, analysis_rate)
    if len(analysis_samples) > analysis_rate * 1200:
        analysis_samples = analysis_samples[: analysis_rate * 1200]

    key_name, root_note = _estimate_key(analysis_samples, analysis_rate)
    energies, onsets, fps = _energy_and_onsets(analysis_samples, analysis_rate)
    envelope = _compress_range(energies)
    tempo_map_bpm, avg_bpm = _estimate_tempo_map(onsets, fps, duration_seconds)

    source_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    rng = random.Random(int(hashlib.sha256((source_id + key_name).encode("utf-8")).hexdigest()[:12], 16))

    if duration_seconds >= 180:
        sample_rate = 8000
        long_mode = True
    elif duration_seconds >= 120:
        sample_rate = 10000
        long_mode = True
    else:
        sample_rate = 12000
        long_mode = False
    total_samples = duration_seconds * sample_rate
    step_starts = _build_step_starts(total_samples, sample_rate, tempo_map_bpm)
    progression_minor = [0, 3, 7, 10, 7, 5, 3, 0]
    progression_major = [0, 4, 7, 11, 7, 5, 4, 0]
    progression = progression_major if "major" in key_name else progression_minor

    stems_float = {name: array("f", [0.0]) * total_samples for name in ATARI_TRACK_NAMES}

    for step_idx, start in enumerate(step_starts):
        step_in_bar = step_idx % 16
        bar_idx = step_idx // 16
        dyn = _sample_envelope(envelope, start * analysis_rate // max(1, sample_rate), len(analysis_samples))
        chord_root = root_note + progression[(bar_idx // 2) % len(progression)]
        density_gate = 0.75 if long_mode else 1.0

        if step_in_bar in (0, 8) or (dyn > 0.55 and step_in_bar == 12 and rng.random() < 0.3):
            _add_kick(stems_float["kick"], start, sample_rate, 0.48 + dyn * 0.5)
        if step_in_bar in (4, 12) and dyn > 0.18 and rng.random() <= density_gate:
            _add_noise_hit(stems_float["snare"], start, int(sample_rate * 0.16), 0.35 + dyn * 0.33, rng, decay=5.2)
            _add_tone(stems_float["snare"], start, int(sample_rate * 0.1), sample_rate, 190.0, 0.12, "triangle", rng)

        if step_in_bar % 2 == 0 and dyn > 0.12 and (not long_mode or step_in_bar % 4 == 0):
            _add_noise_hit(stems_float["hat"], start, int(sample_rate * 0.045), 0.13 + dyn * 0.2, rng, decay=7.5)

        if step_in_bar in (0, 3, 8, 11) and dyn > 0.1:
            bass_note = chord_root - 12 + (0 if step_in_bar in (0, 8) else 7)
            _add_tone(stems_float["bass"], start, int(sample_rate * 0.16), sample_rate, _midi_to_hz(bass_note), 0.22 + dyn * 0.24, "square", rng)

        if step_in_bar % 2 == 0 and dyn > 0.22 and rng.random() <= density_gate:
            arp_offsets = [0, 7, 12, 7]
            arp_note = chord_root + arp_offsets[(step_in_bar // 2) % len(arp_offsets)] + 12
            _add_tone(stems_float["arp"], start, int(sample_rate * 0.11), sample_rate, _midi_to_hz(arp_note), 0.08 + dyn * 0.14, "square", rng)

        if step_in_bar in (0, 8) and dyn > 0.28 and rng.random() <= density_gate:
            for n in (0, 3 if "minor" in key_name else 4, 7):
                _add_tone(stems_float["chord"], start, int(sample_rate * (0.36 if long_mode else 0.62)), sample_rate, _midi_to_hz(chord_root + n + 12), 0.05 + dyn * 0.08, "triangle", rng)

        if step_in_bar in (2, 6, 10, 14) and dyn > 0.34 and rng.random() <= density_gate:
            melody_offsets = [12, 10, 7, 14, 15, 12, 19, 17]
            note = chord_root + melody_offsets[(bar_idx + step_in_bar) % len(melody_offsets)]
            _add_tone(stems_float["lead"], start, int(sample_rate * (0.14 if long_mode else 0.2)), sample_rate, _midi_to_hz(note), 0.08 + dyn * 0.14, "saw", rng)

        if step_in_bar == 15 and dyn > 0.42:
            _add_noise_hit(stems_float["fx"], start, int(sample_rate * 0.2), 0.12 + dyn * 0.13, rng, decay=2.4)

    processed_stems = {}
    for name, data in stems_float.items():
        low = _simple_lowpass(data, alpha=0.17 if name in {"lead", "arp", "hat"} else 0.24)
        crushed = _bitcrush(low, levels=34 if name in {"lead", "arp", "bass"} else 42, hold=2)
        processed_stems[name] = _normalize(crushed, ceiling=0.88)

    mix = array("f", [0.0]) * total_samples
    for name in ATARI_TRACK_NAMES:
        weight = 1.0
        if name == "hat":
            weight = 0.8
        if name == "fx":
            weight = 0.72
        track = processed_stems[name]
        for i in range(total_samples):
            mix[i] += track[i] * weight

    mix = _simple_lowpass(mix, alpha=0.21)
    for i in range(total_samples):
        mix[i] = math.tanh(mix[i] * 1.15)
    mix = _normalize(mix, ceiling=0.9)

    stems_wav = {name: _pcm16_wav_bytes(processed_stems[name], sample_rate) for name in ATARI_TRACK_NAMES}

    return {
        "source_id": source_id,
        "sample_rate": sample_rate,
        "bpm": int(round(avg_bpm)),
        "tempo_map_bpm": [round(v, 2) for v in tempo_map_bpm],
        "duration_seconds": duration_seconds,
        "estimated_key": key_name,
        "stems_wav": stems_wav,
        "mix_wav": _pcm16_wav_bytes(mix, sample_rate),
    }


@protracker_bp.get("/atari-tracker")
def atari_tracker():
    return render_template("atari_tracker.html")


@protracker_bp.post("/api/atari/session")
def atari_session_create():
    upload = flask_request.files.get("source_wav")
    if upload is None or not upload.filename:
        return jsonify({"error": "Attach a WAV file as source_wav"}), 400

    file_bytes = upload.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400
    if len(file_bytes) > 80 * 1024 * 1024:
        return jsonify({"error": "WAV file too large (max 80 MB)"}), 400

    job_id = _start_atari_job(file_bytes, upload.filename)
    return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 202


@protracker_bp.get("/api/atari/job/<job_id>")
def atari_job_status(job_id: str):
    _cleanup_atari_jobs()
    with ATARI_JOB_LOCK:
        job = ATARI_JOB_CACHE.get(job_id)
    if not job:
        return jsonify({"error": "job not found or expired"}), 404
    status = job.get("status", "queued")
    if status == "ready":
        return jsonify({"ok": True, "status": "ready", **(job.get("result") or {})})
    if status == "failed":
        return jsonify({"ok": False, "status": "failed", "error": job.get("error", "Build failed")}), 200
    return jsonify({"ok": True, "status": status}), 200


@protracker_bp.get("/api/atari/session/<session_id>/mix.wav")
def atari_session_mix(session_id: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404
    filename = f"atari-mix-{session['source_id']}.wav"
    try:
        return send_file(session["mix_path"], mimetype="audio/wav", as_attachment=False, download_name=filename)
    except OSError:
        return jsonify({"error": "session file missing"}), 410


@protracker_bp.get("/api/atari/session/<session_id>/stem/<stem_name>.wav")
def atari_session_stem(session_id: str, stem_name: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404
    if stem_name not in ATARI_TRACK_NAMES:
        return jsonify({"error": "unknown stem"}), 404
    filename = f"atari-stem-{stem_name}-{session['source_id']}.wav"
    path = (session.get("stem_paths") or {}).get(stem_name, "")
    try:
        return send_file(path, mimetype="audio/wav", as_attachment=False, download_name=filename)
    except OSError:
        return jsonify({"error": "session file missing"}), 410


@protracker_bp.post("/api/atari/session/<session_id>/export")
def atari_session_export(session_id: str):
    session = _get_atari_session(session_id)
    if not session:
        return jsonify({"error": "session not found or expired"}), 404

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_prefix = f"atari8track-{session['source_id']}-{stamp}"
    with io.BytesIO() as zip_buf:
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            try:
                with open(session["mix_path"], "rb") as f:
                    zf.writestr(f"{file_prefix}/mix.wav", f.read())
            except OSError:
                return jsonify({"error": "session file missing"}), 410
            manifest = {
                "source_name": session["source_name"],
                "source_id": session["source_id"],
                "engine": "WAV-driven Atari-inspired renderer",
                "bpm": session["bpm"],
                "duration_seconds": session["duration_seconds"],
                "estimated_key": session["estimated_key"],
                "tempo_map_bpm": session["tempo_map_bpm"],
                "sample_rate": session["sample_rate"],
                "tracks": ATARI_TRACK_NAMES,
            }
            zf.writestr(f"{file_prefix}/session.json", json.dumps(manifest, indent=2))
            for stem_name in ATARI_TRACK_NAMES:
                path = (session.get("stem_paths") or {}).get(stem_name, "")
                try:
                    with open(path, "rb") as f:
                        zf.writestr(f"{file_prefix}/stems/{stem_name}.wav", f.read())
                except OSError:
                    return jsonify({"error": "session file missing"}), 410
        zip_bytes = zip_buf.getvalue()
    return Response(
        zip_bytes,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_prefix}.zip"'},
    )


@protracker_bp.post("/api/atari/export")
def atari_export():
    payload = flask_request.get_json(silent=True) or {}
    session_id = (payload.get("session_id") or flask_request.form.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    return atari_session_export(session_id)
