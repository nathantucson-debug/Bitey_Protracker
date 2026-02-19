import hashlib
import io
import json
import math
import os
import random
import struct
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
        except Exception as exc:
            _set_job_status(job_id, {"status": "failed", "error": f"Build failed on server: {exc.__class__.__name__}"})

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


def _to_arrayf(samples: list[float]) -> array:
    out = array("f")
    out.extend(float(x) for x in samples)
    return out


def _soft_clip(samples: list[float], drive: float = 1.0) -> array:
    out = array("f", [0.0]) * len(samples)
    for i in range(len(samples)):
        out[i] = math.tanh(samples[i] * drive)
    return out


def _subsample_hold(samples: list[float], hold: int = 2) -> array:
    if hold <= 1:
        return _to_arrayf(samples)
    out = array("f", [0.0]) * len(samples)
    held = 0.0
    for i in range(len(samples)):
        if i % hold == 0:
            held = samples[i]
        out[i] = held
    return out


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


def _wav_audio_format(file_bytes: bytes) -> int:
    if len(file_bytes) < 44 or file_bytes[:4] != b"RIFF" or file_bytes[8:12] != b"WAVE":
        return 0
    idx = 12
    size = len(file_bytes)
    while idx + 8 <= size:
        chunk_id = file_bytes[idx : idx + 4]
        chunk_size = int.from_bytes(file_bytes[idx + 4 : idx + 8], "little", signed=False)
        data_start = idx + 8
        if chunk_id == b"fmt " and data_start + 2 <= size:
            return int.from_bytes(file_bytes[data_start : data_start + 2], "little", signed=False)
        idx = data_start + chunk_size + (chunk_size % 2)
    return 0


def _decode_mono_frame(frame: bytes, channels: int, sample_width: int, is_float32: bool) -> float:
    vals = []
    if sample_width == 1:
        for ch in range(channels):
            vals.append((frame[ch] - 128) / 128.0)
    elif sample_width == 2:
        for ch in range(channels):
            start = ch * 2
            v = int.from_bytes(frame[start : start + 2], "little", signed=True) / 32768.0
            vals.append(v)
    elif sample_width == 3:
        for ch in range(channels):
            start = ch * 3
            b0 = frame[start]
            b1 = frame[start + 1]
            b2 = frame[start + 2]
            value = b0 | (b1 << 8) | (b2 << 16)
            if value & 0x800000:
                value -= 0x1000000
            vals.append(value / 8388608.0)
    elif sample_width == 4:
        for ch in range(channels):
            start = ch * 4
            chunk = frame[start : start + 4]
            if is_float32:
                vals.append(struct.unpack("<f", chunk)[0])
            else:
                vals.append(int.from_bytes(chunk, "little", signed=True) / 2147483648.0)
    else:
        return 0.0
    return max(-1.0, min(1.0, sum(vals) / max(1, len(vals))))


def _read_wav_mono(file_bytes: bytes) -> tuple[list[float], int]:
    audio_format = _wav_audio_format(file_bytes)
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
    elif sample_width == 3:
        frame_width = channels * 3
        mono = []
        for i in range(0, len(raw), frame_width):
            frame = raw[i : i + frame_width]
            if len(frame) < frame_width:
                break
            vals = []
            for ch in range(channels):
                b0 = frame[ch * 3]
                b1 = frame[ch * 3 + 1]
                b2 = frame[ch * 3 + 2]
                value = b0 | (b1 << 8) | (b2 << 16)
                if value & 0x800000:
                    value -= 0x1000000
                vals.append(value / 8388608.0)
            mono.append(max(-1.0, min(1.0, sum(vals) / len(vals))))
    elif sample_width == 4:
        frame_width = channels * 4
        mono = []
        is_float = audio_format == 3
        for i in range(0, len(raw), frame_width):
            frame = raw[i : i + frame_width]
            if len(frame) < frame_width:
                break
            vals = []
            for ch in range(channels):
                chunk = frame[ch * 4 : ch * 4 + 4]
                if is_float:
                    value = struct.unpack("<f", chunk)[0]
                else:
                    value = int.from_bytes(chunk, "little", signed=True) / 2147483648.0
                vals.append(value)
            mono.append(max(-1.0, min(1.0, sum(vals) / len(vals))))
    else:
        raise ValueError("Only 8-bit, 16-bit, 24-bit, or 32-bit WAV files are supported")

    if not mono:
        raise ValueError("WAV file has no audio samples")

    return mono, sample_rate


def _extract_analysis_from_wav(file_bytes: bytes, target_rate: int = 11025, max_seconds: int = 180) -> tuple[list[float], int]:
    audio_format = _wav_audio_format(file_bytes)
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if channels < 1 or sample_rate <= 0:
                raise ValueError("Invalid WAV format")

            if sample_width not in (1, 2, 3, 4):
                raise ValueError("Only 8-bit, 16-bit, 24-bit, or 32-bit WAV files are supported")
            if sample_width == 4 and audio_format not in (1, 3):
                raise ValueError("Unsupported 32-bit WAV encoding")

            max_frames = min(n_frames, max_seconds * sample_rate)
            duration_seconds = max(10, min(max_seconds, int(round(max_frames / sample_rate))))
            is_float32 = audio_format == 3
            ratio = sample_rate / max(1, target_rate)
            next_pick = 0.0
            src_index = 0
            analysis = []
            frame_size = channels * sample_width

            while src_index < max_frames:
                to_read = min(4096, max_frames - src_index)
                raw = wf.readframes(to_read)
                if not raw:
                    break
                frames_in_chunk = len(raw) // frame_size
                for i in range(frames_in_chunk):
                    if src_index + i >= max_frames:
                        break
                    if (src_index + i) < int(next_pick):
                        continue
                    start = i * frame_size
                    frame = raw[start : start + frame_size]
                    analysis.append(_decode_mono_frame(frame, channels, sample_width, is_float32))
                    next_pick += ratio
                src_index += frames_in_chunk
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV file: {exc}") from exc

    if not analysis:
        raise ValueError("WAV file has no audio samples")

    return analysis, duration_seconds


def _rms(block: list[float]) -> float:
    if not block:
        return 0.0
    return math.sqrt(sum(x * x for x in block) / len(block))


def _estimate_pitch_class_hist(samples: list[float], sample_rate: int) -> list[float]:
    frame = 1024
    hop = 2048
    lag_min = max(12, int(sample_rate / 420))
    lag_max = min(frame - 2, int(sample_rate / 80))
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
        for lag in range(lag_min, lag_max, 2):
            corr = 0.0
            for i in range(0, frame - lag, 2):
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
    analysis_rate = 11025
    analysis_samples, duration_seconds = _extract_analysis_from_wav(
        file_bytes=file_bytes, target_rate=analysis_rate, max_seconds=180
    )
    energies, onsets, fps = _energy_and_onsets(analysis_samples, analysis_rate)
    tempo_map_bpm, avg_bpm = _estimate_tempo_map(onsets, fps, duration_seconds)
    key_name, _ = _estimate_key(analysis_samples[: analysis_rate * 30], analysis_rate)

    source_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    sample_rate = 8000 if duration_seconds >= 120 else 12000
    render_samples = _resample_linear(analysis_samples, analysis_rate, sample_rate)
    total_samples = len(render_samples)

    src = _to_arrayf(render_samples)
    low_a = _simple_lowpass(src, alpha=0.03)
    low_b = _simple_lowpass(src, alpha=0.008)
    high = array("f", [0.0]) * total_samples
    mid = array("f", [0.0]) * total_samples
    for i in range(total_samples):
        high[i] = src[i] - low_a[i]
        mid[i] = low_a[i] - low_b[i]

    transient = array("f", [0.0]) * total_samples
    prev = 0.0
    for i in range(total_samples):
        d = abs(src[i] - prev)
        prev = src[i]
        transient[i] = min(1.0, d * 10.0)

    stems_float = {name: array("f", [0.0]) * total_samples for name in ATARI_TRACK_NAMES}
    for i in range(total_samples):
        t = transient[i]
        stems_float["kick"][i] = low_b[i] * (0.6 + 0.9 * t)
        stems_float["snare"][i] = mid[i] * (0.35 + 1.1 * t)
        stems_float["hat"][i] = high[i] * (0.25 + 1.4 * t)
        stems_float["bass"][i] = low_a[i] * (0.95 - 0.45 * t)
        stems_float["arp"][i] = high[i] * 0.45
        stems_float["chord"][i] = mid[i] * 0.95
        stems_float["lead"][i] = high[i] * 0.7 + mid[i] * 0.2
        stems_float["fx"][i] = (src[i] - (low_b[i] + mid[i] + high[i])) * 0.6

    processed_stems = {}
    for name, data in stems_float.items():
        tone = _simple_lowpass(data, alpha=0.22 if name in {"bass", "kick"} else 0.16)
        hold = 3 if sample_rate <= 8000 else 2
        crushed = _bitcrush(_subsample_hold(tone, hold=hold), levels=30 if name in {"lead", "arp"} else 36, hold=2)
        processed_stems[name] = _normalize(_soft_clip(crushed, drive=1.25), ceiling=0.86)

    mix = array("f", [0.0]) * total_samples
    for name in ATARI_TRACK_NAMES:
        weight = 1.0
        if name == "hat":
            weight = 0.7
        if name == "fx":
            weight = 0.45
        track = processed_stems[name]
        for i in range(total_samples):
            mix[i] += track[i] * weight

    mix = _normalize(_soft_clip(_simple_lowpass(mix, alpha=0.2), drive=1.2), ceiling=0.9)
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
