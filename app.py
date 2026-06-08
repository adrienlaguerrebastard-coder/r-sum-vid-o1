import array
import re
import shutil
import subprocess
import threading
import unicodedata
import uuid
from pathlib import Path

import yt_dlp
from flask import Flask, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

AUDIO_SR = 1000
PRE_PEAK_SEC = 6
POST_PEAK_SEC = 9
MIN_CLIP_SEC = 5
MAX_CLIPS = 24
GROUP_GAP_SEC = 3
PEAK_PERCENTILE = 0.75

WHISPER_MODEL_SIZE = "base"
GOAL_PRE_SEC = 6
GOAL_POST_SEC = 8
ACTION_PRE_SEC = 4
ACTION_POST_SEC = 6
GOAL_DEDUPE_SEC = 30
TARGET_MAX_SEC = 300

GOAL_KEYWORDS = [
    "but ", "buts ", "but,", "but!", "but.", "but :", "but de", "but pour",
    "marque", "marqué", "marquer",
    "ouvre le score", "double la mise", "triple la mise",
    "égalise", "égalisation", "égalisateur",
    "filets", "filet", "lucarne",
    "goal", "goooal", "gooal", "goalll", "scores", "scored",
    "dans les buts", "au fond", "dans la lucarne",
    "penalty transformé", "transforme le penalty",
]
EXCITEMENT_KEYWORDS = [
    "incroyable", "magnifique", "magistral", "splendide", "merveilleux",
    "fantastique", "extraordinaire", "spectaculaire", "superbe",
    "remarquable", "phénoménal", "exceptionnel", "génial", "sublime",
    "quel but", "quel arrêt", "quelle frappe", "quelle action", "quelle reprise",
    "what a goal", "amazing", "unbelievable", "incredible",
]
ACTION_KEYWORDS = [
    "frappe", "tir puissant", "reprise", "volée", "demi-volée",
    "tête piquée", "coup franc", "corner décisif",
    "arrêt", "parade", "sauvetage", "détourne",
    "carton rouge", "expulsion", "expulsé",
    "penalty", "pénalty",
]

GOAL_SCORE = 10
EXCITEMENT_SCORE = 3
ACTION_SCORE = 1
MIN_SEGMENT_SCORE = 1


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


GOAL_KW_NORM = [_strip_accents(k) for k in GOAL_KEYWORDS]
EXCITEMENT_KW_NORM = [_strip_accents(k) for k in EXCITEMENT_KEYWORDS]
ACTION_KW_NORM = [_strip_accents(k) for k in ACTION_KEYWORDS]

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
        return _whisper_model


def score_text(text: str) -> tuple[float, bool]:
    text_norm = " " + _strip_accents(text) + " "
    score = 0.0
    is_goal = False
    for kw in GOAL_KW_NORM:
        if kw in text_norm:
            score += GOAL_SCORE
            is_goal = True
    for kw in EXCITEMENT_KW_NORM:
        if kw in text_norm:
            score += EXCITEMENT_SCORE
    for kw in ACTION_KW_NORM:
        if kw in text_norm:
            score += ACTION_SCORE
    return score, is_goal


def transcribe_video(path: Path, job_id: str, video_duration: float) -> list[dict]:
    model = get_whisper_model()
    segments_iter, info = model.transcribe(
        str(path),
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        condition_on_previous_text=False,
    )
    segments: list[dict] = []
    ref_duration = video_duration or info.duration or 1
    for seg in segments_iter:
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
        pct = 50 + int(seg.end / ref_duration * 30)
        update_job(job_id, progress=min(pct, 80))
    return segments


def build_clips_from_transcript(
    segments: list[dict], target: float, duration: float
) -> tuple[list[tuple[float, float]], int, int]:
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    goal_starts: list[float] = []
    last_goal_start = -10000.0
    for seg in sorted_segs:
        _, is_goal = score_text(seg["text"])
        if is_goal and seg["start"] - last_goal_start >= GOAL_DEDUPE_SEC:
            goal_starts.append(seg["start"])
            last_goal_start = seg["start"]

    def in_goal_window(t: float) -> bool:
        return any(g - GOAL_PRE_SEC <= t <= g + GOAL_DEDUPE_SEC for g in goal_starts)

    candidates: list[dict] = []
    for g in goal_starts:
        candidates.append({
            "start": max(0.0, g - GOAL_PRE_SEC),
            "end": min(duration, g + GOAL_POST_SEC),
            "score": GOAL_SCORE * 2,
            "type": "goal",
        })

    for seg in sorted_segs:
        score, is_goal = score_text(seg["text"])
        if score < MIN_SEGMENT_SCORE or is_goal:
            continue
        if in_goal_window(seg["start"]):
            continue
        candidates.append({
            "start": max(0.0, seg["start"] - ACTION_PRE_SEC),
            "end": min(duration, seg["start"] + ACTION_POST_SEC),
            "score": score,
            "type": "action",
        })

    n_goals = len(goal_starts)

    if not candidates:
        return [], 0, 0

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected: list[dict] = []
    total = 0.0
    for c in candidates:
        if c["end"] - c["start"] < MIN_CLIP_SEC:
            continue
        if any(not (c["end"] <= s["start"] or c["start"] >= s["end"]) for s in selected):
            continue
        selected.append(c)
        total += c["end"] - c["start"]
        if len(selected) >= MAX_CLIPS or total >= target:
            break

    selected.sort(key=lambda s: s["start"])
    clips = [(s["start"], s["end"]) for s in selected]

    total = sum(e - s for s, e in clips)
    if total > target:
        excess = total - target
        last_s, last_e = clips[-1]
        new_e = last_e - excess
        if new_e - last_s >= MIN_CLIP_SEC:
            clips[-1] = (last_s, new_e)
        else:
            clips.pop()

    return clips, len(candidates), n_goals

app = Flask(__name__)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def safe_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "video"


def get_video_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio_rms(path: Path) -> list[float] | None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode != 0 or not result.stdout:
        return None

    samples = array.array("h")
    samples.frombytes(result.stdout)
    if not samples:
        return None

    rms_per_sec: list[float] = []
    for i in range(0, len(samples) - AUDIO_SR + 1, AUDIO_SR):
        chunk = samples[i:i + AUDIO_SR]
        s = sum(x * x for x in chunk) / len(chunk)
        rms_per_sec.append(s ** 0.5)
    return rms_per_sec


def find_highlight_clips(rms: list[float], target: float, duration: float) -> list[tuple[float, float]]:
    n = len(rms)
    if n == 0 or duration <= 0:
        return []
    if duration <= target:
        return [(0.0, duration)]

    smoothed = []
    for i in range(n):
        s = max(0, i - 1)
        e = min(n, i + 2)
        smoothed.append(sum(rms[s:e]) / (e - s))

    sorted_vals = sorted(smoothed)
    threshold = sorted_vals[min(n - 1, int(n * PEAK_PERCENTILE))]
    above = [i for i, v in enumerate(smoothed) if v >= threshold]

    if not above:
        ranked = sorted(range(n), key=lambda i: smoothed[i], reverse=True)
        above = sorted(ranked[: max(3, MAX_CLIPS)])

    groups: list[dict] = []
    current = [above[0]]
    for i in above[1:]:
        if i - current[-1] <= GROUP_GAP_SEC:
            current.append(i)
        else:
            peak = max(current, key=lambda x: smoothed[x])
            groups.append({"peak": peak, "score": smoothed[peak]})
            current = [i]
    peak = max(current, key=lambda x: smoothed[x])
    groups.append({"peak": peak, "score": smoothed[peak]})

    groups.sort(key=lambda g: g["score"], reverse=True)

    selected: list[tuple[float, float]] = []
    total = 0.0
    for g in groups[: MAX_CLIPS * 2]:
        peak = g["peak"]
        start = max(0.0, peak - PRE_PEAK_SEC)
        end = min(duration, peak + POST_PEAK_SEC)
        if end - start < MIN_CLIP_SEC:
            continue
        if any(not (end <= s or start >= e) for s, e in selected):
            continue
        selected.append((start, end))
        total += end - start
        if len(selected) >= MAX_CLIPS or total >= target:
            break

    if not selected:
        return [(0.0, min(target, duration))]

    selected.sort()
    total = sum(e - s for s, e in selected)
    if total > target:
        excess = total - target
        last_start, last_end = selected[-1]
        new_end = last_end - excess
        if new_end - last_start >= MIN_CLIP_SEC:
            selected[-1] = (last_start, new_end)
        else:
            selected.pop()

    return selected


LOGO_REGION_W = 0.18
LOGO_REGION_H = 0.13
LOGO_REGION_X = 0.82
LOGO_BLUR_STRENGTH = "boxblur=20:3"


def add_logo_blur(label_in: str, label_out: str) -> list[str]:
    return [
        f"[{label_in}]split=2[{label_in}_o][{label_in}_l]",
        f"[{label_in}_l]crop=iw*{LOGO_REGION_W}:ih*{LOGO_REGION_H}:"
        f"iw*{LOGO_REGION_X}:0,{LOGO_BLUR_STRENGTH}[{label_in}_b]",
        f"[{label_in}_o][{label_in}_b]overlay=W*{LOGO_REGION_X}:0[{label_out}]",
    ]


def build_style_filter(input_label: str, style: str) -> tuple[list[str], str]:
    if style == "crop":
        return [f"[{input_label}]crop=ih*9/16:ih,scale=1080:1920[v]"], "[v]"
    if style == "pad":
        return [
            f"[{input_label}]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black[v]"
        ], "[v]"
    return [
        f"[{input_label}]split=2[bg_src][fg_src]",
        "[bg_src]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=30[bg]",
        "[fg_src]scale=1080:-2[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]",
    ], "[v]"


def build_ffmpeg_manual(input_path: Path, output_path: Path, start: int, duration: int | None, style: str, blur_logo: bool) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_path)]
    if duration:
        cmd += ["-t", str(duration)]
    parts: list[str] = []
    if blur_logo:
        parts.extend(add_logo_blur("0:v", "vclean"))
        style_in = "vclean"
    else:
        style_in = "0:v"
    style_parts, vmap = build_style_filter(style_in, style)
    parts.extend(style_parts)
    cmd += ["-filter_complex", ";".join(parts)]
    cmd += ["-map", vmap, "-map", "0:a?"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


def build_ffmpeg_highlights(input_path: Path, output_path: Path, clips: list[tuple[float, float]], style: str, blur_logo: bool) -> list[str]:
    parts: list[str] = []
    for i, (s, e) in enumerate(clips):
        parts.append(f"[0:v]trim={s:.2f}:{e:.2f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim={s:.2f}:{e:.2f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[vc][ac]")
    if blur_logo:
        parts.extend(add_logo_blur("vc", "vclean"))
        style_in = "vclean"
    else:
        style_in = "vc"
    style_parts, vmap = build_style_filter(style_in, style)
    parts.extend(style_parts)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    cmd += ["-filter_complex", ";".join(parts)]
    cmd += ["-map", vmap, "-map", "[ac]"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


def process_video(job_id: str, url: str, options: dict) -> None:
    download_path = DOWNLOADS_DIR / f"{job_id}.mp4"
    output_path = OUTPUTS_DIR / f"{job_id}.mp4"
    try:
        update_job(job_id, status="downloading", progress=5)

        def hook(d):
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total:
                    pct = int(d.get("downloaded_bytes", 0) / total * 40) + 5
                    update_job(job_id, progress=min(pct, 45))

        ydl_opts = {
            "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
            "outtmpl": str(download_path.with_suffix(".%(ext)s")),
            "merge_output_format": "mp4",
            "quiet": True,
            "noplaylist": True,
            "progress_hooks": [hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title") or "video"

        produced = next(DOWNLOADS_DIR.glob(f"{job_id}.*"), None)
        if produced is None or not produced.exists():
            raise RuntimeError("Téléchargement YouTube échoué")
        if produced.suffix != ".mp4":
            produced.replace(download_path)
        else:
            download_path = produced

        mode = options.get("mode", "auto")
        style = options.get("format", "blur")
        blur_logo = bool(options.get("blur_logo", True))

        if mode == "auto":
            update_job(job_id, status="transcribing", progress=50, title=title)
            target = float(options.get("target_duration") or 60)
            duration = get_video_duration(download_path)
            clips: list[tuple[float, float]] = []
            detection = ""
            try:
                segments = transcribe_video(download_path, job_id, duration)
                clips, n_hits, n_goals = build_clips_from_transcript(segments, target, duration)
                if clips:
                    detection = (
                        f"{len(clips)} clip(s) — {n_goals} but(s) identifié(s), "
                        f"{n_hits} segment(s) clé(s) candidat(s)"
                    )
            except Exception as e:
                detection = f"Whisper indisponible ({type(e).__name__}), repli audio"

            if not clips:
                update_job(job_id, status="analyzing", progress=85)
                rms = extract_audio_rms(download_path)
                if rms and duration > 0:
                    clips = find_highlight_clips(rms, target, duration)
                    if clips and not detection.startswith("Whisper"):
                        detection = f"{len(clips)} moment(s) (repli sur pics audio)"
                if not clips:
                    clips = [(0.0, min(target, duration or target))]
                    detection = detection or "fallback début de vidéo"

            update_job(
                job_id,
                status="processing",
                progress=88,
                clips=[{"start": round(s, 1), "end": round(e, 1)} for s, e in clips],
                detection=detection,
                total_clip_duration=round(sum(e - s for s, e in clips), 1),
            )
            cmd = build_ffmpeg_highlights(download_path, output_path, clips, style, blur_logo)
        else:
            update_job(job_id, status="processing", progress=55, title=title)
            start = int(options.get("start") or 0)
            duration_opt = options.get("duration")
            cmd = build_ffmpeg_manual(
                download_path, output_path, start,
                int(duration_opt) if duration_opt else None, style, blur_logo,
            )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg: {result.stderr.strip()[:500]}")

        update_job(
            job_id,
            status="done",
            progress=100,
            output_path=str(output_path),
            download_name=f"{safe_filename(title)}_tiktok.mp4",
        )
    except Exception as e:
        update_job(job_id, status="error", error=str(e))
    finally:
        if download_path.exists():
            try:
                download_path.unlink()
            except OSError:
                pass


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/process")
def submit():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400

    mode = data.get("mode", "auto")
    style = data.get("format", "blur")
    if style not in {"blur", "crop", "pad"}:
        style = "blur"
    if mode not in {"auto", "manual"}:
        mode = "auto"

    options = {"mode": mode, "format": style, "blur_logo": bool(data.get("blur_logo", True))}
    try:
        if mode == "auto":
            target_raw = data.get("target_duration") or 60
            options["target_duration"] = max(10, min(TARGET_MAX_SEC, int(target_raw)))
        else:
            options["start"] = int(data.get("start") or 0)
            d = data.get("duration")
            options["duration"] = int(d) if d not in (None, "", 0, "0") else None
    except ValueError:
        return jsonify({"error": "Valeurs numériques invalides"}), 400

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "progress": 0}

    threading.Thread(
        target=process_video, args=(job_id, url, options), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.get("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(job)


@app.get("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Vidéo pas prête"}), 404
    return send_file(
        job["output_path"], as_attachment=True,
        download_name=job.get("download_name", "tiktok.mp4"),
        mimetype="video/mp4",
    )


@app.get("/preview/<job_id>")
def preview(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Vidéo pas prête"}), 404
    return send_file(job["output_path"], mimetype="video/mp4")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        print("⚠️  ffmpeg introuvable dans le PATH. Lance via run.ps1.")
    app.run(host="127.0.0.1", port=5000, debug=False)
