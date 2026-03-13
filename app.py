"""
CDMaker — Servidor Web
======================
Inicia o front-end e processa as requisições de geração de vídeo.

Uso:
    python app.py
    Abra:  http://localhost:5000
"""

import shutil
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from main import generate, DEF_FPS, DEF_RPM, DEF_PRESET

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024  # 400 MB (áudios longos)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_jobs: dict = {}
_lock = threading.Lock()

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_AUDIO = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set(job_id: str, **kw) -> None:
    with _lock:
        _jobs[job_id].update(kw)


def _valid_id(job_id: str) -> bool:
    return isinstance(job_id, str) and job_id.isalnum() and 4 <= len(job_id) <= 32


# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def start_generate():
    if "image" not in request.files or "audio" not in request.files:
        return jsonify({"error": "Envie imagem e áudio"}), 400

    img_f = request.files["image"]
    aud_f = request.files["audio"]

    img_ext = Path(img_f.filename or "").suffix.lower()
    aud_ext = Path(aud_f.filename or "").suffix.lower()

    if img_ext not in ALLOWED_IMAGE:
        return jsonify({"error": f"Formato de imagem inválido: {img_ext}"}), 400
    if aud_ext not in ALLOWED_AUDIO:
        return jsonify({"error": f"Formato de áudio inválido: {aud_ext}"}), 400

    job_id  = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    img_path = job_dir / f"cover{img_ext}"
    aud_path = job_dir / f"audio{aud_ext}"
    img_f.save(str(img_path))
    aud_f.save(str(aud_path))

    # Imagens opcionais: arte do disco e fundo personalizado
    disc_img_path = None
    bg_img_path   = None
    for field, prefix in (("disc_image", "disc"), ("bg_image", "bg")):
        f = request.files.get(field)
        if f and f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext in ALLOWED_IMAGE:
                p = job_dir / f"{prefix}{ext}"
                f.save(str(p))
                if field == "disc_image":
                    disc_img_path = str(p)
                else:
                    bg_img_path = str(p)

    title  = (request.form.get("title",  "") or "").strip() or None
    artist = (request.form.get("artist", "") or "").strip() or None
    album  = (request.form.get("album",  "") or "").strip() or None
    handle = (request.form.get("handle", "") or "").strip() or None

    try:
        fps = max(1, min(int(request.form.get("fps", DEF_FPS)), 60))
    except (ValueError, TypeError):
        fps = DEF_FPS

    try:
        rpm = max(0.0, min(float(request.form.get("rpm", DEF_RPM)), 300.0))
    except (ValueError, TypeError):
        rpm = DEF_RPM

    preset = request.form.get("preset", DEF_PRESET)
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        preset = DEF_PRESET

    with _lock:
        _jobs[job_id] = {"status": "running", "progress": 0, "message": "Iniciando…"}

    def _run() -> None:
        try:
            out = str(OUTPUT_DIR / f"{job_id}.mp4")

            def _cb(pct: int, msg: str) -> None:
                _set(job_id, progress=pct, message=msg)

            generate(
                str(img_path), str(aud_path), out,
                title=title, artist=artist, album=album, handle=handle,
                disc_image_path=disc_img_path, bg_image_path=bg_img_path,
                fps=fps, rpm=rpm, preset=preset,
                progress_cb=_cb,
            )
            _set(job_id, status="done", progress=100, message="Concluído!")
        except Exception as exc:
            _set(job_id, status="error", message=str(exc))
        finally:
            shutil.rmtree(str(job_dir), ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id: str):
    if not _valid_id(job_id):
        return jsonify({"status": "not_found"}), 404
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id: str):
    if not _valid_id(job_id):
        return jsonify({"error": "invalid"}), 400
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 400

    out = (OUTPUT_DIR / f"{job_id}.mp4").resolve()
    try:
        out.relative_to(OUTPUT_DIR.resolve())   # path-traversal guard
    except ValueError:
        return jsonify({"error": "Forbidden"}), 403

    if not out.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        str(out),
        as_attachment=True,
        download_name=f"cdmaker_{job_id}.mp4",
        mimetype="video/mp4",
    )


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  🎵  CDMaker  →  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
