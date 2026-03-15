
import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, Response

from main import generate, generate_preview, DEF_FPS, DEF_RPM, DEF_PRESET

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
PROJECTS_DIR = Path("projects")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

_jobs: dict = {}
_lock = threading.Lock()

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_AUDIO = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}

def _set(job_id: str, **kw) -> None:
    with _lock:
        _jobs[job_id].update(kw)

def _valid_id(job_id: str) -> bool:
    return isinstance(job_id, str) and job_id.isalnum() and 4 <= len(job_id) <= 32

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

    vertical = request.form.get("vertical", "").lower() in ("1", "true", "on")

    cancel_ev = threading.Event()
    with _lock:
        _jobs[job_id] = {"status": "running", "progress": 0, "message": "Iniciando…", "cancel_event": cancel_ev}

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
                progress_cb=_cb, vertical=vertical,
                cancel_event=cancel_ev,
            )
            _set(job_id, status="done", progress=100, message="Concluído!")
        except Exception as exc:
            if "__cdmaker_cancelled__" in str(exc):
                _set(job_id, status="cancelled", message="Cancelado pelo usuário")
                (OUTPUT_DIR / f"{job_id}.mp4").unlink(missing_ok=True)
            else:
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
    job.pop("cancel_event", None)
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
        out.relative_to(OUTPUT_DIR.resolve())
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

@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    if not _valid_id(job_id):
        return jsonify({"error": "invalid"}), 400
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    ev = job.get("cancel_event")
    if ev:
        ev.set()
    return jsonify({"ok": True})

@app.route("/preview", methods=["POST"])
def preview():
    if "image" not in request.files:
        return jsonify({"error": "Envie ao menos a imagem de capa"}), 400

    img_f   = request.files["image"]
    img_ext = Path(img_f.filename or "").suffix.lower()
    if img_ext not in ALLOWED_IMAGE:
        return jsonify({"error": "Formato inválido"}), 400

    tmp_dir = UPLOAD_DIR / f"_preview_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        img_path = tmp_dir / f"cover{img_ext}"
        img_f.save(str(img_path))

        disc_path = bg_path = None
        for field, prefix in (("disc_image", "disc"), ("bg_image", "bg")):
            f = request.files.get(field)
            if f and f.filename:
                ext = Path(f.filename).suffix.lower()
                if ext in ALLOWED_IMAGE:
                    p = tmp_dir / f"{prefix}{ext}"
                    f.save(str(p))
                    if field == "disc_image":
                        disc_path = str(p)
                    else:
                        bg_path = str(p)

        title  = (request.form.get("title",  "") or "").strip() or None
        artist = (request.form.get("artist", "") or "").strip() or None
        album  = (request.form.get("album",  "") or "").strip() or None
        handle = (request.form.get("handle", "") or "").strip() or None
        vertical = request.form.get("vertical", "").lower() in ("1", "true", "on")

        png_bytes = generate_preview(
            str(img_path),
            title=title, artist=artist, album=album, handle=handle,
            disc_image_path=disc_path, bg_image_path=bg_path,
            vertical=vertical,
        )
        return Response(png_bytes, mimetype="image/png")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

def _proj_manifest(pid: str) -> Path:
    return PROJECTS_DIR / pid / "manifest.json"

def _read_manifest(pid: str) -> dict | None:
    p = _proj_manifest(pid)
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return None

def _write_manifest(pid: str, data: dict) -> None:
    p = _proj_manifest(pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

@app.route("/projects", methods=["GET"])
def list_projects():
    projects = []
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir() and (d / "manifest.json").exists():
            m = json.loads((d / "manifest.json").read_text("utf-8"))
            projects.append(m)
    projects.sort(key=lambda p: p.get("updated_at", p.get("id", "")), reverse=True)
    return jsonify(projects)

@app.route("/projects", methods=["POST"])
def save_project():
    pid = (request.form.get("id", "") or "").strip()
    if not pid or not pid.isalnum() or len(pid) > 32:
        pid = uuid.uuid4().hex[:12]

    proj_dir = PROJECTS_DIR / pid
    proj_dir.mkdir(parents=True, exist_ok=True)

    file_map = {}
    for field in ("image", "audio", "disc_image", "bg_image"):
        f = request.files.get(field)
        if f and f.filename:
            ext  = Path(f.filename).suffix.lower()
            safe = {"image": "cover", "audio": "audio",
                    "disc_image": "disc", "bg_image": "bg"}[field]

            for old in proj_dir.glob(f"{safe}.*"):
                old.unlink(missing_ok=True)
            dest = proj_dir / f"{safe}{ext}"
            f.save(str(dest))
            file_map[field] = f"{safe}{ext}"

    existing = _read_manifest(pid) or {}
    files = existing.get("files", {})
    files.update(file_map)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "id":         pid,
        "title":      (request.form.get("title",  "") or "").strip(),
        "artist":     (request.form.get("artist", "") or "").strip(),
        "album":      (request.form.get("album",  "") or "").strip(),
        "handle":     (request.form.get("handle", "") or "").strip(),
        "fps":        request.form.get("fps",    str(DEF_FPS)),
        "rpm":        request.form.get("rpm",    str(DEF_RPM)),
        "preset":     request.form.get("preset", DEF_PRESET),
        "vertical":   request.form.get("vertical", "false"),
        "files":      files,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    _write_manifest(pid, manifest)
    return jsonify(manifest)

@app.route("/projects/<pid>", methods=["DELETE"])
def delete_project(pid: str):
    if not _valid_id(pid):
        return jsonify({"error": "invalid"}), 400
    proj_dir = PROJECTS_DIR / pid
    try:
        proj_dir.resolve().relative_to(PROJECTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Forbidden"}), 403
    if proj_dir.exists():
        shutil.rmtree(str(proj_dir), ignore_errors=True)
    return jsonify({"ok": True})

@app.route("/projects/<pid>", methods=["PATCH"])
def patch_project(pid: str):
    if not _valid_id(pid):
        return jsonify({"error": "invalid"}), 400
    existing = _read_manifest(pid)
    if not existing:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    for key in ("title", "artist", "album", "handle", "fps", "rpm", "preset", "vertical"):
        if key in data:
            existing[key] = str(data[key])
    existing["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_manifest(pid, existing)
    return jsonify(existing)

@app.route("/projects/<pid>/file/<fname>")
def project_file(pid: str, fname: str):
    if not _valid_id(pid):
        return jsonify({"error": "invalid"}), 400

    safe_name = Path(fname).name
    fpath = (PROJECTS_DIR / pid / safe_name).resolve()
    try:
        fpath.relative_to(PROJECTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Forbidden"}), 403
    if not fpath.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(fpath))

@app.route("/projects/<pid>/generate", methods=["POST"])
def generate_from_project(pid: str):
    if not _valid_id(pid):
        return jsonify({"error": "invalid"}), 400
    manifest = _read_manifest(pid)
    if not manifest:
        return jsonify({"error": "Projeto não encontrado"}), 404

    files = manifest.get("files", {})
    proj_dir = PROJECTS_DIR / pid

    img_file = files.get("image")
    aud_file = files.get("audio")
    if not img_file or not aud_file:
        return jsonify({"error": "Projeto incompleto — falta capa ou áudio"}), 400

    img_path = proj_dir / img_file
    aud_path = proj_dir / aud_file
    if not img_path.exists() or not aud_path.exists():
        return jsonify({"error": "Arquivos do projeto corrompidos"}), 400

    disc_path = None
    bg_path   = None
    if files.get("disc_image"):
        p = proj_dir / files["disc_image"]
        if p.exists():
            disc_path = str(p)
    if files.get("bg_image"):
        p = proj_dir / files["bg_image"]
        if p.exists():
            bg_path = str(p)

    title  = manifest.get("title")  or None
    artist = manifest.get("artist") or None
    album  = manifest.get("album")  or None
    handle = manifest.get("handle") or None

    try:
        fps = max(1, min(int(manifest.get("fps", DEF_FPS)), 60))
    except (ValueError, TypeError):
        fps = DEF_FPS
    try:
        rpm = max(0.0, min(float(manifest.get("rpm", DEF_RPM)), 300.0))
    except (ValueError, TypeError):
        rpm = DEF_RPM
    preset = manifest.get("preset", DEF_PRESET)
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        preset = DEF_PRESET

    vertical = str(manifest.get("vertical", "false")).lower() in ("1", "true", "on")

    job_id = uuid.uuid4().hex[:12]
    cancel_ev = threading.Event()
    with _lock:
        _jobs[job_id] = {"status": "running", "progress": 0, "message": "Iniciando…", "cancel_event": cancel_ev}

    def _run():
        try:
            out = str(OUTPUT_DIR / f"{job_id}.mp4")
            def _cb(pct, msg):
                _set(job_id, progress=pct, message=msg)
            generate(
                str(img_path), str(aud_path), out,
                title=title, artist=artist, album=album, handle=handle,
                disc_image_path=disc_path, bg_image_path=bg_path,
                fps=fps, rpm=rpm, preset=preset,
                progress_cb=_cb, vertical=vertical,
                cancel_event=cancel_ev,
            )
            _set(job_id, status="done", progress=100, message="Concluído!")
        except Exception as exc:
            if "__cdmaker_cancelled__" in str(exc):
                _set(job_id, status="cancelled", message="Cancelado pelo usuário")
                (OUTPUT_DIR / f"{job_id}.mp4").unlink(missing_ok=True)
            else:
                _set(job_id, status="error", message=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})

if __name__ == "__main__":
    print("\n  🎵  CDMaker  →  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
