#!/usr/bin/env python3
"""
SRT 번역 웹 UI
실행: python3 app.py
접속: http://localhost:5001
"""

import os
import sys
import json
import uuid
import queue
import shutil
import threading
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, send_file

# 스크립트 디렉토리를 경로에 추가 (translate_srt import용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import translate_srt as srt_core
import anthropic

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# 모델별 단가 ($/백만 토큰: input, output)
MODEL_PRICING = {
    "claude-sonnet-4-6":         (3.0,  15.0),
    "claude-opus-4-6":           (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.8,   4.0),
}

# 진행 중인 작업 저장소: job_id → {queue, status, out_srt, out_smi}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# 임시 작업 디렉토리
JOBS_TMP = os.path.join(tempfile.gettempdir(), "srt_jobs")
os.makedirs(JOBS_TMP, exist_ok=True)


# ─── 헬퍼 ────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "")


def reload_srt_config():
    """config.json 변경 후 srt_core 모듈 설정 갱신"""
    cfg = srt_core.load_config()
    srt_core.MODEL = cfg.get("model", "claude-sonnet-4-6")
    srt_core.BATCH_SIZE = cfg.get("batch_size", 20)
    srt_core.SYSTEM_PROMPT = srt_core._build_system_prompt(cfg)


# ─── 라우트: 메인 페이지 ──────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── 라우트: 번역 시작 ───────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".srt"):
        return jsonify({"error": ".srt 파일만 지원합니다"}), 400

    use_context = request.form.get("use_context", "true").lower() == "true"
    reset       = request.form.get("reset", "false").lower() == "true"

    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "API 키가 설정되지 않았습니다. 설정 탭에서 입력하거나 .env 파일을 확인하세요."}), 400

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS_TMP, job_id)
    os.makedirs(job_dir, exist_ok=True)

    srt_path = os.path.join(job_dir, f.filename)
    f.save(srt_path)

    # 이전화 컨텍스트: 원본 SRT 폴더의 .ko.srt 파일 심볼릭 링크 또는 복사
    if use_context:
        for ko_file in Path(BASE_DIR).glob("*.ko.srt"):
            shutil.copy2(str(ko_file), job_dir)

    progress_q = queue.Queue()

    with JOBS_LOCK:
        JOBS[job_id] = {
            "queue":   progress_q,
            "status":  "running",
            "out_srt": None,
            "out_smi": None,
        }

    # 백그라운드 번역 스레드 시작
    thread = threading.Thread(
        target=_run_translation,
        args=(job_id, srt_path, job_dir, api_key, reset, progress_q),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_translation(job_id, srt_path, job_dir, api_key, reset, progress_q):
    try:
        reload_srt_config()
        client = anthropic.Anthropic(api_key=api_key)

        # 진행 상황 처리
        if reset:
            srt_core.clear_progress(srt_path)

        subtitles, start_from = srt_core.load_progress(srt_path)
        if not subtitles or start_from == 0:
            subtitles = srt_core.parse_srt(srt_path)
            start_from = 0

        context = srt_core.build_context_from_previous(job_dir, srt_path)

        pricing = MODEL_PRICING.get(srt_core.MODEL, (3.0, 15.0))

        def on_progress(pct, done, total, cost_usd=0.0):
            progress_q.put({"type": "progress", "pct": pct, "done": done,
                            "total": total, "cost_usd": cost_usd})

        srt_core.translate_all(
            client, subtitles, srt_path,
            context=context,
            batch_size=srt_core.BATCH_SIZE,
            start_from=start_from,
            progress_callback=on_progress,
            model_pricing=pricing,
        )

        base = os.path.splitext(srt_path)[0]
        out_srt = base + ".ko.srt"
        out_smi = base + ".ko.smi"

        srt_core.save_srt(subtitles, out_srt)
        srt_core.save_smi(subtitles, out_smi, title=os.path.basename(base))
        srt_core.clear_progress(srt_path)

        # 완료된 번역 파일을 원본 폴더에도 복사 (다음화 컨텍스트용)
        for f in [out_srt, out_smi]:
            if os.path.exists(f):
                shutil.copy2(f, BASE_DIR)

        with JOBS_LOCK:
            JOBS[job_id]["status"]  = "done"
            JOBS[job_id]["out_srt"] = out_srt
            JOBS[job_id]["out_smi"] = out_smi

        progress_q.put({"type": "done"})

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
        progress_q.put({"type": "error", "message": str(e)})


# ─── 라우트: SSE 진행률 ──────────────────────────────────────
@app.route("/progress/<job_id>")
def progress(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다"}), 404

    def generate():
        q = job["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                continue

            if msg["type"] == "progress":
                data = json.dumps({"pct": msg["pct"], "done": msg["done"],
                                   "total": msg["total"], "cost_usd": msg.get("cost_usd", 0.0)})
                yield f"event: progress\ndata: {data}\n\n"
            elif msg["type"] == "done":
                yield "event: done\ndata: {}\n\n"
                break
            elif msg["type"] == "error":
                data = json.dumps({"message": msg["message"]})
                yield f"event: error_msg\ndata: {data}\n\n"
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── 라우트: 다운로드 ────────────────────────────────────────
@app.route("/download/<job_id>/<ext>")
def download(job_id, ext):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "아직 완료되지 않았습니다"}), 404

    path = job["out_srt"] if ext == "srt" else job["out_smi"]
    if not path or not os.path.exists(path):
        return jsonify({"error": "파일을 찾을 수 없습니다"}), 404

    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


# ─── 라우트: 설정 조회 ───────────────────────────────────────
@app.route("/settings", methods=["GET"])
def get_settings():
    cfg = srt_core.load_config()
    # API 키는 마스킹
    api_key = get_api_key()
    cfg["api_key_masked"] = ("•" * 8 + api_key[-4:]) if len(api_key) > 4 else ""
    return jsonify(cfg)


# ─── 라우트: 설정 저장 ───────────────────────────────────────
@app.route("/settings", methods=["POST"])
def save_settings():
    data = request.get_json()
    config_path = os.path.join(BASE_DIR, "config.json")

    cfg = srt_core.load_config()
    cfg["model"]           = data.get("model", cfg.get("model"))
    cfg["batch_size"]      = int(data.get("batch_size", cfg.get("batch_size", 20)))
    cfg["system_prompt"]   = data.get("system_prompt", cfg.get("system_prompt", ""))
    cfg["character_notes"] = data.get("character_notes", cfg.get("character_notes", {}))

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # API 키 업데이트 (비어있지 않은 경우만)
    new_key = (data.get("api_key") or "").strip()
    if new_key:
        env_path = os.path.join(BASE_DIR, ".env")
        lines = []
        replaced = False
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        lines.append(f"ANTHROPIC_API_KEY={new_key}\n")
                        replaced = True
                    else:
                        lines.append(line)
        if not replaced:
            lines.append(f"ANTHROPIC_API_KEY={new_key}\n")
        with open(env_path, "w") as f:
            f.writelines(lines)
        os.environ["ANTHROPIC_API_KEY"] = new_key

    reload_srt_config()
    return jsonify({"ok": True})


# ─── 실행 ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SRT 자막 번역기 웹 UI")
    print("  http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
