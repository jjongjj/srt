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

# 취소 플래그: job_id → threading.Event
CANCEL_FLAGS: dict = {}

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


def load_config_file() -> dict:
    path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"model": "claude-sonnet-4-6", "batch_size": 20,
            "current_profile": None, "profiles": {}}


def save_config_file(cfg: dict):
    path = os.path.join(BASE_DIR, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def migrate_config_if_needed(cfg: dict) -> dict:
    """구 포맷(profiles 없음)을 profiles 구조로 자동 마이그레이션."""
    if "profiles" not in cfg:
        profile = {
            "name": "기본 프로필",
            "system_prompt": cfg.pop("system_prompt", ""),
            "character_notes": cfg.pop("character_notes", {}),
            "context_folders": [],
            "output_folder": None,
            "glossary_file": None,
            "story_file": None,
        }
        cfg["profiles"] = {"default": profile}
        cfg["current_profile"] = "default"
        save_config_file(cfg)
    return cfg


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
    bilingual   = request.form.get("bilingual", "false").lower() == "true"

    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "API 키가 설정되지 않았습니다. 설정 탭에서 입력하거나 .env 파일을 확인하세요."}), 400

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS_TMP, job_id)
    os.makedirs(job_dir, exist_ok=True)

    srt_path = os.path.join(job_dir, f.filename)
    f.save(srt_path)

    # 현재 프로필의 context_folders 가져오기
    cfg = migrate_config_if_needed(load_config_file())
    profile = srt_core.get_current_profile(cfg)
    context_folders = profile.get("context_folders", []) if use_context else []

    # context_folders가 없을 때는 BASE_DIR fallback (구 방식 호환)
    if use_context and not context_folders:
        for ko_file in Path(BASE_DIR).glob("*.ko.srt"):
            shutil.copy2(str(ko_file), job_dir)
        context_folders_for_job = [job_dir]
    else:
        context_folders_for_job = context_folders

    progress_q = queue.Queue()
    cancel_event = threading.Event()

    with JOBS_LOCK:
        JOBS[job_id] = {
            "queue":   progress_q,
            "status":  "running",
            "out_srt": None,
            "out_smi": None,
        }
        CANCEL_FLAGS[job_id] = cancel_event

    # 백그라운드 번역 스레드 시작
    thread = threading.Thread(
        target=_run_translation,
        args=(job_id, srt_path, job_dir, api_key, reset,
              progress_q, cancel_event, context_folders_for_job, profile, bilingual),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_translation(job_id, srt_path, job_dir, api_key, reset,
                     progress_q, cancel_event, context_folders, profile, bilingual=False):
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

        # 컨텍스트 빌드 (다중 폴더)
        context = srt_core.build_context_from_previous(context_folders, srt_path)

        # 스토리 요약 컨텍스트 추가
        story_context = srt_core.load_recent_story(profile.get("story_file"), max_episodes=3)
        if story_context:
            context += "\n\n" + story_context

        # 용어집 시스템 프롬프트 주입
        cfg = srt_core.load_config()
        glossary_terms = srt_core.load_glossary(profile.get("glossary_file"))
        glossary_text = srt_core.glossary_to_prompt_section(glossary_terms)
        full_system_prompt = srt_core._build_system_prompt(cfg, glossary_text)
        srt_core.SYSTEM_PROMPT = full_system_prompt

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
            cancel_event=cancel_event,
        )

        # 취소 확인
        if cancel_event.is_set():
            # 취소됐어도 진행된 부분 저장
            base = os.path.splitext(srt_path)[0]
            out_srt = base + ".ko.srt"
            out_smi = base + ".ko.smi"
            translated_subs = [s for s in subtitles if s.translated]
            if translated_subs:
                srt_core.save_srt(translated_subs, out_srt, bilingual=bilingual)
                srt_core.save_smi(translated_subs, out_smi, title=os.path.basename(base), bilingual=bilingual)
                with JOBS_LOCK:
                    JOBS[job_id]["status"]  = "cancelled"
                    JOBS[job_id]["out_srt"] = out_srt
                    JOBS[job_id]["out_smi"] = out_smi
            else:
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "cancelled"
            CANCEL_FLAGS.pop(job_id, None)
            progress_q.put({"type": "cancelled"})
            return

        base = os.path.splitext(srt_path)[0]
        out_srt = base + ".ko.srt"
        out_smi = base + ".ko.smi"

        srt_core.save_srt(subtitles, out_srt, bilingual=bilingual)
        srt_core.save_smi(subtitles, out_smi, title=os.path.basename(base), bilingual=bilingual)
        srt_core.clear_progress(srt_path)

        # 완료된 번역 파일 저장
        output_folder = profile.get("output_folder") or BASE_DIR
        for out_f in [out_srt, out_smi]:
            if os.path.exists(out_f):
                dest = os.path.join(output_folder, os.path.basename(out_f))
                os.makedirs(output_folder, exist_ok=True)
                shutil.copy2(out_f, dest)

        # 번역 완료 후 용어집 업데이트 + 스토리 요약 생성 (백그라운드)
        glossary_file = profile.get("glossary_file")
        story_file = profile.get("story_file")
        episode_name = os.path.splitext(os.path.basename(srt_path))[0]

        if glossary_file:
            try:
                srt_core.extract_and_update_glossary(client, subtitles, glossary_file)
            except Exception as e:
                print(f"  ⚠ 용어집 업데이트 오류: {e}", flush=True)

        if story_file:
            try:
                srt_core.generate_episode_summary(client, subtitles, story_file, episode_name)
            except Exception as e:
                print(f"  ⚠ 스토리 요약 오류: {e}", flush=True)

        with JOBS_LOCK:
            JOBS[job_id]["status"]  = "done"
            JOBS[job_id]["out_srt"] = out_srt
            JOBS[job_id]["out_smi"] = out_smi

        CANCEL_FLAGS.pop(job_id, None)
        progress_q.put({"type": "done"})

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
        CANCEL_FLAGS.pop(job_id, None)
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
            elif msg["type"] == "cancelled":
                # 번역된 부분이 있으면 partial 다운로드 가능 여부 전달
                with JOBS_LOCK:
                    job_data = JOBS.get(job_id, {})
                has_partial = bool(job_data.get("out_srt") and
                                   os.path.exists(job_data.get("out_srt", "")))
                data = json.dumps({"has_partial": has_partial})
                yield f"event: cancelled\ndata: {data}\n\n"
                break
            elif msg["type"] == "error":
                data = json.dumps({"message": msg["message"]})
                yield f"event: error_msg\ndata: {data}\n\n"
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── 라우트: 취소 ────────────────────────────────────────────
@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    with JOBS_LOCK:
        flag = CANCEL_FLAGS.get(job_id)
        job = JOBS.get(job_id)
    if not flag:
        return jsonify({"error": "진행 중인 작업이 없습니다"}), 404
    if job and job.get("status") not in ("running",):
        return jsonify({"error": "취소할 수 없는 상태입니다"}), 400
    flag.set()
    return jsonify({"ok": True})


# ─── 라우트: 다운로드 ────────────────────────────────────────
@app.route("/download/<job_id>/<ext>")
def download(job_id, ext):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] not in ("done", "cancelled"):
        return jsonify({"error": "아직 완료되지 않았습니다"}), 404

    path = job["out_srt"] if ext == "srt" else job["out_smi"]
    if not path or not os.path.exists(path):
        return jsonify({"error": "파일을 찾을 수 없습니다"}), 404

    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


# ─── 라우트: 설정 조회 ───────────────────────────────────────
@app.route("/settings", methods=["GET"])
def get_settings():
    cfg = migrate_config_if_needed(load_config_file())
    api_key = get_api_key()
    cfg["api_key_masked"] = ("•" * 8 + api_key[-4:]) if len(api_key) > 4 else ""
    return jsonify(cfg)


# ─── 라우트: 설정 저장 ───────────────────────────────────────
@app.route("/settings", methods=["POST"])
def save_settings():
    data = request.get_json()
    cfg = migrate_config_if_needed(load_config_file())

    cfg["model"]      = data.get("model", cfg.get("model"))
    cfg["batch_size"] = int(data.get("batch_size", cfg.get("batch_size", 20)))

    # API 키 업데이트
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

    save_config_file(cfg)
    reload_srt_config()
    return jsonify({"ok": True})


# ─── 라우트: 프로필 목록 ────────────────────────────────────
@app.route("/profiles", methods=["GET"])
def list_profiles():
    cfg = migrate_config_if_needed(load_config_file())
    profiles = cfg.get("profiles", {})
    current = cfg.get("current_profile")
    return jsonify({
        "profiles": {
            pid: {"id": pid, "name": p.get("name", pid)}
            for pid, p in profiles.items()
        },
        "current_profile": current,
    })


# ─── 라우트: 프로필 생성 ────────────────────────────────────
@app.route("/profiles", methods=["POST"])
def create_profile():
    data = request.get_json()
    cfg = migrate_config_if_needed(load_config_file())
    profiles = cfg.setdefault("profiles", {})

    # ID 자동 생성 (name 기반 slug)
    import re as _re
    name = data.get("name", "새 프로필")
    pid = _re.sub(r"[^a-z0-9-]", "-", name.lower())[:32].strip("-") or uuid.uuid4().hex[:8]
    # 중복 처리
    base_pid = pid
    counter = 1
    while pid in profiles:
        pid = f"{base_pid}-{counter}"
        counter += 1

    profiles[pid] = {
        "name": name,
        "system_prompt": data.get("system_prompt", ""),
        "character_notes": data.get("character_notes", {}),
        "context_folders": data.get("context_folders", []),
        "output_folder": data.get("output_folder") or None,
        "glossary_file": data.get("glossary_file") or None,
        "story_file": data.get("story_file") or None,
    }

    save_config_file(cfg)
    return jsonify({"ok": True, "id": pid})


# ─── 라우트: 프로필 수정 ────────────────────────────────────
@app.route("/profiles/<pid>", methods=["PUT"])
def update_profile(pid):
    data = request.get_json()
    cfg = migrate_config_if_needed(load_config_file())
    profiles = cfg.get("profiles", {})

    if pid not in profiles:
        return jsonify({"error": "프로필을 찾을 수 없습니다"}), 404

    p = profiles[pid]
    for field in ("name", "system_prompt", "character_notes",
                  "context_folders", "output_folder", "glossary_file", "story_file"):
        if field in data:
            p[field] = data[field] if data[field] != "" else None

    save_config_file(cfg)
    reload_srt_config()
    return jsonify({"ok": True})


# ─── 라우트: 프로필 삭제 ────────────────────────────────────
@app.route("/profiles/<pid>", methods=["DELETE"])
def delete_profile(pid):
    cfg = migrate_config_if_needed(load_config_file())
    profiles = cfg.get("profiles", {})

    if pid not in profiles:
        return jsonify({"error": "프로필을 찾을 수 없습니다"}), 404
    if cfg.get("current_profile") == pid:
        return jsonify({"error": "현재 활성 프로필은 삭제할 수 없습니다"}), 400
    if len(profiles) <= 1:
        return jsonify({"error": "마지막 프로필은 삭제할 수 없습니다"}), 400

    del profiles[pid]
    save_config_file(cfg)
    return jsonify({"ok": True})


# ─── 라우트: 프로필 활성화 ──────────────────────────────────
@app.route("/profiles/<pid>/activate", methods=["POST"])
def activate_profile(pid):
    cfg = migrate_config_if_needed(load_config_file())
    if pid not in cfg.get("profiles", {}):
        return jsonify({"error": "프로필을 찾을 수 없습니다"}), 404

    cfg["current_profile"] = pid
    save_config_file(cfg)
    reload_srt_config()
    return jsonify({"ok": True})


# ─── 라우트: 용어집 조회 ────────────────────────────────────
@app.route("/profiles/<pid>/glossary", methods=["GET"])
def get_glossary(pid):
    cfg = migrate_config_if_needed(load_config_file())
    profile = cfg.get("profiles", {}).get(pid)
    if not profile:
        return jsonify({"error": "프로필을 찾을 수 없습니다"}), 404

    terms = srt_core.load_glossary(profile.get("glossary_file"))
    return jsonify({"terms": terms, "count": len(terms)})


# ─── 실행 ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SRT 자막 번역기 웹 UI")
    print("  http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
