#!/usr/bin/env python3
"""
SRT 자막 한국어 번역기 v3
- claude-sonnet-4-6: 캐릭터 말투, 나레이션, 욕설 직역
- 이전화 번역 컨텍스트 자동 로딩 (용어/말투 일관성)
- 중단 후 재시작 지원 (.progress.json)

사용법:
  python3 translate_srt.py episode.srt          # 단일 파일
  python3 translate_srt.py --all ./             # 폴더 내 모든 SRT 순서대로
  python3 translate_srt.py episode.srt --reset  # 저장된 진행 상황 무시하고 처음부터

API 키 설정:
  .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 추가
  또는 export ANTHROPIC_API_KEY=sk-ant-...
"""

import re
import sys
import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from glob import glob
import anthropic

# ─── .env 자동 로드 ───────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _key, _val = _k.strip(), _v.strip()
                if not os.environ.get(_key):
                    os.environ[_key] = _val

# ─── 설정 (config.json에서 로드) ─────────────────────────────
def load_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # fallback defaults
    return {"model": "claude-sonnet-4-6", "batch_size": 20,
            "system_prompt": "", "character_notes": {}}


def get_current_profile(cfg: dict) -> dict:
    """현재 활성 프로필 반환. 구 포맷(profiles 없음) 하위 호환."""
    profiles = cfg.get("profiles")
    if not profiles:
        # 기존 포맷 — 최상위 필드를 그대로 반환
        return {
            "name": "기본",
            "system_prompt": cfg.get("system_prompt", ""),
            "character_notes": cfg.get("character_notes", {}),
            "context_folders": [],
            "output_folder": None,
            "glossary_file": None,
            "story_file": None,
        }
    cur_id = cfg.get("current_profile", next(iter(profiles)))
    p = profiles.get(cur_id, next(iter(profiles.values())))
    # 누락 필드 기본값 채우기
    p.setdefault("context_folders", [])
    p.setdefault("output_folder", None)
    p.setdefault("glossary_file", None)
    p.setdefault("story_file", None)
    return p


def _build_system_prompt(cfg: dict, glossary_text: str = "") -> str:
    profile = get_current_profile(cfg)
    prompt = profile.get("system_prompt", "")
    notes = profile.get("character_notes", {})
    if notes:
        char_section = "\n\n## 캐릭터별 말투\n" + "\n".join(
            f"- **{name}**: {note}" for name, note in notes.items()
        )
        prompt += char_section
    if glossary_text:
        prompt += "\n\n" + glossary_text
    return prompt


_cfg = load_config()
MODEL      = _cfg.get("model", "claude-sonnet-4-6")
BATCH_SIZE = _cfg.get("batch_size", 20)
SYSTEM_PROMPT = _build_system_prompt(_cfg)


# ─── 데이터 구조 ──────────────────────────────────────────────
@dataclass
class Subtitle:
    index: int
    start_ms: int
    end_ms: int
    lines: list       # 원본 줄 목록
    translated: str = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        s = Subtitle(**d)
        return s


# ─── 프랑스어 감지 ────────────────────────────────────────────
_FRENCH_CHARS = re.compile(r'[éèêëàâäùûüîïôöœæçÉÈÊËÀÂÄÙÛÜÎÏÔÖŒÆÇ]')
_FRENCH_WORDS = re.compile(
    r'\b(vous|etes|mais|mes|les|des|du|je|tu|il|nous|ils|elle|elles|'
    r'pour|que|qui|quoi|pardonnez|encore|liberales|dames|donc|'
    r'mignonnes|contraire|maitre|fiche|annees|amour|honnete|'
    r'avant|faudra|comptons|ensemble|nostri|satanas)\b',
    re.IGNORECASE
)

def is_french_text(text: str) -> bool:
    """텍스트가 프랑스어인지 감지. 장면 지문([...])은 제외."""
    clean = re.sub(r'\[.*?\]', '', text).strip()
    if not clean:
        return False
    # 프랑스어 악센트 문자 포함 여부
    if _FRENCH_CHARS.search(clean):
        return True
    # 악센트 없는 프랑스어 단어 2개 이상 매칭
    return len(_FRENCH_WORDS.findall(clean)) >= 2


# ─── SRT 파싱/저장 ────────────────────────────────────────────
def ts_to_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    hms, ms = ts.rsplit(".", 1)
    h, m, s = hms.split(":")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms[:3].ljust(3, "0"))


def ms_to_ts(ms: int) -> str:
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: str) -> list:
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()

    subtitles = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", lines[1])
        if not m:
            continue
        text_lines = [l.strip() for l in lines[2:] if l.strip()]
        if text_lines:
            subtitles.append(Subtitle(idx, ts_to_ms(m.group(1)), ts_to_ms(m.group(2)), text_lines))
    return subtitles


def save_srt(subtitles: list, out_path: str, bilingual: bool = False,
             french_bilingual: bool = False):
    out = []
    for i, s in enumerate(subtitles, 1):
        out.append(str(i))
        out.append(f"{ms_to_ts(s.start_ms)} --> {ms_to_ts(s.end_ms)}")
        orig = " ".join(s.lines) if s.lines else ""
        if french_bilingual and s.lines and is_french_text(orig):
            # 프랑스어 줄: 원문 위에, 한국어 아래
            out.append(orig)
            out.append(s.translated.replace(" / ", "\n"))
        else:
            out.append(s.translated.replace(" / ", "\n"))
            if bilingual and s.lines:
                out.append(orig)
        out.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def save_smi(subtitles: list, out_path: str, title: str = "", bilingual: bool = False,
             french_bilingual: bool = False):
    header = f"""<SAMI>
<HEAD>
<TITLE>{title}</TITLE>
<STYLE TYPE="text/css">
P {{ margin-left:8pt; margin-right:8pt; margin-bottom:2pt; margin-top:2pt;
    text-align:center; font-size:20pt; font-family:'맑은 고딕',Arial;
    font-weight:normal; color:white; background-color:black; }}
.KRCC {{ Name:Korean; lang:ko-KR; SAMI_Type:CC; }}
</STYLE>
</HEAD>
<BODY>
"""
    parts = []
    for s in subtitles:
        html = s.translated.replace(" / ", "<br>")
        orig = " ".join(s.lines) if s.lines else ""
        if french_bilingual and s.lines and is_french_text(orig):
            # 프랑스어 줄: 원문 위에, 한국어 아래
            html = f"<i>{orig}</i><br>" + html
        elif bilingual and s.lines:
            html += f"<br><i>{orig}</i>"
        parts.append(
            f'<SYNC start="{s.start_ms}"><P class="KRCC">{html}</P></SYNC>\n'
            f'<SYNC start="{s.end_ms}"><P class="KRCC">&nbsp;</P></SYNC>'
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(parts) + "\n</BODY>\n</SAMI>\n")


# ─── 진행 상황 저장/로드 ──────────────────────────────────────
def progress_path(srt_path: str) -> str:
    return os.path.splitext(srt_path)[0] + ".progress.json"


def save_progress(srt_path: str, subtitles: list, completed_up_to: int):
    data = {
        "completed_up_to": completed_up_to,
        "subtitles": [s.to_dict() for s in subtitles]
    }
    with open(progress_path(srt_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress(srt_path: str):
    p = progress_path(srt_path)
    if not os.path.exists(p):
        return None, 0
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    subtitles = [Subtitle.from_dict(d) for d in data["subtitles"]]
    return subtitles, data.get("completed_up_to", 0)


def clear_progress(srt_path: str):
    p = progress_path(srt_path)
    if os.path.exists(p):
        os.remove(p)


# ─── 이전화 컨텍스트 추출 ─────────────────────────────────────
def build_context_from_previous(folders, current_file: str) -> str:
    """이미 번역된 *.ko.srt 파일에서 용어/말투 컨텍스트 추출.

    folders: str 또는 list[str] — 스캔할 폴더(들).
             시즌 폴더 여러 개를 넘기면 크로스-시즌 연속성 지원.
    """
    if isinstance(folders, str):
        folders = [folders]

    # 모든 폴더에서 ko.srt 수집 후 파일명 기준 정렬 (시즌1→시즌2 순서)
    all_ko = []
    for folder in folders:
        if folder and os.path.isdir(folder):
            all_ko.extend(glob(os.path.join(folder, "*.ko.srt")))
    all_ko = sorted(set(all_ko), key=lambda p: os.path.basename(p))

    # 현재 파일의 ko.srt는 제외
    base = os.path.splitext(current_file)[0]
    ko_files = [f for f in all_ko if not os.path.abspath(f).startswith(os.path.abspath(base))]

    if not ko_files:
        return ""

    # 최근 2개 화 기준으로 샘플 추출 (토큰 효율)
    selected = ko_files[-2:]
    samples = []
    for kf in selected:
        subs = parse_srt(kf)
        mid = len(subs) // 2
        sampled = subs[:20] + subs[mid:mid+20]
        for s in sampled:
            orig_lines = s.lines if hasattr(s, 'lines') and s.lines else []
            if orig_lines and s.translated:
                samples.append(f"  EN: {' '.join(orig_lines)}")
                samples.append(f"  KO: {s.translated}")

    if not samples:
        return ""

    context = f"""
## 이전 화 번역 참고 (용어/말투 일관성 유지)
아래는 이미 번역된 이전 에피소드 샘플입니다. 동일한 용어, 고유명사, 캐릭터 말투를 그대로 유지하세요.

{chr(10).join(samples[:80])}
"""
    return context


def load_glossary(glossary_file: str) -> dict:
    """용어집 파일 로드. 없으면 빈 딕트 반환."""
    if glossary_file and os.path.exists(glossary_file):
        with open(glossary_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("terms", {})
    return {}


def glossary_to_prompt_section(terms: dict) -> str:
    """용어집 딕트를 시스템 프롬프트용 텍스트로 변환."""
    if not terms:
        return ""
    lines = [f"- {en} → {ko}" for en, ko in sorted(terms.items())]
    return "## 확정 용어집 (반드시 아래 번역을 사용할 것)\n" + "\n".join(lines)


def extract_and_update_glossary(client, subtitles: list, glossary_file: str):
    """번역된 자막에서 고유명사/특수 용어를 추출해 용어집 파일에 누적 저장.

    Claude Haiku 사용 (저비용). 기존 용어집과 병합 — 기존 항목 우선.
    """
    if not glossary_file:
        return

    # EN/KO 쌍 샘플링 (최대 60개)
    pairs = []
    for s in subtitles:
        if s.lines and s.translated:
            pairs.append(f"EN: {' '.join(s.lines)}\nKO: {s.translated}")
        if len(pairs) >= 60:
            break

    if not pairs:
        return

    existing = load_glossary(glossary_file)

    prompt = f"""아래는 드라마 자막 번역 샘플입니다.
고유명사(인물 이름, 지명, 단체명), 드라마 특유 용어만 추출해
JSON 형식으로 반환하세요.

규칙:
- 일반 동사/형용사/부사 제외
- 이미 확정된 용어집에 있는 항목은 그대로 유지
- 형식: {{"영어": "한국어", ...}}
- 항목이 없으면 {{}} 반환

기존 확정 용어집:
{json.dumps(existing, ensure_ascii=False, indent=2)}

번역 샘플:
{chr(10).join(pairs[:40])}

JSON:"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        # JSON 블록 추출
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            new_terms = json.loads(m.group())
            # 기존 항목 우선 병합 (기존값 보존)
            merged = {**new_terms, **existing}
            data = {
                "version": 2,
                "last_updated": __import__("datetime").date.today().isoformat(),
                "terms": merged
            }
            os.makedirs(os.path.dirname(os.path.abspath(glossary_file)), exist_ok=True)
            with open(glossary_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠ 용어집 추출 실패 (무시): {e}", flush=True)


def generate_episode_summary(client, subtitles: list, story_file: str, episode_name: str):
    """번역된 자막으로 에피소드 요약 생성 후 story_file에 누적 저장.

    Claude Haiku 사용 (저비용).
    """
    if not story_file:
        return

    # 자막 샘플 (앞 30개 + 중간 20개 + 끝 20개)
    total = len(subtitles)
    mid = total // 2
    sampled = subtitles[:30] + subtitles[mid:mid+20] + subtitles[max(0, total-20):]
    text_sample = "\n".join(
        s.translated for s in sampled if s.translated
    )

    prompt = f"""아래는 드라마 에피소드 자막(한국어 번역)의 일부입니다.
등장인물, 주요 사건, 감정 흐름을 중심으로 3~5문장 요약을 한국어로 작성하세요.
스포일러 없이 핵심만 간결하게.

에피소드: {episode_name}

자막 샘플:
{text_sample[:3000]}

요약:"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = msg.content[0].text.strip()
        os.makedirs(os.path.dirname(os.path.abspath(story_file)), exist_ok=True)
        with open(story_file, "a", encoding="utf-8") as f:
            f.write(f"\n## {episode_name}\n{summary}\n")
    except Exception as e:
        print(f"  ⚠ 스토리 요약 생성 실패 (무시): {e}", flush=True)


def load_recent_story(story_file: str, max_episodes: int = 3) -> str:
    """story_file에서 최근 N개 에피소드 요약을 읽어 컨텍스트용 텍스트로 반환."""
    if not story_file or not os.path.exists(story_file):
        return ""
    with open(story_file, encoding="utf-8") as f:
        content = f.read()
    # ## 헤더로 분리
    sections = re.split(r"\n(?=## )", content.strip())
    recent = [s.strip() for s in sections if s.strip()][-max_episodes:]
    if not recent:
        return ""
    return "## 이전 화 스토리 요약\n" + "\n\n".join(recent)


# ─── 재저장 (API 호출 없이) ───────────────────────────────────
def resave_with_original(srt_path: str, bilingual: bool = False,
                         french_bilingual: bool = False):
    """원본 .srt + 기존 .ko.srt를 합쳐 새 형식으로 재저장.
    이미 번역 완료된 파일을 API 비용 없이 french_bilingual 모드로 변환.
    """
    base = os.path.splitext(srt_path)[0]
    ko_path = base + ".ko.srt"
    if not os.path.exists(ko_path):
        print(f"  ✗ .ko.srt 없음: {ko_path}")
        return False

    orig_subs = parse_srt(srt_path)
    ko_subs   = parse_srt(ko_path)

    if len(orig_subs) != len(ko_subs):
        print(f"  ⚠ 자막 수 불일치 (원본: {len(orig_subs)}, 번역: {len(ko_subs)}) — 인덱스 매칭 시도")

    # 인덱스 기준 매칭: 원본 lines + 번역 translated 합치기
    merged = []
    for o in orig_subs:
        # 동일 index인 번역 자막 찾기
        tr_sub = next((k for k in ko_subs if k.index == o.index), None)
        if tr_sub:
            o.translated = " ".join(tr_sub.lines)  # ko.srt의 텍스트를 translated로
        else:
            o.translated = " ".join(o.lines)  # 번역 없으면 원문 유지
        merged.append(o)

    out_srt = base + ".ko.srt"
    out_smi = base + ".ko.smi"
    save_srt(merged, out_srt, bilingual=bilingual, french_bilingual=french_bilingual)
    save_smi(merged, out_smi, title=os.path.basename(base),
             bilingual=bilingual, french_bilingual=french_bilingual)
    print(f"  → {out_srt}")
    print(f"  → {out_smi}")
    return True


# ─── 번역 ────────────────────────────────────────────────────
def translate_batch(client, items: list, context: str = "") -> tuple:
    """items: [(batch_idx, lines_list), ...]
    Returns: (results, input_tokens, output_tokens)
    """
    parts = []
    for i, (_, lines) in enumerate(items):
        joined = " / ".join(lines) if len(lines) > 1 else lines[0]
        parts.append(f"[{i+1}|{len(lines)}줄] {joined}")

    system = SYSTEM_PROMPT + context

    prompt = f"""아래 영어 자막을 한국어로 번역하세요.

형식 규칙:
- 각 항목: [번호|줄수] 형식 그대로 유지
- 원본이 N줄이면 " / " 로 N줄 구분 (줄 수 맞추기)
- 번역문만 출력 (설명·주석 없이)
- 욕설은 절대 순화하지 말 것

자막:
{chr(10).join(parts)}

번역:"""

    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    in_tok = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens

    result_text = msg.content[0].text.strip()
    results = [None] * len(items)

    for line in result_text.splitlines():
        m = re.match(r"\[(\d+)\|\d+줄\]\s*(.*)", line.strip())
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(results):
                results[i] = m.group(2).strip()

    for i, (r, (_, orig_lines)) in enumerate(zip(results, items)):
        if not r:
            results[i] = " / ".join(orig_lines)

    return results, in_tok, out_tok


def translate_batch_with_retry(client, items, context="", max_retries=3):
    for attempt in range(max_retries):
        try:
            return translate_batch(client, items, context)
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  ⚠ Rate limit — {wait}초 대기 후 재시도...", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** attempt * 2
            print(f"  ⚠ API 오류 ({e}) — {wait}초 대기 후 재시도...", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  ✗ 예상치 못한 오류: {e}", flush=True)
            break
    print(f"  ✗ 번역 실패 — 원문 유지", flush=True)
    return [" / ".join(lines) for _, lines in items], 0, 0


def translate_all(client, subtitles, srt_path, context="",
                  batch_size=BATCH_SIZE, start_from=0,
                  progress_callback=None,
                  model_pricing=(3.0, 15.0),
                  cancel_event=None):
    """
    progress_callback(pct, done, total, cost_usd) — 진행률 콜백
    model_pricing: (input_price_per_mtok, output_price_per_mtok)
    cancel_event: threading.Event — 세트되면 다음 배치 전에 중단
    """
    total = len(subtitles)
    total_in_tok = 0
    total_out_tok = 0

    for start in range(start_from, total, batch_size):
        # 취소 확인 (배치 시작 전)
        if cancel_event and cancel_event.is_set():
            print("  ⏹ 번역 취소됨", flush=True)
            break

        batch_subs = subtitles[start:start + batch_size]
        items = [(i, s.lines) for i, s in enumerate(batch_subs)]
        end = min(start + batch_size, total)
        pct = end / total * 100
        print(f"  [{pct:5.1f}%] {start+1}~{end}/{total}", flush=True)

        translated, in_tok, out_tok = translate_batch_with_retry(client, items, context)
        total_in_tok += in_tok
        total_out_tok += out_tok
        cost_usd = (total_in_tok / 1e6 * model_pricing[0]
                    + total_out_tok / 1e6 * model_pricing[1])

        for sub, tr in zip(batch_subs, translated):
            sub.translated = tr

        # 배치마다 진행 저장 (중단 복구용)
        save_progress(srt_path, subtitles, end)

        if progress_callback:
            progress_callback(pct, end, total, cost_usd)

    return subtitles


# ─── 파일 처리 ───────────────────────────────────────────────
def process_file(client, srt_path: str, reset: bool = False, bilingual: bool = False,
                 french_bilingual: bool = False, resave: bool = False):
    base = os.path.splitext(srt_path)[0]
    out_srt = base + ".ko.srt"
    out_smi = base + ".ko.smi"
    folder = os.path.dirname(os.path.abspath(srt_path))

    print(f"\n{'='*60}")
    print(f"파일: {os.path.basename(srt_path)}")
    print(f"{'='*60}")

    # --resave: API 없이 기존 번역 재저장
    if resave:
        print("⟳  재저장 모드 (API 호출 없음)", flush=True)
        ok = resave_with_original(srt_path, bilingual=bilingual,
                                  french_bilingual=french_bilingual)
        if ok:
            print("완료!")
        return

    # 이미 완료된 경우 스킵
    if not reset and os.path.exists(out_srt) and os.path.getsize(out_srt) > 1000:
        prog_file = progress_path(srt_path)
        if not os.path.exists(prog_file):
            print("⏭  이미 완료 — 스킵 (--reset 옵션으로 재번역 가능)")
            return

    # 진행 상황 로드 or 새로 파싱
    if not reset:
        subtitles, start_from = load_progress(srt_path)
        if subtitles and start_from > 0:
            print(f"⟳  이어서 번역 ({start_from}/{len(subtitles)}부터)", flush=True)
        else:
            subtitles = parse_srt(srt_path)
            start_from = 0
    else:
        clear_progress(srt_path)
        subtitles = parse_srt(srt_path)
        start_from = 0

    print(f"자막 수: {len(subtitles)}개 | 모델: {MODEL} | 배치: {BATCH_SIZE}개")

    # 이전화 컨텍스트
    context = build_context_from_previous(folder, srt_path)
    if context:
        print(f"이전화 컨텍스트 로드됨 ({len(context)}자)", flush=True)
    print()

    translate_all(client, subtitles, srt_path, context=context, start_from=start_from)

    print("\n저장 중...", flush=True)
    save_srt(subtitles, out_srt, bilingual=bilingual, french_bilingual=french_bilingual)
    save_smi(subtitles, out_smi, title=os.path.basename(base),
             bilingual=bilingual, french_bilingual=french_bilingual)
    clear_progress(srt_path)  # 완료 후 진행 파일 삭제
    print(f"  → {out_srt}")
    print(f"  → {out_smi}")
    print("완료!")


# ─── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SRT → 한국어 SRT + SMI 변환기 (뱀파이어와의 인터뷰 최적화)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python3 translate_srt.py ep1.srt                 # ep1 번역
  python3 translate_srt.py ep2.srt                 # ep2 번역 (ep1 컨텍스트 자동 로드)
  python3 translate_srt.py --all /path/to/srt/     # 전체 폴더 순서대로
  python3 translate_srt.py ep1.srt --reset         # 처음부터 다시 번역

API 키 설정:
  .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 추가
"""
    )
    parser.add_argument("srt_file", nargs="?", help="입력 SRT 파일")
    parser.add_argument("--all", metavar="DIR", help="폴더 내 모든 SRT 파일 처리 (순서대로)")
    parser.add_argument("--reset", action="store_true", help="저장된 진행 상황 무시하고 처음부터")
    parser.add_argument("--api-key", help="Anthropic API 키 (없으면 .env 또는 환경변수 사용)")
    parser.add_argument("--french-bilingual", action="store_true",
                        help="프랑스어 대사 줄에만 원문+한국어 이중 표기")
    parser.add_argument("--resave", action="store_true",
                        help="기존 .ko.srt 재저장 (API 호출 없음). --french-bilingual과 함께 사용")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("오류: API 키가 필요합니다.")
        print("  방법 1: .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 추가")
        print("  방법 2: export ANTHROPIC_API_KEY=sk-ant-...")
        print("  방법 3: --api-key YOUR_KEY 옵션")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.all:
        files = sorted([
            f for f in glob(os.path.join(args.all, "*.srt"))
            if ".ko." not in f
        ])
        if not files:
            print(f"SRT 파일 없음: {args.all}")
            sys.exit(1)
        print(f"발견된 SRT 파일: {len(files)}개")
        for f in files:
            process_file(client, f, reset=args.reset,
                         french_bilingual=args.french_bilingual,
                         resave=args.resave)
        print("\n\n전체 번역 완료!")

    elif args.srt_file:
        if not os.path.exists(args.srt_file):
            print(f"파일 없음: {args.srt_file}")
            sys.exit(1)
        process_file(client, args.srt_file, reset=args.reset,
                     french_bilingual=args.french_bilingual,
                     resave=args.resave)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
