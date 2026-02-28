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

def _build_system_prompt(cfg: dict) -> str:
    prompt = cfg.get("system_prompt", "")
    notes = cfg.get("character_notes", {})
    if notes:
        char_section = "\n\n## 캐릭터별 말투\n" + "\n".join(
            f"- **{name}**: {note}" for name, note in notes.items()
        )
        prompt += char_section
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


def save_srt(subtitles: list, out_path: str):
    out = []
    for i, s in enumerate(subtitles, 1):
        out.append(str(i))
        out.append(f"{ms_to_ts(s.start_ms)} --> {ms_to_ts(s.end_ms)}")
        out.append(s.translated.replace(" / ", "\n"))
        out.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def save_smi(subtitles: list, out_path: str, title: str = ""):
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
def build_context_from_previous(folder: str, current_file: str) -> str:
    """이미 번역된 *.ko.srt 파일에서 용어/말투 컨텍스트 추출"""
    ko_files = sorted(glob(os.path.join(folder, "*.ko.srt")))
    # 현재 파일의 ko.srt는 제외
    base = os.path.splitext(current_file)[0]
    ko_files = [f for f in ko_files if not f.startswith(base)]

    if not ko_files:
        return ""

    # 최근 2개 화 기준으로 샘플 추출 (너무 길면 토큰 낭비)
    selected = ko_files[-2:]
    samples = []
    for kf in selected:
        subs = parse_srt(kf)
        # 앞 20개 + 중간 20개 샘플링
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
                  model_pricing=(3.0, 15.0)):
    """
    progress_callback(pct, done, total, cost_usd) — 진행률 콜백
    model_pricing: (input_price_per_mtok, output_price_per_mtok)
    """
    total = len(subtitles)
    total_in_tok = 0
    total_out_tok = 0

    for start in range(start_from, total, batch_size):
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
def process_file(client, srt_path: str, reset: bool = False):
    base = os.path.splitext(srt_path)[0]
    out_srt = base + ".ko.srt"
    out_smi = base + ".ko.smi"
    folder = os.path.dirname(os.path.abspath(srt_path))

    print(f"\n{'='*60}")
    print(f"파일: {os.path.basename(srt_path)}")
    print(f"{'='*60}")

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
    save_srt(subtitles, out_srt)
    save_smi(subtitles, out_smi, title=os.path.basename(base))
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
            process_file(client, f, reset=args.reset)
        print("\n\n전체 번역 완료!")

    elif args.srt_file:
        if not os.path.exists(args.srt_file):
            print(f"파일 없음: {args.srt_file}")
            sys.exit(1)
        process_file(client, args.srt_file, reset=args.reset)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
