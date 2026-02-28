#!/usr/bin/env python3
"""
시즌1 컨텍스트 부트스트랩 스크립트
-- 이미 번역 완료된 시즌1 .ko.srt 파일로 용어집 + 스토리 요약을 소급 생성 --

사용:
    python3 build_season1_context.py
"""

import os
import sys
import glob

# .env 로드
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# translate_srt 임포트
sys.path.insert(0, str(Path(__file__).parent))
from translate_srt import parse_srt, extract_and_update_glossary, generate_episode_summary, Subtitle
from anthropic import Anthropic

# ─── 설정 ────────────────────────────────────────────────────────
SEASON1_DIR  = "/Users/hyojongchu/Documents/srt/뱀파이어 인터뷰 시즌 1"
GLOSSARY_FILE = "/Users/hyojongchu/Documents/srt/interview-vampire-glossary.json"
STORY_FILE    = "/Users/hyojongchu/Documents/srt/interview-vampire-story.md"

# 에피소드 파일명 패턴 → 표시 이름 매핑
EPISODE_NAMES = {
    "-1-1-": "S1E1 - In Throes of Increasing Wonder",
    "-1-2-": "S1E2 - After the Phantoms of Your Former Self",
    "-1-3-": "S1E3 - Is My Very Nature That of the Devil",
    "-1-4-": "S1E4 - The Ruthless Pursuit of Blood",
    "-1-5-": "S1E5 - A Vile Hunger for Your Hammering Heart",
    "-1-6-": "S1E6 - Like Angels Put in Hell by God",
    "-1-7-": "S1E7 - The Thing Lay Still",
}

def get_episode_name(filename: str) -> str:
    for pattern, name in EPISODE_NAMES.items():
        if pattern in filename:
            return name
    return os.path.basename(filename).replace(".ko.srt", "")

def pair_subtitles(en_path: str, ko_path: str) -> list:
    """EN .srt + KO .ko.srt 를 index 기준으로 페어링.
    Subtitle.lines = 영어, Subtitle.translated = 한국어
    """
    en_subs = parse_srt(en_path)
    ko_subs = parse_srt(ko_path)

    ko_dict = {s.index: " ".join(s.lines) for s in ko_subs}
    paired = []
    for s in en_subs:
        ko_text = ko_dict.get(s.index, "")
        if ko_text:
            s.translated = ko_text
            paired.append(s)
    return paired

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # 번역 완료된 .ko.srt 파일 목록 (이름순 정렬 = 화 순서)
    ko_files = sorted(glob.glob(os.path.join(SEASON1_DIR, "*.ko.srt")))
    if not ko_files:
        print(f"❌ {SEASON1_DIR} 에서 .ko.srt 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"📺 시즌1 에피소드 {len(ko_files)}편 처리 시작\n")

    for ko_path in ko_files:
        en_path = ko_path.replace(".ko.srt", ".srt")
        ep_name = get_episode_name(os.path.basename(ko_path))

        if not os.path.exists(en_path):
            print(f"  ⚠ EN 원본 없음, 건너뜀: {os.path.basename(en_path)}")
            continue

        print(f"📝 {ep_name}")
        paired = pair_subtitles(en_path, ko_path)
        print(f"   페어링된 자막 수: {len(paired)}")

        # 용어집 추출·누적
        print(f"   🔤 용어집 추출 중...", end=" ", flush=True)
        extract_and_update_glossary(client, paired, GLOSSARY_FILE)
        print("완료")

        # 에피소드 요약 생성·누적
        print(f"   📖 스토리 요약 생성 중...", end=" ", flush=True)
        generate_episode_summary(client, paired, STORY_FILE, ep_name)
        print("완료\n")

    # 결과 확인
    import json
    if os.path.exists(GLOSSARY_FILE):
        with open(GLOSSARY_FILE, encoding="utf-8") as f:
            glossary = json.load(f)
        terms = glossary.get("terms", {})
        print(f"✅ 용어집 생성 완료: {GLOSSARY_FILE}")
        print(f"   총 {len(terms)}개 용어 추출됨")
        if terms:
            sample = list(terms.items())[:8]
            for en, ko in sample:
                print(f"   {en} → {ko}")
            if len(terms) > 8:
                print(f"   ... 외 {len(terms)-8}개")
    else:
        print("⚠ 용어집 파일이 생성되지 않았습니다.")

    print()
    if os.path.exists(STORY_FILE):
        with open(STORY_FILE, encoding="utf-8") as f:
            story = f.read()
        ep_count = story.count("## S1E")
        print(f"✅ 스토리 요약 생성 완료: {STORY_FILE}")
        print(f"   총 {ep_count}편 요약 포함됨")
        print()
        print("─" * 60)
        print(story.strip())
        print("─" * 60)
    else:
        print("⚠ 스토리 파일이 생성되지 않았습니다.")

if __name__ == "__main__":
    main()
