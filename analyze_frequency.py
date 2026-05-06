"""
analyze_freq.py

전처리(preprocess.py)를 다시 실행하지 않고
기존 dataset_info.json + gloss_dict.json만으로
글로스 빈도 분석 및 gloss_freq_info.json을 생성합니다.

사용법:
    python analyze_freq.py
    python analyze_freq.py --save_path ./keypoint_data
"""

import json
import os
import argparse
from collections import Counter


SAVE_PATH = "./keypoint_data"


def analyze_freq(save_path: str = SAVE_PATH) -> dict:
    # ── 파일 로드 ──
    dataset_path = os.path.join(save_path, "dataset_info.json")
    dict_path    = os.path.join(save_path, "gloss_dict.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_info.json 없음: {dataset_path}")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"gloss_dict.json 없음: {dict_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    with open(dict_path, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)

    # ── 빈도 계산 ──
    all_gloss_list = [g for item in dataset for g in item['gloss_sequence']]
    freq           = Counter(all_gloss_list)

    # ── 구간별 통계 ──
    freq_dist = {
        "0~1":   sum(1 for c in freq.values() if c <= 1),
        "2~5":   sum(1 for c in freq.values() if 2 <= c <= 5),
        "6~10":  sum(1 for c in freq.values() if 6 <= c <= 10),
        "11~50": sum(1 for c in freq.values() if 11 <= c <= 50),
        "51+":   sum(1 for c in freq.values() if c > 50),
    }

    # ── gloss_dict에 있지만 dataset에 등장 안 한 글로스 확인 ──
    dict_glosses    = set(gloss_dict.keys()) - {"<blank>"}
    dataset_glosses = set(freq.keys())
    missing_in_data = dict_glosses - dataset_glosses   # dict엔 있으나 데이터엔 없음
    extra_in_data   = dataset_glosses - dict_glosses   # 데이터엔 있으나 dict엔 없음 (이상 케이스)

    freq_info = {
        "total_tokens":      len(all_gloss_list),
        "total_types":       len(freq),
        "dict_size":         len(gloss_dict),
        "missing_in_data":   sorted(list(missing_in_data)),   # 0회 등장 글로스
        "extra_in_data":     sorted(list(extra_in_data)),     # dict 미등록 글로스
        "freq_dist":         freq_dist,
        "gloss_freq":        dict(freq),
    }

    # ── 저장 ──
    out_path = os.path.join(save_path, "gloss_freq_info.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(freq_info, f, ensure_ascii=False, indent=4)

    # ── 출력 ──
    print("=" * 50)
    print("글로스 빈도 분석 결과")
    print("=" * 50)
    print(f"  총 샘플 수:          {len(dataset)}")
    print(f"  총 글로스 토큰 수:    {freq_info['total_tokens']}")
    print(f"  고유 글로스 종류:     {freq_info['total_types']}")
    print(f"  gloss_dict 크기:     {freq_info['dict_size']}  (<blank> 포함)")
    print()
    print("  [빈도 구간별 클래스 수]")
    for k, v in freq_dist.items():
        bar = "█" * min(v, 40)
        print(f"    {k:>6}회: {v:4d}개  {bar}")
    print()

    if missing_in_data:
        print(f"  ⚠️  dict에 있으나 데이터에 0회 등장: {len(missing_in_data)}개")
        print(f"     → {list(missing_in_data)[:10]}{'...' if len(missing_in_data)>10 else ''}")
    if extra_in_data:
        print(f"  ⚠️  데이터에 있으나 dict 미등록:    {len(extra_in_data)}개")
        print(f"     → {list(extra_in_data)[:10]}{'...' if len(extra_in_data)>10 else ''}")
    if not missing_in_data and not extra_in_data:
        print("  ✅ dict와 dataset 글로스 일치")

    print()
    print(f"  저장 완료: {out_path}")
    print("=" * 50)

    return freq_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="글로스 빈도 분석 및 gloss_freq_info.json 생성")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help=f"keypoint_data 디렉토리 경로 (기본값: {SAVE_PATH})")
    args = parser.parse_args()

    analyze_freq(save_path=args.save_path)