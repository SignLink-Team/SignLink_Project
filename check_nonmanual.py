"""
check_nonmanual.py
------------------
전처리 결과 비수지 데이터 확인용 스크립트

확인 항목:
    1. dataset_info.json 로드 및 기본 통계
    2. 샘플 1개의 nms_labels 활성화 현황
    3. 전체 데이터셋 비수지 키별 활성화 비율
    4. nonmanual .npy 파일 shape / 값 범위
"""

import json
import numpy as np
import os

SAVE_PATH           = "./keypoint_data"
NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")
INFO_PATH           = os.path.join(SAVE_PATH, "dataset_info.json")

NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "Ebf", "Hno", "Hs"]


def load_dataset_info():
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def segments_to_active_frames(segments, total_frames):
    """구간 리스트 → 활성 프레임 수 합산"""
    count = 0
    for seg in segments:
        count += seg["end_frame"] - seg["start_frame"] + 1
    return min(count, total_frames)


def check_basic_stats(data):
    valid = [d for d in data if "nonmanual_input_dim" in d]
    print("=" * 50)
    print(f"[1] 기본 통계")
    print(f"    총 샘플 수            : {len(data)}")
    print(f"    비수지 전처리 완료    : {len(valid)}")
    print(f"    수지 입력 차원        : {data[0]['input_dim']}")
    print(f"    비수지 입력 차원      : {valid[0]['nonmanual_input_dim']}")
    print(f"    비수지 라벨 키        : {NMS_KEYS}")
    lengths = [d['length'] for d in data]
    print(f"    시퀀스 길이 평균      : {np.mean(lengths):.1f} frames")
    print(f"    시퀀스 길이 범위      : {min(lengths)} ~ {max(lengths)} frames")


def check_sample_labels(data, sample_idx=0):
    valid  = [d for d in data if "nms_labels" in d]
    sample = valid[sample_idx]
    total  = sample['length']

    print("=" * 50)
    print(f"[2] 샘플 확인 (id: {sample['id']})")
    print(f"    총 프레임 수: {total}")
    print(f"    수어 시퀀스 : {sample['gloss_sequence']}")
    print()
    print(f"    {'키':<8} {'활성 프레임':>10} {'비율':>8}  구간")
    print(f"    {'-'*55}")

    for key in NMS_KEYS:
        segments = sample['nms_labels'].get(key, [])
        active   = segments_to_active_frames(segments, total)
        ratio    = active / total * 100
        seg_str  = ', '.join(
            f"{s['start_frame']}~{s['end_frame']}" for s in segments
        ) if segments else "-"
        print(f"    {key:<8} {active:>10} {ratio:>7.1f}%  {seg_str}")


def check_label_distribution(data):
    valid        = [d for d in data if "nms_labels" in d]
    total_frames = sum(d['length'] for d in valid)

    print("=" * 50)
    print(f"[3] 전체 데이터셋 비수지 활성화 비율 (총 {total_frames:,} frames / {len(valid)} 샘플)")
    print(f"    {'키':<8} {'활성 프레임':>12} {'비율':>8}  {'등장 샘플 수':>12}")
    print(f"    {'-'*55}")

    for key in NMS_KEYS:
        active_frames  = sum(
            segments_to_active_frames(d['nms_labels'].get(key, []), d['length'])
            for d in valid
        )
        active_samples = sum(
            1 for d in valid if d['nms_labels'].get(key)
        )
        ratio = active_frames / total_frames * 100
        print(f"    {key:<8} {active_frames:>12,} {ratio:>7.1f}%  {active_samples:>10} / {len(valid)}")


def check_npy_file(data, sample_idx=0):
    valid    = [d for d in data if "nonmanual_feature_file" in d]
    sample   = valid[sample_idx]
    npy_path = os.path.join(NONMANUAL_SAVE_PATH, sample['nonmanual_feature_file'])

    print("=" * 50)
    print(f"[4] .npy 파일 확인 ({sample['nonmanual_feature_file']})")

    if not os.path.exists(npy_path):
        print(f"    파일 없음: {npy_path}")
        return

    arr = np.load(npy_path)
    print(f"    shape  : {arr.shape}  (frames, dims)")
    print(f"    dtype  : {arr.dtype}")
    print(f"    min    : {arr.min():.4f}")
    print(f"    max    : {arr.max():.4f}")
    print(f"    mean   : {arr.mean():.4f}")
    print(f"    NaN 수 : {np.isnan(arr).sum()}")
    print(f"    Inf 수 : {np.isinf(arr).sum()}")


def main():
    if not os.path.exists(INFO_PATH):
        print(f"dataset_info.json 없음: {INFO_PATH}")
        return

    data = load_dataset_info()

    check_basic_stats(data)
    check_sample_labels(data, sample_idx=0)
    check_label_distribution(data)
    check_npy_file(data, sample_idx=0)

    print("=" * 50)


def vecter_check():

    SAVE_PATH           = r"D:\새 폴더 (2)\deeprun\keypoint_data"
    NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")

    with open(os.path.join(SAVE_PATH, "dataset_info.json"), encoding='utf-8') as f:
        info = json.load(f)

    orig = [d for d in info if "nonmanual_feature_file" in d and not d.get("is_aug")]

    total       = len(orig)
    zero_heavy  = 0  # 30% 이상 영벡터
    all_zero    = 0  # 100% 영벡터
    zero_counts = []

    for d in orig:
        feat = np.load(os.path.join(NONMANUAL_SAVE_PATH, d["nonmanual_feature_file"]))
        zero_frames = (feat.sum(axis=1) == 0).sum()
        ratio       = zero_frames / len(feat)
        zero_counts.append(ratio)

        if ratio == 1.0:
            all_zero += 1
        elif ratio >= 0.3:
            zero_heavy += 1

    print(f"전체 원본 샘플     : {total}")
    print(f"100% 영벡터        : {all_zero}")
    print(f"30% 이상 영벡터    : {zero_heavy}")
    print(f"영벡터 비율 평균   : {np.mean(zero_counts):.3f}")
    print(f"영벡터 비율 최대   : {np.max(zero_counts):.3f}")
    print(f"영벡터 0% (정상)   : {sum(1 for r in zero_counts if r == 0)}")

if __name__ == "__main__":
    vecter_check()