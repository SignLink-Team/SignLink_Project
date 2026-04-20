import cv2
import numpy as np
import random

AUG_CONFIG = {
    "noise_prob": 0.5,
    "scale_prob": 0.5,
    "shift_prob": 0.5,
    "time_mask_prob": 0.3,
    "time_warp_prob": 0.3,
}

def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    BGR 이미지를 grayscale로 변환 후
    MediaPipe 호환을 위해 다시 3채널(BGR)로 변환
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def identity(frame: np.ndarray) -> np.ndarray:
    return frame

AUGMENTATIONS = {
    "original": identity,
    "grayscale": to_grayscale,
}

def add_noise(seq, noise_level=0.01):
    noise = np.random.randn(*seq.shape) * noise_level
    return seq + noise
#스케일 변형
def random_scale(seq, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return seq * scale
#위치 이동
def random_shift(seq, shift_range=0.05):
    shifted = seq.copy()
    shift = np.random.uniform(-shift_range, shift_range, size=seq.shape[1])

     # 포즈 (0:99)
    shifted[:, 0:99] += shift[0:99]
    
    # 왼손 좌표 (99:162)
    shifted[:, 99:162] += shift[99:162]
    
    # 오른손 좌표 (177:240)
    shifted[:, 177:240] += shift[177:240]

    return shifted

#시간축 증강 
#1. 일부 프레임 제거
def time_mask(seq, label_segments=None, mask_ratio=0.1, max_try=20):
    seq = seq.copy()
    #전체 시퀀스 길이가 5보다 짧으면 마스킹 안함
    T = seq.shape[0]
    if T < 5:
        return seq

    # segments 없으면 기존 방식
    if not label_segments:
        mask_len = min(int(T * mask_ratio), 3)
        if mask_len < 1:
            return seq
        start = random.randint(1, max(1, T - mask_len))
        seq[start:start + mask_len] = seq[start - 1]
        return seq

    # 5프레임 이상인 글로스만 후보로
    valid_segments = [
        (seg['start_frame'], seg['end_frame'])
        for seg in label_segments
        if seg['end_frame'] - seg['start_frame'] >= 5
    ]

    if not valid_segments:
        return seq

    # 마스킹할 단어 수 랜덤 (1 ~ 전체 후보 수)
    num_to_mask = random.randint(1, len(valid_segments))
    selected_segments = random.sample(valid_segments, num_to_mask)

    print(f"마스킹 단어 수: {num_to_mask} / {len(valid_segments)}")

    for l_start, l_end in selected_segments:
        seg_len = l_end - l_start

        safe_start = l_start + 2  # ← 처음 2프레임 보호
        safe_end   = l_end   - 2  # ← 끝   2프레임 보호

        if safe_end - safe_start < 3:
            continue

        # 글로스 길이의 30%, 최소 2, 최대 5프레임
        mask_len = int(seg_len * 0.3)
        mask_len = max(2, min(mask_len, 5)) 
         # 마스킹 가능 구간 안에 들어오도록 제한
        mask_len = min(mask_len, safe_end - safe_start - 1)
        if mask_len < 1:
            continue                            
        # 글로스 범위 안에서 랜덤 위치
        mask_start = random.randint(safe_start, max(safe_start, safe_end - mask_len))

        # 시작 직전 프레임이 있어야 복사 가능
        if mask_start == 0:
            mask_start = 1

        seq[mask_start:mask_start + mask_len] = seq[mask_start - 1]

    return seq

#2. 속도 변화
def time_warp(seq, warp_range=(0.8, 1.2)):
    T = seq.shape[0]
    factor = np.random.uniform(*warp_range)
    new_T = max(5, int(T * factor))
    indices = np.linspace(0, T - 1, new_T).astype(np.int32)
    return seq[indices]


def augment_keypoints(seq):
    seq = seq.copy()
    if random.random() < AUG_CONFIG["noise_prob"]:
        seq = add_noise(seq)
    if random.random() < AUG_CONFIG["scale_prob"]:
        seq = random_scale(seq)
    if random.random() < AUG_CONFIG["shift_prob"]:
        seq = random_shift(seq)
    if random.random() < AUG_CONFIG["time_mask_prob"]:
        seq = time_mask(seq, label_segments)
    if random.random() < AUG_CONFIG["time_warp_prob"]:
        seq = time_warp(seq)
    return seq

"""
import os
import json
SAVE_PATH = r"D:\새 폴더 (2)\deeprun\keypoint_data2" 

def make_aug_files():
    info_path = os.path.join(SAVE_PATH, "dataset_info.json") #preprocess에서 json에 words 추가후 실행

    if not os.path.exists(info_path):
        print("dataset_info.json 없음 → preprocess.py 먼저 실행하세요")
        return

    with open(info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    total = len(dataset_info)
    success = 0
    skip    = 0

    for i, item in enumerate(dataset_info):
        feature_path = os.path.join(SAVE_PATH, item['feature_file'])
        aug_path     = os.path.join(SAVE_PATH, item['feature_file'].replace('_features.npy', '_aug.npy'))

        # 이미 aug 파일 있으면 스킵
        if os.path.exists(aug_path):
            skip += 1
            continue

        # features 파일 없으면 스킵
        if not os.path.exists(feature_path):
            print(f"[{i+1}/{total}] ⚠️  features 없음: {item['id']}")
            skip += 1
            continue

        try:
            features       = np.load(feature_path)
            label_segments = item.get('words', [])

            aug_features   = augment_keypoints(features, label_segments)

            np.save(aug_path, aug_features)
            success += 1
            print(f"[{i+1}/{total}] ✅ {item['id']} | {features.shape} → {aug_features.shape}")

        except Exception as e:
            print(f"[{i+1}/{total}] ❌ {item['id']} 실패: {e}")
            skip += 1

    print(f"\n완료: 성공 {success}개 / 스킵 {skip}개 / 전체 {total}개")


if __name__ == "__main__":
    make_aug_files()
    """
