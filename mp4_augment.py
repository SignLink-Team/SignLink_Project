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
def time_mask(seq, mask_ratio=0.1):
    seq = seq.copy()
    T = seq.shape[0]
    if T < 5:
        return seq
    
    mask_len = int(T * mask_ratio)
    # 마스크가 시작될 수 있는 인덱스 (최소 1프레임 이후부터 시작해야 이전 프레임 복사 가능)
    start = random.randint(1, max(1, T - mask_len))
    
    # 시작 직전 프레임(start-1)의 값을 가져와서 마스크 구간에 덮어쓰기
    prev_frame = seq[start - 1]
    seq[start:start+mask_len] = prev_frame

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
        seq = time_mask(seq)
    if random.random() < AUG_CONFIG["time_warp_prob"]:
        seq = time_warp(seq)
    return seq
