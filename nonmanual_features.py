"""
nonmanual_features.py
---------------------
비수지(Non-manual) 피처 추출 모듈

468점 전체 FaceMesh 좌표 사용 (vel/acc 없음)
정규화:
    기준점  - 코끝 (1번)
    스케일  - 왼볼(234) ~ 오른볼(454) 거리

출력 차원: 468 × 3 = 1404차원 (float32)

사용법:
    from nonmanual_features import extract_nonmanual, build_nms_label_sequence, NMS_KEYS
"""

import numpy as np
from typing import Dict, List

# 정규화 기준 인덱스
_REF_IDX         = 1    # 코끝
_SCALE_LEFT_IDX  = 234  # 왼볼
_SCALE_RIGHT_IDX = 454  # 오른볼

NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "EBf", "Hno", "Hs"]
_NMS_KEY_LOWER = {k.lower(): i for i, k in enumerate(NMS_KEYS)}

FEATURE_DIM = 468 * 3  # 1404


def extract_nonmanual(face_landmarks, frame_shape) -> np.ndarray:
    """
    FaceMesh 468점 전체를 코끝 기준 정규화하여 1404차원 벡터로 반환.

    Parameters
    ----------
    face_landmarks : mediapipe face_landmarks
        None 이면 영벡터 반환.
    frame_shape : tuple
        cv2 프레임 .shape (인터페이스 통일용, 미사용)

    Returns
    -------
    np.ndarray, shape=(1404,), dtype=float32
    """
    if face_landmarks is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    lm  = face_landmarks.landmark
    ref = lm[_REF_IDX]

    lc = lm[_SCALE_LEFT_IDX]
    rc = lm[_SCALE_RIGHT_IDX]
    face_scale = ((lc.x - rc.x)**2 + (lc.y - rc.y)**2 + (lc.z - rc.z)**2) ** 0.5
    face_scale = face_scale if face_scale > 0.01 else 1.0

    coords = np.array(
        [
            [(lm[i].x - ref.x) / face_scale,
             (lm[i].y - ref.y) / face_scale,
             (lm[i].z - ref.z) / face_scale]
            for i in range(468)
        ],
        dtype=np.float32,
    )  # (468, 3)

    return coords.flatten()  # (1404,)


def build_nms_label_sequence(
    nms_script: Dict[str, List[dict]],
    total_frames: int,
    fps: float,
    start_frame: int = 0,
) -> np.ndarray:
    """
    nms_script 구간 → 프레임별 멀티라벨 (T, 8) 변환.
    """
    labels = np.zeros((total_frames, len(NMS_KEYS)), dtype=np.float32)
    for raw_key, segments in nms_script.items():
        idx = _NMS_KEY_LOWER.get(raw_key.lower())
        if idx is None or not segments:
            continue
        for seg in segments:
            seg_start = max(0, int(seg["start"] * fps) - start_frame)
            seg_end   = min(total_frames - 1, int(seg["end"] * fps) - start_frame)
            if seg_start <= seg_end:
                labels[seg_start:seg_end + 1, idx] = 1.0
    return labels
