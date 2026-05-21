"""
nonmanual_features.py
---------------------
비수지(Non-manual) 피처 추출 모듈

손 랜드마크와 동일한 방식으로 페이스 메쉬 주요 60점을 정규화하여 반환.
전체 468점 대신 비수지 관련 영역만 추려 6000개 수준 데이터셋에 적합한 크기로 경량화.

랜드마크 구성 (총 60점 → 180차원):
    입        20점  - 윤곽 + 내부 (Mo1, Mmo, Mctr)
    눈썹      16점  - 양쪽       (Ebu, Ebf)
    눈        12점  - 양쪽 윤곽  (Ci 보조, 눈 크기)
    볼         8점  - 양볼       (Ci)
    코/기준    4점  - 정규화 기준점

정규화:
    기준점  - 코끝(1번)              : 손의 wrist 와 동일
    스케일  - 왼볼(234) ~ 오른볼(454) 거리 : 손의 hand_length 와 동일

출력:
    extract_nonmanual()           → np.ndarray (180,)  피처
    build_nms_label_sequence()    → np.ndarray (T, 8)  프레임별 멀티라벨
    NMS_KEYS                      → list[str]          라벨 열 순서

사용법:
    from nonmanual_features import extract_nonmanual, build_nms_label_sequence, NMS_KEYS

    feat       = extract_nonmanual(results.face_landmarks, frame.shape)
    nms_labels = build_nms_label_sequence(nms_script, total_frames, fps, start_frame)
"""

import numpy as np
from typing import Dict, List

# ---------------------------------------------------------------------------
# 주요 랜드마크 인덱스
# ---------------------------------------------------------------------------

# 입 윤곽 + 내부 (20점) — Mo1, Mmo, Mctr
MOUTH_IDX = [
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409,
    291, 375, 321, 405, 314,  17,  84, 181,  91, 146,
]

# 눈썹 양쪽 (16점) — Ebu, Ebf
BROW_IDX = [
     55,  65,  52,  53,  46,   # 왼쪽 외곽
     63,  66,  70,             # 왼쪽 중앙
    285, 295, 282, 283, 276,   # 오른쪽 외곽
    293, 296, 336,             # 오른쪽 중앙
]

# 눈 윤곽 양쪽 (12점) — Ci 보조, 눈 크기
EYE_IDX = [
     33, 160, 158, 133, 153, 144,   # 왼쪽
    362, 385, 387, 263, 373, 380,   # 오른쪽
]

# 볼 양쪽 (8점) — Ci
CHEEK_IDX = [
    234, 227, 116, 123,   # 왼볼
    454, 447, 345, 352,   # 오른볼
]

# 코끝 + 기준 보조 (4점)
NOSE_IDX = [1, 2, 98, 327]

# 전체 키 인덱스 (60점 → 180차원)
FACE_KEY_IDX = MOUTH_IDX + BROW_IDX + EYE_IDX + CHEEK_IDX + NOSE_IDX

# 정규화 기준
_REF_IDX         = 1    # 코끝 (기준점)
_SCALE_LEFT_IDX  = 234  # 왼볼
_SCALE_RIGHT_IDX = 454  # 오른볼

# ---------------------------------------------------------------------------
# 비수지 라벨 키 (JSON nms_script 키와 대응)
# ---------------------------------------------------------------------------
NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "Ebf", "Hno", "Hs"]
_NMS_KEY_LOWER = {k.lower(): i for i, k in enumerate(NMS_KEYS)}


# ---------------------------------------------------------------------------
# 피처 추출
# ---------------------------------------------------------------------------

def extract_nonmanual(face_landmarks, frame_shape) -> np.ndarray:
    """
    페이스 메쉬 주요 60점을 정규화하여 180차원 벡터로 반환.

    손 랜드마크와 동일한 정규화 방식:
        coords = (landmark_xyz - ref_xyz) / face_scale

    Parameters
    ----------
    face_landmarks : mediapipe face_landmarks (results.face_landmarks)
        None 이면 영벡터 반환.
    frame_shape : tuple
        cv2 프레임의 .shape — 인터페이스 통일을 위해 유지 (미사용).

    Returns
    -------
    np.ndarray, shape=(180,), dtype=float32
    """
    if face_landmarks is None:
        return np.zeros(len(FACE_KEY_IDX) * 3, dtype=np.float32)

    lm = face_landmarks.landmark

    # 기준점: 코끝
    ref = lm[_REF_IDX]

    # 스케일: 양볼 거리로 얼굴 크기 정규화
    lc = lm[_SCALE_LEFT_IDX]
    rc = lm[_SCALE_RIGHT_IDX]
    face_scale = ((lc.x - rc.x)**2 + (lc.y - rc.y)**2 + (lc.z - rc.z)**2) ** 0.5
    face_scale = face_scale if face_scale > 0.01 else 1.0

    coords = np.array(
        [
            [(lm[i].x - ref.x) / face_scale,
             (lm[i].y - ref.y) / face_scale,
             (lm[i].z - ref.z) / face_scale]
            for i in FACE_KEY_IDX
        ],
        dtype=np.float32,
    )  # (60, 3)

    return coords.flatten()  # (180,)


# ---------------------------------------------------------------------------
# 비수지 라벨 시퀀스 생성
# ---------------------------------------------------------------------------

def build_nms_label_sequence(
    nms_script: Dict[str, List[dict]],
    total_frames: int,
    fps: float,
    start_frame: int = 0,
) -> np.ndarray:
    """
    nms_script(JSON)를 프레임별 멀티라벨 배열로 변환.

    Parameters
    ----------
    nms_script : dict
        JSON 의 "nms_script" 값.
        예: {"Hno": [{"start": 3.4, "end": 3.9}, ...], ...}
    total_frames : int
        추출된 전체 프레임 수 (video_features 길이와 동일).
    fps : float
        영상 FPS.
    start_frame : int
        전처리 시 잘라낸 시작 프레임 오프셋 (margin 적용 후 값).

    Returns
    -------
    np.ndarray, shape=(total_frames, 8), dtype=float32
        열 순서 : NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "Ebf", "Hno", "Hs"]
        활성화 프레임 1.0 / 비활성 0.0
    """
    labels = np.zeros((total_frames, len(NMS_KEYS)), dtype=np.float32)

    for raw_key, segments in nms_script.items():
        idx = _NMS_KEY_LOWER.get(raw_key.lower())
        if idx is None or not segments:
            continue
        for seg in segments:
            seg_start = int(seg["start"] * fps) - start_frame
            seg_end   = int(seg["end"]   * fps) - start_frame
            seg_start = max(0, seg_start)
            seg_end   = min(total_frames - 1, seg_end)
            if seg_start <= seg_end:
                labels[seg_start:seg_end + 1, idx] = 1.0

    return labels