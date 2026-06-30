import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures
from collections import Counter
from scipy.ndimage import gaussian_filter1d

LABEL_DIR = "./data/labels"
VIDEO_DIR = "./data/videos"
SAVE_PATH = "./keypoint_data"

os.makedirs(SAVE_PATH, exist_ok=True)


def get_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    dot_product = np.dot(v1 / norm1, v2 / norm2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def extract_normalized_keypoints(results):
    """
    상대적 벡터(rel_pos)를 추가하여 전역적 위치 정보 보존.
    총 261차원: pose(99) + lh(78) + rh(78) + lh_rel(3) + rh_rel(3)
    """
    nose_coords = np.array([0.0, 0.0, 0.0])
    if results.pose_landmarks:
        pose_res     = results.pose_landmarks.landmark
        ref_nose     = pose_res[0]
        nose_coords  = np.array([ref_nose.x, ref_nose.y, ref_nose.z])
        shoulder_l, shoulder_r = pose_res[11], pose_res[12]
        shoulder_width = np.linalg.norm([
            shoulder_l.x - shoulder_r.x,
            shoulder_l.y - shoulder_r.y,
            shoulder_l.z - shoulder_r.z,
        ])
        pose_scale = shoulder_width if shoulder_width > 0.01 else 1.0
        pose_data  = np.array([
            [(res.x - ref_nose.x) / pose_scale,
             (res.y - ref_nose.y) / pose_scale,
             (res.z - ref_nose.z) / pose_scale]
            for res in pose_res
        ]).flatten()
    else:
        pose_data = np.zeros(33 * 3)

    def process_hand(hand_landmarks, nose_c):
        if not hand_landmarks:
            return np.zeros(78), np.zeros(3)
        wrist      = hand_landmarks.landmark[0]
        rel_pos    = np.array([wrist.x - nose_c[0], wrist.y - nose_c[1], wrist.z - nose_c[2]])
        middle_mcp = hand_landmarks.landmark[9]
        hand_length = np.linalg.norm([
            middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z
        ])
        hand_scale = hand_length if hand_length > 0.01 else 1.0
        coords = np.array([
            [(res.x - wrist.x) / hand_scale,
             (res.y - wrist.y) / hand_scale,
             (res.z - wrist.z) / hand_scale]
            for res in hand_landmarks.landmark
        ])
        angles = []
        joint_indices = [
            [0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
        ]
        for finger in joint_indices:
            for i in range(len(finger) - 2):
                v1 = coords[finger[i+1]] - coords[finger[i]]
                v2 = coords[finger[i+2]] - coords[finger[i+1]]
                angles.append(get_angle(v1, v2))
        return np.concatenate([coords.flatten(), angles]), rel_pos

    lh_data, lh_rel = process_hand(results.left_hand_landmarks,  nose_coords)
    rh_data, rh_rel = process_hand(results.right_hand_landmarks, nose_coords)
    return np.concatenate([pose_data, lh_data, rh_data, lh_rel, rh_rel])


def apply_motion_derivatives(features: np.ndarray) -> np.ndarray:
    """속도 및 가속도 결합 (차원 3배 확장)"""
    velocity     = np.diff(features, axis=0, prepend=features[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    return np.concatenate([features, velocity, acceleration], axis=1)


# [B4 수정] MediaPipe Holistic 객체를 함수 내부에서 생성.
#           ProcessPoolExecutor는 각 워커 프로세스가 독립적으로 실행되므로
#           전역 holistic 객체를 공유하면 내부 TFLite 런타임이 충돌함.
#           함수 내부에서 생성하면 각 프로세스가 자신의 인스턴스를 가짐.
#           (기존 코드도 with 블록 내부에서 생성하지만, 모듈 import 시
#            전역으로 mp.solutions.holistic을 불러오는 것은 무방함.)
def process_single_file(label_file: str):
    filepath = os.path.join(LABEL_DIR, label_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_filename = data.get('vido_file_nm', '') + ".mp4"
    video_path     = os.path.join(VIDEO_DIR, video_filename)
    if not os.path.exists(video_path):
        return None

    gestures = data.get('sign_script', {}).get('sign_gestures_strong', [])
    if not gestures:
        return None

    start_time = min(item['start'] for item in gestures)
    end_time   = max(item['end']   for item in gestures)
    fps        = data.get('potogrf', {}).get('fps', 30)

    margin_frames = int(fps * 0.5)
    start_frame   = max(0, int(start_time * fps) - margin_frames)
    end_frame     = int(end_time * fps) + margin_frames

    gloss_sequence = [item['gloss_id'] for item in gestures]

    # [B10 추가] label_segments 저장: 각 글로스의 프레임 범위를 기록
    #            train.py의 time_mask 글로스 단위 마스킹에 사용됨.
    label_segments = [
        {
            "gloss_id":    item['gloss_id'],
            "start_frame": max(0, int(item['start'] * fps) - start_frame),
            "end_frame":   int(item['end'] * fps) - start_frame,
        }
        for item in gestures
    ]

    video_features = []

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    # [B4 수정] Holistic 객체는 함수 내부(=각 워커 프로세스)에서 생성 — 안전
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame   = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video_features.append(extract_normalized_keypoints(results))
            current_frame += 1
    cap.release()

    if len(video_features) < 5:
        return None

    features          = gaussian_filter1d(np.array(video_features), sigma=1.0, axis=0)
    enhanced_features = apply_motion_derivatives(features)

    feature_save_name = f"{data['id']}_features.npy"
    np.save(os.path.join(SAVE_PATH, feature_save_name), enhanced_features)

    return {
        "id":              data['id'],
        "feature_file":    feature_save_name,
        "gloss_sequence":  gloss_sequence,
        # [B10 추가] label_segments 포함 — train.py time_mask에서 사용
        "label_segments":  label_segments,
        "length":          len(enhanced_features),
        "word_count":      len(gestures),
        "input_dim":       enhanced_features.shape[1],
    }


def preprocess_data():
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    print(f"총 {len(label_files)}개 파일 전처리 시작...")

    # [B4 주의] ProcessPoolExecutor는 각 워커가 독립 프로세스이므로
    #           mediapipe 객체가 함수 내부에서 생성되는 한 안전.
    #           단, Windows에서는 if __name__ == '__main__' 가드가 필수.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [r for r in executor.map(process_single_file, label_files) if r]

    all_gloss_list = [g for r in results for g in r['gloss_sequence']]
    freq           = Counter(all_gloss_list)

    freq_info = {
        "total_tokens": len(all_gloss_list),
        "total_types":  len(freq),
        "freq_dist": {
            "0~1":   sum(1 for c in freq.values() if c <= 1),
            "2~5":   sum(1 for c in freq.values() if 2 <= c <= 5),
            "6~10":  sum(1 for c in freq.values() if 6 <= c <= 10),
            "11~50": sum(1 for c in freq.values() if 11 <= c <= 50),
            "51+":   sum(1 for c in freq.values() if c > 50),
        },
        "gloss_freq": freq,
    }

    with open(os.path.join(SAVE_PATH, "gloss_freq_info.json"), 'w', encoding='utf-8') as f:
        json.dump(freq_info, f, ensure_ascii=False, indent=4)

    all_glosses = sorted(set(all_gloss_list))
    gloss_dict  = {g: i for i, g in enumerate(["<blank>"] + all_glosses)}

    with open(os.path.join(SAVE_PATH, "gloss_dict.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss_dict, f, ensure_ascii=False, indent=4)

    with open(os.path.join(SAVE_PATH, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n전처리 완료.")
    print(f"  입력 차원:      {results[0]['input_dim']}")
    print(f"  전체 클래스 수:  {len(gloss_dict)}  (<blank> 포함, UNK 없음)")
    print(f"  빈도 분포:")
    for k, v in freq_info['freq_dist'].items():
        print(f"    {k}회: {v}개 클래스")


if __name__ == "__main__":
    preprocess_data()