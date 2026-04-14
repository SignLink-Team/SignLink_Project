import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures
from mp4_augment import augment_keypoints, identity, to_grayscale  # [추가] augment 모듈 import

LABEL_DIR = "./data/medical_consult_labels"
VIDEO_DIR = "./data/medical_consult_videos"
SAVE_PATH = "./keypoint_data"

os.makedirs(SAVE_PATH, exist_ok=True)

# [수정] 차원 상수 명시
HAND_FEATURE_DIM = 21 * 3 + 15  # 78
UPPER_BODY_INDICES = list(range(0, 25))
POSE_FEATURE_DIM = len(UPPER_BODY_INDICES) * 3  # 75

def get_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    unit_v1 = v1 / norm1
    unit_v2 = v2 / norm2
    dot_product = np.dot(unit_v1, unit_v2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def extract_normalized_keypoints(results, prev_lh, prev_rh):
    if results.pose_landmarks:
        pose_res = results.pose_landmarks.landmark
        ref_nose = pose_res[0]
        shoulder_l = pose_res[11]
        shoulder_r = pose_res[12]
        shoulder_width = np.linalg.norm([shoulder_l.x - shoulder_r.x, shoulder_l.y - shoulder_r.y, shoulder_l.z - shoulder_r.z])
        pose_scale = shoulder_width if shoulder_width > 0.01 else 1.0

        # [수정] 상체 랜드마크(0~24)만 추출
        pose_data = np.array([[(pose_res[i].x - ref_nose.x)/pose_scale, (pose_res[i].y - ref_nose.y)/pose_scale, (pose_res[i].z - ref_nose.z)/pose_scale]
                             for i in UPPER_BODY_INDICES]).flatten()
    else:
        pose_data = np.zeros(POSE_FEATURE_DIM)

    def process_hand(hand_landmarks, prev_data):
        if not hand_landmarks:
            return prev_data
        
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        hand_length = np.linalg.norm([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z])
        hand_scale = hand_length if hand_length > 0.01 else 1.0

        coords = np.array([[(res.x - wrist.x)/hand_scale, (res.y - wrist.y)/hand_scale, (res.z - wrist.z)/hand_scale] 
                          for res in hand_landmarks.landmark])
        
        angles = []
        joint_indices = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
        for finger in joint_indices:
            for i in range(len(finger) - 2):
                v1 = coords[finger[i+1]] - coords[finger[i]]
                v2 = coords[finger[i+2]] - coords[finger[i+1]]
                angles.append(get_angle(v1, v2))
        
        return np.concatenate([coords.flatten(), angles])

    lh_data = process_hand(results.left_hand_landmarks, prev_lh)
    rh_data = process_hand(results.right_hand_landmarks, prev_rh)
    
    return np.concatenate([pose_data, lh_data, rh_data]), lh_data, rh_data

def extract_features_from_video(video_path, start_frame, end_frame, frame_fn):
    video_features = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    # [수정] 차원 상수 사용
    prev_lh = np.zeros(HAND_FEATURE_DIM)
    prev_rh = np.zeros(HAND_FEATURE_DIM)

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame_fn(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints, prev_lh, prev_rh = extract_normalized_keypoints(results, prev_lh, prev_rh)
            video_features.append(keypoints)
            current_frame += 1

    cap.release()
    return np.array(video_features) if video_features else None

def process_single_file(label_file):
    try:
        filepath = os.path.join(LABEL_DIR, label_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_filename = data.get('vido_file_nm', '') + ".mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        if not os.path.exists(video_path): return None

        gestures = data.get('sign_script', {}).get('sign_gestures_strong', [])
        if not gestures: return None

        start_time = min(item['start'] for item in gestures)
        end_time = max(item['end'] for item in gestures)

        # [수정] 실제 영상 fps 사용
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        start_frame = int(start_time * actual_fps)
        end_frame = int(end_time * actual_fps)

        # [수정] 방향 정보는 메타로 분리, 라벨은 기본 글로스만 사용 + sentence_loc 보존
        gloss_sequence = []
        direction_meta = []
        for item in gestures:
            gloss_sequence.append(item['gloss_id'])
            direction = item.get('direction', {})
            direction_meta.append({
                "gloss_id": item['gloss_id'],
                "start": item['start'],
                "end": item['end'],
                "source": direction.get('source', ''),
                "target": direction.get('target', ''),
                "sentence_loc": item.get('sentence_loc')
            })

        features      = extract_features_from_video(video_path, start_frame, end_frame, identity)
        gray_features = extract_features_from_video(video_path, start_frame, end_frame, to_grayscale)

        if features is None or gray_features is None:
            return None

        np.save(os.path.join(SAVE_PATH, f"{data['id']}_features.npy"),  features)
        np.save(os.path.join(SAVE_PATH, f"{data['id']}_gray.npy"),      gray_features)
        np.save(os.path.join(SAVE_PATH, f"{data['id']}_aug.npy"),       augment_keypoints(features))
        np.save(os.path.join(SAVE_PATH, f"{data['id']}_gray_aug.npy"),  augment_keypoints(gray_features))

        return {
            "id": data['id'],
            "feature_file":  f"{data['id']}_features.npy",
            "gray_file":     f"{data['id']}_gray.npy",
            "aug_file":      f"{data['id']}_aug.npy",
            "gray_aug_file": f"{data['id']}_gray_aug.npy",
            "gloss_sequence": gloss_sequence,
            "direction_meta": direction_meta,
            "length": len(features)
        }

    except Exception as e:
        print(f"[오류] {label_file}: {e}")
        return None

def preprocess_data():
    if not os.path.exists(LABEL_DIR):
        print(f"오류: 라벨 폴더({LABEL_DIR})를 찾을 수 없습니다.")
        return

    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    dataset_summary = []
    all_glosses = set()

    print(f"총 {len(label_files)}개의 라벨 파일을 찾았습니다. 전처리를 시작합니다...")

    # [수정] ProcessPoolExecutor → ThreadPoolExecutor, 워커 수 제한
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_file, label_files))
        
    for res in results:
        if res is not None:
            dataset_summary.append(res)
            for gloss in res['gloss_sequence']:
                all_glosses.add(gloss)

    gloss_list = ["<blank>"] + sorted(list(all_glosses))
    gloss_dict = {gloss: i for i, gloss in enumerate(gloss_list)}
    
    with open(os.path.join(SAVE_PATH, "gloss_dict.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss_dict, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(SAVE_PATH, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=4)
        
    print(f"전처리 완료! 추출된 데이터가 {SAVE_PATH} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_data()