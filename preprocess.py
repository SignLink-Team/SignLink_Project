import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures
from scipy.ndimage import gaussian_filter1d

LABEL_DIR = "./data/labels"
VIDEO_DIR = "./data/videos"
SAVE_PATH = "./keypoint_data"

os.makedirs(SAVE_PATH, exist_ok=True)

def get_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    unit_v1 = v1 / norm1
    unit_v2 = v2 / norm2
    dot_product = np.dot(unit_v1, unit_v2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def extract_normalized_keypoints(results, prev_lh, prev_rh):
    nose_coords = np.array([0, 0, 0])
    if results.pose_landmarks:
        pose_res = results.pose_landmarks.landmark
        ref_nose = pose_res[0]
        nose_coords = np.array([ref_nose.x, ref_nose.y, ref_nose.z])
        shoulder_l, shoulder_r = pose_res[11], pose_res[12]
        shoulder_width = np.linalg.norm([shoulder_l.x - shoulder_r.x, shoulder_l.y - shoulder_r.y, shoulder_l.z - shoulder_r.z])
        pose_scale = shoulder_width if shoulder_width > 0.01 else 1.0
        pose_data = np.array([[(res.x - ref_nose.x)/pose_scale, (res.y - ref_nose.y)/pose_scale, (res.z - ref_nose.z)/pose_scale] 
                             for res in pose_res]).flatten()
    else:
        pose_data = np.zeros(33 * 3)

    def process_hand(hand_landmarks, prev_data, nose_c):
        if not hand_landmarks: return prev_data, 0.0
        wrist = hand_landmarks.landmark[0]
        dist_to_nose = np.linalg.norm([wrist.x - nose_c[0], wrist.y - nose_c[1], wrist.z - nose_c[2]])
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
        return np.concatenate([coords.flatten(), angles]), dist_to_nose

    lh_data, lh_dist = process_hand(results.left_hand_landmarks, prev_lh, nose_coords)
    rh_data, rh_dist = process_hand(results.right_hand_landmarks, prev_rh, nose_coords)
    return np.concatenate([pose_data, lh_data, rh_data, [lh_dist, rh_dist]]), lh_data, rh_data

def apply_gaussian_smoothing(seq, sigma=1.0):
    return gaussian_filter1d(seq, sigma=sigma, axis=0)

def process_single_file(label_file):
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
    word_count = len(gestures) # 단어 수 저장
    fps = data.get('potogrf', {}).get('fps', 30)
    
    margin_frames = int(fps * 0.2)
    start_frame = max(0, int(start_time * fps) - margin_frames)
    end_frame = int(end_time * fps) + margin_frames

    gloss_sequence = [item['gloss_id'] for item in gestures]
    video_features = []
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    prev_lh, prev_rh = np.zeros(78), np.zeros(78)
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            # [속도 개선] 연산 전 이미지를 640 해상도로 강제 축소하여 OpenCV 병목 현상 제거
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            keypoints, prev_lh, prev_rh = extract_normalized_keypoints(results, prev_lh, prev_rh)
            video_features.append(keypoints)
            current_frame += 1
            
    cap.release()

    if len(video_features) < 5: return None

    # 1. 스무딩까지 적용
    base_features = apply_gaussian_smoothing(np.array(video_features))
    
    feature_save_name = f"{data['id']}_features.npy"
    np.save(os.path.join(SAVE_PATH, feature_save_name), base_features)
    
    # word_count를 json에 같이 저장하여 train.py에서 쓸 수 있게 함
    return {
        "id": data['id'],
        "feature_file": feature_save_name,
        "gloss_sequence": gloss_sequence,
        "length": len(base_features),
        "word_count": word_count
    }

def preprocess_data():
    if not os.path.exists(LABEL_DIR): return
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    dataset_summary = []
    all_glosses = set()

    print(f"총 {len(label_files)}개 파일 고속 전처리 시작 (해상도 최적화, 증강 생략)...")

    # CPU 코어를 최대한 활용
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, label_files))
        
    for res in results:
        if res is not None:
            dataset_summary.append(res)
            for gloss in res['gloss_sequence']:
                all_glosses.add(gloss)

    gloss_dict = {gloss: i for i, gloss in enumerate(["<blank>"] + sorted(list(all_glosses)))}
    with open(os.path.join(SAVE_PATH, "gloss_dict.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss_dict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(SAVE_PATH, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=4)
        
    print(f"전처리 완료! (용량 절약 및 속도 개선 완료, 총 데이터 수: {len(dataset_summary)})")

if __name__ == "__main__":
    preprocess_data()