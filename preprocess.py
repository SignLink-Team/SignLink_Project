import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures

# --- 상대 경로 설정 ---
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

# [수정 1] 이전 프레임의 손 데이터를 매개변수(prev_lh, prev_rh)로 받습니다.
def extract_normalized_keypoints(results, prev_lh, prev_rh):
    if results.pose_landmarks:
        pose_res = results.pose_landmarks.landmark
        ref_nose = pose_res[0]
        shoulder_l = pose_res[11]
        shoulder_r = pose_res[12]
        shoulder_width = np.linalg.norm([shoulder_l.x - shoulder_r.x, shoulder_l.y - shoulder_r.y, shoulder_l.z - shoulder_r.z])
        pose_scale = shoulder_width if shoulder_width > 0.01 else 1.0
        
        pose_data = np.array([[(res.x - ref_nose.x)/pose_scale, (res.y - ref_nose.y)/pose_scale, (res.z - ref_nose.z)/pose_scale] 
                             for res in pose_res]).flatten()
    else:
        pose_data = np.zeros(33 * 3)

    def process_hand(hand_landmarks, prev_data):
        # MediaPipe가 손을 놓치면 0으로 만들지 않고 직전 프레임(prev_data)을 그대로 유지!
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
    
    # 특징 벡터 배열과 함께, 다음 프레임을 위해 현재 손 데이터를 반환합니다.
    return np.concatenate([pose_data, lh_data, rh_data]), lh_data, rh_data

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
    
    fps = data.get('potogrf', {}).get('fps', 30)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # [수정 2] 방향성 동사 처리: source와 target이 있으면 라벨에 병합
    gloss_sequence = []
    for item in gestures:
        base_gloss = item['gloss_id']
        direction = item.get('direction', {})
        
        src = direction.get('source', '')
        tgt = direction.get('target', '')
        
        # 방향 정보가 비어있지 않으면 "단어_시작_끝" 형태로 구체화 (예: 돕다1_1_2)
        if src and tgt:
            specific_gloss = f"{base_gloss}_{src}_{tgt}"
            gloss_sequence.append(specific_gloss)
        else:
            gloss_sequence.append(base_gloss)

    video_features = []
    
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    # 결측치 방어를 위한 초기 빈 배열 (손 좌표 21개*3차원 + 각도 15개 = 78차원)
    prev_lh = np.zeros(78)
    prev_rh = np.zeros(78)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # 이전 프레임의 손 데이터를 넘겨주고, 갱신된 손 데이터를 다시 받아옴
            keypoints, prev_lh, prev_rh = extract_normalized_keypoints(results, prev_lh, prev_rh)
            video_features.append(keypoints)
            current_frame += 1
            
    cap.release()

    if len(video_features) == 0: return None

    feature_save_name = f"{data['id']}_features.npy"
    np.save(os.path.join(SAVE_PATH, feature_save_name), np.array(video_features))
    
    return {
        "id": data['id'],
        "feature_file": feature_save_name,
        "gloss_sequence": gloss_sequence,
        "length": len(video_features)
    }

def preprocess_data():
    if not os.path.exists(LABEL_DIR):
        print(f"오류: 라벨 폴더({LABEL_DIR})를 찾을 수 없습니다.")
        return

    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    dataset_summary = []
    all_glosses = set()

    print(f"총 {len(label_files)}개의 라벨 파일을 찾았습니다. 전처리를 시작합니다...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
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