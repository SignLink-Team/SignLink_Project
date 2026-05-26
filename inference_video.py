import os
import json
import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# 기존 모델 아키텍처 로드
from model import SignLanguageModel

# ==========================================
# 1. preprocess.py와 완벽히 동일한 피처 추출 로직
# ==========================================
def get_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    dot_product = np.dot(v1 / norm1, v2 / norm2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def extract_normalized_keypoints(results):
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
    velocity     = np.diff(features, axis=0, prepend=features[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    return np.concatenate([features, velocity, acceleration], axis=1)

# ==========================================
# 2. CTC 예측 디코더
# ==========================================
def ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0):
    decoded_sequence = []
    previous_idx = -1
    for idx in predictions:
        if idx != previous_idx and idx != blank_idx:
            decoded_sequence.append(idx_to_gloss.get(idx, "<UNK>"))
        previous_idx = idx
    return decoded_sequence

# ==========================================
# 3. 메인 영상 추론 프로세스
# ==========================================
def predict_video(video_path, model_path, dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎬 [영상 분석 시작] 입력 영상: {video_path}")
    print(f"💻 연산 디바이스: {device}")

    with open(dict_path, 'r', encoding='utf-8') as f:
        gloss_to_idx = json.load(f)
    
    if "<UNK>" not in gloss_to_idx:
        gloss_to_idx["<UNK>"] = len(gloss_to_idx)
        
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    vocab_size = len(gloss_to_idx)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {video_path}")
        return

    raw_video_features = []
    print("⏳ MediaPipe 랜드마크 추출 중...")
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = extract_normalized_keypoints(results)
            raw_video_features.append(keypoints)
            
    cap.release()
    print(f"📸 총 {len(raw_video_features)} 프레임 추출 완료.")

    if len(raw_video_features) < 5:
        print("❌ 영상의 프레임 수가 너무 적어 추론을 진행할 수 없습니다.")
        return

    features = gaussian_filter1d(np.array(raw_video_features), sigma=1.0, axis=0)
    enhanced_features = apply_motion_derivatives(features)

    model = SignLanguageModel(input_dim=783, hidden_dim=512, num_classes=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    feature_tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(feature_tensor)
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    final_predicted_words = ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0)

    print("\n" + "="*60)
    print("✨ [수어 단어 예측 결과] ✨")
    print("="*60)
    print(f"🗣️  최종 변환된 단어 리스트: {final_predicted_words}")
    print("="*60)
    
    return final_predicted_words

if __name__ == "__main__":
    # 🌟 [추가됨] GUI 창이 뜨지만 배경에 빈 Tkinter 윈도우가 남지 않도록 숨김 처리
    root = tk.Tk()
    root.withdraw()
    
    print("📂 분석할 수어 영상을 선택 창에서 골라주세요...")
    # 파일 탐색기 열기 (mp4, avi, mov, mkv 확장자 필터링)
    selected_video_path = filedialog.askopenfilename(
        title="분석할 수어 영상 선택",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )
    
    # 파일을 선택하지 않고 창을 닫았을 때 예외 처리
    if not selected_video_path:
        print("❌ 영상 선택이 취소되었습니다. 프로그램을 종료합니다.")
    else:
        MODEL_WEIGHT_PATH = "best_sign_model.pth"
        GLOSS_DICT_PATH = "./keypoint_data/gloss_dict.json"

        predict_video(selected_video_path, MODEL_WEIGHT_PATH, GLOSS_DICT_PATH)