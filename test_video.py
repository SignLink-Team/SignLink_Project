import cv2
import torch
import numpy as np
import json
import os
import mediapipe as mp
from model import SignLanguageModel
from preprocess import extract_normalized_keypoints, HAND_FEATURE_DIM # 기존 상수 가져오기

# --- 설정 ---
MODEL_PATH = "./keypoint_data/sign_model_best.pth"
DICT_PATH = "./keypoint_data/gloss_dict.json"
VIDEO_PATH = "test_video.mp4"

def predict_saved_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 사전 및 모델 로드
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)
    idx_to_gloss = {v: k for k, v in gloss_dict.items()}
    
    model = SignLanguageModel(
        input_dim=231, hidden_dim=256, num_layers=3, 
        num_classes=len(gloss_dict)
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 영상 전처리 (기존 extract_features_from_video 로직 활용)
    cap = cv2.VideoCapture(video_path)
    video_features = []
    prev_lh = np.zeros(HAND_FEATURE_DIM)
    prev_rh = np.zeros(HAND_FEATURE_DIM)
    
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 학습 때와 동일하게 RGB 변환 후 전처리
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints, prev_lh, prev_rh = extract_normalized_keypoints(results, prev_lh, prev_rh)
            video_features.append(keypoints)
    cap.release()

    # 3. 예측 실행
    if len(video_features) > 10:
        input_tensor = torch.tensor(np.array(video_features), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            # CTC Decode (중복 제거 로직 포함)
            pred_indices = torch.argmax(logits, dim=2)[0].cpu().numpy()
            
            decoded_result = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != prev_idx and idx != 0: # 0은 <blank>
                    decoded_result.append(idx_to_gloss[idx])
                prev_idx = idx
            
            print(f"\n🎬 영상 분석 결과: {' '.join(decoded_result)}")
    else:
        print("❌ 영상이 너무 짧아 분석할 수 없습니다.")

if __name__ == "__main__":
    predict_saved_video(VIDEO_PATH)