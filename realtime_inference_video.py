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

# MediaPipe 그리기 도구 설정
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ==========================================
# 1. preprocess.py 규격 일치 피처 추출 로직
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

def ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0):
    decoded_sequence = []
    previous_idx = -1
    for idx in predictions:
        if idx != previous_idx and idx != blank_idx:
            decoded_sequence.append(idx_to_gloss.get(idx, "<UNK>"))
        previous_idx = idx
    return decoded_sequence

# ==========================================
# 2. 실시간 재생 및 라이브 추론 루프
# ==========================================
def realtime_video_inference(video_path, model_path, dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 사전 및 모델 초기화
    with open(dict_path, 'r', encoding='utf-8') as f:
        gloss_to_idx = json.load(f)
    if "<UNK>" not in gloss_to_idx:
        gloss_to_idx["<UNK>"] = len(gloss_to_idx)
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    
    model = SignLanguageModel(input_dim=783, hidden_dim=512, num_classes=len(gloss_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {video_path}")
        return

    raw_video_features = []
    current_predicted_words = []
    frame_idx = 0

    print("\n" + "="*60)
    print("📺 [실시간 수어 인식 시스템 가동]")
    print("👉 OpenCV 영상 창을 확인하세요. (종료하려면 영상 창에서 'q' 입력)")
    print("="*60 + "\n")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            # 해상도 정규화 및 처리
            display_frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # 1. 피처 추출 및 누적
            keypoints = extract_normalized_keypoints(results)
            raw_video_features.append(keypoints)
            
            # 2. 랜드마크 시각화 (화면에 선 그리기)
            mp_drawing.draw_landmarks(display_frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # 3. 실시간 CTC 추론 (연산 부하를 줄이기 위해 2프레임마다 수행)
            if len(raw_video_features) >= 15 and frame_idx % 2 == 0:
                feat_arr = np.array(raw_video_features)
                # 현재까지 쌓인 프레임에 시계열 필터 및 속도 결합 적용
                features = gaussian_filter1d(feat_arr, sigma=1.0, axis=0)
                enhanced_features = apply_motion_derivatives(features)
                
                # 텐서 변환 및 예측
                feature_tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(feature_tensor)
                    predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
                
                # 단어 디코딩
                current_predicted_words = ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0)
            
            # 4. 실시간 상태 터미널 및 화면 텍스트 출력
            # 터미널에 한 줄 피드로 밀어내며 지우기 (\r 효과)
            print(f"\r⏳ [Frame {frame_idx:03d}] 실시간 번역 결과 ➡️  {current_predicted_words}", end="", flush=True)
            
            # 화면 상단에 영문 가이드 레이아웃 추가
            cv2.putText(display_frame, f"Frame: {frame_idx} | Evaluating...", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # 화면 표시 및 키 입력 대기 (실시간 프레임 레이트 체감용)
            cv2.imshow("Real-time Sign Language Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 사용자에 의해 중단되었습니다.")
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("\n\n🏁 영상 분석이 전행 완료되었습니다!")
    print(f"🎬 최종 인식된 수어 문장: {current_predicted_words}")
    print("="*60)

if __name__ == "__main__":
    # 파일 탐색기 초기화
    root = tk.Tk()
    root.withdraw()
    
    print("📂 실시간 분석을 수행할 테스트셋 영상을 선택해 주세요...")
    video_path = filedialog.askopenfilename(
        title="실시간 테스트 영상 선택",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    
    if video_path:
        MODEL_WEIGHT_PATH = "best_sign_model.pth"
        GLOSS_DICT_PATH = "./keypoint_data/gloss_dict.json"
        realtime_video_inference(video_path, MODEL_WEIGHT_PATH, GLOSS_DICT_PATH)
    else:
        print("❌ 영상 선택이 취소되었습니다.")