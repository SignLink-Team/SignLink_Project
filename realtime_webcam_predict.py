import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


PROJECT_DIR = r"C:\sign_language_learning"
MODEL_PATH = os.path.join(PROJECT_DIR, "best_sign_model.pth")
GLOSS_DICT_PATH = os.path.join(PROJECT_DIR, "keypoint_data", "gloss_dict.json")
GLOSS_MAPPING_PATH = os.path.join(PROJECT_DIR, "keypoint_data", "gloss_mapping.json")

CAMERA_INDEX = 0
HIDDEN_DIM = 512
MIN_FRAMES = 5

sys.path.insert(0, PROJECT_DIR)
from model import SignLanguageModel  # noqa: E402
from preprocess import (  # noqa: E402
    apply_motion_derivatives as preprocess_motion_derivatives,
    extract_normalized_keypoints as preprocess_keypoints,
)


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def build_inference_dict():
    with open(GLOSS_DICT_PATH, "r", encoding="utf-8") as f:
        old_gloss_to_idx = json.load(f)

    mapping_dict = {}
    if os.path.exists(GLOSS_MAPPING_PATH):
        with open(GLOSS_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)

    blank_tokens = [k for k, v in old_gloss_to_idx.items() if v == 0]
    if not blank_tokens:
        raise ValueError("gloss_dict.json에서 index 0인 blank 토큰을 찾지 못했습니다.")

    blank_token = blank_tokens[0]
    active_glosses = set()
    for original_gloss in old_gloss_to_idx:
        if original_gloss == blank_token:
            continue
        active_glosses.add(mapping_dict.get(original_gloss, original_gloss))

    active_glosses.add("<UNK>")
    active_glosses.discard(blank_token)

    gloss_to_idx = {blank_token: 0}
    for idx, gloss in enumerate(sorted(active_glosses), start=1):
        gloss_to_idx[gloss] = idx

    return gloss_to_idx


def load_model(device):
    gloss_to_idx = build_inference_dict()
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    blank_idx = gloss_to_idx.get("<blank>", 0)
    num_classes = len(gloss_to_idx)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    checkpoint_classes = checkpoint["fc.weight"].shape[0]
    if checkpoint_classes != num_classes:
        raise ValueError(
            f"모델 클래스 수({checkpoint_classes})와 사전 클래스 수({num_classes})가 다릅니다."
        )

    model = SignLanguageModel(input_dim=783, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, idx_to_gloss, blank_idx, num_classes


def ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0):
    decoded = []
    previous_idx = -1

    for idx in predictions:
        idx = int(idx)
        if idx != previous_idx and idx != blank_idx:
            decoded.append(idx_to_gloss.get(idx, "<UNK>"))
        previous_idx = idx

    return decoded


def make_model_features(raw_features):
    features = gaussian_filter1d(np.array(raw_features), sigma=1.0, axis=0)
    return preprocess_motion_derivatives(features)


def predict_sequence(model, raw_features, idx_to_gloss, blank_idx, device):
    if len(raw_features) < MIN_FRAMES:
        return []

    enhanced_features = make_model_features(raw_features)
    feature_tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(feature_tensor)
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    return ctc_greedy_decode(predictions, idx_to_gloss, blank_idx)


def resize_like_preprocess(frame):
    width = 640
    height = int(width * frame.shape[0] / frame.shape[1])
    return cv2.resize(frame, (width, height))


def draw_skeleton(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_overlay(frame, is_recording, frame_count, last_words, device, num_classes):
    mode = "RECORDING" if is_recording else "READY"
    color = (0, 70, 255) if is_recording else (30, 120, 30)
    result_text = " / ".join(last_words[-8:]) if last_words else "No prediction yet"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (20, 20, 20), -1)
    cv2.circle(frame, (28, 30), 10, color, -1)
    cv2.putText(frame, mode, (46, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(
        frame,
        f"Buffered frames: {frame_count} | Device: {device} | Classes: {num_classes}",
        (16, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        frame,
        f"Result: {result_text}",
        (16, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "SPACE: start/stop+predict   R: reset   Q: quit",
        (16, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
    )


def run_webcam_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_gloss, blank_idx, num_classes = load_model(device)
    print(f"Device: {device} | Classes: {num_classes}")
    print("SPACE: start/stop and predict | R: reset | Q: quit")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")

    is_recording = False
    raw_features = []
    last_words = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = resize_like_preprocess(frame)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            keypoints = preprocess_keypoints(results)

            if is_recording:
                raw_features.append(keypoints)

            draw_skeleton(display_frame, results)
            draw_overlay(
                display_frame,
                is_recording,
                len(raw_features),
                last_words,
                device,
                num_classes,
            )
            cv2.imshow("Realtime Webcam Sign Prediction", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                raw_features = []
                last_words = []
                is_recording = False
                print("Reset.")
            if key == 32:
                if not is_recording:
                    raw_features = []
                    last_words = []
                    is_recording = True
                    print("Recording started.")
                else:
                    is_recording = False
                    print(f"Recording stopped. Frames: {len(raw_features)}")
                    last_words = predict_sequence(model, raw_features, idx_to_gloss, blank_idx, device)
                    print("Prediction:", " ".join(last_words) if last_words else "No words detected")

    cap.release()
    cv2.destroyAllWindows()


def main():
    try:
        run_webcam_prediction()
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
