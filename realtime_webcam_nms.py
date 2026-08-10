"""
webcam_combined.py
-------------------
수어 단어 인식(Holistic + CTC beam search)과
비수지 신호 분류(FaceMesh + Transformer)를
하나의 웹캠 루프에서 "동시에" 실행하는 통합 스크립트.
(수지 신호 타임스탬프 출력 기능 추가)
"""

import argparse
import collections
import json
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 공통 설정
# ---------------------------------------------------------------------------
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
torch.serialization.add_safe_globals([np.dtype])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 수어 단어 인식 모델 경로/설정 ---
PROJECT_DIR = r"C:\sign_language_learning"
SIGN_MODEL_PATH = os.path.join(PROJECT_DIR, "best_sign_model.pth")
GLOSS_DICT_PATH = os.path.join(PROJECT_DIR, "keypoint_data", "gloss_dict.json")
GLOSS_MAPPING_PATH = os.path.join(PROJECT_DIR, "keypoint_data", "gloss_mapping.json")

HIDDEN_DIM = 512
MIN_FRAMES = 5
BEAM_WIDTH = 10
TOKEN_CONFIDENCE_THRESHOLD = 0.05
SEQUENCE_CONFIDENCE_THRESHOLD = 0.00
REMOVE_UNK = True

sys.path.insert(0, PROJECT_DIR)
from model import SignLanguageModel  # noqa: E402
from preprocess import (  # noqa: E402
    apply_motion_derivatives as preprocess_motion_derivatives,
    extract_normalized_keypoints as preprocess_keypoints,
)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# --- 비수지(NMS) 모델 경로/설정 ---
NMS_CKPT_PATH = os.path.join(PROJECT_DIR, "best_nms_finetune_5.pt")
NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "EBf", "Hno", "Hs"]
NMS_KEY_THRESHOLD = {
    "Mo1": 0.42, "Mmo": 0.42, "Mctr": 0.42, "Ci": 0.40,
    "Ebu": 0.45, "EBf": 0.52, "Hno": 0.50, "Hs": 0.40,
}
NMS_KEY_COLORS = {
    "Mo1": (50, 100, 255), "Mmo": (50, 180, 255), "Mctr": (100, 220, 255),
    "Ci": (255, 200, 50), "Ebu": (150, 255, 50), "EBf": (255, 180, 100),
    "Hno": (255, 80, 200), "Hs": (150, 80, 255),
}

from nonmanual_features import extract_nonmanual  # noqa: E402


# ===========================================================================
# 1) 비수지(NMS) 모델 정의 / 로드 / 추론
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class NMSClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_labels, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.dropout(self.input_proj(x))
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)


def load_nms_model():
    ckpt = torch.load(NMS_CKPT_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt["cfg"]
    model = NMSClassifier(
        input_dim=cfg["input_dim"], d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], num_labels=len(NMS_KEYS), dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[NMS] 모델 로드: epoch {ckpt['epoch']}, macro_F1 {ckpt['macro_f1']:.4f}")
    return model


@torch.no_grad()
def infer_nms(model, feat_window):
    seq = gaussian_filter1d(np.array(feat_window, dtype=np.float32), sigma=1.0, axis=0)
    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs[-1]


def draw_nms_panel(frame, probs, smoothed_probs, y_top=130):
    h, w = frame.shape[:2]
    panel_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Non-Manual", (8, y_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (0, y_top + 26), (panel_w, y_top + 26), (80, 80, 80), 1)

    for i, key in enumerate(NMS_KEYS):
        thr = NMS_KEY_THRESHOLD[key]
        prob = smoothed_probs[i]
        active = prob >= thr
        color = NMS_KEY_COLORS[key] if active else (80, 80, 80)
        y = y_top + 42 + i * 28

        bar_max = panel_w - 16
        bar_w = int(prob * bar_max)
        cv2.rectangle(frame, (8, y), (8 + bar_max, y + 16), (50, 50, 50), -1)
        cv2.rectangle(frame, (8, y), (8 + bar_w, y + 16), color, -1)

        label = f"{key}  {prob:.2f}"
        cv2.putText(frame, label, (12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)

        if active:
            cv2.circle(frame, (panel_w - 10, y + 8), 5, color, -1)

    return frame


# ===========================================================================
# 2) 수어 단어 인식 모델 정의 / 로드 / 디코딩
# ===========================================================================

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


def load_sign_model(device):
    gloss_to_idx = build_inference_dict()
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    blank_idx = gloss_to_idx.get("<blank>", 0)
    num_classes = len(gloss_to_idx)

    checkpoint = torch.load(SIGN_MODEL_PATH, map_location=device)
    checkpoint_classes = checkpoint["fc.weight"].shape[0]
    if checkpoint_classes != num_classes:
        raise ValueError(
            f"모델 클래스 수({checkpoint_classes})와 사전 클래스 수({num_classes})가 다릅니다."
        )

    model = SignLanguageModel(input_dim=783, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[SIGN] 모델 로드 완료: classes={num_classes}")
    return model, idx_to_gloss, blank_idx, num_classes


def ctc_beam_search_decode(log_probs, idx_to_gloss, blank_idx=0, beam_width=BEAM_WIDTH):
    T, C = log_probs.shape
    # beams: {seq: (log_prob, probs_list, frames_list)}
    beams = {(): (0.0, [], [])}

    for t in range(T):
        new_beams = {}
        frame_log_probs = log_probs[t]

        for seq, (seq_log_prob, seq_probs, seq_frames) in beams.items():
            for c in range(C):
                token_log_prob = float(frame_log_probs[c])
                new_log_prob = seq_log_prob + token_log_prob

                if c == blank_idx:
                    new_seq = seq
                    new_probs = seq_probs
                    new_frames = seq_frames
                elif seq and seq[-1] == c:
                    new_seq = seq
                    new_probs = list(seq_probs)
                    new_frames = list(seq_frames)
                    current_prob = float(np.exp(token_log_prob))
                    # 동일한 단어가 연속될 때, 가장 확률이 높은 프레임을 타임스탬프로 업데이트
                    if current_prob > new_probs[-1]:
                        new_probs[-1] = current_prob
                        new_frames[-1] = t
                else:
                    new_seq = seq + (c,)
                    new_probs = seq_probs + [float(np.exp(token_log_prob))]
                    new_frames = seq_frames + [t]

                if new_seq in new_beams:
                    prev_log_prob, prev_probs, prev_frames = new_beams[new_seq]
                    merged = np.logaddexp(prev_log_prob, new_log_prob)
                    keep_probs = new_probs if len(new_probs) >= len(prev_probs) else prev_probs
                    keep_frames = new_frames if len(new_frames) >= len(prev_frames) else prev_frames
                    new_beams[new_seq] = (merged, keep_probs, keep_frames)
                else:
                    new_beams[new_seq] = (new_log_prob, new_probs, new_frames)

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1][0], reverse=True)[:beam_width]
        )

    if not beams:
        return []

    best_seq, (_, best_probs, best_frames) = max(beams.items(), key=lambda x: x[1][0])

    result = []
    for i, token_idx in enumerate(best_seq):
        gloss = idx_to_gloss.get(token_idx, "<UNK>")
        prob = best_probs[i] if i < len(best_probs) else 0.0
        f_idx = best_frames[i] if i < len(best_frames) else 0
        result.append((gloss, prob, f_idx))  # 프레임 인덱스 추가 반환

    return result


def filter_predictions(raw_predictions):
    if not raw_predictions:
        return []

    mean_conf = np.mean([p for _, p, _ in raw_predictions])
    if mean_conf < SEQUENCE_CONFIDENCE_THRESHOLD:
        print(f"[FILTER] 시퀀스 평균 신뢰도 낮음 ({mean_conf:.3f} < {SEQUENCE_CONFIDENCE_THRESHOLD}) → 전체 버림")
        return []

    filtered = []
    for gloss, prob, f_idx in raw_predictions:
        if REMOVE_UNK and gloss == "<UNK>":
            print("[FILTER] <UNK> 제거")
            continue
        if prob < TOKEN_CONFIDENCE_THRESHOLD:
            print(f"[FILTER] '{gloss}' conf={prob:.3f} < {TOKEN_CONFIDENCE_THRESHOLD} → 제거")
            continue
        filtered.append(gloss)

    return filtered


def make_model_features(raw_features):
    features = gaussian_filter1d(np.array(raw_features), sigma=1.0, axis=0)
    return preprocess_motion_derivatives(features)


def predict_sequence(model, raw_features, idx_to_gloss, blank_idx, device):
    if len(raw_features) < MIN_FRAMES:
        return [], [], 0

    enhanced = make_model_features(raw_features)
    tensor = torch.tensor(enhanced, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    raw_predictions = ctc_beam_search_decode(log_probs, idx_to_gloss, blank_idx, BEAM_WIDTH)
    filtered = filter_predictions(raw_predictions)

    # 필터링된 단어 리스트, 원본 예측 튜플(단어, 확률, 프레임), 총 프레임(T) 반환
    return filtered, raw_predictions, log_probs.shape[0]


def draw_skeleton(frame, holistic_results):
    mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_sign_header(frame, is_recording, frame_count, last_words, last_raw_preds, device, num_classes):
    mode = "RECORDING" if is_recording else "READY"
    color = (0, 70, 255) if is_recording else (30, 120, 30)
    result_text = " / ".join(last_words[-8:]) if last_words else "아직 예측 결과 없음"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (20, 20, 20), -1)
    cv2.circle(frame, (28, 30), 10, color, -1)
    cv2.putText(frame, mode, (46, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(
        frame,
        f"Frames: {frame_count} | Device: {device} | Classes: {num_classes}",
        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
    )

    if last_raw_preds:
        # last_raw_preds 튜플 구조 변경 반영 (gloss, prob, f_idx)
        conf_text = "  ".join([f"{g}({p:.2f})" for g, p, _ in last_raw_preds[:5]])
        cv2.putText(
            frame, f"Raw(beam): {conf_text}", (16, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1,
        )

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("AppleGothic.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

    draw.text((16, 83), f"결과: {result_text}", font=font, fill=(0, 255, 255))
    frame[:] = np.array(img_pil)

    cv2.putText(
        frame, "SPACE: sign start/stop+predict   r: reset all   q: quit",
        (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
    )


class LandmarkInterpolator:
    def __init__(self):
        self.prev_keypoints = None

    def update(self, current_keypoints):
        if self.prev_keypoints is None:
            self.prev_keypoints = current_keypoints.copy()
            return current_keypoints

        current = current_keypoints.copy()
        zero_ratio = np.mean(np.abs(current) < 1e-6)

        if zero_ratio > 0.6:
            current = self.prev_keypoints.copy()
        else:
            self.prev_keypoints = current.copy()

        return current


def print_nms_timeline(nms_segments):
    if not nms_segments:
        print("── 비수지 신호 타임라인: 감지된 신호 없음 ──")
        return

    ordered = sorted(nms_segments, key=lambda seg: seg[1])
    print("── 비수지 신호 타임라인 ──")
    for key, start, end in ordered:
        print(f"  {key:<6} {start:6.2f}s ~ {end:6.2f}s  (지속 {end - start:5.2f}s)")


def resize_like_preprocess(frame):
    width = 640
    height = int(width * frame.shape[0] / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# ===========================================================================
# 3) 통합 메인 루프
# ===========================================================================

def run(camera_idx=0, nms_window_size=60, save_path=None, nms_infer_stride=1):
    nms_model = load_nms_model()
    sign_model, idx_to_gloss, blank_idx, num_classes = load_sign_model(DEVICE)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"카메라 열기 실패 (index: {camera_idx})")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None

    # --- NMS 상태 ---
    feat_buffer = collections.deque(maxlen=nms_window_size)
    prev_face_landmarks = None
    smooth_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)
    alpha = 0.3

    # --- 수어 상태 ---
    is_recording = False
    raw_features = []
    last_words = []
    last_raw_preds = []
    interpolator = LandmarkInterpolator()

    # --- 비수지 타임라인 상태 ---
    recording_start_time = None
    nms_key_active = {key: False for key in NMS_KEYS}
    nms_open_start = {key: None for key in NMS_KEYS}
    nms_segments = []

    frame_idx = 0
    nms_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)

    print("통합 웹캠 추론 시작")
    print("  SPACE : 수어 녹화 시작/종료(+예측)")
    print("  r     : 전체 버퍼/상태 리셋")
    print("  q     : 종료")
    print(
        f"[파라미터] beam_width={BEAM_WIDTH} | token_conf={TOKEN_CONFIDENCE_THRESHOLD} | "
        f"seq_conf={SEQUENCE_CONFIDENCE_THRESHOLD} | remove_unk={REMOVE_UNK} | "
        f"nms_window={nms_window_size} | nms_infer_stride={nms_infer_stride}"
    )

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = resize_like_preprocess(frame)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            holistic_results = holistic.process(rgb_frame)
            facemesh_results = face_mesh.process(rgb_frame)

            if writer is None and save_path:
                h, w = display_frame.shape[:2]
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            # ---------------- 수어 단어 인식 파이프라인 ----------------
            keypoints = preprocess_keypoints(holistic_results)
            keypoints = interpolator.update(keypoints)
            if is_recording:
                raw_features.append(keypoints)

            # ---------------- 비수지(NMS) 파이프라인 ----------------
            if facemesh_results.multi_face_landmarks:
                face_landmarks = facemesh_results.multi_face_landmarks[0]
                prev_face_landmarks = face_landmarks
            else:
                face_landmarks = prev_face_landmarks

            nms_feat = extract_nonmanual(face_landmarks, display_frame.shape)
            feat_buffer.append(nms_feat)
            frame_idx += 1

            if len(feat_buffer) >= 10:
                if frame_idx % nms_infer_stride == 0:
                    nms_probs = infer_nms(nms_model, list(feat_buffer))
                    smooth_probs = alpha * nms_probs + (1 - alpha) * smooth_probs
            else:
                nms_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)

            # ---------------- 비수지 타임라인 추적 (녹화 중일 때만) ----------------
            if is_recording and recording_start_time is not None:
                elapsed = time.time() - recording_start_time
                for i, key in enumerate(NMS_KEYS):
                    active = bool(smooth_probs[i] >= NMS_KEY_THRESHOLD[key])
                    was_active = nms_key_active[key]

                    if active and not was_active:
                        nms_open_start[key] = elapsed
                        nms_key_active[key] = True
                    elif not active and was_active:
                        start = nms_open_start[key]
                        if start is not None:
                            nms_segments.append((key, start, elapsed))
                        nms_key_active[key] = False
                        nms_open_start[key] = None

            # ---------------- 렌더링 ----------------
            draw_skeleton(display_frame, holistic_results)
            draw_sign_header(
                display_frame, is_recording, len(raw_features),
                last_words, last_raw_preds, DEVICE, num_classes,
            )
            draw_nms_panel(display_frame, nms_probs, smooth_probs, y_top=130)

            cv2.putText(
                display_frame, f"nms_buf:{len(feat_buffer)}/{nms_window_size}",
                (display_frame.shape[1] - 170, display_frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
            )

            cv2.imshow("Combined: Sign Word + Non-Manual", display_frame)

            if writer:
                writer.write(display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("r"):
                raw_features = []
                last_words = []
                last_raw_preds = []
                is_recording = False
                feat_buffer.clear()
                smooth_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)
                recording_start_time = None
                nms_key_active = {key: False for key in NMS_KEYS}
                nms_open_start = {key: None for key in NMS_KEYS}
                nms_segments = []
                print("전체 리셋 (수어 + 비수지)")

            elif key == 32:  # SPACE
                if not is_recording:
                    raw_features = []
                    last_words = []
                    last_raw_preds = []
                    is_recording = True
                    interpolator = LandmarkInterpolator()

                    recording_start_time = time.time()
                    nms_key_active = {key: False for key in NMS_KEYS}
                    nms_open_start = {key: None for key in NMS_KEYS}
                    nms_segments = []

                    print("Recording started.")
                else:
                    is_recording = False
                    elapsed = time.time() - recording_start_time if recording_start_time else 0
                    print(f"Recording stopped. Frames: {len(raw_features)}, Time: {elapsed:.2f}s")

                    if recording_start_time is not None:
                        for key in NMS_KEYS:
                            if nms_key_active[key] and nms_open_start[key] is not None:
                                nms_segments.append((key, nms_open_start[key], elapsed))
                            nms_key_active[key] = False
                            nms_open_start[key] = None

                    # predict_sequence가 T_len(총 프레임 수)도 반환하도록 수정됨
                    last_words, last_raw_preds, T_len = predict_sequence(
                        sign_model, raw_features, idx_to_gloss, blank_idx, DEVICE
                    )

                    print("── Beam Search 결과 (필터 전) ──")
                    for gloss, prob, f_idx in last_raw_preds:
                        marker = "✓" if gloss in last_words else "✗"
                        print(f"  {marker} {gloss:<20} conf={prob:.3f}")
                    print(f"── 최종 결과: {' / '.join(last_words) if last_words else 'No words detected'}")
                    print()

                    # 수지 신호(단어) 타임라인 출력 추가
                    print("── 수지 신호(단어) 타임라인 ──")
                    if not last_words:
                        print("  감지된 수지 신호 없음")
                    else:
                        time_per_frame = elapsed / T_len if T_len > 0 else 0
                        for gloss, prob, f_idx in last_raw_preds:
                            if gloss in last_words:
                                timestamp = f_idx * time_per_frame
                                print(f"  {gloss:<10} 발생 시점: {timestamp:5.2f}s (conf: {prob:.3f})")
                    print()

                    print_nms_timeline(nms_segments)
                    print()

    cap.release()
    if writer:
        writer.release()
        print(f"저장 완료: {save_path}")
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본: 0)")
    parser.add_argument("--window", type=int, default=60, help="NMS 슬라이딩 윈도우 프레임 수 (기본: 60)")
    parser.add_argument("--save", type=str, default=None, help="결과 영상 저장 경로")
    parser.add_argument(
        "--nms-stride", type=int, default=2,
        help="비수지 Transformer 추론을 몇 프레임마다 실행할지 (기본: 2, 1이면 매 프레임)",
    )
    args = parser.parse_args()
    run(
        camera_idx=args.camera,
        nms_window_size=args.window,
        save_path=args.save,
        nms_infer_stride=args.nms_stride,
    )