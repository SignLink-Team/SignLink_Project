"""
inference_nms.py
----------------
비수지 추론 코드 - 학습 전처리와 완전히 동일한 방식 적용

모드:
    1. dataset_info.json 에서 랜덤 샘플 선택 → 전처리된 npy 그대로 추론
    2. 영상 파일 직접 입력 → 학습과 동일한 전처리 후 추론
    3. dataset_info.json 샘플의 영상을 직접 재전처리하여 추론 (성능 검증용)

사용법:
    # 랜덤 샘플로 검증 (전처리된 npy 사용)
    python inference_nms.py --mode sample

    # 랜덤 샘플 영상 재전처리 후 추론 (파이프라인 검증)
    python inference_nms.py --mode sample --reprocess

    # 특정 영상 추론
    python inference_nms.py --mode video --video path/to/video.mp4 --output result.mp4
"""

import os
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
from scipy.ndimage import gaussian_filter1d

from nonmanual_features import extract_nonmanual

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

CKPT_PATH           = r"D:\새 폴더 (2)\deepRun\keypoint_data\checkpoints_nms\best_nms.pt"
SAVE_PATH           = r"D:\새 폴더 (2)\deepRun\keypoint_data"
NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")
INFO_PATH           = os.path.join(SAVE_PATH, "dataset_info.json")
VIDEO_DIR           = r"D:\새 폴더 (2)\deepRun\data\videos"

NMS_KEYS  = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "EBf", "Hno", "Hs"]
THRESHOLD = 0.45   # 0.5 보다 낮게 설정 (모델 출력 범위 고려)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEY_COLORS = {
    "Mo1":  (255, 100,  50),
    "Mmo":  (255, 180,  50),
    "Mctr": (255, 220, 100),
    "Ci":   ( 50, 200, 255),
    "Ebu":  ( 50, 255, 150),
    "EBf":  (100, 180, 255),
    "Hno":  (200,  80, 255),
    "Hs":   (255,  80, 150),
}


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

class NMSClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_labels, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.dropout(self.input_proj(x))
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)


def load_model():
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg   = ckpt["cfg"]
    model = NMSClassifier(
        input_dim  = cfg["input_dim"],
        d_model    = cfg["d_model"],
        nhead      = cfg["nhead"],
        num_layers = cfg["num_layers"],
        num_labels = len(NMS_KEYS),
        dropout    = cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"모델 로드 완료 (epoch {ckpt['epoch']}, macro_F1: {ckpt['macro_f1']:.4f})")
    return model


# ---------------------------------------------------------------------------
# 전처리 (학습 시와 완전히 동일)
# ---------------------------------------------------------------------------

def apply_motion_derivatives(features):
    velocity     = np.diff(features, axis=0, prepend=features[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    return np.concatenate([features, velocity, acceleration], axis=1)


def extract_features(video_path, start_frame, end_frame):
    """
    학습 전처리(preprocess_nms.py)와 완전히 동일한 방식:
        - start_frame ~ end_frame 구간만 처리
        - confidence 0.3
        - prev_landmarks 대체
        - gaussian_filter1d + motion_derivatives
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 없음: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    feat_seq = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    prev_landmarks = None

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            resized = cv2.resize(
                frame, (640, int(640 * frame.shape[0] / frame.shape[1])),
                interpolation=cv2.INTER_LINEAR
            )
            results = face_mesh.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                prev_landmarks = face_landmarks
            else:
                face_landmarks = prev_landmarks

            feat_seq.append(extract_nonmanual(face_landmarks, resized.shape))
            current_frame += 1
    cap.release()

    if len(feat_seq) < 5:
        raise ValueError("유효 프레임 부족")

    features = gaussian_filter1d(np.array(feat_seq), sigma=1.0, axis=0)
    features = apply_motion_derivatives(features)
    return frames, features, fps


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, features):
    x      = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    logits = model(x)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, 8)


# ---------------------------------------------------------------------------
# 정답 라벨 → dense 변환
# ---------------------------------------------------------------------------

def segments_to_dense(nms_labels, total_frames):
    label = np.zeros((total_frames, len(NMS_KEYS)), dtype=np.float32)
    for i, key in enumerate(NMS_KEYS):
        for seg in nms_labels.get(key, []):
            s = max(0, seg["start_frame"])
            e = min(total_frames - 1, seg["end_frame"])
            label[s:e + 1, i] = 1.0
    return label


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------

def print_summary(probs, gt_labels, fps, sample):
    T = len(probs)
    print(f"\n샘플 ID     : {sample['id']}")
    print(f"수어 시퀀스 : {sample['gloss_sequence']}")
    print(f"총 프레임   : {T}")
    print(f"\n{'키':<8} {'예측 구간':<40} {'정답 구간'}")
    print("-" * 90)

    for i, key in enumerate(NMS_KEYS):
        # 예측 구간
        active = (probs[:, i] >= THRESHOLD).astype(int)
        diff   = np.diff(active, prepend=0, append=0)
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0] - 1
        pred_str = ", ".join(
            f"{s/fps:.2f}s~{e/fps:.2f}s" for s, e in zip(starts, ends)
        ) if len(starts) > 0 else "없음"

        # 정답 구간
        gt_str = ", ".join(
            f"{seg['start_frame']/fps:.2f}s~{seg['end_frame']/fps:.2f}s"
            for seg in sample.get("nms_labels", {}).get(key, [])
        ) or "없음"

        print(f"{key:<8} {pred_str:<40} {gt_str}")

    # F1 계산
    if gt_labels is not None:
        from sklearn.metrics import f1_score
        pred_bin = (probs >= THRESHOLD).astype(int)
        print(f"\n{'키':<8} {'F1':>6}")
        print("-" * 20)
        f1s = []
        for i, key in enumerate(NMS_KEYS):
            f1 = f1_score(gt_labels[:, i], pred_bin[:, i], zero_division=0)
            f1s.append(f1)
            print(f"{key:<8} {f1:>6.3f}")
        print(f"{'macro':<8} {np.mean(f1s):>6.3f}")


def draw_overlay(frame, probs, gt_row, frame_idx):
    active = [(NMS_KEYS[i], probs[i]) for i in range(len(NMS_KEYS)) if probs[i] >= THRESHOLD]

    if active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 20 + len(active) * 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for j, (key, prob) in enumerate(active):
        color  = KEY_COLORS.get(key, (255, 255, 255))
        bar_w  = int(prob * 150)
        y      = 30 + j * 30
        is_gt  = gt_row is not None and gt_row[NMS_KEYS.index(key)] == 1.0
        prefix = "✓" if is_gt else " "
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 18), color, -1)
        cv2.putText(frame, f"{prefix}{key}: {prob:.2f}",
                    (15, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"frame: {frame_idx}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    return frame


# ---------------------------------------------------------------------------
# 모드별 실행
# ---------------------------------------------------------------------------

def run_sample(reprocess=False):
    """dataset_info.json 에서 랜덤 샘플 선택"""
    with open(INFO_PATH, encoding='utf-8') as f:
        info = json.load(f)

    # 원본 + 비수지 전처리 완료 + 영벡터 아닌 샘플
    candidates = []
    for d in info:
        if d.get("is_aug") or "nonmanual_feature_file" not in d:
            continue
        feat = np.load(os.path.join(NONMANUAL_SAVE_PATH, d["nonmanual_feature_file"]))
        if feat.mean() > 0.001:
            candidates.append(d)

    sample = random.choice(candidates)
    print(f"\n선택된 샘플: {sample['id']}")

    model = load_model()

    if reprocess:
        # 영상 직접 재전처리
        video_path = os.path.join(VIDEO_DIR, sample['id'] + ".mp4")
        if not os.path.exists(video_path):
            print(f"영상 없음: {video_path}")
            return

        # start_frame / end_frame 재계산 (preprocess_nms.py 와 동일)
        fps_label = 30  # 라벨 JSON 에서 읽어야 하지만 근사값 사용
        cap = cv2.VideoCapture(video_path)
        fps_label = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # nms_labels 에서 전체 구간 추정
        all_segs = [
            seg for segs in sample.get("nms_labels", {}).values() for seg in segs
        ]
        if all_segs:
            start_frame = max(0, min(s["start_frame"] for s in all_segs) - int(fps_label * 0.5))
            end_frame   = max(s["end_frame"] for s in all_segs) + int(fps_label * 0.5)
        else:
            start_frame = 0
            end_frame   = sample["length"] - 1

        print(f"재전처리 구간: {start_frame} ~ {end_frame} 프레임")
        frames, features, fps = extract_features(video_path, start_frame, end_frame)
        print(f"재전처리 완료: {features.shape}")
    else:
        # 전처리된 npy 그대로 사용
        features = np.load(
            os.path.join(NONMANUAL_SAVE_PATH, sample["nonmanual_feature_file"])
        ).astype(np.float32)
        frames   = None
        fps      = 30.0
        print(f"전처리된 npy 사용: {features.shape}")

    probs     = run_inference(model, features)
    T         = len(probs)
    gt_labels = segments_to_dense(sample.get("nms_labels", {}), T)

    print_summary(probs, gt_labels, fps, sample)

    # 영상 출력 (reprocess 모드에서만)
    if reprocess and frames:
        for idx in range(min(len(frames), T)):
            gt_row = gt_labels[idx] if idx < len(gt_labels) else None
            frame  = draw_overlay(frames[idx].copy(), probs[idx], gt_row, idx)
            cv2.imshow("NMS Inference (✓=정답)", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def run_video(video_path, output_path=None):
    """영상 파일 직접 입력"""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = load_model()
    frames, features, fps = extract_features(video_path, 0, total - 1)
    probs = run_inference(model, features)

    print_summary(probs, None, fps, {"id": os.path.basename(video_path), "gloss_sequence": [], "nms_labels": {}})

    if output_path:
        h, w   = frames[0].shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for idx in range(min(len(frames), len(probs))):
            frame = draw_overlay(frames[idx].copy(), probs[idx], None, idx)
            writer.write(frame)
        writer.release()
        print(f"\n결과 영상 저장: {output_path}")
    else:
        for idx in range(min(len(frames), len(probs))):
            frame = draw_overlay(frames[idx].copy(), probs[idx], None, idx)
            cv2.imshow("NMS Inference", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="sample", choices=["sample", "video"])
    parser.add_argument("--reprocess",  action="store_true", help="영상 재전처리 후 추론 (sample 모드)")
    parser.add_argument("--video",      default=None,  help="영상 경로 (video 모드)")
    parser.add_argument("--output",     default=None,  help="결과 영상 저장 경로")
    args = parser.parse_args()

    if args.mode == "sample":
        run_sample(reprocess=args.reprocess)
    else:
        if not args.video:
            print("--video 경로를 지정하세요")
        else:
            run_video(args.video, args.output)