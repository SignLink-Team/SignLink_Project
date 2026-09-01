"""
실시간 수어 번역기 - 스트리밍(확정 후 전진) + 멀티프로세싱 버전

기존 코드 대비 변경점:
  - 수지(단어) 인식을 SPACE로 녹화 시작/종료하는 수동 방식에서,
    AsyncSignPredictor 를 통한 자동 실시간 스트리밍 방식으로 변경
  - 좌표 추출(mediapipe, 메인 프로세스)과 예측(모델 forward, 별도 프로세스)을
    OS 프로세스 단위로 완전히 분리 -> GIL의 영향을 받지 않고 실제로 다른 CPU 코어에서 동시 실행
  - 예측이 실시간 프레임 도착 속도보다 느려지면(백로그 발생), 입력 큐가 가장 오래된
    프레임부터 버리고 최신 프레임으로 따라잡는다 -> "지금 시점"과 "예측 중인 시점"의
    간격이 무한정 벌어지는 문제(예: 사용자는 75프레임인데 3,6,9프레임을 예측 중)를 방지
  - NMS(비수지) 파이프라인은 기존 구조 그대로 유지 (이미 슬라이딩 윈도우 방식이라 손댈 필요 없음)

핵심 아이디어 (Decode-and-Commit, 워커 프로세스 내부):
  1) 메인 프로세스: mediapipe로 좌표 추출 -> 큐에 넣기만 함 (즉시 리턴, 절대 안 막힘)
  2) 워커 프로세스: buffer가 stride만큼 늘어날 때마다 스냅샷을 떠서 greedy decode(가벼움)로
     "안정됐는지"만 확인 -> beam search(무거움)는 확정 직전 딱 한 번만 호출
  3) 첫 토큰이 stability_k번 연속으로 같은 단어 + 거의 같은 끝 프레임이면 확정
  4) 확정된 gloss는 결과 큐에 담아 메인 프로세스가 poll_committed()로 꺼내 화면에 표시,
     buffer는 확정 지점(+margin)까지만 잘라내고 그 이후 새로 들어온 프레임은 그대로 보존
  5) 손 움직임이 애매해서 안정 기준을 못 채우면 max_wait 프레임을 넘는 순간 강제로 확정 (타임아웃)
  6) 입력 큐가 FRAME_QUEUE_MAXSIZE를 넘으면 가장 오래된 프레임을 버림 (실시간성 최우선)
"""

import argparse
import collections
import json
import math
import multiprocessing as mlp
import os
import queue
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

PROJECT_DIR = r"C:\ai\sign_language_learning"
SIGN_MODEL_PATH = os.path.join(PROJECT_DIR, "best_sign_model.pth")
GLOSS_DICT_PATH = os.path.join(PROJECT_DIR, "gloss_dict.json")
GLOSS_MAPPING_PATH = os.path.join(PROJECT_DIR, "gloss_mapping.json")

HIDDEN_DIM = 512
MIN_FRAMES = 5
BEAM_WIDTH = 10
TOKEN_CONFIDENCE_THRESHOLD = 0.05
REMOVE_UNK = True

# --- 스트리밍(확정 후 전진) 관련 파라미터 ---
STREAM_STRIDE = 8          # 몇 프레임마다 추론할지
STREAM_STABILITY_K = 3     # 첫 토큰이 연속 몇 번 같아야 "확정 후보"로 볼지
STREAM_STABILITY_TOL = 2   # 확정 판단 시 끝 프레임 위치가 이 값(프레임) 이내로 흔들리는 건 허용
STREAM_MARGIN = 3          # 확정 지점에서 몇 프레임 여유를 두고 buffer를 자를지
STREAM_MAX_WAIT = 75       # buffer가 이 프레임 수를 넘으면 안정 여부와 상관없이 강제 확정 (타임아웃)
STREAM_HARD_CAP = 90      # 이 값을 넘으면 무조건 오래된 프레임을 잘라내는 안전장치 (렉 방지 최후 방어선)

sys.path.insert(0, PROJECT_DIR)
from model import SignLanguageModel  # noqa: E402
from preprocess import (  # noqa: E402
    apply_motion_derivatives as preprocess_motion_derivatives,
    extract_normalized_keypoints as preprocess_keypoints,
)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# --- 비수지(NMS) 모델 경로/설정 (기존과 동일) ---
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
# 1) 비수지(NMS) 모델 - 기존과 동일, 변경 없음
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
# 2) 수지(단어) 인식 모델 정의 / 로드 / CTC 디코딩 (모델·디코더 자체는 기존과 동일)
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
    """buffer 전체(누적된 프레임)를 매번 처음부터 다시 디코딩한다.
    반환되는 각 튜플의 f_idx는 '현재 buffer 기준' 프레임 인덱스이다."""
    T, C = log_probs.shape
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
        result.append((gloss, prob, f_idx))

    return result


def ctc_greedy_decode(log_probs, idx_to_gloss, blank_idx=0):
    """
    안정성 체크용 초경량 디코딩. 각 토큰에 대해 "이 토큰이 이미 끝났다고 확신할 수 있는지"
    (closed) 여부까지 반환한다.

    핵심: 예전 버전은 "확률이 가장 높은 프레임 위치가 흔들리지 않는지"로 안정성을 판단했는데,
    이건 같은 동작을 오래 유지할 때 계속 같은 결과를 반환해서 "같은 단어가 무한 반복 확정"되는
    버그의 원인이었다. 대신 여기서는 CTC collapse 규칙 그대로 "이 gloss 다음에 실제로 blank나
    다른 클래스가 나왔는가"를 직접 확인한다. 그게 확인된 토큰만 closed=True이고, 그 토큰은
    이미 끝난 게 확실하므로 즉시 확정해도 안전하다. buffer 맨 끝까지 이어지는 마지막 토큰은
    아직 끝났는지 알 수 없으므로 closed=False (계속 관찰 필요).
    """
    pred_ids = np.argmax(log_probs, axis=-1)  # (T,)
    T = len(pred_ids)

    tokens = []  # 각 원소: {"cls", "best_prob", "best_frame", "end"}
    prev = None
    for t in range(T):
        c = int(pred_ids[t])
        if c == blank_idx:
            prev = None
            continue
        prob = float(np.exp(log_probs[t, c]))
        if c != prev:
            tokens.append({"cls": c, "best_prob": prob, "best_frame": t, "end": t})
            prev = c
        else:
            tok = tokens[-1]
            tok["end"] = t
            if prob > tok["best_prob"]:
                tok["best_prob"] = prob
                tok["best_frame"] = t

    result = []
    for i, tok in enumerate(tokens):
        is_last = (i == len(tokens) - 1)
        # 마지막 토큰인데 buffer 끝까지 이어져 있으면(뒤에 blank/다른 클래스 없음) 아직 미완료
        closed = (not is_last) or (tok["end"] < T - 1)
        gloss = idx_to_gloss.get(tok["cls"], "<UNK>")
        result.append((gloss, tok["best_prob"], tok["end"], closed))
    return result


def make_model_features(raw_features):
    features = gaussian_filter1d(np.array(raw_features), sigma=1.0, axis=0)
    return preprocess_motion_derivatives(features)


# ===========================================================================
# 3) [신규] 스트리밍(확정 후 전진) 디코더
# ===========================================================================

# ===========================================================================
# 3) [신규] 스트리밍(확정 후 전진) 디코더 - 별도 프로세스(진짜 다른 코어)에서 실행
# ===========================================================================
#
# 좌표 추출(mediapipe)은 메인 프로세스, 예측(모델 forward)은 별도 프로세스로 완전히 분리.
# threading은 GIL 때문에 진짜 병렬이 아니라서 여기선 multiprocessing을 쓴다.
#
# 또한 "밀리면 오래된 프레임을 버리고 최신으로 따라잡는" 정책을 추가했다.
# 예측이 실시간보다 느리면 큐가 꽉 차는데, 이때 오래된 프레임을 버리지 않고 계속
# FIFO로 다 처리하려고 하면 "지금 몇 프레임인지"와 "무엇을 예측 중인지"의 간격이
# 시간이 갈수록 계속 벌어진다 (사용자가 겪은 3,6,9프레임 vs 75프레임 문제).
# 그래서 입력 큐 자체를 크기 제한하고, 꽉 찼을 때 가장 오래된 프레임을 버린 뒤
# 새 프레임을 넣는다 -> 워커는 항상 "최근 N초 이내"의 데이터만 보게 되고,
# 처리 속도가 못 따라가도 지연 시간이 무한정 벌어지지 않고 일정 수준에서 수렴한다.
# (대신 그 사이 있었던 수어 일부는 유실될 수 있음 -> 실시간성과의 트레이드오프)

# 워커 프로세스에 얼마나 뒤처진(stale) 데이터까지 허용할지 -> 이 프레임 수를 넘는 백로그는 버림
FRAME_QUEUE_MAXSIZE = 24  # CPU 2코어에서는 긴 백로그를 허용하지 않음

# 동작 시작/종료 감지
MOTION_START_THRESHOLD = 0.004
MOTION_END_THRESHOLD = 0.002
MOTION_START_FRAMES = 3
MOTION_END_FRAMES = 8
MIN_COLLECT_FRAMES = 12
POST_COMMIT_COOLDOWN_FRAMES = 8


def _sign_worker_main(frame_queue, result_queue, stop_event, cfg, debug_queue=None):
    """수지 모델 worker: CPU 2코어 + motion gate + 중복 확정 방지."""
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign_model, idx_to_gloss, blank_idx, _ = load_sign_model(device)

    stride = cfg["stride"]
    min_frames = cfg["min_frames"]
    max_wait = cfg["max_wait"]
    hard_cap = cfg["hard_cap"]

    buffer = []
    global_frame_offset = 0
    prev_kp = None

    motion_active = False
    start_count = 0
    end_count = 0
    cooldown = 0

    def motion_score(prev, curr):
        if prev is None:
            return 0.0
        a = np.asarray(prev, dtype=np.float32)
        b = np.asarray(curr, dtype=np.float32)
        if a.shape != b.shape:
            return 0.0
        return float(np.mean(np.abs(a - b)))

    def forward_log_probs(frames):
        enhanced = make_model_features(frames)
        tensor = torch.from_numpy(
            np.asarray(enhanced, dtype=np.float32)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sign_model(tensor)
            return F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    try:
        while not stop_event.is_set():
            try:
                kp = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break

            if kp is None:
                break

            current_motion = motion_score(prev_kp, kp)
            prev_kp = np.asarray(kp, dtype=np.float32).copy()

            # ------------------------------------------------------------
            # IDLE: 움직임이 없으면 모델에 넣지 않는다.
            # ------------------------------------------------------------
            if not motion_active:
                if cooldown > 0:
                    cooldown -= 1
                    continue

                if current_motion >= MOTION_START_THRESHOLD:
                    start_count += 1
                else:
                    start_count = 0

                if start_count >= MOTION_START_FRAMES:
                    motion_active = True
                    start_count = 0
                    end_count = 0
                    buffer = [kp]
                else:
                    continue

            else:
                # --------------------------------------------------------
                # COLLECTING
                # --------------------------------------------------------
                buffer.append(kp)

                if len(buffer) > hard_cap:
                    drop = len(buffer) - hard_cap
                    buffer = buffer[drop:]
                    global_frame_offset += drop

                if current_motion <= MOTION_END_THRESHOLD:
                    end_count += 1
                else:
                    end_count = 0

                enough = len(buffer) >= min_frames
                ended = end_count >= MOTION_END_FRAMES
                timeout = len(buffer) >= max_wait

                # CPU 2코어: 너무 잦은 inference 방지
                should_infer = (
                    enough and (
                        len(buffer) == min_frames
                        or len(buffer) % stride == 0
                        or ended
                        or timeout
                    )
                )

                if not should_infer:
                    continue

                log_probs = forward_log_probs(buffer)
                greedy_preds = ctc_greedy_decode(
                    log_probs, idx_to_gloss, blank_idx
                )
                greedy_preds = [
                    p for p in greedy_preds
                    if not (REMOVE_UNK and p[0] == "<UNK>")
                ]

                if debug_queue is not None:
                    try:
                        while True:
                            debug_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        debug_queue.put_nowait(
                            [(g, p, e) for g, p, e, _ in greedy_preds]
                        )
                    except Exception:
                        pass

                if not greedy_preds:
                    if timeout:
                        buffer.clear()
                        motion_active = False
                        start_count = 0
                        end_count = 0
                        cooldown = POST_COMMIT_COOLDOWN_FRAMES
                    continue

                first_gloss, first_prob, first_end, first_closed = greedy_preds[0]

                # 첫 토큰이 실제로 닫혔거나 동작이 끝난 경우에만 확정.
                can_commit = first_closed and len(buffer) >= MIN_COLLECT_FRAMES

                if not (can_commit or ended or timeout):
                    continue

                # 확정 직전만 beam search
                beam_preds = ctc_beam_search_decode(
                    log_probs, idx_to_gloss, blank_idx, BEAM_WIDTH
                )
                beam_preds = [
                    p for p in beam_preds
                    if not (REMOVE_UNK and p[0] == "<UNK>")
                ]

                if beam_preds:
                    final_gloss, final_prob, _ = beam_preds[0]
                else:
                    final_gloss, final_prob = first_gloss, first_prob

                if final_prob >= TOKEN_CONFIDENCE_THRESHOLD:
                    global_end = global_frame_offset + first_end
                    try:
                        result_queue.put_nowait(
                            (final_gloss, final_prob, global_end)
                        )
                    except Exception:
                        pass

                # --------------------------------------------------------
                # 가장 중요:
                # 확정 즉시 buffer를 비우고 IDLE로 돌아간다.
                # 손을 계속 가만히 두어도 같은 단어를 다시 확정하지 않는다.
                # --------------------------------------------------------
                buffer.clear()
                motion_active = False
                start_count = 0
                end_count = 0
                cooldown = POST_COMMIT_COOLDOWN_FRAMES

    finally:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


class AsyncSignPredictor:
    """메인 프로세스용 handle. 수지 모델은 별도 process에서 실행."""

    def __init__(self, stride=STREAM_STRIDE, min_frames=MIN_FRAMES,
                 stability_k=STREAM_STABILITY_K, stability_tol=STREAM_STABILITY_TOL,
                 margin=STREAM_MARGIN, max_wait=STREAM_MAX_WAIT, hard_cap=STREAM_HARD_CAP):
        self.cfg = dict(
            stride=stride,
            min_frames=min_frames,
            stability_k=stability_k,
            stability_tol=stability_tol,
            margin=margin,
            max_wait=max_wait,
            hard_cap=hard_cap,
        )
        self._frame_queue = mlp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._result_queue = mlp.Queue(maxsize=64)
        self._debug_queue = mlp.Queue(maxsize=1)
        self._stop_event = mlp.Event()
        self._process = None
        self.dropped_frame_count = 0

    def start(self):
        if self._process is not None and self._process.is_alive():
            return

        self._process = mlp.Process(
            target=_sign_worker_main,
            args=(
                self._frame_queue,
                self._result_queue,
                self._stop_event,
                self.cfg,
                self._debug_queue,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self):
        """worker 정상 종료 후 살아 있으면 강제 종료."""
        proc = self._process
        if proc is None:
            return

        print("[SIGN] worker 종료 요청")
        self._stop_event.set()

        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass

        proc.join(timeout=1.5)

        if proc.is_alive():
            print("[SIGN] worker 강제 종료")
            proc.terminate()
            proc.join(timeout=1.0)

        for q in (self._frame_queue, self._result_queue, self._debug_queue):
            try:
                q.close()
            except Exception:
                pass

        for q in (self._frame_queue, self._result_queue, self._debug_queue):
            try:
                q.join_thread()
            except Exception:
                pass

        self._process = None
        print("[SIGN] worker 종료 완료")

    def push_frame(self, keypoints):
        if self._process is None or not self._process.is_alive():
            return

        try:
            self._frame_queue.put_nowait(keypoints)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self.dropped_frame_count += 1
            except (queue.Empty, EOFError, OSError):
                pass

            try:
                self._frame_queue.put_nowait(keypoints)
            except (queue.Full, EOFError, OSError):
                pass

    def poll_committed(self):
        results = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
        return results

    def poll_last_raw_preds(self):
        try:
            return self._debug_queue.get_nowait()
        except queue.Empty:
            return []
        except (EOFError, OSError):
            return []

    def reset(self):
        self.stop()
        self._frame_queue = mlp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._result_queue = mlp.Queue(maxsize=64)
        self._debug_queue = mlp.Queue(maxsize=1)
        self._stop_event = mlp.Event()
        self.dropped_frame_count = 0
        self.start()

    def approx_backlog(self):
        try:
            return self._frame_queue.qsize()
        except (NotImplementedError, OSError):
            return -1


# ===========================================================================
# 4) 렌더링 유틸 (기존과 거의 동일, 헤더 문구만 스트리밍에 맞게 수정)
# ===========================================================================

def draw_skeleton(frame, holistic_results):
    mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_sign_header(frame, sentence_words, decoder, device, num_classes):
    color = (0, 150, 255)
    result_text = " / ".join(sentence_words[-8:]) if sentence_words else "인식 대기 중..."

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (20, 20, 20), -1)
    cv2.circle(frame, (28, 30), 10, color, -1)
    cv2.putText(frame, "STREAMING", (46, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(
        frame,
        f"Queue 백로그: {decoder.approx_backlog()}f | Device: {device} | Classes: {num_classes}",
        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
    )

    raw_preds_preview = decoder.poll_last_raw_preds()
    if raw_preds_preview:
        conf_text = "  ".join([f"{g}({p:.2f})" for g, p, _ in raw_preds_preview[:5]])
        cv2.putText(
            frame, f"Raw(greedy): {conf_text}", (16, 110),
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
        frame, "SPACE: 문장 구분(현재 문장 확정 후 새 문장 시작)   r: 전체 리셋   q: 종료",
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
# 5) 통합 메인 루프
# ===========================================================================

def run(camera_idx=1, nms_window_size=60, save_path=None, nms_infer_stride=1):
    nms_model = load_nms_model()

    # 수지 모델 자체는 이제 메인 프로세스에서 로드하지 않는다 (워커 프로세스가 따로 로드함).
    # 화면 표시용 클래스 개수만 가볍게 계산.
    num_classes = len(build_inference_dict())

    sign_predictor = AsyncSignPredictor()
    sign_predictor.start()  # 예측 워커 "프로세스" 시작 (진짜 다른 코어에서 동작)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(f"카메라 열기 실패 (index: {camera_idx})")
        sign_predictor.stop()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None

    # --- NMS 상태 (기존과 동일) ---
    feat_buffer = collections.deque(maxlen=nms_window_size)
    prev_face_landmarks = None
    smooth_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)
    alpha = 0.3

    # --- 수지 상태 (스트리밍) ---
    sentence_words = []          # 현재 문장으로 확정된 gloss들
    committed_log = []           # (gloss, prob, global_frame) 전체 로그 (문장 구분과 무관하게 누적)
    interpolator = LandmarkInterpolator()

    # --- 비수지 타임라인 상태 ---
    session_start_time = time.time()
    nms_key_active = {key: False for key in NMS_KEYS}
    nms_open_start = {key: None for key in NMS_KEYS}
    nms_segments = []

    frame_idx = 0
    nms_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)

    print("통합 웹캠 추론 시작 (실시간 스트리밍 모드)")
    print("  SPACE : 현재 문장 확정(구분) 후 새 문장 시작")
    print("  r     : 전체 버퍼/상태 리셋")
    print("  q     : 종료")
    print(
        f"[스트리밍 파라미터] stride={STREAM_STRIDE} | stability_k={STREAM_STABILITY_K} | "
        f"tol={STREAM_STABILITY_TOL}f | margin={STREAM_MARGIN}f | max_wait={STREAM_MAX_WAIT}f | "
        f"hard_cap={STREAM_HARD_CAP}f | "
        f"token_conf={TOKEN_CONFIDENCE_THRESHOLD} | remove_unk={REMOVE_UNK} | "
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

            # ---------------- 수지 단어 인식: 항상 실행 (녹화 토글 없음) ----------------
            keypoints = preprocess_keypoints(holistic_results)
            keypoints = interpolator.update(keypoints)

            # 프레임을 큐에 넣기만 함 (즉시 리턴, 예측은 별도 프로세스가 다른 코어에서 처리)
            sign_predictor.push_frame(keypoints)

            # 워커가 그동안 확정해놓은 결과가 있으면 꺼내온다 (블로킹 없음)
            newly_committed = sign_predictor.poll_committed()
            for gloss, prob, global_end in newly_committed:
                sentence_words.append(gloss)
                elapsed = time.time() - session_start_time
                committed_log.append((gloss, prob, global_end, elapsed))
                print(f"[확정] {gloss:<15} conf={prob:.3f}  (frame={global_end}, t={elapsed:.2f}s)")

            # ---------------- 비수지(NMS) 파이프라인 (기존과 동일) ----------------
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

            # ---------------- 비수지 타임라인 추적 (항상, 세션 전체 기준) ----------------
            elapsed = time.time() - session_start_time
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
            draw_sign_header(display_frame, sentence_words, sign_predictor, DEVICE, num_classes)
            draw_nms_panel(display_frame, nms_probs, smooth_probs, y_top=130)

            cv2.putText(
                display_frame, f"nms_buf:{len(feat_buffer)}/{nms_window_size}",
                (display_frame.shape[1] - 170, display_frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
            )

            cv2.imshow("Combined: Sign Word + Non-Manual (Streaming)", display_frame)

            if writer:
                writer.write(display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("r"):
                sign_predictor.reset()
                sentence_words = []
                committed_log = []
                feat_buffer.clear()
                smooth_probs = np.zeros(len(NMS_KEYS), dtype=np.float32)
                session_start_time = time.time()
                nms_key_active = {key: False for key in NMS_KEYS}
                nms_open_start = {key: None for key in NMS_KEYS}
                nms_segments = []
                print("전체 리셋 (수어 + 비수지)")

            elif key == 32:  # SPACE: 문장 구분
                print(f"── 문장 확정: {' / '.join(sentence_words) if sentence_words else '(빈 문장)'}")
                print_nms_timeline(nms_segments)
                print()
                sentence_words = []
                nms_segments = []
                # 진행 중이던 buffer(아직 확정 안 된 프레임)는 유지 -> 문장 사이 gloss 유실 방지
                # 완전히 끊고 싶다면 아래 주석 해제:
                # sign_predictor.reset()

    sign_predictor.stop()  # 워커 프로세스 정상 종료
    cap.release()
    if writer:
        writer.release()
        print(f"저장 완료: {save_path}")
    cv2.destroyAllWindows()

    print()
    print("── 전체 확정 로그 ──")
    for gloss, prob, global_end, elapsed in committed_log:
        print(f"  {gloss:<15} conf={prob:.3f}  frame={global_end}  t={elapsed:.2f}s")
    print("종료")


if __name__ == "__main__":
    mlp.freeze_support()  # Windows에서 멀티프로세싱 안전하게 쓰기 위한 관례적 호출
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