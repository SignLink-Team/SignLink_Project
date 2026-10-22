from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import io
import json
import multiprocessing as mlp
import os
import queue
import sys
import tempfile
import math
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# 0. 경로 / 모델 설정
# ============================================================================

AI_SERVER_DIR = Path(__file__).resolve().parent
ROOT_DIR = AI_SERVER_DIR.parent

MODEL_DIR_CANDIDATES = [
    AI_SERVER_DIR,
    AI_SERVER_DIR / "model",
    ROOT_DIR / "model",
]

MODEL_DIR = next(
    (path for path in MODEL_DIR_CANDIDATES if (path / "best_sign_model.pth").exists()),
    AI_SERVER_DIR,
)

KEYPOINT_DIR = MODEL_DIR / "keypoint_data"

MODEL_PATH = MODEL_DIR / "best_sign_model.pth"
GLOSS_DICT_PATH = KEYPOINT_DIR / "gloss_dict.json"
GLOSS_MAPPING_PATH = KEYPOINT_DIR / "gloss_mapping.json"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model import SignLanguageModel  # noqa: E402
from llm_client import LLM_TIMEOUT_SECONDS, translate_gloss  # noqa: E402
from preprocess import apply_motion_derivatives, extract_normalized_keypoints  # noqa: E402

# 비수지(NMS)는 보조 기능이다. nonmanual_features.py가 없거나 import에 실패해도
# 수지 인식 서버 자체는 정상 기동해야 하므로 실패를 삼키고 NMS만 비활성화한다.
try:
    from nonmanual_features import extract_nonmanual  # noqa: E402
    NMS_FEATURES_AVAILABLE = True
    NMS_FEATURES_ERROR = None
except Exception as _nms_import_exc:  # noqa: BLE001
    extract_nonmanual = None  # type: ignore[assignment]
    NMS_FEATURES_AVAILABLE = False
    NMS_FEATURES_ERROR = repr(_nms_import_exc)

try:
    from mediapipe.framework.formats import landmark_pb2 as _landmark_pb2
except Exception:  # noqa: BLE001
    _landmark_pb2 = None


# ============================================================================
# 1. 공통 설정
# ============================================================================

HIDDEN_DIM = 512
MIN_FRAMES = 12

# 예전 단독 실시간 테스트 코드와 동일
BEAM_WIDTH = 10
TOKEN_CONFIDENCE_THRESHOLD = 0.05
REMOVE_UNK = True

STREAM_STRIDE = 8
STREAM_MARGIN = 3
STREAM_MAX_WAIT = 30
STREAM_HARD_CAP = 40

FRAME_QUEUE_MAXSIZE = 8

# 예전 단독 테스트의 motion gate.
# front_still이 오는 정상 경로에서는 이 값들이 아예 쓰이지 않는다.
# 프론트가 still을 안 보내는 구버전 클라이언트에 대한 폴백 전용이므로,
# 프론트의 motion score(다른 계산식/스케일)를 그대로 복사해 넣지 말 것 —
# 필요하면 서버 자체 motion_score 분포를 따로 실측해서 재보정한다.
MOTION_START_THRESHOLD = 0.004
MOTION_END_THRESHOLD = 0.002
MOTION_START_FRAMES = 3
MOTION_END_FRAMES = 8
MIN_COLLECT_FRAMES = 12
POST_COMMIT_COOLDOWN_FRAMES = 0

# front_still/motion 시작·종료 판정과 무관한 독립 안전장치.
# 지금 디코딩하려는 buffer 구간 자체의 프레임 간 diff 표준편차가 이보다
# 낮으면(=사실상 정지/센서 잡음) 추론 자체를 건너뛴다. raw 261차원
# keypoint 스케일 기준이며, 실측 데이터로 재보정이 필요한 값이다.
#
# 0.01은 실측 없이 잡은 추정치였고, 실제 활발한 수어 동작 중에도 이
# 값보다 낮은 구간이 있어 추론 자체가 스킵되면서 실시간 스트리밍
# 커밋이 전부 막히고 stream_end의 force_flush에서만 한꺼번에 쏟아지는
# 문제가 있었다. [MOTION] 로그로 실제 활동 중 std 분포를 확인해
# 재보정하기 전까지는 사실상 항상 통과하도록 낮춰둔다(게이트 로직과
# 계측 로그는 유지 — 값만 무력화).
MIN_BUFFER_MOTION_STD = 0.0001

STREAM_STABILITY_K = 3
# stride(8프레임)마다 한 번씩만 추론하므로, 연속된 두 관측치의 global_end는
# 구조적으로 거의 항상 stride만큼 벌어진다. 이 값이 stride보다 작으면
# stable-k가 설계상 거의 통과할 수 없어 매번 timeout(느린 경로)로만
# 빠지게 된다 — 실측 로그에서 모든 커밋이 timeout이었던 주된 원인.
# stride + 여유분으로 설정한다.
STABILITY_END_TOL = STREAM_STRIDE + 4
# 실측 confidence 분포(연속 수어 실사용 로그)가 0.15~0.52 사이였고
# 0.50을 넘는 경우가 거의 없어, stable-k가 사실상 죽어있었다. 이 값을
# 낮추면 확정이 빨라지지만 오탐(정지/잡음을 단어로 오확정) 가능성도
# 커진다 — 실제로 0.35로 낮췄다가 정지 구간에서 엉뚱한 gloss가 대량
# 확정되는 문제가 발생해 0.45로 다시 올렸다. 배포 후 실측하며 재조정할 것.
STABILITY_PROB_THRESHOLD = 0.45

# ----------------------------------------------------------------------------
# 비수지(NMS) 설정 — 통합 웹캠 스크립트(async_realtime_webcam_fixed.py)와 동일 값
# ----------------------------------------------------------------------------
NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "EBf", "Hno", "Hs"]
# 서버(timeline/active 판정)와 프론트(src/nms.ts)의 값은 항상 같게 유지할 것.
NMS_KEY_THRESHOLD = {
    "Mo1": 0.42, "Mmo": 0.42, "Mctr": 0.42, "Ci": 0.40,
    "Ebu": 0.45, "EBf": 0.52, "Hno": 0.50, "Hs": 0.40,
}
NMS_WINDOW_SIZE = 60
NMS_MIN_FRAMES = 10
NMS_INFER_STRIDE = 2
NMS_SMOOTH_ALPHA = 0.3
NMS_FACE_VALUES = 468 * 3          # FaceMesh 468점 x (x, y, z)
NMS_QUEUE_MAXSIZE = 8
CAMERA_FPS = 30.0                  # 프론트 CAMERA_EXTRACTION_FPS와 동일(타임라인 초 환산용)

NMS_ENABLED = os.getenv("SIGNLINK_NMS_ENABLED", "true").lower() == "true"
NMS_CKPT_PATH = Path(
    os.getenv("SIGNLINK_NMS_CKPT", str(MODEL_DIR / "best_nms_finetune_5.pt"))
)
NMS_AVAILABLE = NMS_ENABLED and NMS_FEATURES_AVAILABLE and NMS_CKPT_PATH.exists()

# ----------------------------------------------------------------------------
# 녹화(세션) 전체 디코딩 / LLM 전달 설정
# ----------------------------------------------------------------------------
# 한 세션(start~end)에서 보관할 최대 프레임 수(30fps 기준 약 3분). 넘으면 이후 프레임은 녹화하지 않는다.
MAX_RECORD_FRAMES = 5400
# 최종(전체 시퀀스) 디코딩에서 <UNK>를 버릴지. 기존 실시간 경로와 같게 REMOVE_UNK를 따른다.
# False로 바꾸면 <UNK>가 결과에 남아 게이트웨이의 requires_review(UnkSolver) 경로로 간다.
FINAL_REMOVE_UNK = REMOVE_UNK

# 비수지 키의 한글 의미. LLM 프롬프트에 "코드(의미)" 형태로 들어간다.
# 비워 두면 코드명만 전달되어 LLM이 해석할 수 없으니 반드시 채울 것.
NMS_KEY_LABELS: dict[str, str] = {
    "Mo1": "",
    "Mmo": "",
    "Mctr": "",
    "Ci": "",
    "Ebu": "",
    "EBf": "",
    "Hno": "",
    "Hs": "",
}
# 비수지 감지는 smoothing 때문에 실제 표정보다 늦게 나오므로, 단어 구간과 겹침을 판정할 때 허용하는 여유(초)
NMS_ALIGN_TOLERANCE_SEC = 0.4
# LLM에 전달할 때 이보다 짧은 비수지 구간은 깜빡임 잡음으로 보고 제외(UI용 nms_segments에는 모두 남김)
NMS_MIN_SEGMENT_SEC = 0.2

PREDICTION_OUTPUT_DIR = ROOT_DIR / "ai_server" / "prediction_outputs"
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_LLM = os.getenv("SIGNLINK_SKIP_LLM", "true").lower() == "true"


# ============================================================================
# 2. FastAPI
# ============================================================================

app = FastAPI(title="SignLink AI Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: list[list[float]] | None = None
    landmarks: dict[str, Any] | None = None


# ============================================================================
# 3. 수지 모델 / 사전
# ============================================================================

def build_inference_dict() -> dict[str, int]:
    with GLOSS_DICT_PATH.open("r", encoding="utf-8") as file:
        old_gloss_to_idx: dict[str, int] = json.load(file)

    mapping_dict: dict[str, str] = {}

    if GLOSS_MAPPING_PATH.exists():
        with GLOSS_MAPPING_PATH.open("r", encoding="utf-8") as file:
            mapping_dict = json.load(file)

    blank_tokens = [
        key for key, value in old_gloss_to_idx.items()
        if value == 0
    ]

    if not blank_tokens:
        raise ValueError(
            "gloss_dict.json must contain a blank token at index 0."
        )

    blank_token = blank_tokens[0]

    active_glosses = {
        mapping_dict.get(gloss, gloss)
        for gloss in old_gloss_to_idx
        if gloss != blank_token
    }

    active_glosses.add("<UNK>")
    active_glosses.discard(blank_token)

    gloss_to_idx = {blank_token: 0}

    for idx, gloss in enumerate(sorted(active_glosses), start=1):
        gloss_to_idx[gloss] = idx

    return gloss_to_idx


def load_sign_model(
    device: torch.device,
) -> tuple[SignLanguageModel, dict[int, str], int, int]:
    gloss_to_idx = build_inference_dict()

    idx_to_gloss = {
        value: key
        for key, value in gloss_to_idx.items()
    }

    blank_idx = gloss_to_idx.get("<blank>", 0)
    num_classes = len(gloss_to_idx)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    checkpoint_classes = checkpoint["fc.weight"].shape[0]

    if checkpoint_classes != num_classes:
        raise ValueError(
            f"Model class count({checkpoint_classes}) and "
            f"dictionary class count({num_classes}) do not match."
        )

    model = SignLanguageModel(
        input_dim=783,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(
        f"[SIGN] model loaded: "
        f"classes={num_classes}, device={device}",
        flush=True,
    )

    return model, idx_to_gloss, blank_idx, num_classes


def make_model_features(raw_features: np.ndarray) -> np.ndarray:
    """
    예전 단독 테스트와 동일한 전처리 순서:
      raw 261
        -> gaussian_filter1d(sigma=1.0)
        -> motion derivatives
        -> 783
    """
    features = gaussian_filter1d(
        np.asarray(raw_features, dtype=np.float32),
        sigma=1.0,
        axis=0,
    )

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features, got shape={features.shape}"
        )

    if features.shape[1] == 261:
        features = apply_motion_derivatives(features)

    return np.asarray(features, dtype=np.float32)


# ============================================================================
# 4. CTC Greedy / Beam Search
#    예전 단독 테스트 코드의 핵심 로직
# ============================================================================

def ctc_greedy_decode(
    log_probs: np.ndarray,
    idx_to_gloss: dict[int, str],
    blank_idx: int = 0,
) -> list[tuple[str, float, int, bool]]:
    """
    Greedy CTC decoding.

    반환:
      (gloss, best_prob, end_frame, closed)

    closed:
      해당 토큰 뒤에 blank 또는 다른 class가 실제로 나온 경우 True.
      마지막 토큰이 buffer 끝까지 이어지는 경우 False.
    """
    pred_ids = np.argmax(log_probs, axis=-1)
    T = len(pred_ids)

    tokens: list[dict[str, Any]] = []
    prev: int | None = None

    for t in range(T):
        c = int(pred_ids[t])

        if c == blank_idx:
            prev = None
            continue

        prob = float(np.exp(log_probs[t, c]))

        if c != prev:
            tokens.append(
                {
                    "cls": c,
                    "best_prob": prob,
                    "best_frame": t,
                    "end": t,
                }
            )
            prev = c
        else:
            token = tokens[-1]
            token["end"] = t

            if prob > token["best_prob"]:
                token["best_prob"] = prob
                token["best_frame"] = t

    result: list[tuple[str, float, int, bool]] = []

    for i, token in enumerate(tokens):
        is_last = i == len(tokens) - 1

        closed = (
            (not is_last)
            or token["end"] < T - 1
        )

        gloss = idx_to_gloss.get(
            int(token["cls"]),
            "<UNK>",
        )

        result.append(
            (
                gloss,
                float(token["best_prob"]),
                int(token["end"]),
                bool(closed),
            )
        )

    return result


def ctc_greedy_spans(
    log_probs: np.ndarray, idx_to_gloss: dict[int, str], blank_idx: int = 0
) -> list[dict[str, Any]]:
    """전체 시퀀스용 greedy CTC. 토큰별 (gloss, 최고 확률, 시작/끝 프레임 인덱스)를 반환한다."""
    pred_ids = np.argmax(log_probs, axis=-1)
    spans: list[dict[str, Any]] = []
    prev: int | None = None

    for t, c in enumerate(pred_ids.tolist()):
        if c == blank_idx:
            prev = None
            continue

        prob = float(np.exp(log_probs[t, c]))

        if c != prev:
            spans.append(
                {"gloss": idx_to_gloss.get(int(c), "<UNK>"), "prob": prob, "start": t, "end": t}
            )
            prev = c
        else:
            span = spans[-1]
            span["end"] = t
            if prob > span["prob"]:
                span["prob"] = prob

    return spans


def ctc_beam_search_decode(
    log_probs: np.ndarray,
    idx_to_gloss: dict[int, str],
    blank_idx: int = 0,
    beam_width: int = BEAM_WIDTH,
) -> list[tuple[str, float, int]]:
    """
    예전 단독 테스트의 beam search.

    buffer 전체를 다시 처음부터 디코딩하고,
    beam search는 확정 직전에만 호출한다.
    """
    T, C = log_probs.shape

    # seq -> (log probability, token probs, token frames)
    beams: dict[
        tuple[int, ...],
        tuple[float, list[float], list[int]],
    ] = {
        (): (0.0, [], [])
    }

    for t in range(T):
        new_beams: dict[
            tuple[int, ...],
            tuple[float, list[float], list[int]],
        ] = {}

        frame_log_probs = log_probs[t]

        for seq, (
            seq_log_prob,
            seq_probs,
            seq_frames,
        ) in beams.items():

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

                    current_prob = float(
                        np.exp(token_log_prob)
                    )

                    if (
                        new_probs
                        and current_prob > new_probs[-1]
                    ):
                        new_probs[-1] = current_prob
                        new_frames[-1] = t

                else:
                    new_seq = seq + (c,)
                    new_probs = seq_probs + [
                        float(np.exp(token_log_prob))
                    ]
                    new_frames = seq_frames + [t]

                if new_seq in new_beams:
                    (
                        prev_log_prob,
                        prev_probs,
                        prev_frames,
                    ) = new_beams[new_seq]

                    merged = np.logaddexp(
                        prev_log_prob,
                        new_log_prob,
                    )

                    keep_probs = (
                        new_probs
                        if len(new_probs) >= len(prev_probs)
                        else prev_probs
                    )

                    keep_frames = (
                        new_frames
                        if len(new_frames) >= len(prev_frames)
                        else prev_frames
                    )

                    new_beams[new_seq] = (
                        merged,
                        keep_probs,
                        keep_frames,
                    )

                else:
                    new_beams[new_seq] = (
                        new_log_prob,
                        new_probs,
                        new_frames,
                    )

        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )[:beam_width]
        )

    if not beams:
        return []

    best_seq, (
        _,
        best_probs,
        best_frames,
    ) = max(
        beams.items(),
        key=lambda item: item[1][0],
    )

    result: list[tuple[str, float, int]] = []

    for i, token_idx in enumerate(best_seq):
        gloss = idx_to_gloss.get(
            int(token_idx),
            "<UNK>",
        )

        prob = (
            best_probs[i]
            if i < len(best_probs)
            else 0.0
        )

        frame_idx = (
            best_frames[i]
            if i < len(best_frames)
            else 0
        )

        result.append(
            (
                gloss,
                float(prob),
                int(frame_idx),
            )
        )

    return result


# ============================================================================
# 5. 별도 프로세스 수지 스트리밍 worker
# ============================================================================

def _sign_worker_main(
    frame_queue: Any,
    result_queue: Any,
    stop_event: Any,
    latest_frame_id: Any,
    cfg: dict[str, Any],
    debug_queue: Any = None,
) -> None:
    """
    예전 단독 테스트의 AsyncSignPredictor worker를
    WebSocket 서버용으로 이식.

    핵심:
      - 프레임을 계속 buffer에 누적
      - stride마다 greedy
      - 첫 token이 closed인지 확인
      - 안정적으로 닫히면 beam search
      - 확정 후 buffer를 전체 삭제하지 않고
        확정 지점 + margin까지만 전진
      - 남은 프레임은 다음 단어의 문맥으로 유지
    """
    try:
        # CPU inference라면 코어 수에 맞춰 intra-op 스레드를 늘린다.
        # 고정값 2는 근거 없는 절충값이었고, 배포 환경마다 재튜닝이 필요하다.
        cpu_threads = max(2, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        print(f"[SIGN WORKER] torch threads: {cpu_threads}", flush=True)
    except RuntimeError:
        pass

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("[SIGN WORKER] before load_sign_model", flush=True)
    sign_model, idx_to_gloss, blank_idx, _ = load_sign_model(device)

    print(
        f"[SIGN WORKER] model loaded: classes={len(idx_to_gloss)}",
        flush=True,
    )

    # 서버 startup이 모델 로드 완료를 기다릴 수 있도록 신호를 보낸다.
    try:
        result_queue.put_nowait(("__ready__",))
    except Exception:
        pass

    stride = int(cfg["stride"])
    min_frames = int(cfg["min_frames"])
    margin = int(cfg["margin"])
    max_wait = int(cfg["max_wait"])
    hard_cap = int(cfg["hard_cap"])
    stability_k = int(cfg["stability_k"])
    stability_prob_threshold = float(cfg["stability_prob_threshold"])
    stability_end_tol = int(cfg["stability_end_tol"])

    buffer: list[np.ndarray] = []
    buffer_frame_ids: list[int] = []

    # 카메라에서 실제로 추출된 frame 번호를 그대로 유지한다.
    # idle 동안 전송하지 않은 frame도 번호가 증가하므로,
    # AI가 반환하는 frame_id는 실제 카메라 frame 번호와 일치한다.

    prev_kp: np.ndarray | None = None

    motion_active = False
    start_count = 0
    end_count = 0
    cooldown = 0
    frames_since_infer = 0
    stability_history: list[tuple[str, float, int, bool]] = []
    blocked_gloss: str | None = None

    infer_counter = 0

    def motion_score(
        prev: np.ndarray | None,
        curr: np.ndarray,
    ) -> float:
        if prev is None:
            return 0.0

        a = np.asarray(prev, dtype=np.float32)
        b = np.asarray(curr, dtype=np.float32)

        if a.shape != b.shape:
            return 0.0

        return float(
            np.mean(np.abs(a - b))
        )

    def buffer_has_real_motion(frames: list[np.ndarray]) -> bool:
        """buffer 구간 전체에 실제 움직임이 있는지 확인한다.

        front_still이나 프레임 단위 motion_score(연속 두 프레임 diff)는
        still 필드가 오지 않거나(현재 still=None으로 계속 오고 있음) 서버
        자체 threshold 캘리브레이션이 안 맞으면 무력화된다. 이 검사는 그와
        무관하게, 지금 디코딩하려는 buffer 구간 자체의 keypoint 분산을
        직접 봐서 "이 구간에 정말 신호가 있는가"를 판단하는 독립적인
        안전장치다. 분산이 낮으면(=거의 정지/잡음) 애초에 forward pass도
        beam도 돌리지 않고 추론 자체를 건너뛴다 — CPU도 아끼고, 정지
        구간에서 엉뚱한 gloss가 확정되는 것도 원천 차단한다.
        """
        if len(frames) < 2:
            return False
        arr = np.asarray(frames, dtype=np.float32)
        # 프레임 간 diff의 표준편차: 손이 실제로 움직이는 구간은 크고,
        # 정지 상태의 센서 잡음은 작게 나온다.
        diffs = np.diff(arr, axis=0)
        return float(np.std(diffs)) >= MIN_BUFFER_MOTION_STD

    def forward_log_probs(
        frames: list[np.ndarray],
    ) -> np.ndarray:
        enhanced = make_model_features(
            np.asarray(frames, dtype=np.float32)
        )

        tensor = torch.from_numpy(
            enhanced
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = sign_model(tensor)
            log_probs = (
                F.log_softmax(logits, dim=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        # CTC 시간축과 입력 frame 축이 1:1인지 먼저 검증한다.
        assert log_probs.shape[0] == len(frames), (
            "CTC temporal dimension mismatch: "
            f"model_T={log_probs.shape[0]}, input_T={len(frames)}"
        )
        return log_probs

    def clear_motion_state() -> None:
        nonlocal motion_active, start_count, end_count, cooldown
        nonlocal frames_since_infer, stability_history, blocked_gloss
        motion_active = False
        start_count = 0
        end_count = 0
        cooldown = POST_COMMIT_COOLDOWN_FRAMES
        frames_since_infer = 0
        stability_history = []
        blocked_gloss = None

    def commit_from_buffer(
        greedy_preds: list[tuple[str, float, int, bool]],
        buffer_frame_ids_snapshot: list[int],
        beam_result: tuple[str, float, int] | None = None,
        force: bool = False,
        reason: str = "stable-k",
    ) -> tuple[str, float, int] | None:
        """이미 계산된 결과로 첫 token을 확정한다. force는 안정성 요건만 면제한다."""
        nonlocal buffer, buffer_frame_ids, stability_history, blocked_gloss

        if len(buffer) < MIN_COLLECT_FRAMES or not greedy_preds:
            return None

        first_gloss, first_prob, first_end, _ = greedy_preds[0]

        # force 경로에서도 중복 gloss는 절대 다시 확정하지 않는다.
        if blocked_gloss is not None and first_gloss == blocked_gloss:
            return None

        if not force:
            if first_prob < stability_prob_threshold:
                return None
            final_gloss, final_prob = first_gloss, first_prob
        else:
            # force는 confidence/stable 요건만 면제. beam은 보조값일 뿐이다.
            if beam_result is not None and beam_result[0] == first_gloss:
                final_gloss, final_prob = beam_result[0], beam_result[1]
            else:
                final_gloss, final_prob = first_gloss, first_prob

        if final_prob < TOKEN_CONFIDENCE_THRESHOLD:
            return None
        if not (0 <= first_end < len(buffer_frame_ids_snapshot)):
            print("[SIGN] invalid CTC end index", {"first_end": first_end, "buffer_len": len(buffer_frame_ids_snapshot)}, flush=True)
            return None

        global_end = int(buffer_frame_ids_snapshot[first_end])
        consume = min(len(buffer), first_end + margin + 1)
        if consume <= 0:
            return None

        buffer = buffer[consume:]
        buffer_frame_ids = buffer_frame_ids[consume:]
        blocked_gloss = final_gloss
        stability_history = []

        print("[SIGN] commit", {
            "gloss": final_gloss,
            "confidence": round(float(final_prob), 4),
            "global_end": global_end,
            "reason": reason,
            "force": force,
            "remaining_buffer": len(buffer),
        }, flush=True)
        return final_gloss, float(final_prob), int(global_end)

    def force_flush(reason: str) -> None:
        """문장 종료 시 남은 buffer를 가능한 만큼 flush한다."""
        nonlocal buffer, buffer_frame_ids
        guard = 0
        while len(buffer) >= MIN_COLLECT_FRAMES and guard < 8:
            guard += 1
            log_probs = forward_log_probs(buffer)
            greedy_preds = [p for p in ctc_greedy_decode(log_probs, idx_to_gloss, blank_idx)
                            if not (REMOVE_UNK and p[0] == "<UNK>")]
            if not greedy_preds:
                break
            snapshot_ids = list(buffer_frame_ids)
            beam_preds = [p for p in ctc_beam_search_decode(log_probs, idx_to_gloss, blank_idx, BEAM_WIDTH)
                          if not (REMOVE_UNK and p[0] == "<UNK>")]
            committed = commit_from_buffer(
                greedy_preds, snapshot_ids,
                beam_result=beam_preds[0] if beam_preds else None,
                force=True, reason=reason,
            )
            if committed is None:
                break
            try:
                result_queue.put_nowait(committed)
            except Exception:
                pass
        clear_motion_state()

    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            except (
                EOFError,
                OSError,
            ):
                break

            if item is None:
                break

            # --------------------------------------------------------
            # control message
            # --------------------------------------------------------
            if isinstance(item, tuple):
                command = item[0]

                if command == "__boundary__":
                    # 직전 inference와 동일 buffer로 K를 가짜로 채우지 않는다.
                    if len(buffer) >= MIN_COLLECT_FRAMES and frames_since_infer >= stride:
                        try:
                            log_probs = forward_log_probs(buffer)
                            preds = [p for p in ctc_greedy_decode(log_probs, idx_to_gloss, blank_idx)
                                     if not (REMOVE_UNK and p[0] == "<UNK>")]
                            if preds:
                                snapshot_ids = list(buffer_frame_ids)
                                gloss, prob, _, closed = preds[0]
                                if (closed and prob >= stability_prob_threshold and
                                    not (blocked_gloss is not None and gloss == blocked_gloss)):
                                    beam_preds = [p for p in ctc_beam_search_decode(log_probs, idx_to_gloss, blank_idx, BEAM_WIDTH)
                                                  if not (REMOVE_UNK and p[0] == "<UNK>")]
                                    committed = commit_from_buffer(
                                        preds, snapshot_ids,
                                        beam_result=beam_preds[0] if beam_preds else None,
                                        force=False, reason="boundary-stable",
                                    )
                                    if committed:
                                        try: result_queue.put_nowait(committed)
                                        except Exception: pass
                                        frames_since_infer = 0
                        except Exception as exc:
                            print(f"[SIGN] boundary decode error: {exc}", flush=True)
                    continue

                if command == "__flush__":
                    force_flush("stream_end")
                    continue

                if command == "__reset__":
                    buffer = []
                    buffer_frame_ids = []
                    prev_kp = None
                    clear_motion_state()
                    continue

                continue

            # --------------------------------------------------------
            # 실제 keypoint frame
            # --------------------------------------------------------
            if not isinstance(item, dict):
                continue

            frame_id = item.get("frame_id")
            keypoints = item.get("keypoints")
            # 프론트가 이미 calibration된 기준으로 계산해 보낸 정지 여부.
            # 값이 있으면 서버 자체 motion_score보다 이걸 신뢰한다.
            # 필드가 없는(구버전) 클라이언트는 None이 되어 기존 로직으로 폴백한다.
            front_still = item.get("still")

            if not isinstance(frame_id, int):
                continue

            kp = np.asarray(
                keypoints,
                dtype=np.float32,
            )

            if kp.ndim != 1 or kp.shape[0] != 261:
                continue

            # AI worker가 카메라보다 크게 뒤처지면 오래된 buffer를 버리고
            # 최신 프레임 묶음으로 즉시 따라잡는다.
            try:
                newest_received = int(latest_frame_id.value)
            except Exception:
                newest_received = frame_id

            if newest_received - frame_id >= int(cfg["lag_threshold"]):
                recent_items = [(frame_id, kp)]
                try:
                    while True:
                        newer = frame_queue.get_nowait()
                        if isinstance(newer, dict) and isinstance(newer.get("frame_id"), int):
                            newer_kp = np.asarray(newer.get("keypoints"), dtype=np.float32)
                            if newer_kp.ndim == 1 and newer_kp.shape[0] == 261:
                                recent_items.append((int(newer["frame_id"]), newer_kp))
                except (queue.Empty, EOFError, OSError):
                    pass

                recent_items = recent_items[-int(cfg["fast_forward_frames"]):]
                buffer = [x[1] for x in recent_items]
                buffer_frame_ids = [x[0] for x in recent_items]
                frame_id = buffer_frame_ids[-1]
                kp = buffer[-1]
                prev_kp = None
                motion_active = True
                start_count = 0
                end_count = 0
                frames_since_infer = 0
                stability_history = []
                blocked_gloss = None
                print(
                    "[SIGN] fast-forward",
                    {
                        "latest_camera_frame": newest_received,
                        "worker_frame": frame_id,
                        "gap": newest_received - frame_id,
                        "kept_frames": len(buffer),
                    },
                    flush=True,
                )

            current_motion = motion_score(
                prev_kp,
                kp,
            )

            prev_kp = kp.copy()

            # --------------------------------------------------------
            # IDLE
            # --------------------------------------------------------
            if not motion_active:
                if cooldown > 0:
                    cooldown -= 1
                    start_count = 0
                    continue

                # front_still이 있으면(프론트가 이미 calibration한 판정)
                # 그걸 우선 사용하고, 없으면 서버 자체 motion_score로 폴백한다.
                if front_still is not None:
                    is_moving = not front_still
                else:
                    is_moving = current_motion >= MOTION_START_THRESHOLD

                if is_moving:
                    start_count += 1
                else:
                    start_count = 0

                if start_count >= MOTION_START_FRAMES:
                    motion_active = True
                    start_count = 0
                    end_count = 0
                    frames_since_infer = 0
                    stability_history = []
                    buffer.append(kp)
                    buffer_frame_ids.append(frame_id)

                continue

            # --------------------------------------------------------
            # COLLECTING
            # --------------------------------------------------------
            buffer.append(kp)
            buffer_frame_ids.append(frame_id)

            if len(buffer) > hard_cap:
                drop = len(buffer) - hard_cap
                buffer = buffer[drop:]
                buffer_frame_ids = buffer_frame_ids[drop:]

            if (
                front_still
                if front_still is not None
                else current_motion <= MOTION_END_THRESHOLD
            ):
                end_count += 1
            else:
                end_count = 0

            frames_since_infer += 1

            enough = len(buffer) >= min_frames
            ended = end_count >= MOTION_END_FRAMES
            timeout = len(buffer) >= max_wait

            should_infer = (
                enough and (
                    frames_since_infer >= stride
                    or ended
                    or timeout
                )
            )

            if not should_infer:
                continue

            # front_still/motion 임계값과 무관한 독립 안전장치: 지금
            # buffer 구간 자체에 실질적인 움직임이 없으면(=정지/잡음)
            # 추론을 아예 건너뛴다. 이게 없으면 "정지 상태에서도 계속
            # 엉뚱한 gloss가 확정되는" 문제가 남는다.
            has_motion = buffer_has_real_motion(buffer)
            if infer_counter % 20 == 0:
                # 캘리브레이션용: MIN_BUFFER_MOTION_STD가 실제 데이터와
                # 맞는지 확인하려면 이 로그의 std 값 분포를 보고 조정한다.
                arr_dbg = np.asarray(buffer, dtype=np.float32)
                std_dbg = float(np.std(np.diff(arr_dbg, axis=0))) if len(buffer) >= 2 else 0.0
                print(
                    f"[MOTION] buffer_std={std_dbg:.5f} threshold={MIN_BUFFER_MOTION_STD} "
                    f"has_motion={has_motion} buffer_len={len(buffer)}",
                    flush=True,
                )

            if not has_motion:
                if ended:
                    # 진짜 motion 정지인데 buffer 자체에 실질적 신호가
                    # 없었다는 뜻 — 잡음/정지 구간이었으므로 디코딩 없이
                    # 그냥 IDLE로 복귀한다. 여기서 force_flush를 부르면
                    # 잡음을 단어로 오확정할 위험이 있다.
                    clear_motion_state()
                # can_commit/timeout 판정 자체가 무의미하므로 건너뛴다.
                continue

            infer_counter += 1

            log_probs = forward_log_probs(
                buffer
            )

            greedy_preds = ctc_greedy_decode(
                log_probs,
                idx_to_gloss,
                blank_idx,
            )

            greedy_preds = [
                pred
                for pred in greedy_preds
                if not (
                    REMOVE_UNK
                    and pred[0] == "<UNK>"
                )
            ]

            if not greedy_preds:
                if timeout or ended:
                    clear_motion_state()
                continue

            frames_since_infer = 0

            first_gloss, first_prob, first_end, first_closed = greedy_preds[0]
            snapshot_ids = list(buffer_frame_ids)

            if not (0 <= first_end < len(snapshot_ids)):
                print("[SIGN] invalid CTC end index", {"first_end": first_end, "buffer_len": len(snapshot_ids)}, flush=True)
                stability_history = []
                continue

            # 반드시 global camera frame id로 history를 저장한다.
            global_end = int(snapshot_ids[first_end])

            if (first_prob >= stability_prob_threshold and
                not (blocked_gloss is not None and first_gloss == blocked_gloss)):
                stability_history.append(
                    (first_gloss, float(first_prob), global_end, bool(first_closed))
                )
                if len(stability_history) > stability_k:
                    stability_history = stability_history[-stability_k:]
            else:
                stability_history = []

            stable = False
            if len(stability_history) >= stability_k:
                glosses = [x[0] for x in stability_history]
                probs = [x[1] for x in stability_history]
                ends = [x[2] for x in stability_history]
                stable = (
                    len(set(glosses)) == 1
                    and all(p >= stability_prob_threshold for p in probs)
                    and max(ends) - min(ends) <= stability_end_tol
                )

            # Beam은 일반 확정 gate가 아니다. 실제로 쓰이는 곳은 timeout
            # 강제 확정 시 confidence 보정뿐이다(stable-k는 안 쓰고, ended는
            # force_flush가 자체적으로 계산한다). 매 추론마다 돌리면 순수
            # Python 구현인 beam search가 CPU를 크게 잡아먹어 실시간 처리를
            # 못 따라가고 lag가 누적되므로, 꼭 필요한 timeout 시점에만 실행한다.
            beam_result = None
            if timeout and first_closed:
                beam_preds = [p for p in ctc_beam_search_decode(log_probs, idx_to_gloss, blank_idx, BEAM_WIDTH)
                              if not (REMOVE_UNK and p[0] == "<UNK>")]
                if beam_preds:
                    beam_result = beam_preds[0]

            if debug_queue is not None:
                try:
                    while True: debug_queue.get_nowait()
                except Exception: pass
                try:
                    debug_queue.put_nowait({
                        "greedy": first_gloss,
                        "greedy_prob": float(first_prob),
                        "end": int(first_end),
                        "global_end": global_end,
                        "closed": bool(first_closed),
                        "beam": beam_result[0] if beam_result else None,
                        "beam_score": float(beam_result[1]) if beam_result else None,
                        "stable_k": len(stability_history),
                        "stable": bool(stable),
                        "stability_threshold": float(stability_prob_threshold),
                    })
                except Exception: pass

            can_commit = (
                stable and
                first_gloss != (blocked_gloss or "") and
                len(buffer) >= MIN_COLLECT_FRAMES
            )

            if ended:
                # 진짜 motion 정지 시점. 새 프레임이 더 안 들어올 수 있으므로
                # 지금 이 순간 buffer에 남아있는 내용을 전부 소진해야 한다.
                # 토큰 하나만 확정하고 넘어가면, 그 뒤에 남은 완성된 단어가
                # 있어도 다음 motion이 시작되기 전까지는 영영 검사되지 않는다.
                force_flush("motion-end")
                continue

            if can_commit or timeout:
                committed = commit_from_buffer(
                    greedy_preds,
                    snapshot_ids,
                    beam_result=beam_result,
                    force=timeout,
                    reason=("stable-k" if can_commit else "timeout"),
                )
                if committed:
                    try: result_queue.put_nowait(committed)
                    except Exception: pass
                    if timeout:
                        # timeout은 진짜 경계가 아니라 buffer가 가득 차서
                        # 강제로 자른 것뿐이므로, blocked_gloss는 유지해서
                        # 같은 연속 동작이 반복 확정되지 않게 막는다.
                        # motion 상태/버퍼는 계속 유지하되 stability만 리셋.
                        stability_history = []

    except Exception as exc:
        print(
            f"[SIGN] worker fatal error: {exc}",
            flush=True,
        )

    finally:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================================
# 6. AsyncSignPredictor
# ============================================================================
class AsyncSignPredictor:
    """
    WebSocket 서버에서 사용하는 수지 predictor.

    모델 forward는 별도 multiprocessing worker에서 실행한다.
    """

    def __init__(
        self,
        stride: int = STREAM_STRIDE,
        min_frames: int = MIN_FRAMES,
        margin: int = STREAM_MARGIN,
        max_wait: int = STREAM_MAX_WAIT,
        hard_cap: int = STREAM_HARD_CAP,
        lag_threshold: int = 12,
        fast_forward_frames: int = 16,
    ) -> None:
        self.cfg = {
            "stride": stride,
            "min_frames": min_frames,
            "margin": margin,
            "max_wait": max_wait,
            "hard_cap": hard_cap,
            "stability_k": STREAM_STABILITY_K,
            "stability_prob_threshold": STABILITY_PROB_THRESHOLD,
            "stability_end_tol": STABILITY_END_TOL,
            "lag_threshold": lag_threshold,
            "fast_forward_frames": fast_forward_frames,
        }

        self._frame_queue = mlp.Queue(
            maxsize=FRAME_QUEUE_MAXSIZE
        )

        self._result_queue = mlp.Queue(
            maxsize=64
        )

        self._debug_queue = mlp.Queue(
            maxsize=1
        )

        self._stop_event = mlp.Event()
        self._latest_frame_id = mlp.Value("q", -1)
        self._process: mlp.Process | None = None

        self.dropped_frame_count = 0

    def start(self) -> None:
        if (
            self._process is not None
            and self._process.is_alive()
        ):
            return

        self._stop_event.clear()
        self._latest_frame_id.value = -1

        self._process = mlp.Process(
            target=_sign_worker_main,
            args=(
                self._frame_queue,
                self._result_queue,
                self._stop_event,
                self._latest_frame_id,
                self.cfg,
                self._debug_queue,
            ),
            daemon=True,
        )

        self._process.start()

        print(
            "[SIGN] prediction worker started",
            flush=True,
        )

    def stop(self) -> None:
        proc = self._process

        if proc is None:
            return

        self._stop_event.set()

        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            # A full frame queue must not prevent the shutdown sentinel from
            # reaching the worker. Drop one stale frame and try once more.
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(None)
            except (
                queue.Empty,
                queue.Full,
                EOFError,
                OSError,
            ):
                pass
        except (
            EOFError,
            OSError,
        ):
            pass

        proc.join(timeout=10)

        if proc.is_alive():
            print(
                "[SIGN] worker did not stop gracefully; terminating",
                flush=True,
            )
            proc.terminate()
            proc.join(timeout=3.0)

        for q in (
            self._frame_queue,
            self._result_queue,
            self._debug_queue,
        ):
            try:
                # Never wait for a multiprocessing queue feeder thread after
                # a worker was interrupted during camera disconnect cleanup.
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

        self._process = None

    def push_frame(
        self,
        keypoints: list[float] | np.ndarray,
        frame_id: int,
        still: bool | None = None,
    ) -> None:
        if (
            self._process is None
            or not self._process.is_alive()
        ):
            return

        kp = np.asarray(
            keypoints,
            dtype=np.float32,
        )

        try:
            self._latest_frame_id.value = int(frame_id)
        except Exception:
            pass

        item = {"frame_id": int(frame_id), "keypoints": kp, "still": still}

        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self.dropped_frame_count += 1
            except (
                queue.Empty,
                EOFError,
                OSError,
            ):
                pass

            try:
                self._frame_queue.put_nowait(item)
            except (
                queue.Full,
                EOFError,
                OSError,
            ):
                pass

    def latest_frame_id(self) -> int:
        try:
            return int(self._latest_frame_id.value)
        except Exception:
            return -1

    def word_boundary(self) -> None:
        """
        프론트의 word_boundary를 받되
        기존 main.py처럼 buffer를 잘라버리지 않는다.

        worker가 예전 streaming decoder 방식으로
        현재 buffer 문맥을 유지한다.
        """
        try:
            self._frame_queue.put_nowait(
                ("__boundary__",)
            )
        except queue.Full:
            # boundary는 프레임처럼 버리면 안 되므로
            # 큐가 꽉 찬 경우 오래된 프레임 하나를 버리고 넣는다.
            try:
                self._frame_queue.get_nowait()
            except (
                queue.Empty,
                EOFError,
                OSError,
            ):
                pass

            try:
                self._frame_queue.put_nowait(
                    ("__boundary__",)
                )
            except Exception:
                pass

    def flush(self) -> None:
        try:
            self._frame_queue.put_nowait(
                ("__flush__",)
            )
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except Exception:
                pass

            try:
                self._frame_queue.put_nowait(
                    ("__flush__",)
                )
            except Exception:
                pass

    def reset(self) -> None:
        try:
            self._latest_frame_id.value = -1
        except Exception:
            pass
        try:
            self._frame_queue.put_nowait(
                ("__reset__",)
            )
        except Exception:
            pass

    def poll_committed(
        self,
    ) -> list[tuple[str, float, int]]:
        results = []

        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            except (
                EOFError,
                OSError,
            ):
                break

            # 제어 신호(__ready__ 등)는 커밋 결과가 아니므로 걸러낸다.
            if (
                isinstance(item, tuple)
                and len(item) == 1
                and isinstance(item[0], str)
                and item[0].startswith("__")
            ):
                continue

            results.append(item)

        return results

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """모델 로드가 끝나 worker가 __ready__를 보낼 때까지 대기한다.

        서버 startup에서 한 번만 호출한다(블로킹이므로 to_thread로 감쌀 것).
        """
        import time

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                item = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return False

            if item == ("__ready__",):
                return True
            # __ready__보다 먼저 다른 게 올 일은 없지만, 방어적으로 버린다.

        return False

    def poll_last_raw_preds(self) -> list:
        try:
            return self._debug_queue.get_nowait()
        except queue.Empty:
            return []
        except (
            EOFError,
            OSError,
        ):
            return []

    def approx_backlog(self) -> int:
        try:
            return self._frame_queue.qsize()
        except (
            NotImplementedError,
            OSError,
        ):
            return -1


# ============================================================================
# 6-1. 비수지(NMS) — 모델 / 별도 프로세스 worker / predictor
#
# 수지 worker와 완전히 분리된 두 번째 프로세스에서 실행한다. 같은 프로세스에
# 넣으면 NMS Transformer 추론이 수지 추론 지연에 그대로 더해지기 때문이다.
# NMS는 보조 기능이므로 로드/추론이 실패해도 수지 인식은 계속 동작해야 한다.
# ============================================================================

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
        return x + self.pe[:, : x.size(1)]


class NMSClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_labels, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.dropout(self.input_proj(x))
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)


def load_nms_model(device: torch.device) -> NMSClassifier:
    ckpt = torch.load(NMS_CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = NMSClassifier(
        input_dim=cfg["input_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        num_labels=len(NMS_KEYS),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"[NMS] model loaded: epoch {ckpt.get('epoch')}, "
        f"macro_F1 {float(ckpt.get('macro_f1', 0.0)):.4f}, device={device}",
        flush=True,
    )
    return model


@torch.no_grad()
def infer_nms(model: NMSClassifier, device: torch.device, feat_window) -> np.ndarray:
    seq = gaussian_filter1d(np.array(feat_window, dtype=np.float32), sigma=1.0, axis=0)
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs[-1]


def _build_face_landmarks(face_xyz: np.ndarray):
    """브라우저에서 받은 (468, 3) 배열을 FaceMesh 결과(NormalizedLandmarkList)와
    같은 인터페이스로 감싼다. extract_nonmanual은 기존 스크립트와 동일한 코드를
    그대로 쓰므로, 학습 때와 같은 특징 추출 경로가 유지된다."""
    if _landmark_pb2 is not None:
        landmark_list = _landmark_pb2.NormalizedLandmarkList()
        for x, y, z in face_xyz.tolist():
            landmark_list.landmark.add(x=x, y=y, z=z)
        return landmark_list

    ns = types.SimpleNamespace
    return ns(landmark=[ns(x=x, y=y, z=z) for x, y, z in face_xyz.tolist()])


def _nms_frame_shape(face_w: Any, face_h: Any) -> tuple[int, int, int]:
    """기존 스크립트의 resize_like_preprocess(가로 640 고정)와 같은 shape를 만든다.
    extract_nonmanual이 shape로 종횡비를 반영하므로 원본 비율이 중요하다."""
    try:
        w = float(face_w)
        h = float(face_h)
        if w > 0 and h > 0:
            return (int(640 * h / w), 640, 3)
    except (TypeError, ValueError):
        pass
    return (480, 640, 3)


def _put_latest(q: Any, item: Any) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass


def _nms_worker_main(frame_queue: Any, result_queue: Any, stop_event: Any, cfg: dict[str, Any]) -> None:
    try:
        torch.set_num_threads(int(cfg.get("threads", 1)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[NMS WORKER] before load_nms_model", flush=True)
    try:
        model = load_nms_model(device)
    except Exception as exc:  # noqa: BLE001
        print(f"[NMS WORKER] model load failed: {exc!r}", flush=True)
        try:
            result_queue.put_nowait(("__failed__", repr(exc)))
        except Exception:
            pass
        return

    try:
        result_queue.put_nowait(("__ready__",))
    except Exception:
        pass

    window = int(cfg["window"])
    min_frames = int(cfg["min_frames"])
    stride = int(cfg["stride"])
    alpha = float(cfg["alpha"])

    feat_buffer: deque = deque(maxlen=window)
    smooth = np.zeros(len(NMS_KEYS), dtype=np.float32)
    prev_face = None
    count = 0
    reported_feature_error = False

    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break

            if item is None:
                break

            if len(item) == 1:  # ("__reset__",)
                feat_buffer.clear()
                smooth = np.zeros(len(NMS_KEYS), dtype=np.float32)
                prev_face = None
                count = 0
                continue

            frame_id, face_xyz, face_w, face_h = item

            # 이번 프레임에서 얼굴이 안 잡히면 직전 얼굴을 재사용한다(기존 스크립트와 동일).
            if face_xyz is not None:
                face_landmarks = _build_face_landmarks(face_xyz)
                prev_face = face_landmarks
            else:
                face_landmarks = prev_face

            if face_landmarks is None:
                continue

            try:
                feat = extract_nonmanual(face_landmarks, _nms_frame_shape(face_w, face_h))
            except Exception as exc:  # noqa: BLE001
                if not reported_feature_error:
                    print(f"[NMS WORKER] extract_nonmanual failed: {exc!r}", flush=True)
                    reported_feature_error = True
                continue

            feat_buffer.append(feat)
            count += 1

            if len(feat_buffer) < min_frames or count % stride != 0:
                continue

            try:
                probs = infer_nms(model, device, list(feat_buffer))
            except Exception as exc:  # noqa: BLE001
                print(f"[NMS WORKER] inference failed: {exc!r}", flush=True)
                continue

            smooth = alpha * probs + (1 - alpha) * smooth
            _put_latest(
                result_queue,
                {
                    "frame_id": int(frame_id),
                    "probs": [round(float(p), 4) for p in probs],
                    "smoothed": [round(float(p), 4) for p in smooth],
                },
            )
    finally:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


class AsyncNmsPredictor:
    """수지 predictor와 같은 구조의 비수지 predictor. 결과는 항상 최신 1건만 의미가 있다."""

    def __init__(self) -> None:
        self.cfg = {
            "threads": max(1, (os.cpu_count() or 4) // 4),
            "window": NMS_WINDOW_SIZE,
            "min_frames": NMS_MIN_FRAMES,
            "stride": NMS_INFER_STRIDE,
            "alpha": NMS_SMOOTH_ALPHA,
        }
        self._frame_queue = mlp.Queue(maxsize=NMS_QUEUE_MAXSIZE)
        self._result_queue = mlp.Queue(maxsize=8)
        self._stop_event = mlp.Event()
        self._process: mlp.Process | None = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_event.clear()
        self._process = mlp.Process(
            target=_nms_worker_main,
            args=(self._frame_queue, self._result_queue, self._stop_event, self.cfg),
            daemon=True,
        )
        self._process.start()
        print("[NMS] worker started", flush=True)

    def stop(self) -> None:
        proc = self._process
        if proc is None:
            return
        self._stop_event.set()
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3.0)
        for q in (self._frame_queue, self._result_queue):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        self._process = None

    def push(self, frame_id: int, face_flat: list[float], face_w: Any, face_h: Any) -> None:
        if self._process is None or not self._process.is_alive():
            return
        face_xyz = np.asarray(face_flat, dtype=np.float32).reshape(-1, 3)
        # 가득 차면 가장 오래된 프레임을 버리고 최신으로 따라잡는다(실시간성 우선).
        _put_latest(self._frame_queue, (int(frame_id), face_xyz, face_w, face_h))

    def poll_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            if isinstance(item, dict):
                latest = item
        return latest

    def reset(self) -> None:
        while True:
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
        try:
            self._frame_queue.put_nowait(("__reset__",))
        except Exception:
            pass

    def wait_ready(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return False
            if item == ("__ready__",):
                return True
            if isinstance(item, tuple) and item and item[0] == "__failed__":
                print(f"[NMS] worker reported failure: {item[1:]}", flush=True)
                return False
        return False


def _new_nms_timeline() -> dict[str, Any]:
    return {
        "first_frame": None,
        "last_t": 0.0,
        "open": {key: None for key in NMS_KEYS},
        "peak": {key: 0.0 for key in NMS_KEYS},
        "segments": [],
    }


def _update_nms_timeline(state: dict[str, Any], frame_id: int, smoothed: list[float]) -> list[bool]:
    """smoothed 확률이 키별 임계값을 넘는 구간을 세션 상대 시간(초)으로 기록한다.
    시간축은 카메라 frame_id 기준이라 전송이 중단된 정지 구간도 실제 시간으로 반영된다.
    first_frame은 세션 첫 프레임(frame 핸들러에서 설정)이라 단어 시간축과 원점이 같다."""
    if state["first_frame"] is None:
        state["first_frame"] = frame_id
    t = max(0.0, (frame_id - state["first_frame"]) / CAMERA_FPS)

    flags: list[bool] = []
    for i, key in enumerate(NMS_KEYS):
        value = float(smoothed[i])
        active = value >= NMS_KEY_THRESHOLD[key]
        start = state["open"][key]

        if active:
            state["peak"][key] = max(state["peak"][key], value)
            if start is None:
                state["open"][key] = t
        elif start is not None:
            state["segments"].append(
                {
                    "key": key,
                    "start": round(start, 2),
                    "end": round(t, 2),
                    "peak": round(state["peak"][key], 2),
                }
            )
            state["open"][key] = None
            state["peak"][key] = 0.0

        flags.append(bool(active))

    state["last_t"] = t
    return flags


def _close_nms_timeline(state: dict[str, Any]) -> list[dict[str, Any]]:
    for key in NMS_KEYS:
        start = state["open"][key]
        if start is not None:
            state["segments"].append(
                {
                    "key": key,
                    "start": round(start, 2),
                    "end": round(state["last_t"], 2),
                    "peak": round(state["peak"][key], 2),
                }
            )
            state["open"][key] = None
            state["peak"][key] = 0.0
    return sorted(state["segments"], key=lambda s: (s["start"], s["key"]))


def _recorded_time(idx: int, frame_ids: list[int], origin: int) -> float:
    """녹화 시퀀스의 프레임 인덱스를 세션 상대 시간(초)으로 바꾼다."""
    if not frame_ids:
        return 0.0
    idx = min(max(int(idx), 0), len(frame_ids) - 1)
    return max(0.0, (frame_ids[idx] - origin) / CAMERA_FPS)


def _nms_label(key: str) -> str:
    label = NMS_KEY_LABELS.get(key, "")
    return f"{key}({label})" if label else key


def build_nms_context(
    words_timed: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    duration_sec: float,
) -> dict[str, Any]:
    """녹화 전체의 비수지 타임라인을 단어 구간에 맞춰 정렬하고 LLM에 넘길 형태로 만든다.

    반환 dict
      words    : 단어별 {gloss, start, end, nms:[{key,start,end,peak}]}
      segments : LLM에 전달할 비수지 구간(짧은 잡음 제외)
      unaligned: 어느 단어와도 겹치지 않은 비수지 구간
      text     : 프롬프트에 그대로 붙일 수 있는 한국어 블록
    """
    segs = [s for s in segments if s["end"] - s["start"] >= NMS_MIN_SEGMENT_SEC]
    tol = NMS_ALIGN_TOLERANCE_SEC

    aligned_words = [{**w, "nms": []} for w in words_timed]
    unaligned: list[dict[str, Any]] = []

    for seg in segs:
        hit = False
        for w in aligned_words:
            if seg["end"] > w["start"] - tol and seg["start"] < w["end"] + tol:
                w["nms"].append(seg)
                hit = True
        if not hit:
            unaligned.append(seg)

    def fmt(seg: dict[str, Any]) -> str:
        peak = seg.get("peak")
        strength = f", 강도 {peak:.2f}" if peak is not None else ""
        return f"{_nms_label(seg['key'])} {seg['start']:.2f}~{seg['end']:.2f}초{strength}"

    lines = [f"[비수지 타임라인] 수어 녹화 전체(총 {duration_sec:.2f}초) 기준"]
    if not segs:
        lines.append("감지된 비수지 신호 없음")
    else:
        for i, w in enumerate(aligned_words, 1):
            nms_text = " / ".join(fmt(s) for s in w["nms"]) if w["nms"] else "없음"
            lines.append(f"{i}. {w['gloss']} ({w['start']:.2f}~{w['end']:.2f}초) 비수지: {nms_text}")
        if unaligned:
            lines.append("단어와 겹치지 않는 비수지: " + " / ".join(fmt(s) for s in unaligned))

    return {
        "duration_sec": round(duration_sec, 2),
        "words": aligned_words,
        "segments": segs,
        "unaligned": unaligned,
        "text": "\n".join(lines),
    }


# ============================================================================
# 7. 일반 HTTP/비스트리밍 InferenceEngine
# ============================================================================

class InferenceEngine:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model: SignLanguageModel | None = None
        self.idx_to_gloss: dict[int, str] = {}
        self.blank_idx = 0
        self.num_classes = 0

    def load(self) -> None:
        if self.model is not None:
            return

        (
            self.model,
            self.idx_to_gloss,
            self.blank_idx,
            self.num_classes,
        ) = load_sign_model(self.device)

    def build_model_features(
        self,
        raw_features: np.ndarray,
    ) -> np.ndarray:
        return make_model_features(
            raw_features
        )

    def decode_words_only(
        self,
        raw_features: np.ndarray,
        save_npy: bool = False,
    ) -> tuple[
        list[str],
        float,
        int,
        Path | None,
    ]:
        """
        단독 동영상/NPY 테스트용.

        여기서는 기존 HTTP predict 동작을 유지하기 위해
        전체 입력을 한 번에 greedy CTC한다.
        실시간 WebSocket은 AsyncSignPredictor를 사용한다.
        """
        if self.model is None:
            self.load()

        raw_features = np.asarray(
            raw_features,
            dtype=np.float32,
        )

        if len(raw_features) < MIN_FRAMES:
            print(
                f"[ai] not enough frames: "
                f"{len(raw_features)}",
                flush=True,
            )
            return [], 0.0, len(raw_features), None

        enhanced_features = self.build_model_features(
            raw_features
        )

        npy_path: Path | None = None

        if save_npy:
            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            npy_path = (
                PREDICTION_OUTPUT_DIR
                / f"prediction_{stamp}.npy"
            )

            np.save(
                npy_path,
                enhanced_features,
            )

        tensor = torch.tensor(
            enhanced_features,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=2,
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=2,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

            confidence = float(
                probabilities.max(
                    dim=2
                ).values.mean()
                .detach()
                .cpu()
                .item()
            )

        words = self._ctc_decode(
            predictions
        )

        print(
            "[ai] full decode",
            {
                "raw_frames": len(raw_features),
                "feature_shape": list(
                    enhanced_features.shape
                ),
                "words": words,
                "confidence": confidence,
            },
            flush=True,
        )

        return (
            words,
            confidence,
            len(raw_features),
            npy_path,
        )

    def predict_raw_features(
        self,
        raw_features: np.ndarray,
        save_npy: bool = False,
    ) -> dict[str, Any]:
        (
            words,
            confidence,
            frame_count,
            npy_path,
        ) = self.decode_words_only(
            raw_features,
            save_npy=save_npy,
        )

        result = self._response(
            words,
            frame_count=frame_count,
            confidence=confidence,
        )

        if npy_path:
            result["npy_path"] = str(
                npy_path
            )
            result["npy_saved"] = True

        return result

    def predict_video(
        self,
        video_path: Path,
    ) -> dict[str, Any]:
        raw_features = extract_video_keypoints(
            video_path
        )

        return self.predict_raw_features(
            raw_features,
            save_npy=True,
        )

    def _ctc_decode(
        self,
        predictions: np.ndarray,
    ) -> list[str]:
        decoded: list[str] = []

        previous_idx = -1

        for idx in predictions:
            int_idx = int(idx)

            if (
                int_idx != previous_idx
                and int_idx != self.blank_idx
            ):
                decoded.append(
                    self.idx_to_gloss.get(
                        int_idx,
                        "<UNK>",
                    )
                )

            previous_idx = int_idx

        return decoded

    def decode_recording(self, raw_features: np.ndarray) -> dict[str, Any]:
        """녹화해 둔 전체 시퀀스를 한 번에 디코딩한다(녹화 스크립트의 predict_sequence와 같은 방식).

        반환 tokens의 start/end는 입력 시퀀스(raw_features)의 프레임 인덱스다.
        """
        if self.model is None:
            self.load()

        raw = np.asarray(raw_features, dtype=np.float32)
        if len(raw) < MIN_FRAMES:
            return {"tokens": [], "confidence": 0.0}

        enhanced = self.build_model_features(raw)
        tensor = (
            torch.from_numpy(np.asarray(enhanced, dtype=np.float32)).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            logits = self.model(tensor)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        spans = ctc_greedy_spans(log_probs, self.idx_to_gloss, self.blank_idx)
        if FINAL_REMOVE_UNK:
            spans = [s for s in spans if s["gloss"] != "<UNK>"]

        # 모델 출력 길이가 입력과 다르면(다운샘플링) 인덱스를 입력 프레임 기준으로 환산한다.
        scale = len(raw) / max(1, log_probs.shape[0])
        last = len(raw) - 1
        tokens = [
            {
                "gloss": s["gloss"],
                "prob": float(s["prob"]),
                "start": min(last, int(round(s["start"] * scale))),
                "end": min(last, int(round(s["end"] * scale))),
            }
            for s in spans
        ]
        confidence = float(np.mean([t["prob"] for t in tokens])) if tokens else 0.0

        print(
            "[ai] recording decode",
            {
                "frames": len(raw),
                "model_T": int(log_probs.shape[0]),
                "tokens": [(t["gloss"], round(t["prob"], 2)) for t in tokens],
            },
            flush=True,
        )
        return {"tokens": tokens, "confidence": confidence}

    def _response(
        self,
        words: list[str],
        frame_count: int,
        confidence: float,
    ) -> dict[str, Any]:
        gloss_result = " ".join(words)

        return {
            "type": "prediction",
            "text": gloss_result,
            "words": words,
            "gloss_result": gloss_result,
            "confidence": confidence,
            "frame_count": frame_count,
            "device": str(self.device),
            "translation_candidates": [],
        }


engine = InferenceEngine()


# ============================================================================
# 8. LLM
# ============================================================================

def _translate_gloss_accepts_nms() -> bool:
    """llm_client.translate_gloss가 nms_context 인자를 받는지 확인한다(구버전이면 비수지는 건너뜀)."""
    try:
        params = inspect.signature(translate_gloss).parameters
    except (TypeError, ValueError):
        return False
    return "nms_context" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


TRANSLATE_ACCEPTS_NMS = _translate_gloss_accepts_nms()


async def enrich_with_llm(
    result: dict[str, Any],
) -> dict[str, Any]:
    words = result.get("words") or []

    gloss_result = (
        result.get("gloss_result")
        or " ".join(words)
    )

    confidence = float(
        result.get("confidence") or 0
    )

    llm_kwargs: dict[str, Any] = {}
    nms_context = result.get("nms_context")
    if nms_context:
        if TRANSLATE_ACCEPTS_NMS:
            llm_kwargs["nms_context"] = nms_context
        else:
            print(
                "[ai-llm] translate_gloss가 nms_context를 받지 않아 비수지는 LLM에 전달되지 않습니다 "
                "(llm_client.py 수정 필요)",
                flush=True,
            )

    llm_result = await translate_gloss(
        words,
        gloss_result,
        confidence,
        **llm_kwargs,
    )

    enriched = {
        **result
    }

    enriched["text"] = llm_result["text"]
    enriched["translated_text"] = (
        llm_result["text"]
    )
    enriched["translation_candidates"] = (
        llm_result["translation_candidates"]
    )
    enriched["llm_used"] = (
        llm_result["llm_used"]
    )
    enriched["llm_error"] = (
        llm_result["llm_error"]
    )

    print(
        "[ai-llm] result",
        {
            "llm_used": enriched["llm_used"],
            "llm_error": enriched["llm_error"],
            "gloss_result": gloss_result,
            "text": enriched["text"],
            "candidate_count": len(
                enriched["translation_candidates"]
            ),
        },
        flush=True,
    )

    return enriched


async def maybe_enrich_with_llm(
    result: dict[str, Any],
) -> dict[str, Any]:
    if SKIP_LLM:
        words = result.get("words") or []

        gloss_result = (
            result.get("gloss_result")
            or " ".join(words)
        )

        result["text"] = gloss_result
        result["translated_text"] = (
            gloss_result
        )
        result["translation_candidates"] = []
        result["llm_used"] = False
        result["llm_error"] = None

        print(
            "[ai-llm] skipped "
            "(SIGNLINK_SKIP_LLM=true)",
            {
                "gloss_result": gloss_result
            },
            flush=True,
        )

        return result

    return await enrich_with_llm(
        result
    )


# ============================================================================
# 9. Video / NPY
# ============================================================================

def extract_video_keypoints(
    video_path: Path,
) -> np.ndarray:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    features: list[np.ndarray] = []

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                break

            width = 640
            height = int(
                width
                * frame.shape[0]
                / frame.shape[1]
            )

            resized = cv2.resize(
                frame,
                (width, height),
            )

            results = holistic.process(
                cv2.cvtColor(
                    resized,
                    cv2.COLOR_BGR2RGB,
                )
            )

            features.append(
                extract_normalized_keypoints(
                    results
                )
            )

    cap.release()

    if not features:
        return np.empty(
            (0, 261),
            dtype=np.float32,
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def decode_data_url_base64(
    data_base64: str,
) -> bytes:
    if "," in data_base64:
        data_base64 = data_base64.split(
            ",",
            1,
        )[1]

    return base64.b64decode(
        data_base64
    )


def suffix_from_content_type(
    content_type: str | None,
) -> str:
    if content_type == "video/webm":
        return ".webm"

    if content_type == "video/mp4":
        return ".mp4"

    if content_type == "application/x-npy":
        return ".npy"

    return ".bin"


# ============================================================================
# 10. FastAPI lifecycle
# ============================================================================

# 세션(웹소켓 연결)마다 프로세스를 새로 띄우고 죽이던 기존 방식은
# 모델 로드 시간(수 초)이 세션 길이보다 길면 예측이 아예 안 나오는
# 문제를 일으킨다. 서버가 떠 있는 동안 단 하나의 worker 프로세스를
# 상시 유지하고, 세션 경계는 reset()으로만 처리한다.
sign_predictor: AsyncSignPredictor | None = None
sign_predictor_ready = False

# 비수지(NMS) worker도 같은 방식으로 상시 유지한다. 비활성/로드 실패 시 None 또는 not ready.
nms_predictor: AsyncNmsPredictor | None = None
nms_predictor_ready = False


@app.on_event("startup")
async def startup() -> None:
    # HTTP / upload prediction용 모델 로드
    await asyncio.to_thread(
        engine.load
    )

    print(
        f"[ai] SIGNLINK_SKIP_LLM={SKIP_LLM}",
        flush=True,
    )

    print(
        "[ai] streaming decoder:",
        {
            "stride": STREAM_STRIDE,
            "margin": STREAM_MARGIN,
            "max_wait": STREAM_MAX_WAIT,
            "hard_cap": STREAM_HARD_CAP,
            "beam_width": BEAM_WIDTH,
        },
        flush=True,
    )

    # 스트리밍 worker를 서버 시작 시 1회 띄우고 모델 로드가
    # 끝날 때까지 대기한다. 이후 세션들은 이 worker를 재사용한다.
    global sign_predictor, sign_predictor_ready, nms_predictor, nms_predictor_ready

    sign_predictor = AsyncSignPredictor()
    await asyncio.to_thread(sign_predictor.start)

    # 비수지 worker는 수지 worker와 병렬로 로드한다(서로 기다리지 않는다).
    if NMS_AVAILABLE:
        nms_predictor = AsyncNmsPredictor()
        await asyncio.to_thread(nms_predictor.start)
        asyncio.create_task(_wait_nms_ready_in_background())
    else:
        if not NMS_ENABLED:
            nms_reason = "SIGNLINK_NMS_ENABLED=false"
        elif not NMS_FEATURES_AVAILABLE:
            nms_reason = f"nonmanual_features import 실패: {NMS_FEATURES_ERROR}"
        else:
            nms_reason = f"체크포인트 없음: {NMS_CKPT_PATH}"
        print(f"[ai] NMS 비활성화 — {nms_reason}", flush=True)

    print("[ai] waiting for sign model to load...", flush=True)

    # Windows spawn 방식은 worker 프로세스를 띄울 때마다 torch/cv2/mediapipe를
    # 처음부터 다시 import해야 해서(콜드스타트) 30초를 넘기는 경우가 흔하다.
    # 60초까지는 startup을 블로킹해서 기다리되, 그래도 못 받으면 서버를
    # 계속 죽은 채로 두지 않고 백그라운드에서 계속 기다리다가 나중에
    # 로드가 끝나는 순간 자동으로 ready 상태가 되도록 한다. 이게 없으면
    # 최초 대기가 한 번 타임아웃된 뒤로는, worker가 그 뒤에 실제로 로드를
    # 마쳐도 그 신호를 읽어줄 코드가 없어 서버 재시작 전까지 영원히
    # "not ready" 상태에 머무르는 버그가 있었다.
    sign_predictor_ready = await asyncio.to_thread(
        sign_predictor.wait_ready,
        60.0,
    )

    if sign_predictor_ready:
        print("[ai] sign model ready", flush=True)
    else:
        print(
            "[ai] sign model still loading after 60s; will keep waiting in "
            "the background and become ready automatically once it "
            "finishes (no restart needed)",
            flush=True,
        )
        asyncio.create_task(_wait_ready_in_background())


async def _wait_ready_in_background() -> None:
    global sign_predictor_ready

    predictor = sign_predictor
    if predictor is None:
        return

    ready = await asyncio.to_thread(predictor.wait_ready, 600.0)
    sign_predictor_ready = ready

    if ready:
        print("[ai] sign model ready (background wait)", flush=True)
    else:
        print(
            "[ai] sign model failed to report ready within extended "
            "timeout; check the worker process for errors",
            flush=True,
        )


async def _wait_nms_ready_in_background() -> None:
    global nms_predictor_ready

    predictor = nms_predictor
    if predictor is None:
        return

    ready = await asyncio.to_thread(predictor.wait_ready, 600.0)
    nms_predictor_ready = ready
    if ready:
        print("[ai] NMS model ready", flush=True)
    else:
        print("[ai] NMS model failed to load; NMS disabled (sign recognition unaffected)", flush=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    global sign_predictor, sign_predictor_ready, nms_predictor, nms_predictor_ready

    if nms_predictor is not None:
        await asyncio.to_thread(nms_predictor.stop)
        nms_predictor = None
    nms_predictor_ready = False

    if sign_predictor is not None:
        await asyncio.to_thread(sign_predictor.stop)
        sign_predictor = None

    sign_predictor_ready = False


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": str(engine.device),
        "num_classes": engine.num_classes,
        "model_path": str(MODEL_PATH),
        "skip_llm": SKIP_LLM,
        "sign_model_ready": sign_predictor_ready,
        "nms_enabled": NMS_AVAILABLE,
        "nms_ready": nms_predictor_ready,
        "llm_model": os.getenv(
            "SIGNLINK_LLM_MODEL",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
        "llm_base_url": os.getenv(
            "SIGNLINK_LLM_BASE_URL",
            "",
        ),
        "llm_timeout_seconds": (
            LLM_TIMEOUT_SECONDS
        ),
    }


@app.post("/api/v1/predict")
async def predict_http(
    body: PredictRequest,
) -> dict[str, Any]:
    if body.features is not None:
        features = np.asarray(
            body.features,
            dtype=np.float32,
        )

        result = await asyncio.to_thread(
            engine.predict_raw_features,
            features,
        )

        return await maybe_enrich_with_llm(
            result
        )

    return {
        "type": "prediction",
        "text": "",
        "words": [],
        "gloss_result": "",
        "confidence": 0.0,
        "message": (
            "Send features or use "
            "/ws/predict for video prediction."
        ),
    }


# ============================================================================
# 11. WebSocket 실시간 수지 예측
# ============================================================================

@app.websocket("/ws/predict")
async def predict_websocket(
    websocket: WebSocket,
) -> None:
    await websocket.accept()

    predictor: AsyncSignPredictor | None = None

    accumulated_words: list[str] = []
    accumulated_confidences: list[float] = []
    accumulated_frame_count = 0
    nms_timeline = _new_nms_timeline()
    # 녹화: 세션(start~end) 동안 받은 모든 프레임. end에서 전체를 한 번에 디코딩한다.
    recorded_frames: list[np.ndarray] = []
    recorded_frame_ids: list[int] = []
    recording_truncated = False

    try:
        while True:
            print("[AI WS] before receive", flush=True)
            message = await websocket.receive_json()
            print(
                f"[AI WS] received: {message.get('type')}",
                flush=True,
            )
            message_type = message.get("type")

            # ------------------------------------------------------------
            # START
            # ------------------------------------------------------------
            if message_type == "start":
                print("[AI WS] START received", flush=True)

                # 세션마다 프로세스를 새로 띄우지 않는다. 서버 시작 시
                # 만들어 둔 상시 worker를 재사용하고, buffer/motion 상태만
                # reset()으로 지운다. 이렇게 하면 매 세션 모델 재로드로
                # 인해 세션이 끝날 때까지 예측이 하나도 안 나오는 문제가
                # 사라진다.
                predictor = sign_predictor

                if predictor is None or not sign_predictor_ready:
                    print(
                        "[AI WS] sign model not ready yet; "
                        "rejecting start",
                        flush=True,
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                "sign model is still loading, "
                                "try again shortly"
                            ),
                        }
                    )
                    continue

                predictor.reset()

                print("[AI WS] predictor reset for new session", flush=True)

                recorded_frames = []
                recorded_frame_ids = []
                recording_truncated = False
                nms_timeline = _new_nms_timeline()
                if nms_predictor is not None and nms_predictor_ready:
                    nms_predictor.reset()

                accumulated_words = []
                accumulated_confidences = []
                accumulated_frame_count = 0

                await websocket.send_json(
                    {
                        "type": "stream_started"
                    }
                )

                print(
                    "[AI WS] stream started "
                    "(decode-and-commit worker)",
                    flush=True,
                )

                continue

            # ------------------------------------------------------------
            # FRAME
            # ------------------------------------------------------------
            if message_type == "frame":
                keypoints = message.get(
                    "keypoints"
                )
                frame_id = message.get("frame_id")
                # 게이트웨이(translate.py/ai_client.py)가 보낸 정지 여부.
                # 이게 빠져있던 게 "still이 게이트웨이까지는 도달하는데
                # worker에서는 항상 None으로 보이는" 문제의 실제 원인이었다
                # — 여기서 안 읽으면 뒤에서 아무리 읽으려 해도 이미 없다.
                still = message.get("still")

                if not isinstance(
                    keypoints,
                    list,
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                "keypoints "
                                "must be a list"
                            ),
                        }
                    )
                    continue

                if not isinstance(frame_id, int) or frame_id < 0:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "frame_id must be a non-negative integer",
                        }
                    )
                    continue
                print(
                    f"[ai] received frame {frame_id} "
                    f"with {len(keypoints)} keypoints",
                )

                if len(keypoints) != 261:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                "expected 261 "
                                f"keypoints, got "
                                f"{len(keypoints)}"
                            ),
                        }
                    )
                    continue

                if predictor is None:
                    # 프론트가 start보다 frame을 먼저 보내는 경우에도
                    # 상시 worker를 그대로 재사용한다(새 프로세스를
                    # 만들지 않는다).
                    if sign_predictor is None or not sign_predictor_ready:
                        continue
                    predictor = sign_predictor
                    predictor.reset()

                predictor.push_frame(
                    keypoints,
                    frame_id,
                    still,
                )

                # 녹화: 실시간 단어 표시와 별개로, end에서 전체 디코딩할 수 있게 모든 프레임을 보관한다.
                if nms_timeline["first_frame"] is None:
                    nms_timeline["first_frame"] = frame_id  # 단어/비수지 시간축의 공통 원점
                if len(recorded_frames) < MAX_RECORD_FRAMES:
                    recorded_frames.append(np.asarray(keypoints, dtype=np.float32))
                    recorded_frame_ids.append(int(frame_id))
                elif not recording_truncated:
                    recording_truncated = True
                    print(f"[ai] recording buffer full ({MAX_RECORD_FRAMES} frames); later frames are not recorded", flush=True)

                # ------------------------------------------------------
                # 비수지(NMS): 수지와 독립된 별도 worker로 보낸다.
                # 어떤 이유로 실패해도 수지 인식 흐름은 절대 막지 않는다.
                # ------------------------------------------------------
                if nms_predictor is not None and nms_predictor_ready:
                    try:
                        face = message.get("face")
                        if isinstance(face, list) and len(face) == NMS_FACE_VALUES:
                            nms_predictor.push(
                                frame_id,
                                face,
                                message.get("face_w"),
                                message.get("face_h"),
                            )

                        nms_result = nms_predictor.poll_latest()
                        if nms_result is not None:
                            nms_frame_id = int(nms_result.get("frame_id", frame_id))
                            nms_smoothed = nms_result["smoothed"]
                            nms_active = _update_nms_timeline(
                                nms_timeline, nms_frame_id, nms_smoothed
                            )
                            await websocket.send_json(
                                {
                                    "type": "nms",
                                    "frame_id": nms_frame_id,
                                    "probs": nms_result["probs"],
                                    "smoothed": nms_smoothed,
                                    "active": nms_active,
                                }
                            )
                    except WebSocketDisconnect:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        print(f"[AI WS] NMS error (ignored): {exc!r}", flush=True)

                # worker에서 이미 확정된 단어를
                # non-blocking으로 꺼낸다.
                newly_committed = (
                    predictor.poll_committed()
                )

                for (
                    gloss,
                    confidence,
                    global_end,
                ) in newly_committed:
                    accumulated_words.append(
                        gloss
                    )

                    accumulated_confidences.append(
                        float(confidence)
                    )

                    accumulated_frame_count = max(
                        accumulated_frame_count,
                        int(global_end),
                    )

                    print(
                        "[ai] committed",
                        {
                            "gloss": gloss,
                            "confidence": round(
                                float(confidence),
                                4,
                            ),
                            "frame_id": global_end,
                            "global_end": global_end,
                            "latest_camera_frame": predictor.latest_frame_id(),
                            "lag_frames": max(0, predictor.latest_frame_id() - int(global_end)),
                            "accumulated": (
                                accumulated_words
                            ),
                        },
                        flush=True,
                    )

                # 확정 결과 + 현재 greedy 1위를 partial로 송출한다.
                # partial은 임시 표시용이며 accumulated_words에는 넣지 않는다.
                raw_partial = predictor.poll_last_raw_preds()
                partial_gloss = None
                partial_confidence = 0.0
                if isinstance(raw_partial, dict):
                    partial_gloss = raw_partial.get("greedy")
                    partial_confidence = float(raw_partial.get("greedy_prob") or 0.0)

                if accumulated_words or partial_gloss:
                    await websocket.send_json(
                        {
                            "type": "partial",
                            "words": accumulated_words,
                            "gloss_result": " ".join(accumulated_words),
                            "partial_gloss": partial_gloss,
                            "partial_confidence": partial_confidence,
                            "latest_camera_frame": predictor.latest_frame_id(),
                            "latest_predicted_frame": (
                                int(accumulated_frame_count)
                                if accumulated_frame_count > 0
                                else None
                            ),
                            "lag_frames": (
                                max(0, predictor.latest_frame_id() - int(accumulated_frame_count))
                                if accumulated_frame_count > 0
                                else 0
                            ),
                        }
                    )

                continue

            # ------------------------------------------------------------
            # WORD BOUNDARY
            # ------------------------------------------------------------
            if message_type == "word_boundary":
                if predictor is None:
                    continue

                # 중요:
                # 여기서 frame buffer를 decode하고 clear하지 않는다.
                # 예전 standalone streaming decoder처럼
                # worker가 현재 문맥을 계속 유지한다.
                predictor.word_boundary()

                # 이미 확정되어 있는 결과만 즉시 전달
                newly_committed = (
                    predictor.poll_committed()
                )

                for (
                    gloss,
                    confidence,
                    global_end,
                ) in newly_committed:
                    accumulated_words.append(
                        gloss
                    )

                    accumulated_confidences.append(
                        float(confidence)
                    )

                    accumulated_frame_count = max(
                        accumulated_frame_count,
                        int(global_end),
                    )

                print(
                    "[ai] word_boundary "
                    "(buffer preserved)",
                    {
                        "accumulated": (
                            accumulated_words
                        ),
                    },
                    flush=True,
                )

                await websocket.send_json(
                    {
                        "type": "partial",
                        "words": (
                            accumulated_words
                        ),
                        "gloss_result": (
                            " ".join(
                                accumulated_words
                            )
                        ),
                    }
                )

                continue

            # ------------------------------------------------------------
            # END
            # ------------------------------------------------------------
            if message_type == "end":
                print(
                    "[ai] stream ended. "
                    f"recorded_frames={len(recorded_frames)} live_words={accumulated_words}",
                    flush=True,
                )

                nms_active = nms_predictor is not None and nms_predictor_ready

                # (1) 비수지 worker가 아직 처리 중인 마지막 프레임 결과를 짧게 기다려 반영한다.
                if nms_active:
                    for _ in range(10):
                        await asyncio.sleep(0.02)
                        try:
                            tail = nms_predictor.poll_latest()
                            if tail is not None:
                                _update_nms_timeline(
                                    nms_timeline, int(tail.get("frame_id", 0)), tail["smoothed"]
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(f"[ai] NMS tail error (ignored): {exc!r}", flush=True)
                            break

                # (2) 녹화해 둔 전체 시퀀스를 한 번에 디코딩해 최종 gloss를 만든다.
                #     실시간 표시용 스트리밍 결과(accumulated_words)는 중간 미리보기일 뿐이다.
                tokens: list[dict[str, Any]] = []
                decode_confidence = 0.0
                final_source = "recording"

                if len(recorded_frames) >= MIN_FRAMES:
                    try:
                        decoded = await asyncio.to_thread(
                            engine.decode_recording, np.stack(recorded_frames)
                        )
                        tokens = decoded["tokens"]
                        decode_confidence = float(decoded["confidence"])
                    except Exception as exc:  # noqa: BLE001
                        # 전체 디코딩이 예외로 실패한 경우에만 실시간 결과로 대체한다.
                        print(f"[ai] recording decode failed, using live words: {exc!r}", flush=True)
                        final_source = "stream_fallback"

                origin = nms_timeline["first_frame"]
                words_timed: list[dict[str, Any]] = []
                if final_source == "recording" and origin is not None:
                    for tok in tokens:
                        words_timed.append(
                            {
                                "gloss": tok["gloss"],
                                "prob": round(tok["prob"], 3),
                                "start": round(_recorded_time(tok["start"], recorded_frame_ids, origin), 2),
                                "end": round(
                                    _recorded_time(tok["end"], recorded_frame_ids, origin) + 1.0 / CAMERA_FPS,
                                    2,
                                ),
                            }
                        )

                if final_source == "recording":
                    final_words = [w["gloss"] for w in words_timed]
                    final_confidence = decode_confidence
                else:
                    final_words = list(accumulated_words)
                    final_confidence = (
                        sum(accumulated_confidences) / len(accumulated_confidences)
                        if accumulated_confidences
                        else 0.0
                    )

                # (3) 녹화 전체의 비수지 타임라인을 단어에 맞춰 정렬해 LLM 입력으로 만든다.
                nms_segments = _close_nms_timeline(nms_timeline) if nms_active else []
                nms_context = None
                if nms_active and final_source == "recording" and words_timed:
                    duration = _recorded_time(len(recorded_frame_ids) - 1, recorded_frame_ids, origin)
                    nms_context = build_nms_context(words_timed, nms_segments, duration)
                    print("[ai] NMS -> LLM context:\n" + nms_context["text"], flush=True)

                result = engine._response(
                    final_words,
                    frame_count=len(recorded_frames),
                    confidence=final_confidence,
                )

                result["type"] = "translation"
                result["final_source"] = final_source
                result["live_words"] = list(accumulated_words)
                result["words_timed"] = [
                    {"gloss": w["gloss"], "start": w["start"], "end": w["end"]} for w in words_timed
                ]
                result["nms_segments"] = nms_segments
                if recording_truncated:
                    result["recording_truncated"] = True
                if nms_context is not None:
                    result["nms_context"] = nms_context

                result = await maybe_enrich_with_llm(result)

                # LLM 입력용 구조체는 클라이언트로 보낼 필요가 없다(프롬프트 블록만 남김).
                ctx = result.pop("nms_context", None)
                if ctx is not None:
                    result["nms_prompt"] = ctx["text"]

                await websocket.send_json(result)

                # 상시 worker는 죽이지 않는다. buffer/motion 상태만
                # 정리해 다음 세션이 깨끗하게 시작하도록 한다.
                if predictor is not None:
                    predictor.reset()
                    predictor = None

                accumulated_words = []
                accumulated_confidences = []
                accumulated_frame_count = 0

                nms_timeline = _new_nms_timeline()
                recorded_frames = []
                recorded_frame_ids = []
                recording_truncated = False
                if nms_predictor is not None and nms_predictor_ready:
                    nms_predictor.reset()

                continue

            # ------------------------------------------------------------
            # VIDEO
            # ------------------------------------------------------------
            if message_type == "predict_video":
                content_type = message.get(
                    "content_type"
                )

                payload = (
                    decode_data_url_base64(
                        message[
                            "data_base64"
                        ]
                    )
                )

                suffix = suffix_from_content_type(
                    content_type
                )

                temp_path: Path | None = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:
                        temp_file.write(payload)
                        temp_path = Path(
                            temp_file.name
                        )

                    result = (
                        await asyncio.to_thread(
                            engine.predict_video,
                            temp_path,
                        )
                    )

                    result = (
                        await maybe_enrich_with_llm(
                            result
                        )
                    )

                    await websocket.send_json(
                        result
                    )

                finally:
                    if temp_path is not None:
                        try:
                            temp_path.unlink(
                                missing_ok=True
                            )
                        except Exception:
                            pass

                continue

            # ------------------------------------------------------------
            # NPY
            # ------------------------------------------------------------
            if message_type == "predict_npy":
                payload = (
                    decode_data_url_base64(
                        message[
                            "data_base64"
                        ]
                    )
                )

                features = np.load(
                    io.BytesIO(payload)
                )

                result = (
                    await asyncio.to_thread(
                        engine.predict_raw_features,
                        features,
                        True,
                    )
                )

                result = (
                    await maybe_enrich_with_llm(
                        result
                    )
                )

                await websocket.send_json(
                    result
                )

                continue

            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Unknown message type: "
                        f"{message_type}"
                    ),
                }
            )

    except WebSocketDisconnect:
        return

    except Exception as exc:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
        except Exception:
            pass

    finally:
        # 연결이 끊기거나 예외가 나도 상시 worker 프로세스는 유지한다.
        # buffer/motion 상태만 정리해 다음 연결이 깨끗하게 시작하게 한다.
        if predictor is not None:
            predictor.reset()
        if nms_predictor is not None and nms_predictor_ready:
            nms_predictor.reset()


# ============================================================================
# 12. 실행
# ============================================================================

if __name__ == "__main__":
    mlp.freeze_support()

    # 직접 python main.py 실행할 경우 uvicorn 사용
    import uvicorn

    host = os.getenv(
        "SIGNLINK_AI_HOST",
        "0.0.0.0",
    )

    port = int(
        os.getenv(
            "SIGNLINK_AI_PORT",
            "8001",
        )
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
    )