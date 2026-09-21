from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import multiprocessing as mlp
import os
import queue
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# 0. 경로 / 모델 설정
# ============================================================================

# AI 서버를 어느 위치에서 실행하더라도 모델 파일을 찾을 수 있도록
# 현재 파일, ai_server/model, 프로젝트 루트/model 순서로 모델 경로를 탐색한다.
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


# ============================================================================
# 1. 공통 설정
# ============================================================================

HIDDEN_DIM = 512
# 너무 짧은 입력은 하나의 수어 동작으로 보기 어려우므로 추론하지 않는다.
MIN_FRAMES = 5
# 실제 추론에 사용한 783차원 특징 배열을 확인할 수 있도록 npy 파일로 보관한다.

# 예전 단독 실시간 테스트 코드와 동일
BEAM_WIDTH = 10
TOKEN_CONFIDENCE_THRESHOLD = 0.05
REMOVE_UNK = True

STREAM_STRIDE = 8
STREAM_MARGIN = 3
STREAM_MAX_WAIT = 75
STREAM_HARD_CAP = 90

FRAME_QUEUE_MAXSIZE = 24

# 예전 단독 테스트의 motion gate
MOTION_START_THRESHOLD = 0.004
MOTION_END_THRESHOLD = 0.002
MOTION_START_FRAMES = 3
MOTION_END_FRAMES = 8
MIN_COLLECT_FRAMES = 12
POST_COMMIT_COOLDOWN_FRAMES = 8
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
    """HTTP 예측 API가 받는 요청 형식."""

    features: list[list[float]] | None = None
    landmarks: dict[str, Any] | None = None


# ============================================================================
# 3. 수지 모델 / 사전
# ============================================================================

def build_inference_dict() -> dict[str, int]:
    """학습 사전과 글로스 통합 매핑으로 추론용 클래스 사전을 만든다."""
    with GLOSS_DICT_PATH.open("r", encoding="utf-8") as file:
        old_gloss_to_idx: dict[str, int] = json.load(file)

    # gloss_mapping.json이 있으면 서로 같은 의미로 정리한 글로스를 대표 글로스로 통합한다.
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
    # 같은 대표 글로스로 매핑되는 항목은 set을 이용해 한 번만 남긴다.
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
    """글로스 사전과 학습 가중치를 읽어 추론 모델을 준비한다."""

    gloss_to_idx = build_inference_dict()

    idx_to_gloss = {
        value: key
        for key, value in gloss_to_idx.items()
    }

    blank_idx = gloss_to_idx.get("<blank>", 0)
    num_classes = len(gloss_to_idx)

    # 체크포인트의 분류층 크기와 글로스 사전 크기가 다르면
    # 클래스 번호가 어긋나므로 서버 시작 단계에서 실패시킨다.
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
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sign_model, idx_to_gloss, blank_idx, _ = load_sign_model(device)

    stride = int(cfg["stride"])
    min_frames = int(cfg["min_frames"])
    margin = int(cfg["margin"])
    max_wait = int(cfg["max_wait"])
    hard_cap = int(cfg["hard_cap"])

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
            return (
                F.log_softmax(logits, dim=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

    def clear_motion_state() -> None:
        nonlocal motion_active
        nonlocal start_count
        nonlocal end_count
        nonlocal cooldown

        motion_active = False
        start_count = 0
        end_count = 0
        cooldown = POST_COMMIT_COOLDOWN_FRAMES

    def commit_from_buffer(
        beam_result: tuple[str, float, int] | None = None,
        force: bool = False,
        reason: str = "closed",
    ) -> tuple[str, float, int] | None:
        """현재 buffer의 첫 token을 확정한다.

        일반 확정은 메인 루프에서 이미 계산한 Beam 결과를 전달받는다.
        따라서 확정 과정에서 Beam Search를 다시 실행하지 않는다.
        """
        nonlocal buffer
        nonlocal buffer_frame_ids

        if len(buffer) < MIN_COLLECT_FRAMES:
            return None

        log_probs = forward_log_probs(buffer)
        greedy_preds = [
            pred
            for pred in ctc_greedy_decode(
                log_probs, idx_to_gloss, blank_idx
            )
            if not (REMOVE_UNK and pred[0] == "<UNK>")
        ]

        if not greedy_preds:
            return None

        first_gloss, first_prob, first_end, first_closed = greedy_preds[0]

        if not force:
            if not first_closed or beam_result is None:
                return None
            beam_gloss, beam_prob, _ = beam_result
            if first_gloss != beam_gloss:
                return None
            final_gloss, final_prob = beam_gloss, beam_prob
        else:
            if beam_result is not None:
                final_gloss, final_prob, _ = beam_result
            else:
                final_gloss, final_prob = first_gloss, first_prob

        if final_prob < TOKEN_CONFIDENCE_THRESHOLD:
            return None

        global_end = int(buffer_frame_ids[first_end])
        consume = min(len(buffer), first_end + margin + 1)
        if consume <= 0:
            return None

        buffer = buffer[consume:]
        buffer_frame_ids = buffer_frame_ids[consume:]

        print(
            "[SIGN] commit",
            {
                "gloss": final_gloss,
                "confidence": round(float(final_prob), 4),
                "global_end": global_end,
                "reason": reason,
                "force": force,
                "remaining_buffer": len(buffer),
            },
            flush=True,
        )

        return final_gloss, float(final_prob), int(global_end)

    def force_flush(
        reason: str,
    ) -> None:
        """
        문장 종료 시 남아 있는 buffer를
        가능한 만큼 commit한다.
        """
        nonlocal buffer
        nonlocal buffer_frame_ids

        # 예전 로직과 동일하게 한 번에 첫 token만 확정하고,
        # 남은 문맥은 반복해서 확인한다.
        guard = 0

        while (
            len(buffer) >= MIN_COLLECT_FRAMES
            and guard < 8
        ):
            guard += 1

            log_probs = forward_log_probs(buffer)

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
                break

            first_gloss, first_prob, first_end, first_closed = (
                greedy_preds[0]
            )

            # end 상황에서는 closed 여부와 무관하게
            # 마지막 남은 token도 확정할 수 있게 한다.
            beam_preds = ctc_beam_search_decode(
                log_probs,
                idx_to_gloss,
                blank_idx,
                BEAM_WIDTH,
            )

            beam_preds = [
                pred
                for pred in beam_preds
                if not (
                    REMOVE_UNK
                    and pred[0] == "<UNK>"
                )
            ]

            if beam_preds:
                final_gloss, final_prob, _ = beam_preds[0]
            else:
                final_gloss = first_gloss
                final_prob = first_prob

            if final_prob < TOKEN_CONFIDENCE_THRESHOLD:
                break

            global_end = int(buffer_frame_ids[first_end])

            consume = min(
                len(buffer),
                first_end + margin + 1,
            )

            if consume <= 0:
                break

            buffer = buffer[consume:]
            buffer_frame_ids = buffer_frame_ids[consume:]

            try:
                result_queue.put_nowait(
                    (
                        final_gloss,
                        float(final_prob),
                        int(global_end),
                    )
                )
            except Exception:
                pass

            print(
                "[SIGN] flush commit",
                {
                    "gloss": final_gloss,
                    "confidence": round(
                        float(final_prob),
                        4,
                    ),
                    "global_end": global_end,
                    "reason": reason,
                    "remaining_buffer": len(buffer),
                },
                flush=True,
            )

        clear_motion_state()

    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(
                    timeout=0.05
                )
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
                    # buffer는 유지한다. 이미 closed된 첫 token만
                    # Greedy + Beam agreement로 한 번 확인한다.
                    if len(buffer) >= MIN_COLLECT_FRAMES:
                        try:
                            log_probs = forward_log_probs(buffer)
                            preds = [
                                pred
                                for pred in ctc_greedy_decode(
                                    log_probs, idx_to_gloss, blank_idx
                                )
                                if not (REMOVE_UNK and pred[0] == "<UNK>")
                            ]

                            if preds:
                                gloss, _, _, closed = preds[0]
                                if closed:
                                    beam_preds = [
                                        pred
                                        for pred in ctc_beam_search_decode(
                                            log_probs,
                                            idx_to_gloss,
                                            blank_idx,
                                            BEAM_WIDTH,
                                        )
                                        if not (REMOVE_UNK and pred[0] == "<UNK>")
                                    ]
                                    beam_result = beam_preds[0] if beam_preds else None

                                    if (
                                        beam_result is not None
                                        and gloss == beam_result[0]
                                    ):
                                        committed = commit_from_buffer(
                                            beam_result=beam_result,
                                            reason="boundary-closed",
                                        )
                                        if committed:
                                            try:
                                                result_queue.put_nowait(committed)
                                            except Exception:
                                                pass
                        except Exception as exc:
                            print(
                                f"[SIGN] boundary decode error: {exc}",
                                flush=True,
                            )
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
                    continue

                if (
                    current_motion
                    >= MOTION_START_THRESHOLD
                ):
                    start_count += 1
                else:
                    start_count = 0

                if start_count >= MOTION_START_FRAMES:
                    motion_active = True
                    start_count = 0
                    end_count = 0

                    # 시작 프레임부터 수집
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
                current_motion
                <= MOTION_END_THRESHOLD
            ):
                end_count += 1
            else:
                end_count = 0

            enough = (
                len(buffer) >= min_frames
            )

            ended = (
                end_count >= MOTION_END_FRAMES
            )

            timeout = (
                len(buffer) >= max_wait
            )

            should_infer = (
                enough
                and (
                    len(buffer) == min_frames
                    or len(buffer) % stride == 0
                    or ended
                    or timeout
                )
            )

            if not should_infer:
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

            first_gloss, first_prob, first_end, first_closed = greedy_preds[0]

            # closed일 때만 Beam Search를 정확히 한 번 실행한다.
            # 이 결과를 debug와 commit 양쪽에서 재사용한다.
            beam_result = None
            beam_agree = False

            if first_closed:
                beam_preds = [
                    pred
                    for pred in ctc_beam_search_decode(
                        log_probs,
                        idx_to_gloss,
                        blank_idx,
                        BEAM_WIDTH,
                    )
                    if not (REMOVE_UNK and pred[0] == "<UNK>")
                ]
                if beam_preds:
                    beam_result = beam_preds[0]
                    beam_agree = (
                        first_gloss == beam_result[0]
                    )

            # 디버그 값은 모든 변수가 정의된 뒤 기록한다.
            if debug_queue is not None:
                try:
                    while True:
                        debug_queue.get_nowait()
                except Exception:
                    pass

                try:
                    debug_queue.put_nowait(
                        {
                            "greedy": first_gloss,
                            "greedy_prob": float(first_prob),
                            "end": int(first_end),
                            "closed": bool(first_closed),
                            "beam": (
                                beam_result[0]
                                if beam_result is not None
                                else None
                            ),
                            "beam_score": (
                                float(beam_result[1])
                                if beam_result is not None
                                else None
                            ),
                            "beam_agree": bool(beam_agree),
                        }
                    )
                except Exception:
                    pass

            # 일반 단어는 closed + Beam agreement일 때만 확정.
            # motion-end / timeout은 force 경로로 남은 token을 처리한다.
            can_commit = (
                first_closed
                and beam_result is not None
                and beam_agree
            )

            if can_commit or ended or timeout:
                committed = commit_from_buffer(
                    beam_result=beam_result,
                    force=(ended or timeout),
                    reason=(
                        "beam-agree"
                        if can_commit
                        else "motion-end"
                        if ended
                        else "timeout"
                    ),
                )

                if committed:
                    try:
                        result_queue.put_nowait(committed)
                    except Exception:
                        pass

                    # 일반 단어 commit은 같은 motion segment를 유지한다.
                    # IDLE 복귀는 실제 motion-end / timeout에서만 한다.
                    if ended or timeout:
                        clear_motion_state()

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
        lag_threshold: int = 45,
        fast_forward_frames: int = 16,
    ) -> None:
        self.cfg = {
            "stride": stride,
            "min_frames": min_frames,
            "margin": margin,
            "max_wait": max_wait,
            "hard_cap": hard_cap,
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

        item = {"frame_id": int(frame_id), "keypoints": kp}

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
                results.append(
                    self._result_queue.get_nowait()
                )
            except queue.Empty:
                break
            except (
                EOFError,
                OSError,
            ):
                break

        return results

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


# 서버 프로세스에서 모델 인스턴스를 한 번만 생성해 요청마다 재사용한다.
engine = InferenceEngine()


# ============================================================================
# 8. LLM
# ============================================================================

async def enrich_with_llm(
    result: dict[str, Any],
) -> dict[str, Any]:
    """인식한 글로스를 sLLM으로 자연스러운 한국어 문장으로 변환한다."""

    words = result.get("words") or []

    gloss_result = (
        result.get("gloss_result")
        or " ".join(words)
    )

    confidence = float(
        result.get("confidence") or 0
    )

    llm_result = await translate_gloss(
        words,
        gloss_result,
        confidence,
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
    """동영상 전체에서 MediaPipe Holistic 키포인트 [T, 261]을 만든다."""

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
    """데이터 URL 또는 순수 Base64 문자열을 디코딩한다."""

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
    """임시 파일에 사용할 확장자를 MIME 타입에서 선택한다."""

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

@app.on_event("startup")
async def startup() -> None:
    # 첫 요청의 모델 로딩 지연을 막기 위해 HTTP/upload 모델을 미리 로드한다.
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


@app.get("/health")
async def health() -> dict[str, Any]:
    """모델, 실행 장치와 LLM 연결 설정을 확인한다."""

    return {
        "ok": True,
        "device": str(engine.device),
        "num_classes": engine.num_classes,
        "model_path": str(MODEL_PATH),
        "skip_llm": SKIP_LLM,
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
    """완성된 키포인트 배열을 한 번에 받아 예측한다."""

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
    """실시간 프레임과 동영상/NPY 입력을 처리하는 WebSocket API."""

    await websocket.accept()

    predictor: AsyncSignPredictor | None = None

    accumulated_words: list[str] = []
    accumulated_confidences: list[float] = []
    accumulated_frame_count = 0

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            # ------------------------------------------------------------
            # START
            # ------------------------------------------------------------
            if message_type == "start":
                if predictor is not None:
                    predictor.stop()

                predictor = AsyncSignPredictor()
                predictor.start()

                accumulated_words = []
                accumulated_confidences = []
                accumulated_frame_count = 0

                await websocket.send_json(
                    {
                        "type": "stream_started"
                    }
                )

                print(
                    "[ai] stream started "
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
                    # 프론트가 start보다 frame을 먼저 보내는
                    # 경우에도 안전하게 시작
                    predictor = AsyncSignPredictor()
                    predictor.start()

                predictor.push_frame(
                    keypoints,
                    frame_id,
                )

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

                if newly_committed:
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
                            "latest_camera_frame": predictor.latest_frame_id(),
                            "latest_predicted_frame": (
                                int(accumulated_frame_count)
                                if accumulated_frame_count > 0
                                else None
                            ),
                            "lag_frames": (
                                max(
                                    0,
                                    predictor.latest_frame_id()
                                    - int(accumulated_frame_count),
                                )
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
                    f"accumulated so far: "
                    f"{accumulated_words}",
                    flush=True,
                )

                if predictor is not None:
                    # worker가 남은 buffer를 flush.
                    predictor.flush()

                    # multiprocessing queue에 들어간
                    # flush command가 처리될 시간을 짧게 양보.
                    for _ in range(20):
                        await asyncio.sleep(0.01)

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

                        if (
                            not newly_committed
                        ):
                            # 한 번 더 polling
                            await asyncio.sleep(
                                0.01
                            )

                avg_confidence = (
                    sum(
                        accumulated_confidences
                    )
                    / len(
                        accumulated_confidences
                    )
                    if accumulated_confidences
                    else 0.0
                )

                result = engine._response(
                    accumulated_words,
                    frame_count=(
                        accumulated_frame_count
                    ),
                    confidence=avg_confidence,
                )

                result["type"] = "translation"

                result = (
                    await maybe_enrich_with_llm(
                        result
                    )
                )

                await websocket.send_json(
                    result
                )

                if predictor is not None:
                    await asyncio.to_thread(
                        predictor.stop
                    )
                    predictor = None

                accumulated_words = []
                accumulated_confidences = []
                accumulated_frame_count = 0

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
        if predictor is not None:
            await asyncio.to_thread(
                predictor.stop
            )


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
