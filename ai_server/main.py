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

STREAM_STABILITY_K = 3
STABILITY_PROB_THRESHOLD = 0.50
STABILITY_END_TOL = 6

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
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
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

            # Beam은 일반 확정 gate가 아니다. debug/force flush 보조용으로만 사용한다.
            beam_result = None
            if first_closed:
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

            if can_commit or ended or timeout:
                committed = commit_from_buffer(
                    greedy_preds,
                    snapshot_ids,
                    beam_result=beam_result,
                    force=(ended or timeout),
                    reason=("stable-k" if can_commit else "motion-end" if ended else "timeout"),
                )
                if committed:
                    try: result_queue.put_nowait(committed)
                    except Exception: pass
                    # ended(진짜 motion 정지)는 새 세그먼트의 시작이므로
                    # blocked_gloss를 포함해 상태를 전부 초기화한다.
                    # timeout은 진짜 경계가 아니라 buffer가 가득 차서
                    # 강제로 자른 것뿐이므로, blocked_gloss는 유지해서
                    # 같은 연속 동작이 반복 확정되지 않게 막는다.
                    if ended:
                        clear_motion_state()
                    elif timeout:
                        # motion 상태/버퍼는 계속 유지하되 stability만 리셋
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


engine = InferenceEngine()


# ============================================================================
# 8. LLM
# ============================================================================

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
    global sign_predictor, sign_predictor_ready

    sign_predictor = AsyncSignPredictor()
    await asyncio.to_thread(sign_predictor.start)

    print("[ai] waiting for sign model to load...", flush=True)

    sign_predictor_ready = await asyncio.to_thread(
        sign_predictor.wait_ready,
        30.0,
    )

    if sign_predictor_ready:
        print("[ai] sign model ready", flush=True)
    else:
        print(
            "[ai] sign model did not report ready within timeout; "
            "streaming predictions may be unavailable until it finishes loading",
            flush=True,
        )


@app.on_event("shutdown")
async def shutdown() -> None:
    global sign_predictor, sign_predictor_ready

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

                # 상시 worker는 죽이지 않는다. buffer/motion 상태만
                # 정리해 다음 세션이 깨끗하게 시작하도록 한다.
                if predictor is not None:
                    predictor.reset()
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
        # 연결이 끊기거나 예외가 나도 상시 worker 프로세스는 유지한다.
        # buffer/motion 상태만 정리해 다음 연결이 깨끗하게 시작하게 한다.
        if predictor is not None:
            predictor.reset()


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