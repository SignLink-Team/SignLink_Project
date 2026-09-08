from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d

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

HIDDEN_DIM = 512
# 너무 짧은 입력은 하나의 수어 동작으로 보기 어려우므로 추론하지 않는다.
MIN_FRAMES = 5
# 실제 추론에 사용한 783차원 특징 배열을 확인할 수 있도록 npy 파일로 보관한다.
PREDICTION_OUTPUT_DIR = ROOT_DIR / "ai_server" / "prediction_outputs"
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SignLink AI Server", version="0.1.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class PredictRequest(BaseModel):
  """HTTP 예측 API가 받는 요청 형식.

  features는 [프레임 수, 특징 수] 형태이며, 현재 landmarks 필드는 향후 확장을 위해 남겨 두었다.
  """

  features: list[list[float]] | None = None
  landmarks: dict[str, Any] | None = None


class InferenceEngine:
  """키포인트 전처리, PyTorch 모델 추론, CTC 디코딩을 담당한다."""

  def __init__(self) -> None:
    # CUDA를 사용할 수 있으면 GPU, 그렇지 않으면 CPU에서 추론한다.
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model: SignLanguageModel | None = None
    self.idx_to_gloss: dict[int, str] = {}
    self.blank_idx = 0
    self.num_classes = 0

  def load(self) -> None:
    """글로스 사전과 학습된 가중치를 읽어 추론 모델을 준비한다."""

    gloss_to_idx = self._build_inference_dict()
    self.idx_to_gloss = {value: key for key, value in gloss_to_idx.items()}
    self.blank_idx = gloss_to_idx.get("<blank>", 0)
    self.num_classes = len(gloss_to_idx)

    # 체크포인트의 마지막 분류층 크기와 글로스 사전 크기가 다르면
    # 클래스 번호가 서로 어긋나므로 서버 시작 단계에서 바로 실패시킨다.
    checkpoint = torch.load(MODEL_PATH, map_location=self.device)
    checkpoint_classes = checkpoint["fc.weight"].shape[0]
    if checkpoint_classes != self.num_classes:
      raise ValueError(
        f"Model class count({checkpoint_classes}) and dictionary class count({self.num_classes}) do not match."
      )

    # 좌표 261 + 속도 261 + 가속도 261 = 총 783차원을 모델 입력으로 사용한다.
    model = SignLanguageModel(input_dim=783, hidden_dim=HIDDEN_DIM, num_classes=self.num_classes)
    model.load_state_dict(checkpoint)
    model.to(self.device)
    # Dropout과 BatchNorm을 추론 모드로 전환한다.
    model.eval()
    self.model = model

  def _build_inference_dict(self) -> dict[str, int]:
    """학습 사전과 글로스 통합 매핑을 이용해 추론용 클래스 사전을 만든다."""

    with GLOSS_DICT_PATH.open("r", encoding="utf-8") as file:
      old_gloss_to_idx: dict[str, int] = json.load(file)

    # gloss_mapping.json이 있으면 서로 같은 의미로 정리한 글로스를 대표 글로스로 통합한다.
    mapping_dict: dict[str, str] = {}
    if GLOSS_MAPPING_PATH.exists():
      with GLOSS_MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping_dict = json.load(file)

    blank_tokens = [key for key, value in old_gloss_to_idx.items() if value == 0]
    if not blank_tokens:
      raise ValueError("gloss_dict.json must contain a blank token at index 0.")

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

  def build_model_features(self, raw_features: np.ndarray) -> np.ndarray:
    """프레임별 키포인트를 평활화하고 모델 입력 특징으로 확장한다.

    브라우저에서 받은 원본 키포인트는 [T, 261]이고, 속도와 가속도를 붙인 결과는 [T, 783]이다.
    이미 783차원으로 전처리된 npy 입력은 다시 확장하지 않는다.
    """

    # 시간축(axis=0)의 순간적인 키포인트 흔들림을 줄인다.
    smoothed_features = gaussian_filter1d(np.asarray(raw_features), sigma=1.0, axis=0)
    if smoothed_features.shape[1] == 261:
      return apply_motion_derivatives(smoothed_features)
    return smoothed_features

  def predict_raw_features(self, raw_features: np.ndarray, save_npy: bool = False) -> dict[str, Any]:
    """전체 프레임 시퀀스를 모델에 넣어 글로스 목록과 신뢰도를 반환한다."""

    # startup 훅을 거치지 않고 직접 호출된 경우에도 최초 요청에서 모델을 준비한다.
    if self.model is None:
      self.load()

    if len(raw_features) < MIN_FRAMES:
      print(f"[ai] not enough frames: {len(raw_features)}", flush=True)
      return self._response([], frame_count=len(raw_features), confidence=0.0)

    enhanced_features = self.build_model_features(raw_features)
    npy_path: Path | None = None
    if save_npy:
      # 웹캠/동영상 예측에 사용된 특징을 재현 및 디버깅용으로 저장한다.
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
      npy_path = PREDICTION_OUTPUT_DIR / f"prediction_{stamp}.npy"
      np.save(npy_path, enhanced_features)

    # 모델 입력 형태: [T, 783] -> [배치 1, T, 783]
    tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      # logits 형태는 [배치, T, 글로스 클래스 수]이다.
      logits = self.model(tensor)
      probabilities = torch.softmax(logits, dim=2)
      # 각 프레임에서 확률이 가장 높은 글로스 클래스 번호를 선택한다.
      predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
      # 현재 신뢰도는 문장 전체 확률이 아니라 프레임별 최고 확률의 평균이다.
      confidence = float(probabilities.max(dim=2).values.mean().detach().cpu().item())

    # 프레임마다 반복 출력된 클래스 번호를 실제 글로스 시퀀스로 축약한다.
    words = self._ctc_decode(predictions)
    result = self._response(words, frame_count=len(raw_features), confidence=confidence)
    if npy_path:
      result["npy_path"] = str(npy_path)
      result["npy_saved"] = True
    print(
      "[ai] prediction",
      {
        "raw_frames": len(raw_features),
        "feature_shape": list(enhanced_features.shape),
        "words": words,
        "confidence": confidence,
        "npy_path": str(npy_path) if npy_path else None,
      },
      flush=True,
    )
    return result

  def predict_video(self, video_path: Path) -> dict[str, Any]:
    """업로드된 동영상에서 키포인트를 추출한 뒤 동일한 추론 경로를 사용한다."""

    raw_features = extract_video_keypoints(video_path)
    return self.predict_raw_features(raw_features, save_npy=True)

  def _ctc_decode(self, predictions: np.ndarray) -> list[str]:
    """연속 중복 클래스와 CTC blank를 제거하고 클래스 번호를 글로스로 변환한다."""

    decoded: list[str] = []
    previous_idx = -1
    for idx in predictions:
      int_idx = int(idx)
      if int_idx != previous_idx and int_idx != self.blank_idx:
        decoded.append(self.idx_to_gloss.get(int_idx, "<UNK>"))
      previous_idx = int_idx
    return decoded

  def _response(self, words: list[str], frame_count: int, confidence: float) -> dict[str, Any]:
    """모든 입력 방식에서 공통으로 사용하는 기본 예측 응답을 만든다."""

    return {
      "type": "prediction",
      "text": " ".join(words),
      "words": words,
      "gloss_result": " ".join(words),
      "confidence": confidence,
      "frame_count": frame_count,
      "device": str(self.device),
      "translation_candidates": [],
    }


# 서버 프로세스에서 모델 인스턴스를 한 번만 생성해 요청마다 재사용한다.
engine = InferenceEngine()


async def enrich_with_llm(result: dict[str, Any]) -> dict[str, Any]:
  """AI 모델이 인식한 글로스를 sLLM으로 자연스러운 한국어 문장으로 변환한다."""

  words = result.get("words") or []
  gloss_result = result.get("gloss_result") or " ".join(words)
  confidence = float(result.get("confidence") or 0)
  llm_result = await translate_gloss(words, gloss_result, confidence)
  enriched = {**result}
  # text는 클라이언트가 바로 표시할 최종 문장이고,
  # gloss_result에는 LLM 적용 전 모델의 글로스 결과를 그대로 보존한다.
  enriched["text"] = llm_result["text"]
  enriched["translated_text"] = llm_result["text"]
  enriched["translation_candidates"] = llm_result["translation_candidates"]
  enriched["llm_used"] = llm_result["llm_used"]
  enriched["llm_error"] = llm_result["llm_error"]
  print(
    "[ai-llm] result",
    {
      "llm_used": enriched["llm_used"],
      "llm_error": enriched["llm_error"],
      "gloss_result": gloss_result,
      "text": enriched["text"],
      "candidate_count": len(enriched["translation_candidates"]),
    },
    flush=True,
  )
  return enriched


def extract_video_keypoints(video_path: Path) -> np.ndarray:
  """동영상의 모든 프레임을 읽어 MediaPipe Holistic 키포인트 [T, 261]을 만든다."""

  cap = cv2.VideoCapture(str(video_path))
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
      height = int(width * frame.shape[0] / frame.shape[1])
      resized = cv2.resize(frame, (width, height))
      # OpenCV는 BGR, MediaPipe는 RGB 이미지를 사용한다.
      results = holistic.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
      features.append(extract_normalized_keypoints(results))

  cap.release()
  if not features:
    return np.empty((0, 261), dtype=np.float32)
  return np.asarray(features, dtype=np.float32)


def decode_data_url_base64(data_base64: str) -> bytes:
  """data:...;base64, 접두사가 포함된 문자열과 순수 Base64 문자열을 모두 디코딩한다."""

  if "," in data_base64:
    data_base64 = data_base64.split(",", 1)[1]
  return base64.b64decode(data_base64)


def suffix_from_content_type(content_type: str | None) -> str:
  """임시 파일을 만들 때 MIME 타입에 맞는 확장자를 선택한다."""

  if content_type == "video/webm":
    return ".webm"
  if content_type == "video/mp4":
    return ".mp4"
  if content_type == "application/x-npy":
    return ".npy"
  return ".bin"


@app.on_event("startup")
async def startup() -> None:
  # 첫 요청에서 모델 로딩 지연이 발생하지 않도록 서버 시작 시 미리 로드한다.
  await asyncio.to_thread(engine.load)


@app.get("/health")
async def health() -> dict[str, Any]:
  """모델, 실행 장치, LLM 연결 설정을 확인하는 상태 점검 API."""

  return {
    "ok": True,
    "device": str(engine.device),
    "num_classes": engine.num_classes,
    "model_path": str(MODEL_PATH),
    "llm_model": os.getenv("SIGNLINK_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    "llm_base_url": os.getenv("SIGNLINK_LLM_BASE_URL", ""),
    "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
  }


@app.post("/api/v1/predict")
async def predict_http(body: PredictRequest) -> dict[str, Any]:
  """완성된 키포인트 배열을 한 번에 받아 예측하는 HTTP API."""

  if body.features is not None:
    features = np.asarray(body.features, dtype=np.float32)
    # PyTorch 추론은 동기 작업이므로 별도 스레드에서 실행해 이벤트 루프 정체를 줄인다.
    result = await asyncio.to_thread(engine.predict_raw_features, features)
    return await enrich_with_llm(result)
  return {
    "type": "prediction",
    "text": "",
    "words": [],
    "gloss_result": "",
    "confidence": 0.0,
    "message": "Send features or use /ws/predict for video prediction.",
  }


@app.websocket("/ws/predict")
async def predict_websocket(websocket: WebSocket) -> None:
  """웹캠 프레임, 동영상 또는 npy 입력을 처리하는 WebSocket API.

  웹캠은 start -> frame 반복 -> end 순서로 통신한다. frame 수신 중에는 예측하지 않고,
  end를 받은 시점에 연결별 frame_buffer 전체를 한 번에 추론한다.
  """

  await websocket.accept()
  # 각 WebSocket 연결마다 독립적인 프레임 버퍼를 사용한다.
  frame_buffer: list[list[float]] = []
  try:
    while True:
      message = await websocket.receive_json()
      message_type = message.get("type")

      if message_type == "start":
        # 새 촬영이 시작되면 이전 촬영에서 남은 키포인트를 제거한다.
        frame_buffer = []
        continue

      if message_type == "frame":
        # 프론트엔드가 MediaPipe로 추출한 한 프레임의 261차원 키포인트를 받는다.
        keypoints = message.get("keypoints")
        if not isinstance(keypoints, list):
          await websocket.send_json({"type": "error", "message": "keypoints must be a list"})
          continue
        if len(keypoints) != 261:
          await websocket.send_json({"type": "error", "message": f"expected 261 keypoints, got {len(keypoints)}"})
          continue
        frame_buffer.append([float(value) for value in keypoints])
        # 30프레임마다 진행 상황만 출력하며, 이 시점에는 모델 추론을 수행하지 않는다.
        if len(frame_buffer) % 30 == 0:
          print(f"[ai] buffered frames: {len(frame_buffer)}", flush=True)
        continue

      if message_type == "end":
        # 촬영 종료 시 누적 배열 [T, 261]을 만들어 한 번 예측하고 LLM 번역까지 수행한다.
        print(f"[ai] stream ended. total frames: {len(frame_buffer)}", flush=True)
        features = np.asarray(frame_buffer, dtype=np.float32)
        result = await asyncio.to_thread(engine.predict_raw_features, features, True)
        result = await enrich_with_llm(result)
        await websocket.send_json(result)
        # 결과 전송 후 다음 촬영을 위해 버퍼를 초기화한다.
        frame_buffer = []
        continue

      if message_type == "predict_video":
        # Base64 동영상을 임시 파일로 저장한 뒤 서버 측 MediaPipe로 키포인트를 추출한다.
        content_type = message.get("content_type")
        payload = decode_data_url_base64(message["data_base64"])
        suffix = suffix_from_content_type(content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
          temp_file.write(payload)
          temp_path = Path(temp_file.name)

        result = await asyncio.to_thread(engine.predict_video, temp_path)
        result = await enrich_with_llm(result)
        await websocket.send_json(result)
        continue

      if message_type == "predict_npy":
        # 이미 추출된 특징 npy는 동영상 디코딩 없이 곧바로 추론한다.
        payload = decode_data_url_base64(message["data_base64"])
        features = np.load(io.BytesIO(payload))
        result = await asyncio.to_thread(engine.predict_raw_features, features, True)
        result = await enrich_with_llm(result)
        await websocket.send_json(result)
        continue

      await websocket.send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
  except WebSocketDisconnect:
    # 사용자가 페이지를 닫는 등 정상적으로 연결이 끊긴 경우 별도 오류 응답은 보내지 않는다.
    return
  except Exception as exc:
    # 처리 중 발생한 예외는 클라이언트가 표시할 수 있는 WebSocket 오류 메시지로 변환한다.
    await websocket.send_json({"type": "error", "message": str(exc)})
