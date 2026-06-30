from __future__ import annotations

import asyncio
import base64
import io
import json
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
from preprocess import apply_motion_derivatives, extract_normalized_keypoints  # noqa: E402

HIDDEN_DIM = 512
MIN_FRAMES = 5
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
  features: list[list[float]] | None = None
  landmarks: dict[str, Any] | None = None


class InferenceEngine:
  def __init__(self) -> None:
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model: SignLanguageModel | None = None
    self.idx_to_gloss: dict[int, str] = {}
    self.blank_idx = 0
    self.num_classes = 0

  def load(self) -> None:
    gloss_to_idx = self._build_inference_dict()
    self.idx_to_gloss = {value: key for key, value in gloss_to_idx.items()}
    self.blank_idx = gloss_to_idx.get("<blank>", 0)
    self.num_classes = len(gloss_to_idx)

    checkpoint = torch.load(MODEL_PATH, map_location=self.device)
    checkpoint_classes = checkpoint["fc.weight"].shape[0]
    if checkpoint_classes != self.num_classes:
      raise ValueError(
        f"Model class count({checkpoint_classes}) and dictionary class count({self.num_classes}) do not match."
      )

    model = SignLanguageModel(input_dim=783, hidden_dim=HIDDEN_DIM, num_classes=self.num_classes)
    model.load_state_dict(checkpoint)
    model.to(self.device)
    model.eval()
    self.model = model

  def _build_inference_dict(self) -> dict[str, int]:
    with GLOSS_DICT_PATH.open("r", encoding="utf-8") as file:
      old_gloss_to_idx: dict[str, int] = json.load(file)

    mapping_dict: dict[str, str] = {}
    if GLOSS_MAPPING_PATH.exists():
      with GLOSS_MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping_dict = json.load(file)

    blank_tokens = [key for key, value in old_gloss_to_idx.items() if value == 0]
    if not blank_tokens:
      raise ValueError("gloss_dict.json must contain a blank token at index 0.")

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

  def build_model_features(self, raw_features: np.ndarray) -> np.ndarray:
    smoothed_features = gaussian_filter1d(np.asarray(raw_features), sigma=1.0, axis=0)
    if smoothed_features.shape[1] == 261:
      return apply_motion_derivatives(smoothed_features)
    return smoothed_features

  def predict_raw_features(self, raw_features: np.ndarray, save_npy: bool = False) -> dict[str, Any]:
    if self.model is None:
      self.load()

    if len(raw_features) < MIN_FRAMES:
      print(f"[ai] not enough frames: {len(raw_features)}", flush=True)
      return self._response([], frame_count=len(raw_features), confidence=0.0)

    enhanced_features = self.build_model_features(raw_features)
    npy_path: Path | None = None
    if save_npy:
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
      npy_path = PREDICTION_OUTPUT_DIR / f"prediction_{stamp}.npy"
      np.save(npy_path, enhanced_features)

    tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      logits = self.model(tensor)
      probabilities = torch.softmax(logits, dim=2)
      predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
      confidence = float(probabilities.max(dim=2).values.mean().detach().cpu().item())

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
    raw_features = extract_video_keypoints(video_path)
    return self.predict_raw_features(raw_features, save_npy=True)

  def _ctc_decode(self, predictions: np.ndarray) -> list[str]:
    decoded: list[str] = []
    previous_idx = -1
    for idx in predictions:
      int_idx = int(idx)
      if int_idx != previous_idx and int_idx != self.blank_idx:
        decoded.append(self.idx_to_gloss.get(int_idx, "<UNK>"))
      previous_idx = int_idx
    return decoded

  def _response(self, words: list[str], frame_count: int, confidence: float) -> dict[str, Any]:
    return {
      "type": "prediction",
      "text": " ".join(words),
      "words": words,
      "gloss_result": " ".join(words),
      "confidence": confidence,
      "frame_count": frame_count,
      "device": str(self.device),
    }


engine = InferenceEngine()


def extract_video_keypoints(video_path: Path) -> np.ndarray:
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
      results = holistic.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
      features.append(extract_normalized_keypoints(results))

  cap.release()
  if not features:
    return np.empty((0, 261), dtype=np.float32)
  return np.asarray(features, dtype=np.float32)


def decode_data_url_base64(data_base64: str) -> bytes:
  if "," in data_base64:
    data_base64 = data_base64.split(",", 1)[1]
  return base64.b64decode(data_base64)


def suffix_from_content_type(content_type: str | None) -> str:
  if content_type == "video/webm":
    return ".webm"
  if content_type == "video/mp4":
    return ".mp4"
  if content_type == "application/x-npy":
    return ".npy"
  return ".bin"


@app.on_event("startup")
async def startup() -> None:
  await asyncio.to_thread(engine.load)


@app.get("/health")
async def health() -> dict[str, Any]:
  return {
    "ok": True,
    "device": str(engine.device),
    "num_classes": engine.num_classes,
    "model_path": str(MODEL_PATH),
  }


@app.post("/api/v1/predict")
async def predict_http(body: PredictRequest) -> dict[str, Any]:
  if body.features is not None:
    features = np.asarray(body.features, dtype=np.float32)
    return await asyncio.to_thread(engine.predict_raw_features, features)
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
  await websocket.accept()
  frame_buffer: list[list[float]] = []
  try:
    while True:
      message = await websocket.receive_json()
      message_type = message.get("type")

      if message_type == "start":
        frame_buffer = []
        continue

      if message_type == "frame":
        keypoints = message.get("keypoints")
        if not isinstance(keypoints, list):
          await websocket.send_json({"type": "error", "message": "keypoints must be a list"})
          continue
        if len(keypoints) != 261:
          await websocket.send_json({"type": "error", "message": f"expected 261 keypoints, got {len(keypoints)}"})
          continue
        frame_buffer.append([float(value) for value in keypoints])
        if len(frame_buffer) % 30 == 0:
          print(f"[ai] buffered frames: {len(frame_buffer)}", flush=True)
        continue

      if message_type == "end":
        print(f"[ai] stream ended. total frames: {len(frame_buffer)}", flush=True)
        features = np.asarray(frame_buffer, dtype=np.float32)
        result = await asyncio.to_thread(engine.predict_raw_features, features, True)
        await websocket.send_json(result)
        frame_buffer = []
        continue

      if message_type == "predict_video":
        content_type = message.get("content_type")
        payload = decode_data_url_base64(message["data_base64"])
        suffix = suffix_from_content_type(content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
          temp_file.write(payload)
          temp_path = Path(temp_file.name)

        result = await asyncio.to_thread(engine.predict_video, temp_path)
        await websocket.send_json(result)
        continue

      if message_type == "predict_npy":
        payload = decode_data_url_base64(message["data_base64"])
        features = np.load(io.BytesIO(payload))
        result = await asyncio.to_thread(engine.predict_raw_features, features, True)
        await websocket.send_json(result)
        continue

      await websocket.send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
  except WebSocketDisconnect:
    return
  except Exception as exc:
    await websocket.send_json({"type": "error", "message": str(exc)})
