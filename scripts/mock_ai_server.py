"""Minimal AI server mock for local development (port 8001)."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Server Mock")


class PredictRequest(BaseModel):
    landmarks: dict


@app.post("/api/v1/predict")
async def predict(payload: PredictRequest) -> dict:
    hand_count = len(payload.landmarks.get("hands", []))
    return {
        "text": f"[mock] 손 랜드마크 {hand_count}개 수신",
        "confidence": 0.95,
    }
