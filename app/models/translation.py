from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


CATEGORY_DEFAULT = "\uae30\ud0c0"


class TranslationCreate(BaseModel):
    session_id: str | None = None
    patient_id: str | None = None
    gloss_result: str = ""
    translated_text: str = Field(min_length=1)
    confidence: float | None = None
    category: str = CATEGORY_DEFAULT


class TranslationPublic(BaseModel):
    id: str
    log_id: str
    medical_id: str
    session_id: str | None = None
    patient_id: str | None = None
    gloss_result: str = ""
    translated_text: str
    confidence: float | None = None
    category: str = CATEGORY_DEFAULT
    input_time: datetime
    timestamp: datetime


class LandmarksFrame(BaseModel):
    hands: list[list[float]] = Field(default_factory=list)
    pose: list[list[float]] = Field(default_factory=list)
    face: list[list[float]] = Field(default_factory=list)


class WSLandmarksMessage(BaseModel):
    type: Literal["landmarks"] = "landmarks"
    session_id: str
    frame: LandmarksFrame
    timestamp: float | None = None
    auto_save: bool = True


class WSTranslationMessage(BaseModel):
    type: Literal["translation"] = "translation"
    text: str
    confidence: float | None = None
    session_id: str | None = None
    saved: bool = False
    log_id: str | None = None


class WSErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str


class WSConnectedMessage(BaseModel):
    type: Literal["connected"] = "connected"
    message: str = "WebSocket connected"


def translation_document_to_public(doc: dict) -> TranslationPublic:
    input_time = doc.get("input_time") or doc.get("timestamp") or datetime.now(timezone.utc)
    return TranslationPublic(
        id=str(doc["_id"]),
        log_id=str(doc.get("_id")),
        medical_id=doc.get("medical_id") or doc.get("doctor_id", ""),
        session_id=doc.get("session_id"),
        patient_id=doc.get("patient_id"),
        gloss_result=doc.get("gloss_result", ""),
        translated_text=doc["translated_text"],
        confidence=doc.get("confidence"),
        category=doc.get("category", CATEGORY_DEFAULT),
        input_time=input_time,
        timestamp=input_time,
    )
