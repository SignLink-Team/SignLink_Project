from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    patient_id: str | None = None
    title: str = Field(default="Consultation", max_length=200)


class SessionPublic(BaseModel):
    id: str
    doctor_id: str
    medical_id: int | None = None
    patient_id: str | None = None
    title: str
    status: Literal["active", "closed"] = "active"
    created_at: datetime
    closed_at: datetime | None = None


def session_document_to_public(doc: dict) -> SessionPublic:
    return SessionPublic(
        id=str(doc["_id"]),
        doctor_id=doc["doctor_id"],
        medical_id=doc.get("medical_id"),
        patient_id=doc.get("patient_id"),
        title=doc["title"],
        status=doc.get("status", "active"),
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
        closed_at=doc.get("closed_at"),
    )
