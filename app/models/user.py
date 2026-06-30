from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=100)
    role: Literal["doctor", "patient"]


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: Literal["doctor", "patient"]
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def user_document_to_public(doc: dict) -> UserPublic:
    return UserPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        role=doc["role"],
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
    )
