from datetime import datetime, timedelta, timezone
from typing import Annotated

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.auth.jwt import create_access_token, hash_password, verify_password
from app.database import get_database
from app.models.user import TokenResponse, UserCreate, UserLogin, UserPublic, user_document_to_public

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate) -> UserPublic:
    database = get_database()
    existing = await database.user.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    now = datetime.now(timezone.utc)
    doc = {
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "name": payload.name,
        "role": payload.role,
        "created_at": now,
    }
    result = await database.user.insert_one(doc)
    doc["_id"] = result.inserted_id
    return user_document_to_public(doc)


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin) -> TokenResponse:
    database = get_database()
    user = await database.user.find_one({"email": payload.email.lower()})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user_id = str(user["_id"])
    token = create_access_token(
        subject=user_id,
        extra_claims={"email": user["email"], "role": user["role"]},
    )

    await database.auth_sessions.insert_one(
        {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc),
            "type": "auth",
        }
    )

    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserPublic)
async def get_me(current_user: Annotated[dict, Depends(get_current_user)]) -> UserPublic:
    database = get_database()
    user = await database.user.find_one({"_id": ObjectId(current_user["id"])})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_document_to_public(user)
