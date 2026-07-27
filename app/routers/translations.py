from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo import ReturnDocument

from app.auth.dependencies import require_role
from app.database import get_database
from app.models.translation import (
    TranslationCreate,
    TranslationPublic,
    TranslationUpdate,
    translation_document_to_public,
)
from app.services.counters import next_sequence

router = APIRouter(prefix="/translations", tags=["translations"])

CATEGORY_ALL = "\uc804\uccb4"
CATEGORY_DEFAULT = "\uae30\ud0c0"


def infer_category(text: str) -> str:
    rules = [
        ("\ub450\ud1b5", ["\uba38\ub9ac", "\ub450\ud1b5", "\uc5b4\uc9c0"]),
        ("\ud638\ud761\uae30", ["\uae30\uce68", "\uc228", "\ud638\ud761", "\uac00\ub798"]),
        ("\uc804\uc2e0/\uac10\uae30", ["\uc5f4", "\uac10\uae30", "\ubab8\uc0b4", "\ucd25"]),
        ("\uc18c\ud654\uae30", ["\ubc30", "\ubcf5\ubd80", "\uc18d", "\uc18c\ud654", "\uad6c\ud1a0"]),
        ("\uadfc\uace8\uaca9\uacc4", ["\ud5c8\ub9ac", "\ud314", "\ub2e4\ub9ac", "\uad00\uc808", "\ud1b5\uc99d"]),
        ("\uc54c\ub808\ub974\uae30", ["\uc54c\ub808\ub974\uae30", "\ub450\ub4dc\ub7ec\uae30", "\uac00\ub824"]),
        ("\uc774\ube44\uc778\ud6c4\uacfc", ["\ubaa9", "\uadc0", "\ucf54", "\uc0bc\ud0a4"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return CATEGORY_DEFAULT


@router.post("", response_model=TranslationPublic, status_code=status.HTTP_201_CREATED)
async def create_translation(
    payload: TranslationCreate,
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> TranslationPublic:
    database = get_database()
    now = datetime.now(timezone.utc)
    log_id = await next_sequence("translation_log_id", 1)
    doc = {
        "log_id": log_id,
        "medical_id": current_user["medical_id"],
        "doctor_id": current_user["id"],
        "session_id": payload.session_id,
        "patient_id": payload.patient_id or "none",
        "input_time": now,
        "gloss_result": payload.gloss_result,
        "translated_text": payload.translated_text,
        "confidence": payload.confidence,
        "category": payload.category or infer_category(payload.translated_text),
    }
    result = await database.translation_log.insert_one(doc)
    doc["_id"] = result.inserted_id
    return translation_document_to_public(doc)


@router.patch("/{log_id}", response_model=TranslationPublic)
async def update_translation(
    log_id: int,
    payload: TranslationUpdate,
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> TranslationPublic:
    database = get_database()
    update: dict = {}
    if payload.patient_id is not None:
        update["patient_id"] = payload.patient_id.strip() or "none"

    if not update:
        doc = await database.translation_log.find_one({"log_id": log_id, "medical_id": current_user["medical_id"]})
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found")
        return translation_document_to_public(doc)

    doc = await database.translation_log.find_one_and_update(
        {"log_id": log_id, "medical_id": current_user["medical_id"]},
        {"$set": update},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found")
    return translation_document_to_public(doc)


@router.get("", response_model=list[TranslationPublic])
async def list_translations(
    current_user: Annotated[dict, Depends(require_role("doctor"))],
    session_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[TranslationPublic]:
    database = get_database()
    query: dict = {"medical_id": current_user["medical_id"]}
    if session_id:
        query["session_id"] = session_id
    if category and category != CATEGORY_ALL:
        query["category"] = category
    if search:
        query["$or"] = [
            {"translated_text": {"$regex": search, "$options": "i"}},
            {"gloss_result": {"$regex": search, "$options": "i"}},
        ]

    cursor = database.translation_log.find(query).sort("input_time", -1).limit(limit)
    return [translation_document_to_public(doc) async for doc in cursor]


@router.get("/{log_id}", response_model=TranslationPublic)
async def get_translation(
    log_id: int,
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> TranslationPublic:
    database = get_database()
    doc = await database.translation_log.find_one({"log_id": log_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found")
    if doc.get("medical_id") != current_user["medical_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return translation_document_to_public(doc)


@router.delete("/{log_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_translation(
    log_id: int,
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> None:
    database = get_database()
    result = await database.translation_log.delete_one({"log_id": log_id, "medical_id": current_user["medical_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found")


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_translations(
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> None:
    database = get_database()
    await database.translation_log.delete_many({"medical_id": current_user["medical_id"]})
