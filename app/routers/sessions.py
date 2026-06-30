from datetime import datetime, timedelta, timezone
from typing import Annotated

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user, require_role
from app.database import get_database
from app.models.session import SessionCreate, SessionPublic, session_document_to_public

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionPublic, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: Annotated[dict, Depends(require_role("doctor"))],
) -> SessionPublic:
    database = get_database()
    doc = {
        "doctor_id": current_user["id"],
        "patient_id": payload.patient_id,
        "title": payload.title,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "closed_at": None,
    }
    result = await database.consultation_sessions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return session_document_to_public(doc)


@router.get("", response_model=list[SessionPublic])
async def list_sessions(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> list[SessionPublic]:
    database = get_database()
    if current_user["role"] == "doctor":
        query = {"doctor_id": current_user["id"]}
    else:
        query = {"patient_id": current_user["id"]}

    cursor = database.consultation_sessions.find(query).sort("created_at", -1)
    return [session_document_to_public(doc) async for doc in cursor]


@router.post("/{session_id}/join", response_model=SessionPublic)
async def join_session(
    session_id: str,
    current_user: Annotated[dict, Depends(require_role("patient"))],
) -> SessionPublic:
    database = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id") from exc

    doc = await database.consultation_sessions.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if doc.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session is not active")
    if doc.get("patient_id") and doc["patient_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already has a patient")

    if not doc.get("patient_id"):
        await database.consultation_sessions.update_one(
            {"_id": oid},
            {"$set": {"patient_id": current_user["id"]}},
        )
        doc["patient_id"] = current_user["id"]

    return session_document_to_public(doc)


@router.get("/{session_id}", response_model=SessionPublic)
async def get_session(
    session_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> SessionPublic:
    database = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id") from exc

    doc = await database.consultation_sessions.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if current_user["role"] == "doctor" and doc["doctor_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    if current_user["role"] == "patient" and doc.get("patient_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return session_document_to_public(doc)
