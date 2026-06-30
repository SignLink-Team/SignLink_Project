import json
from datetime import datetime, timedelta, timezone
from typing import Annotated

from bson import ObjectId
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.auth.jwt import decode_access_token
from app.database import get_database
from app.models.translation import WSErrorMessage, WSTranslationMessage
from app.routers.translations import infer_category
from app.services.ai_client import AIServerError, AIStreamClient, predict_sign

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    def register(self, session_id: str, websocket: WebSocket) -> None:
        self.active_connections.setdefault(session_id, []).append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        connections = self.active_connections.get(session_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self.active_connections.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict) -> None:
        for connection in self.active_connections.get(session_id, []):
            await connection.send_json(message)


manager = ConnectionManager()


def _authenticate_token(token: str | None) -> dict:
    if not token:
        raise ValueError("Missing token")
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise ValueError("Invalid token payload")
    return {
        "id": user_id,
        "email": payload.get("email"),
        "role": payload.get("role"),
    }


@router.websocket("/ws/translate")
async def translate_websocket(
    websocket: WebSocket,
    token: Annotated[str | None, Query()] = None,
) -> None:
    try:
        user = _authenticate_token(token)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await websocket.send_json({"type": "connected", "message": "WebSocket connected"})

    session_id: str | None = None
    ai_stream: AIStreamClient | None = None
    active_consultation: dict | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            msg_type = data.get("type")

            if msg_type == "join":
                join_session_id = data.get("session_id")
                if not join_session_id:
                    await websocket.send_json(
                        WSErrorMessage(message="session_id is required").model_dump()
                    )
                    continue
                manager.register(join_session_id, websocket)
                session_id = join_session_id
                await websocket.send_json(
                    {"type": "joined", "session_id": join_session_id, "message": "Session joined"}
                )
                continue

            if msg_type == "start":
                session_id = data.get("session_id")
                if not session_id:
                    await websocket.send_json(
                        WSErrorMessage(message="session_id is required").model_dump()
                    )
                    continue

                manager.register(session_id, websocket)
                database = get_database()
                try:
                    session_oid = ObjectId(session_id)
                except Exception:
                    await websocket.send_json(WSErrorMessage(message="Invalid session_id").model_dump())
                    continue

                active_consultation = await database.consultation_sessions.find_one({"_id": session_oid})
                if not active_consultation:
                    await websocket.send_json(WSErrorMessage(message="Session not found").model_dump())
                    continue

                if ai_stream:
                    await ai_stream.close()
                ai_stream = AIStreamClient()
                try:
                    await ai_stream.connect()
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    ai_stream = None
                    continue

                await websocket.send_json({"type": "stream_started", "session_id": session_id})
                continue

            if msg_type == "frame":
                keypoints = data.get("keypoints")
                if not ai_stream or not session_id:
                    await websocket.send_json(WSErrorMessage(message="stream is not started").model_dump())
                    continue
                if not isinstance(keypoints, list):
                    await websocket.send_json(WSErrorMessage(message="keypoints must be a list").model_dump())
                    continue
                try:
                    await ai_stream.add_frame(keypoints)
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    continue
                await websocket.send_json({"type": "frame_received"})
                continue

            if msg_type == "end":
                if not ai_stream or not session_id or not active_consultation:
                    await websocket.send_json(WSErrorMessage(message="stream is not started").model_dump())
                    continue

                try:
                    ai_result = await ai_stream.finish()
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    continue
                finally:
                    await ai_stream.close()
                    ai_stream = None

                words = ai_result.get("words") or []
                gloss_result = ai_result.get("gloss_result") or " ".join(words)
                translated_text = (
                    ai_result.get("text")
                    or ai_result.get("translated_text")
                    or gloss_result
                    or ""
                )
                confidence = ai_result.get("confidence")
                auto_save = data.get("auto_save", True)
                print(
                    "[backend] AI result",
                    {
                        "session_id": session_id,
                        "frame_count": ai_result.get("frame_count"),
                        "words": words,
                        "text": translated_text,
                        "confidence": confidence,
                    },
                    flush=True,
                )

                saved_log_id = None
                if auto_save:
                    database = get_database()
                    result = await database.translation_log.insert_one(
                        {
                            "medical_id": active_consultation.get("doctor_id"),
                            "doctor_id": active_consultation.get("doctor_id"),
                            "session_id": session_id,
                            "patient_id": active_consultation.get("patient_id"),
                            "input_time": datetime.now(timezone.utc),
                            "gloss_result": gloss_result,
                            "translated_text": translated_text,
                            "confidence": confidence,
                            "category": infer_category(translated_text),
                            "frame_count": ai_result.get("frame_count"),
                            "npy_path": ai_result.get("npy_path"),
                        }
                    )
                    saved_log_id = str(result.inserted_id)

                response = WSTranslationMessage(
                    text=translated_text,
                    confidence=confidence,
                    session_id=session_id,
                    saved=auto_save,
                    log_id=saved_log_id,
                ).model_dump()
                response["words"] = words
                response["gloss_result"] = gloss_result
                response["frame_count"] = ai_result.get("frame_count")

                await manager.broadcast(session_id, response)
                continue

            if msg_type != "landmarks":
                await websocket.send_json(
                    WSErrorMessage(message="Unsupported message type").model_dump()
                )
                continue

            session_id = data.get("session_id")
            frame = data.get("frame")
            if not session_id or not isinstance(frame, dict):
                await websocket.send_json(
                    WSErrorMessage(message="session_id and frame are required").model_dump()
                )
                continue

            manager.register(session_id, websocket)

            database = get_database()
            try:
                session_oid = ObjectId(session_id)
            except Exception:
                await websocket.send_json(
                    WSErrorMessage(message="Invalid session_id").model_dump()
                )
                continue

            consultation = await database.consultation_sessions.find_one({"_id": session_oid})
            if not consultation:
                await websocket.send_json(
                    WSErrorMessage(message="Session not found").model_dump()
                )
                continue

            try:
                ai_result = await predict_sign(frame)
            except AIServerError as exc:
                await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                continue

            translated_text = ai_result.get("text") or ai_result.get("translated_text") or ""
            confidence = ai_result.get("confidence")
            auto_save = data.get("auto_save", True)

            saved_log_id = None
            if auto_save:
                result = await database.translation_log.insert_one(
                    {
                        "medical_id": consultation.get("doctor_id"),
                        "doctor_id": consultation.get("doctor_id"),
                        "session_id": session_id,
                        "patient_id": consultation.get("patient_id"),
                        "input_time": datetime.now(timezone.utc),
                        "gloss_result": ai_result.get("gloss_result") or ai_result.get("gloss") or "",
                        "translated_text": translated_text,
                        "confidence": confidence,
                        "category": infer_category(translated_text),
                    }
                )
                saved_log_id = str(result.inserted_id)

            response = WSTranslationMessage(
                text=translated_text,
                confidence=confidence,
                session_id=session_id,
                saved=auto_save,
                log_id=saved_log_id,
            ).model_dump()

            await manager.broadcast(session_id, response)

    except WebSocketDisconnect:
        if ai_stream:
            await ai_stream.close()
        if session_id:
            manager.disconnect(session_id, websocket)
    except Exception:
        if ai_stream:
            await ai_stream.close()
        if session_id:
            manager.disconnect(session_id, websocket)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
