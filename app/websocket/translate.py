import asyncio
import json
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Annotated

from bson import ObjectId
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.auth.jwt import decode_access_token
from app.config import settings
from app.database import get_database
from app.models.translation import WSErrorMessage, WSTranslationMessage
from app.routers.translations import infer_category
from app.services.ai_client import AIServerError, AIStreamClient, predict_sign
from app.services.counters import next_sequence

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    def register(self, session_id: str, websocket: WebSocket) -> None:
        connections = self.active_connections.setdefault(session_id, [])
        if websocket not in connections:
            connections.append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        connections = self.active_connections.get(session_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self.active_connections.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict) -> None:
        connections = list(self.active_connections.get(session_id, []))
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(session_id, connection)


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
    ai_receiver_task: asyncio.Task[None] | None = None
    ai_final_result: asyncio.Future[dict] | None = None
    active_consultation: dict | None = None

    async def stop_ai_stream() -> None:
        nonlocal ai_stream, ai_receiver_task, ai_final_result

        receiver_task = ai_receiver_task
        ai_receiver_task = None

        if receiver_task and receiver_task is not asyncio.current_task():
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task

        if ai_final_result and not ai_final_result.done():
            ai_final_result.cancel()
        ai_final_result = None

        stream = ai_stream
        ai_stream = None
        if stream:
            await stream.close()

    async def relay_ai_messages(
        stream: AIStreamClient,
        final_result: asyncio.Future[dict],
        stream_session_id: str,
    ) -> None:
        try:
            while True:
                message = await stream.receive()
                message_type = message.get("type")

                if message_type == "partial":
                    await websocket.send_json(
                        {
                            **message,
                            "session_id": stream_session_id,
                        }
                    )
                    continue

                if message_type in {"translation", "error"}:
                    if not final_result.done():
                        final_result.set_result(message)

                    if message_type == "error":
                        await websocket.send_json(
                            WSErrorMessage(
                                message=message.get(
                                    "message",
                                    "AI server prediction failed",
                                )
                            ).model_dump()
                        )
                    return

                print(
                    f"[BACKEND] ignored AI websocket message: {message_type}",
                    flush=True,
                )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_message = (
                str(exc)
                if isinstance(exc, AIServerError)
                else f"AI server websocket receive failed: {exc}"
            )
            if not final_result.done():
                final_result.set_result(
                    {
                        "type": "error",
                        "message": error_message,
                    }
                )
            with suppress(Exception):
                await websocket.send_json(
                    WSErrorMessage(message=error_message).model_dump()
                )

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

                await stop_ai_stream()
                new_ai_stream = AIStreamClient()
                try:
                    await new_ai_stream.connect()
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    continue

                ai_stream = new_ai_stream
                ai_final_result = asyncio.get_running_loop().create_future()
                ai_receiver_task = asyncio.create_task(
                    relay_ai_messages(
                        ai_stream,
                        ai_final_result,
                        session_id,
                    ),
                    name=f"ai-stream-receiver-{session_id}",
                )

                await websocket.send_json({"type": "stream_started", "session_id": session_id})
                continue

            if msg_type == "frame":
                keypoints = data.get("keypoints")
                frame_id = data.get("frame_id")
                print(
                    "[BACKEND] frame received:",
                    {
                        "frame_id": frame_id,
                        "values": len(keypoints) if isinstance(keypoints, list) else None,
                    },
                    flush=True,
                )
                if ai_stream is None or session_id is None or ai_final_result is None:
                    print("[BACKEND] ERROR: ai_stream is None", flush=True)
                    await websocket.send_json(WSErrorMessage(message="stream is not started").model_dump())
                    continue
                if ai_final_result.done():
                    await stop_ai_stream()
                    await websocket.send_json(
                        WSErrorMessage(message="AI stream is no longer active").model_dump()
                    )
                    continue
                if not isinstance(keypoints, list):
                    await websocket.send_json(WSErrorMessage(message="keypoints must be a list").model_dump())
                    continue
                if len(keypoints) != 261:
                    await websocket.send_json(
                        WSErrorMessage(
                            message=f"expected 261 keypoints, got {len(keypoints)}"
                        ).model_dump()
                    )
                    continue
                if type(frame_id) is not int or frame_id < 0:
                    await websocket.send_json(
                        WSErrorMessage(
                            message="frame_id must be a non-negative integer"
                        ).model_dump()
                    )
                    continue
                try:
                    await ai_stream.add_frame(keypoints, frame_id)
                    print("[BACKEND] frame sent to AI server", flush=True)
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    continue
                await websocket.send_json(
                    {
                        "type": "frame_received",
                        "frame_id": frame_id,
                    }
                )
                continue

            if msg_type == "word_boundary":
                # 짧은 정지(단어 경계): AI서버에게 "지금까지 쌓인 프레임을 디코딩해서
                # gloss만 누적해두라"고 알린다. LLM은 호출 안 되고, 세션/연결은 그대로 유지된다.
                if ai_stream is None or session_id is None or ai_final_result is None:
                    await websocket.send_json(WSErrorMessage(message="stream is not started").model_dump())
                    continue
                if ai_final_result.done():
                    await stop_ai_stream()
                    await websocket.send_json(
                        WSErrorMessage(message="AI stream is no longer active").model_dump()
                    )
                    continue
                try:
                    await ai_stream.word_boundary()
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    continue
                await websocket.send_json({"type": "word_boundary_received"})
                continue

            if msg_type == "end":
                if (
                    ai_stream is None
                    or ai_final_result is None
                    or session_id is None
                    or not active_consultation
                ):
                    await websocket.send_json(WSErrorMessage(message="stream is not started").model_dump())
                    continue

                try:
                    await ai_stream.request_finish()
                    ai_result = await asyncio.wait_for(
                        asyncio.shield(ai_final_result),
                        timeout=settings.ai_stream_result_timeout_seconds,
                    )
                except AIServerError as exc:
                    await websocket.send_json(WSErrorMessage(message=str(exc)).model_dump())
                    await stop_ai_stream()
                    continue
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        WSErrorMessage(
                            message="AI server final response timed out"
                        ).model_dump()
                    )
                    await stop_ai_stream()
                    continue

                await stop_ai_stream()

                if ai_result.get("type") == "error":
                    # relay_ai_messages already forwarded this error.
                    continue

                if ai_result.get("type") != "translation":
                    await websocket.send_json(
                        WSErrorMessage(
                            message="AI server returned an invalid final response"
                        ).model_dump()
                    )
                    continue

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
                translation_candidates = ai_result.get("translation_candidates") or ai_result.get("candidates") or []
                requires_review = bool(translation_candidates) or "unk" in f"{translated_text} {gloss_result}".lower()
                print(
                    "[backend] AI result",
                    {
                        "session_id": session_id,
                        "frame_count": ai_result.get("frame_count"),
                        "words": words,
                        "text": translated_text,
                        "confidence": confidence,
                        "llm_used": ai_result.get("llm_used"),
                        "llm_error": ai_result.get("llm_error"),
                        "translation_candidates": len(translation_candidates),
                    },
                    flush=True,
                )

                saved_log_id = None
                if auto_save and not requires_review:
                    database = get_database()
                    saved_log_id = await next_sequence("translation_log_id", 1)
                    medical_id = active_consultation.get("medical_id")
                    if not medical_id:
                        doctor = await database.user.find_one({"_id": ObjectId(active_consultation.get("doctor_id"))})
                        medical_id = int(doctor.get("medical_id", 0)) if doctor else 0
                    result = await database.translation_log.insert_one(
                        {
                            "log_id": saved_log_id,
                            "medical_id": medical_id,
                            "doctor_id": active_consultation.get("doctor_id"),
                            "session_id": session_id,
                            "patient_id": active_consultation.get("patient_id") or "none",
                            "input_time": datetime.now(timezone.utc),
                            "gloss_result": gloss_result,
                            "translated_text": translated_text,
                            "confidence": confidence,
                            "category": infer_category(translated_text),
                            "translation_candidates": translation_candidates,
                        }
                    )

                response = WSTranslationMessage(
                    text=translated_text,
                    confidence=confidence,
                    session_id=session_id,
                    saved=auto_save and not requires_review,
                    log_id=saved_log_id,
                ).model_dump()
                response["words"] = words
                response["gloss_result"] = gloss_result
                response["frame_count"] = ai_result.get("frame_count")
                response["translation_candidates"] = translation_candidates
                response["requires_review"] = requires_review
                response["llm_used"] = ai_result.get("llm_used")
                response["llm_error"] = ai_result.get("llm_error")

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
                saved_log_id = await next_sequence("translation_log_id", 1)
                medical_id = consultation.get("medical_id")
                if not medical_id:
                    doctor = await database.user.find_one({"_id": ObjectId(consultation.get("doctor_id"))})
                    medical_id = int(doctor.get("medical_id", 0)) if doctor else 0
                result = await database.translation_log.insert_one(
                    {
                        "log_id": saved_log_id,
                        "medical_id": medical_id,
                        "doctor_id": consultation.get("doctor_id"),
                        "session_id": session_id,
                        "patient_id": consultation.get("patient_id") or "none",
                        "input_time": datetime.now(timezone.utc),
                        "gloss_result": ai_result.get("gloss_result") or ai_result.get("gloss") or "",
                        "translated_text": translated_text,
                        "confidence": confidence,
                        "category": infer_category(translated_text),
                    }
                )

            response = WSTranslationMessage(
                text=translated_text,
                confidence=confidence,
                session_id=session_id,
                saved=auto_save,
                log_id=saved_log_id,
            ).model_dump()

            await manager.broadcast(session_id, response)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[BACKEND] websocket error: {exc}", flush=True)
        with suppress(Exception):
            await websocket.send_json(
                WSErrorMessage(message="WebSocket processing failed").model_dump()
            )
        with suppress(Exception):
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await stop_ai_stream()
        if session_id:
            manager.disconnect(session_id, websocket)
