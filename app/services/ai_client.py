import asyncio
import json

import httpx
import websockets

from app.config import settings


class AIServerError(Exception):
    pass


async def predict_sign(landmarks: dict) -> dict:
    try:
        async with httpx.AsyncClient(
            timeout=settings.ai_request_timeout_seconds
        ) as client:
            response = await client.post(
                settings.ai_predict_url,
                json={"landmarks": landmarks},
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as exc:
        raise AIServerError(
            f"AI server request failed: {exc}"
        ) from exc


class AIStreamClient:
    def __init__(self):
        self.websocket = None

    async def connect(self):
        url = settings.ai_predict_ws_url

        try:
            print(
                "[AI CLIENT] connecting:",
                url,
                flush=True,
            )

            self.websocket = await websockets.connect(
                url,
                open_timeout=settings.ai_request_timeout_seconds,
                close_timeout=settings.ai_request_timeout_seconds,
            )

            print("[AI CLIENT] websocket connected", flush=True)

            await self.send({"type": "start"})

            response = await asyncio.wait_for(
                self.receive(),
                timeout=settings.ai_request_timeout_seconds,
            )

            if response.get("type") == "error":
                raise AIServerError(
                    response.get("message", "AI server failed to start the stream")
                )

            if response.get("type") != "stream_started":
                raise AIServerError(
                    "AI server returned an unexpected start response: "
                    f"{response.get('type')}"
                )

            print("[AI CLIENT] stream started", flush=True)
            return response

        except AIServerError:
            await self.close()
            raise
        except Exception as exc:
            await self.close()
            raise AIServerError(
                f"AI server websocket connect failed: {exc}"
            ) from exc

    async def send(self, payload):
        if self.websocket is None:
            raise AIServerError(
                "AI server websocket is not connected"
            )

        message_type = payload.get("type")

        print(
            f"[AI CLIENT] sending: {message_type}",
            flush=True,
        )

        try:
            raw = json.dumps(payload, ensure_ascii=False)

            await self.websocket.send(raw)

            print(
                f"[AI CLIENT] sent: {message_type}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[AI CLIENT] send failed: {message_type} / {exc}",
                flush=True,
            )

            raise AIServerError(
                f"AI server websocket send failed: {exc}"
            ) from exc

    async def receive(self) -> dict:
        if self.websocket is None:
            raise AIServerError(
                "AI server websocket is not connected"
            )

        try:
            raw = await self.websocket.recv()
        except Exception as exc:
            raise AIServerError(
                f"AI server websocket receive failed: {exc}"
            ) from exc

        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AIServerError(
                "AI server returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise AIServerError(
                "AI server returned an invalid websocket message"
            )

        return data

    async def add_frame(self, keypoints, frame_id: int):
        print(
            f"[AI CLIENT] add_frame: id={frame_id}, values={len(keypoints)}",
            flush=True,
        )

        await self.send(
            {
                "type": "frame",
                "keypoints": keypoints,
                "frame_id": frame_id,
            }
        )

        print(
            "[AI CLIENT] add_frame complete",
            flush=True,
        )

    async def word_boundary(self):
        await self.send(
            {
                "type": "word_boundary",
            }
        )

    async def request_finish(self):
        await self.send(
            {
                "type": "end",
            }
        )

    async def finish(self):
        """Finish a stream when no external receive loop is running."""
        await self.request_finish()

        while True:
            data = await self.receive()
            message_type = data.get("type")

            if message_type == "error":
                raise AIServerError(
                    data.get("message", "AI server prediction failed")
                )

            if message_type == "translation":
                return data

    async def close(self):
        if self.websocket is not None:
            try:
                await self.websocket.close()
            finally:
                self.websocket = None
