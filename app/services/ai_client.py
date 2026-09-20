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
                settings.ai_server_url + settings.ai_predict_path,
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
        try:
            print(
                "[AI CLIENT] connecting:",
                settings.ai_server_ws_url + settings.ai_predict_ws_path,
                flush=True,
            )

            self.websocket = await websockets.connect(
                settings.ai_server_ws_url + settings.ai_predict_ws_path,
                open_timeout=settings.ai_request_timeout_seconds,
            )

            print("[AI CLIENT] websocket connected", flush=True)

            await self.send({"type": "start"})

            print("[AI CLIENT] start sent", flush=True)

        except Exception as exc:
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
            raw = json.dumps(payload)

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

    async def add_frame(self, keypoints):
        print(
            f"[AI CLIENT] add_frame: {len(keypoints)}",
            flush=True,
        )

        await self.send(
            {
                "type": "frame",
                "keypoints": keypoints,
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

    async def finish(self):
        if self.websocket is None:
            raise AIServerError(
                "AI server websocket is not connected"
            )

        await self.send(
            {
                "type": "end",
            }
        )

        try:
            raw = await self.websocket.recv()

        except Exception as exc:
            raise AIServerError(
                f"AI server websocket receive failed: {exc}"
            ) from exc

        data = json.loads(raw)

        if data.get("type") == "error":
            raise AIServerError(
                data.get(
                    "message",
                    "AI server prediction failed",
                )
            )

        return data

    async def close(self):
        if self.websocket is not None:
            try:
                await self.websocket.close()
            finally:
                self.websocket = None