import httpx
import websockets

from app.config import settings


class AIServerError(Exception):
    pass


async def predict_sign(landmarks: dict) -> dict:
    """Forward landmark data to the AI server and return its JSON response."""
    try:
        async with httpx.AsyncClient(timeout=settings.ai_request_timeout_seconds) as client:
            response = await client.post(
                settings.ai_predict_url,
                json={"landmarks": landmarks},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise AIServerError(f"AI server request failed: {exc}") from exc


class AIStreamClient:
    def __init__(self) -> None:
        self.websocket = None

    async def connect(self) -> None:
        try:
            self.websocket = await websockets.connect(
                settings.ai_predict_ws_url,
                open_timeout=settings.ai_request_timeout_seconds,
            )
            await self.send({"type": "start"})
        except Exception as exc:
            raise AIServerError(f"AI server websocket connect failed: {exc}") from exc

    async def send(self, payload: dict) -> None:
        if self.websocket is None:
            raise AIServerError("AI server websocket is not connected")
        try:
            await self.websocket.send_json(payload)
        except AttributeError:
            import json

            await self.websocket.send(json.dumps(payload))
        except Exception as exc:
            raise AIServerError(f"AI server websocket send failed: {exc}") from exc

    async def add_frame(self, keypoints: list[float]) -> None:
        await self.send({"type": "frame", "keypoints": keypoints})

    async def finish(self) -> dict:
        if self.websocket is None:
            raise AIServerError("AI server websocket is not connected")
        await self.send({"type": "end"})
        try:
            raw = await self.websocket.recv()
        except Exception as exc:
            raise AIServerError(f"AI server websocket receive failed: {exc}") from exc

        import json

        data = json.loads(raw)
        if data.get("type") == "error":
            raise AIServerError(data.get("message", "AI server prediction failed"))
        return data

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
