from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Sign Language Backend"
    debug: bool = True

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "sign_language"

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    ai_server_url: str = "http://host.docker.internal:8001"
    ai_server_ws_url: str = "ws://host.docker.internal:8001"
    ai_predict_path: str = "/api/v1/predict"
    ai_predict_ws_path: str = "/ws/predict"
    ai_request_timeout_seconds: float = 5.0
    # websocket.py의 "end" 핸들러가 AI 서버의 최종 "translation" 응답을
    # 기다릴 때 쓰는 타임아웃. 이 속성이 없으면 스트림 종료 시점마다
    # AttributeError로 웹소켓이 끊기고 최종 결과가 프론트에 전달되지
    # 못한다. AI 서버의 force_flush는 최악의 경우 buffer를 최대 8회
    # 반복 디코딩(각 회차 forward + beam search)하므로, CPU 환경에서는
    # 그 시간을 넉넉히 덮을 수 있게 잡는다. 실측 후 조정할 것.
    ai_stream_result_timeout_seconds: float = 15.0

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def ai_predict_url(self) -> str:
        return f"{self.ai_server_url.rstrip('/')}{self.ai_predict_path}"

    @property
    def ai_predict_ws_url(self) -> str:
        return f"{self.ai_server_ws_url.rstrip('/')}{self.ai_predict_ws_path}"


settings = Settings()