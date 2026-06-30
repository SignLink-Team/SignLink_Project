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

    ai_server_url: str = "http://localhost:8001"
    ai_server_ws_url: str = "ws://localhost:8001"
    ai_predict_path: str = "/api/v1/predict"
    ai_predict_ws_path: str = "/ws/predict"
    ai_request_timeout_seconds: float = 5.0

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
