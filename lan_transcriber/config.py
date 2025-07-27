from pydantic.v1 import BaseSettings, Field


class Settings(BaseSettings):
    APP_ENV: str = Field("prod", env="APP_ENV")
    PROM_PORT: int = Field(8001, env="PROM_PORT")
    FETCH_INTERVAL_SEC: int = Field(300, env="FETCH_INTERVAL_SEC")

    class Config:
        case_sensitive = True
