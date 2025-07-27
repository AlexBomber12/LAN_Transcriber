from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    APP_ENV: str = Field("prod", env="APP_ENV")
    DATA_DIR: str = Field("data", env="DATA_DIR")
    PROM_PORT: int = Field(8001, env="PROM_PORT")
    FETCH_INTERVAL_SEC: int = Field(300, env="FETCH_INTERVAL_SEC")

    class Config:
        case_sensitive = True
        extra = "allow"        # lets tests pass arbitrary kwargs
