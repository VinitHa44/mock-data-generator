from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """
    Application settings.
    You can override these values by creating a .env file in the root directory.
    """
    # API settings
    PROJECT_NAME: str = "Mock Data Generator"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # LLM and Model Settings
    GGUF_MODEL_PATH: str = "/Users/vishwasbheda/Developer/Personal Project/Mock_Data_Generation/MDG_4.0/MDG/model/ggml-model-Q4_0.gguf"
    MODERATION_MODEL: str = "KoalaAI/Text-Moderation"
    PROMPT_INJECTION_MODEL: str = "protectai/deberta-v3-base-prompt-injection-v2"
    
    # Services
    REDIS_URL: str = "redis://localhost:6379"
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DB_NAME: str = "mdg_logs"
    OPENVERSE_API_URL: str = "https://api.openverse.org/v1/images/"
    MODEL_SERVER_URL: str = "http://localhost:8001"

    # Caching
    CACHE_GROUP_HASH_LIMIT: int = 50  # k-entry limit for hashes in a group
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_TIMEFRAME: int = 60  # in seconds

    # Logging
    LOG_LEVEL: str = "INFO"
    INTERMEDIATE_SAVE_PATH: str = "intermediate_steps"

    class Config:
        case_sensitive = True
        # To load from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Create intermediate save path if it doesn't exist
os.makedirs(settings.INTERMEDIATE_SAVE_PATH, exist_ok=True)