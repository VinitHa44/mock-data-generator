import os

from pydantic_settings import BaseSettings


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
    GGUF_MODEL_PATH: str = "model/phi35-finetuned-Q4_K_M.gguf"
    MODERATION_MODEL: str = "KoalaAI/Text-Moderation"
    PROMPT_INJECTION_MODEL: str = (
        "protectai/deberta-v3-base-prompt-injection-v2"
    )
    
    # LLM Pool Configuration for Concurrent Processing
    LLM_POOL_SIZE: int = 2  # Reduced from 3 for better stability
    LLM_BATCH_SIZE: int = 25  # Reduced from 50 for better reliability
    LLM_BATCH_THRESHOLD: int = 15  # Use batch processing for counts > this value
    
    # LLM Generation Parameters for Stability
    LLM_TEMPERATURE: float = 0.6  # Further reduced for more consistent output
    LLM_TOP_P: float = 0.85  # More conservative for better JSON structure
    LLM_MAX_TOKENS: int = -1  # No token limit for complete generation
    LLM_RETRY_ATTEMPTS: int = 3  # Number of retry attempts for failed batches
    
    # Performance Tuning (adjust based on your hardware)
    # For 500 samples per user:
    # - LLM_POOL_SIZE=2: Allows 2 concurrent generations (more stable)
    # - LLM_BATCH_SIZE=25: Processes 500 samples in 20 parallel batches
    # - Expected performance: ~15-20 seconds for 500 samples vs 60+ seconds sequentially

    # Services
    REDIS_URL: str = "redis://localhost:6379"
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DB_NAME: str = "mdg_logs"
    OPENVERSE_API_URL: str = "https://api.openverse.org/v1/images/"
    MODEL_SERVER_URL: str = "http://localhost:8001"

    # Caching
    CACHE_GROUP_HASH_LIMIT: int = 50  # k-entry limit for hashes in a group
    # CACHE_EXPIRATION_SECONDS: int = 7 * 24 * 60 * 60  # 1 week in seconds
    CACHE_EXPIRATION_SECONDS: int = 2 * 60  # 2 min in seconds

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_TIMEFRAME: int = 60  # in seconds

    # Logging
    LOG_LEVEL: str = "INFO"
    INTERMEDIATE_SAVE_PATH: str = "intermediate_steps"

    # CORS Settings
    CORS_ALLOW_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]

    # Image Processing Settings
    IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]
    IMAGE_CONTENT_TYPES: list = ['image/', 'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/svg']
    IMAGE_KEYWORDS: list = [
        "image", "img", "picture", "photo", "pic", "avatar", "thumbnail", 
        "logo", "image_url", "img_url", "photo_url", "avatar_url", 
        "logo_url", "profile_pic"
    ]
    GENERIC_URL_PARTS: list = [
        'images', 'img', 'assets', 'static', 'uploads', 'content',
        'v1', 'v2', 'v3', 'media', 'photos', 'pictures', 'examples', 
        'example', 'photo', 'photos', 'image', 'contents'
    ]
    ERROR_INDICATORS: list = [
        "these aren't the droids you're looking for", "not found", "error",
        "page not found", "404", "forbidden", "access denied", "maintenance",
        "temporarily unavailable", "service unavailable", "bad request", "unauthorized"
    ]
    MIN_IMAGE_SIZE_BYTES: int = 1000  # Minimum size for valid image content
    IMAGE_VALIDATION_TIMEOUT: int = 8  # seconds for HEAD request
    IMAGE_FETCH_TIMEOUT: int = 10  # seconds for GET request
    IMAGE_CONTENT_CHECK_SIZE: int = 500  # characters to check for error content

    # ID Generation Settings
    ID_PATTERN_REGEXES: list = [
        (r"^([a-zA-Z_]+)(\d+)$", ""),
        (r"^([a-zA-Z_]+)-(\d+)$", "-"),
        (r"^([a-zA-Z_]+):(\d+)$", ":"),
        (r"^([a-zA-Z_]+)_(\d+)$", "_"),
        (r"^([a-zA-Z_]+)\.(\d+)$", "."),
        # Complex alphanumeric patterns
        (r"^([a-zA-Z_]+):([a-zA-Z0-9\-]+)$", ":"),
        (r"^([a-zA-Z_]+)-([a-zA-Z0-9\-]+)$", "-"),
        (r"^([a-zA-Z_]+)_([a-zA-Z0-9\-]+)$", "_"),
        (r"^([a-zA-Z_]+)\.([a-zA-Z0-9\-]+)$", "."),
        # Date-based patterns
        (r"^([a-zA-Z_]+)-(\d{4}-\d{2}-\d{2}-\d+)$", "-"),
        (r"^([a-zA-Z_]+)-(\d{4}-Q\d-\d+)$", "-"),
        (r"^([a-zA-Z_]+)_(\d{4}_\d{2}_\d{2}_\d+)$", "_"),
        # Revision patterns
        (r"^([a-zA-Z_]+)-(\d+)-([a-zA-Z0-9]+)$", "-"),
        (r"^([a-zA-Z_]+)_(\d+)_([a-zA-Z0-9]+)$", "_"),
    ]
    ID_KEY_PATTERNS: list = ["id", "_id", "Id", "ID"]  # Keys that should be treated as IDs

    # LLM System Prompt
    LLM_SYSTEM_PROMPT: str = (
        "You are an expert assistant for generating synthetic data. "
        "Analyze the user's request and generate a detailed, high-quality synthetic dataset entry."
    )

    # Data Generation Settings
    MAX_GENERATION_ATTEMPTS: int = 3
    LARGE_RESPONSE_THRESHOLD: int = 8000  # characters
    MAX_TOKENS_PER_ITEM: int = 200
    MAX_TOTAL_TOKENS: int = 8192

    # Debug Settings
    DEBUG_OUTPUT_DIR: str = "debug_outputs"
    BLOOM_FILTER_FILE: str = "bloom_filter.pkl"

    # SSL Settings
    DISABLE_SSL_DOMAINS: list = ["flickr.com", "staticflickr.com"]

    class Config:
        case_sensitive = True
        # To load from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Create intermediate save path if it doesn't exist
os.makedirs(settings.INTERMEDIATE_SAVE_PATH, exist_ok=True)
