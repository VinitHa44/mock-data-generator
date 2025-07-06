from datetime import datetime

import motor.motor_asyncio
from app.config.settings import settings
from app.utils.logging_config import get_logger
from pymongo.errors import ConnectionFailure

logger = get_logger(__name__)


class LogRepository:
    def __init__(self, mongo_uri: str, db_name: str):
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = self.client[db_name]
            self.token_log_collection = self.db["token_logs"]
            logger.info("MongoDB client initialized, connection will be established on first use.")
        except Exception as e:
            logger.error("Could not initialize MongoDB client", error=str(e))
            self.client = None
            self.db = None

    async def log_token_usage(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
    ):
        """
        Log the number of input and output tokens for a request.
        """
        if self.db is None:
            logger.warning(
                "Skipping token logging due to no MongoDB connection."
            )
            return

        log_entry = {
            "request_id": request_id,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "timestamp": datetime.utcnow(),
        }
        try:
            await self.token_log_collection.insert_one(log_entry)
            logger.info("Logged token usage to MongoDB", request_id=request_id)
        except Exception as e:
            logger.error("Failed to log token usage to MongoDB", error=str(e))


# Singleton instance
log_repository = LogRepository(settings.MONGO_URI, settings.MONGO_DB_NAME)
