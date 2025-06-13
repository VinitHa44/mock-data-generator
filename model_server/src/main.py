import app.services.patch_ssl
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import json
from pydantic import BaseModel
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextPayload(BaseModel):
    text: str

class ModerationService:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    logger.info("Initializing ModerationService...")
                    # These model names should ideally come from a config
                    self.text_moderation_pipeline = pipeline("text-classification", model="KoalaAI/Text-Moderation")
                    self.injection_tokenizer = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
                    self.injection_model = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
                    self.injection_pipeline = pipeline(
                        "text-classification",
                        model=self.injection_model,
                        tokenizer=self.injection_tokenizer,
                    )
                    self.initialized = True
                    logger.info("ModerationService initialized successfully.")

moderation_service = ModerationService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    logger.info("Model server starting up...")
    moderation_service
    logger.info("Model server started successfully.")
    yield
    # Clean up the models and release the resources
    logger.info("Model server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/validate_harmful")
async def validate_harmful(payload: TextPayload):
    results = moderation_service.text_moderation_pipeline(payload.text, top_k=None)
    # The KoalaAI/Text-Moderation model returns 'OK' for safe content.
    # We should iterate through all labels and if any non-'OK' label has a high score,
    # we consider it harmful.
    for result in results:
        if result['label'] != 'OK' and result['score'] > 0.7:
            logger.warning(f"Harmful content detected: {result}")
            return {"is_harmful": True, "details": result}
    
    # If no harmful labels are found, we can return the top scoring label, which should be 'OK'.
    top_result = max(results, key=lambda x: x['score'])
    return {"is_harmful": False, "details": top_result}

@app.post("/validate_injection")
async def validate_injection(payload: TextPayload):
    result = moderation_service.injection_pipeline(payload.text)
    if result and result[0]['label'] == 'INJECTION' and result[0]['score'] > 0.8:
        logger.warning(f"Prompt injection attempt detected: {result}")
        return {"is_injection": True, "details": result}
    return {"is_injection": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 