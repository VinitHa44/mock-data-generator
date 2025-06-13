import os
import json
import asyncio
import functools
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from llama_cpp import Llama

from app.config.settings import settings
from app.prompts.prompts import USER_PROMPT_TEMPLATE, FINE_TUNING_INSTRUCTIONS
from app.utils.logging_config import get_logger
from app.utils.app_exceptions import LLMGenerationError

logger = get_logger(__name__)

class LLMInterface(ABC):
    @abstractmethod
    async def generate_mock_data(self, input_data: List[Dict], count: int) -> List[Dict]:
        pass

class LocalLLMService(LLMInterface):
    _instance = None
    _llm = None
    _user_prompt_template = USER_PROMPT_TEMPLATE
    _fine_tuning_instructions = FINE_TUNING_INSTRUCTIONS

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_schema_for_value(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Recursively generates a JSON schema for a given value, with special
        pattern enforcement for image-related fields.
        """
        # First, handle structural types recursively.
        if isinstance(value, dict):
            properties = {k: self._get_schema_for_value(k, v) for k, v in value.items()}
            return {"type": "object", "properties": properties, "required": list(value.keys())}
        
        if isinstance(value, list):
            if not value:
                return {"type": "array", "items": {}}
            # Recurse on the first item, losing the parent key as it's not relevant for items in a list.
            return {"type": "array", "items": self._get_schema_for_value("", value[0])}

        # Now, handle primitive types and perform image detection only on strings.
        if isinstance(value, str):
            key_lower = key.lower()
            val_lower = value.lower()

            # --- Start of Image Field Detection Logic ---
            is_image = False
            key_triggers = ['image', 'img', 'picture', 'photo', 'pic', 'avatar', 'thumbnail', 'logo']
            if any(trigger in key_lower for trigger in key_triggers):
                is_image = True

            if not is_image:
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']
                if any(val_lower.endswith(ext) for ext in image_extensions):
                    is_image = True
            
            if not is_image:
                url_keywords = ['/images/', '/photos/', 'unsplash.com', 'pexels.com', 'pixabay.com']
                if any(keyword in val_lower for keyword in url_keywords):
                    is_image = True
            
            if not is_image and key_lower.endswith(('_url', '_uri')):
                 if any(kw in val_lower for kw in ['image', 'photo', 'picture']):
                    is_image = True
            # --- End of Image Field Detection Logic ---

            if is_image:
                return {
                    "type": "string",
                    "description": "A string of 1-5 keywords separated by plus signs.",
                    "pattern": r"^([a-zA-Z_]+)(?:\+[a-zA-Z_]+){0,4}$"
                }
            
            # Default for regular strings
            return {"type": "string"}

        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        
        # Fallback for other types like None.
        return {"type": "string"}

    def _create_json_schema_from_sample(self, sample_data: Dict[str, Any], count: int) -> Dict[str, Any]:
        """Dynamically creates a JSON schema, enforcing item count and image patterns."""
        item_schema = self._get_schema_for_value("root", sample_data)
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": item_schema,
                    "minItems": count,
                    "maxItems": count
                }
            },
            "required": ["data"]
        }

    async def initialize(self):
        if self._llm is None:
            logger.info("Initializing and loading GGUF model via llama-cpp-python...")
            try:
                model_path = settings.GGUF_MODEL_PATH
                if not os.path.exists(model_path):
                    raise LLMGenerationError(f"GGUF model not found at path: {model_path}")

                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=8192,
                    n_gpu_layers=-1,
                    n_batch=512,
                    verbose=True,
                    chat_format="chatml"
                )
                logger.info("GGUF model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load GGUF model", error=str(e), exc_info=True)
                raise LLMGenerationError("Could not load the GGUF language model.")

    async def _run_generation(self, input_data: List[Dict], count: int) -> List[Dict]:
        await self.initialize()
        
        instruction_template = random.choice(self._fine_tuning_instructions)
        instruction = instruction_template.format(n=count)

        user_prompt_content = self._user_prompt_template.format(
            instruction=instruction,
            input_text=json.dumps(input_data, indent=2),
            count=count
        )
        
        messages = [{"role": "user", "content": user_prompt_content}]
        
        json_schema = None
        if input_data:
            try:
                json_schema = self._create_json_schema_from_sample(input_data[0], count)
                logger.info("Successfully created JSON schema with item count and image pattern enforcement.")
            except Exception as e:
                logger.warning(f"Could not create JSON schema: {e}", exc_info=True)

        max_tokens = -1
        logger.info(f"Invoking GGUF model via Llama.cpp. Generating {count} items.", max_tokens=max_tokens)

        try:
            loop = asyncio.get_running_loop()
            
            response_format = {"type": "json_object", "schema": json_schema} if json_schema else None

            llm_call = functools.partial(
                self._llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["<|im_end|>"],
                response_format=response_format,
            )
            
            output = await loop.run_in_executor(None, llm_call)
            
            generated_text = output['choices'][0]['message']['content'].strip()

            try:
                parsed_output = json.loads(generated_text)
                return parsed_output.get('data', [])
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from grammar-constrained output", error=str(e), content=generated_text)
                return []

        except Exception as e:
            logger.error("Error during Llama.cpp generation", error=str(e), exc_info=True)
            raise LLMGenerationError(f"Failed to generate mock data with GGUF model. Error: {e}")

    async def generate_mock_data(self, input_data: List[Dict], count: int) -> List[Dict]:
        return await self._run_generation(input_data, count)

llm_service = LocalLLMService()