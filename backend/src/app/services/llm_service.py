import os
import json
import re
import asyncio
import functools
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from llama_cpp import Llama
from app.config.settings import settings
from app.prompts.prompts import GENERATION_PROMPT
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
    _generation_prompt_template = GENERATION_PROMPT

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        if self._llm is None:
            logger.info("Initializing and loading GGUF model via llama-cpp-python...")
            try:
                model_path = settings.GGUF_MODEL_PATH
                if not os.path.exists(model_path):
                    raise LLMGenerationError(f"GGUF model not found at path: {model_path}")

                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,        # Context size
                    n_gpu_layers=-1,   # Offload all layers to GPU
                    n_batch=512,       # Batch size
                    verbose=True
                )
                logger.info("GGUF model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load GGUF model", error=str(e), exc_info=True)
                raise LLMGenerationError("Could not load the GGUF language model.")

    def _build_prompt(self, input_data: List[Dict], count: int) -> str:
        input_text = json.dumps(input_data, indent=2)
        
        prompt = self._generation_prompt_template.format(
            count=count,
            input_text=input_text
        )
        
        # This prompt structure is critical for the chat-finetuned model
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _extract_json_array(self, text: str) -> List[Dict]:
        """Extracts a JSON array from the model's raw output with improved handling of various formats."""
        logger.info("Attempting to parse JSON from LLM response")
        
        # Try to parse as a JSON array first (most ideal case)
        try:
            # Remove any extra brackets at the beginning
            if text.strip().startswith("[["):
                text = text.strip()[1:]
            if text.strip().endswith("]]"):
                text = text.strip()[:-1]
                
            # Clean potentially harmful whitespace or control characters
            text = text.strip()
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            logger.debug("Failed to parse direct JSON array, trying alternative methods")
        
        # Try to find individual JSON objects and combine them
        try:
            # Look for patterns like {...}\n\n{...} or consecutive JSON objects
            json_objects = []
            object_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            matches = re.finditer(object_pattern, text)
            
            for match in matches:
                try:
                    obj = json.loads(match.group(0))
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
                    
            if json_objects:
                logger.info(f"Successfully extracted {len(json_objects)} JSON objects")
                return json_objects
        except Exception as e:
            logger.warning(f"Error in regex-based JSON extraction: {str(e)}")
        
        # Final attempt: try to clean and restructure the output
        try:
            # Look for array-like structures and fix them
            if "[" in text and "]" in text:
                start_idx = text.find("[")
                end_idx = text.rfind("]")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_text = text[start_idx:end_idx+1]
                    # Fix double brackets if present
                    if json_text.startswith("[[") and not json_text.startswith("[[["):
                        json_text = json_text[1:]
                    if json_text.endswith("]]") and not json_text.endswith("]]]"):
                        json_text = json_text[:-1]
                        
                    result = json.loads(json_text)
                    if isinstance(result, list):
                        return result
        except json.JSONDecodeError as e:
            logger.error(f"All JSON parsing attempts failed: {str(e)}")
            
        logger.warning("Could not extract any valid JSON from the response", content=text)
        return []

    async def _run_generation(self, input_data: List[Dict], count: int) -> List[Dict]:
        await self.initialize() # Ensure model is loaded
        
        prompt = self._build_prompt(input_data, count)
        
        # Estimate required tokens to prevent cutting off the response
        max_tokens = min(8000, max(1024, count * 250)) 

        logger.info(f"Invoking GGUF model via Llama.cpp. Generating {count} items.", max_tokens=max_tokens)

        try:
            loop = asyncio.get_running_loop()
            
            # functools.partial is the correct way to pass kwargs to run_in_executor
            llm_call = functools.partial(
                self._llm,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["<|im_end|>"],
                echo=False,
            )
            
            output = await loop.run_in_executor(None, llm_call)
            
            generated_text = output['choices'][0]['text'].strip()
            
            return self._extract_json_array(generated_text)

        except Exception as e:
            logger.error("Error during Llama.cpp generation", error=str(e), exc_info=True)
            raise LLMGenerationError(f"Failed to generate mock data with GGUF model. Error: {e}")

    async def generate_mock_data(self, input_data: List[Dict], count: int) -> List[Dict]:
        return await self._run_generation(input_data, count)

# Singleton instance
llm_service = LocalLLMService() 