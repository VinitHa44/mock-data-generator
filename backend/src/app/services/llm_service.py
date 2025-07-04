import asyncio
import functools
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import threading

from app.config.settings import settings
from app.prompts.prompts import create_finetuned_prompt_content, get_system_prompt, get_grammar_rules
from app.utils.app_exceptions import LLMGenerationError
from app.utils.logging_config import get_logger
from app.utils.performance_monitor import PerformanceTracker
from llama_cpp import Llama, LlamaGrammar

logger = get_logger(__name__)


class LLMInterface(ABC):
    @abstractmethod
    async def generate_mock_data(
        self, input_data: List[Dict], count: int
    ) -> List[Dict]:
        pass


class LocalLLMService(LLMInterface):
    _instance = None
    _lock = threading.Lock()
    _model_pool = []
    _pool_size = 0
    _executor = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    self._pool_size = getattr(settings, 'LLM_POOL_SIZE', 3)
                    self._executor = ThreadPoolExecutor(max_workers=self._pool_size)
                    self._initialized = True

    def _is_image_key(self, key: str) -> bool:
        """Determines if a key likely refers to an image."""
        key_lower = key.lower()
        return any(key_lower.endswith(suffix) for suffix in settings.IMAGE_KEYWORDS)

    def _build_gbnf_rule_for_value(
        self,
        key: str,
        value: Any,
        rules: Dict[str, str],
        rule_name_prefix: str = "",
    ) -> str:
        """
        Recursively builds GBNF rules for a given value, creating named, reusable
        rules for complex types like objects and arrays to produce an efficient grammar.
        """
        value_type = type(value).__name__
        rule_name = f"{rule_name_prefix}-{key}" if rule_name_prefix else key

        if isinstance(value, dict):
            # For objects, create a new named rule.
            obj_rule_name = f"{rule_name}-object"
            if obj_rule_name not in rules:
                # Add a placeholder to break recursion if the same object is nested.
                rules[obj_rule_name] = ""
                properties = []
                for i, (k, v) in enumerate(value.items()):
                    separator = ' "," ws ' if i < len(value) - 1 else ""
                    # Recursively get the rule for the property's value.
                    value_rule = self._build_gbnf_rule_for_value(
                        k, v, rules, rule_name_prefix=obj_rule_name
                    )
                    # JSON key must be a quoted string.
                    properties.append(
                        f'"\\"{k}\\"" ws ":" ws {value_rule}{separator}'
                    )

                # Define the object rule.
                rules[obj_rule_name] = (
                    f'{obj_rule_name} ::= "{{" ws {"".join(properties)} "}}" ws'
                )
            return obj_rule_name

        elif isinstance(value, list):
            # For arrays, create a rule for its items.
            if not value:
                return '"[]" ws'  # Handle empty arrays.

            # Assume all items in the array have the same structure as the first.
            # Pass the original key 'key' instead of a derived key so that _is_image_key
            # can be applied correctly to the items within the list.
            item_rule_name = self._build_gbnf_rule_for_value(
                key, value[0], rules, rule_name_prefix=rule_name_prefix
            )
            array_rule_name = f"array-of-{item_rule_name}"

            # Define the array rule, referencing the item rule.
            # This handles a variable number of items. For fixed-size, a different rule is needed.
            if array_rule_name not in rules:
                rules[array_rule_name] = (
                    f'{array_rule_name} ::= "[" ws {item_rule_name} ("," ws {item_rule_name})* ws "]"'
                )
            return array_rule_name

        elif isinstance(value, str):
            # All strings are handled by the 'string' rule now.
            return "string"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            # Fallback for null or other unexpected types.
            return "null"

    def _create_gbnf_grammar_from_sample(
        self, sample_data: Dict[str, Any], count: int
    ) -> str:
        """
        Dynamically creates a GBNF grammar string from a sample data object. This
        generates a structured and efficient grammar by creating named, reusable
        rules for objects and arrays, preventing massive, inline rule definitions.
        """
        rules = {}
        # Start building the grammar from the top-level 'item' structure.
        item_rule_name = self._build_gbnf_rule_for_value(
            "item", sample_data, rules
        )

        # Define the root of the grammar, which expects a 'data' array.
        root_and_data_array_rules = [
            'root ::= "{" ws "\\"data\\":" ws data-array "}"',
            # This rule enforces the exact number of items requested.
            f'data-array ::= "[" ws {item_rule_name} ("," ws {item_rule_name}){{{count - 1}}} ws "]"',
        ]

        # Define primitive types that form the building blocks of the JSON.
        # Enhanced string rule to prevent control characters
        primitive_rules = list(get_grammar_rules().values())

        # Combine the root, dynamically generated rules, and primitives into the final grammar.
        # The dynamic rules are placed in the middle, as they may depend on primitives
        # and are required by the root.
        final_grammar = "\n".join(
            root_and_data_array_rules + list(rules.values()) + primitive_rules
        )
        logger.debug(f"Generated GBNF Grammar:\n{final_grammar}")
        return final_grammar

    def _create_llm_instance(self) -> Llama:
        """Create a new LLM instance with optimized settings."""
        model_path = settings.GGUF_MODEL_PATH
        if not os.path.exists(model_path):
            raise LLMGenerationError(f"GGUF model not found at path: {model_path}")

        return Llama(
            model_path=model_path,
            n_ctx=4096,  # Reduced context for better memory efficiency
            n_gpu_layers=-1,
            n_batch=256,  # Reduced batch size for better concurrency
            verbose=False,  # Disable verbose for production
            chat_format="chatml",
            n_threads=2,  # Limit threads per instance
        )

    async def initialize(self):
        """Initialize the model pool with multiple instances."""
        if not self._model_pool:
            logger.info(f"Initializing LLM model pool with {self._pool_size} instances...")
            
            for i in range(self._pool_size):
                try:
                    llm_instance = self._create_llm_instance()
                    self._model_pool.append(llm_instance)
                    logger.info(f"Created LLM instance {i+1}/{self._pool_size}")
                except Exception as e:
                    logger.error(f"Failed to create LLM instance {i+1}: {e}")
                    raise LLMGenerationError(f"Could not initialize LLM model pool: {e}")
            
            logger.info("LLM model pool initialized successfully")

    def _get_available_model(self) -> Llama:
        """Get an available model from the pool (simple round-robin for now)."""
        if not self._model_pool:
            raise LLMGenerationError("Model pool not initialized")
        
        # Simple round-robin selection - in production, you might want more sophisticated load balancing
        model = self._model_pool.pop(0)
        self._model_pool.append(model)
        return model

    async def _run_generation_with_model(
        self, llm_instance: Llama, input_data: List[Dict], count: int
    ) -> List[Dict]:
        """Run generation with a specific model instance."""
        prompt_content = create_finetuned_prompt_content(input_data, count)

        # This system prompt must match the one used during fine-tuning
        # to properly activate the model's specialized knowledge.
        system_prompt = get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content},
        ]

        grammar = None
        if input_data:
            try:
                # Use the new, robust GBNF grammar generation.
                gbnf_string = self._create_gbnf_grammar_from_sample(
                    input_data[0], count
                )
                grammar = LlamaGrammar.from_string(gbnf_string, verbose=False)
                logger.debug("Successfully created GBNF grammar from sample data.")
            except Exception as e:
                logger.warning(f"Could not create GBNF grammar: {e}")

        # Set a reasonable max_tokens to prevent extremely large responses
        max_tokens = min(settings.MAX_TOTAL_TOKENS, count * settings.MAX_TOKENS_PER_ITEM)  # Limit tokens based on count
        logger.debug(f"Invoking GGUF model with GBNF grammar. Generating {count} items with max_tokens={max_tokens}.")

        try:
            loop = asyncio.get_running_loop()

            llm_call = functools.partial(
                llm_instance.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                stop=["<|im_end|>"],
                grammar=grammar,
            )

            output = await loop.run_in_executor(self._executor, llm_call)

            generated_text = output["choices"][0]["message"]["content"].strip()

            # Check for truncation issues
            if len(generated_text) > settings.LARGE_RESPONSE_THRESHOLD:  # Large response threshold
                logger.warning(f"Large response detected ({len(generated_text)} chars), checking for truncation")
                generated_text = self._handle_large_response(generated_text)

            # Clean and parse the generated text
            parsed_output = self._parse_and_clean_json(generated_text)
            
            # If parsing failed completely, save the raw output for debugging
            if not parsed_output.get("data"):
                self._save_debug_output(generated_text, count)
            
            return parsed_output.get("data", [])

        except Exception as e:
            logger.error(
                "Error during Llama.cpp generation", error=str(e), exc_info=True
            )
            raise LLMGenerationError(
                f"Failed to generate mock data with GGUF model. Error: {e}"
            )

    def _handle_large_response(self, text: str) -> str:
        """Handle large responses that might be truncated."""
        import re
        
        # Look for the last complete object in the response
        # Find all complete JSON objects
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        objects = re.findall(object_pattern, text)
        
        if objects:
            # Find the last complete object
            last_complete_object = objects[-1]
            last_object_end = text.rfind(last_complete_object) + len(last_complete_object)
            
            # Truncate the text to the end of the last complete object
            if last_object_end < len(text):
                logger.info(f"Truncating response at last complete object (position {last_object_end})")
                text = text[:last_object_end]
                
                # Ensure the JSON structure is complete
                if text.endswith(','):
                    text = text[:-1]
                if not text.endswith(']'):
                    text += ']'
                if not text.endswith('}'):
                    text += '}'
        
        return text

    def _save_debug_output(self, text: str, count: int):
        """Save raw LLM output for debugging when parsing fails."""
        import os
        import uuid
        
        debug_dir = settings.DEBUG_OUTPUT_DIR
        os.makedirs(debug_dir, exist_ok=True)
        
        filename = f"{debug_dir}/failed_parse_{count}items_{uuid.uuid4().hex[:8]}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Failed JSON parse for {count} items\n")
                f.write("=" * 50 + "\n")
                f.write(text)
            
            logger.error(f"Saved failed parse output to {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug output: {e}")

    def _parse_and_clean_json(self, text: str) -> Dict:
        """
        Parse and clean JSON output from LLM, handling common issues like
        control characters, incomplete JSON, and malformed structures.
        """
        if not text:
            logger.warning("Empty text received from LLM")
            return {"data": []}
        
        # Log the raw text for debugging (truncated)
        logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}...")
        
        # First, try to find JSON content in the text
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start == -1 or json_end == -1:
            logger.warning("No valid JSON structure found in LLM output")
            return {"data": []}
        
        # Extract the JSON portion
        json_text = text[json_start:json_end + 1]
        
        # Try multiple parsing strategies
        parsing_strategies = [
            ("direct_parse", lambda: self._try_direct_parse(json_text)),
            ("cleaned_parse", lambda: self._try_cleaned_parse(json_text)),
            ("chunked_parse", lambda: self._try_chunked_parse(json_text)),
            ("regex_extract", lambda: self._try_regex_extract(json_text)),
            ("manual_construct", lambda: self._try_manual_construct(json_text)),
        ]
        
        for strategy_name, strategy_func in parsing_strategies:
            try:
                result = strategy_func()
                if result and result.get("data"):
                    logger.info(f"Successfully parsed JSON using {strategy_name}")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                continue
        
        logger.error("All JSON parsing strategies failed")
        return {"data": []}

    def _try_direct_parse(self, json_text: str) -> Dict:
        """Try direct JSON parsing."""
        return json.loads(json_text)

    def _try_cleaned_parse(self, json_text: str) -> Dict:
        """Try parsing after cleaning control characters."""
        cleaned_text = self._clean_json_text(json_text)
        return json.loads(cleaned_text)

    def _try_chunked_parse(self, json_text: str) -> Dict:
        """Try parsing by breaking into smaller chunks."""
        import re
        
        # Look for the data array specifically
        data_match = re.search(r'"data"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
        if not data_match:
            return {"data": []}
        
        data_content = data_match.group(1)
        
        # Split the data content into individual objects
        objects = []
        brace_count = 0
        current_object = ""
        
        for char in data_content:
            current_object += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Complete object found
                    try:
                        obj = json.loads(current_object.strip())
                        objects.append(obj)
                    except json.JSONDecodeError:
                        # Try to fix common issues in this object
                        fixed_obj = self._fix_object_json(current_object.strip())
                        if fixed_obj:
                            objects.append(fixed_obj)
                    current_object = ""
        
        # Handle incomplete object at the end (truncated response)
        if current_object.strip() and brace_count > 0:
            logger.warning("Detected truncated JSON object, attempting to complete it")
            completed_obj = self._complete_truncated_object(current_object.strip())
            if completed_obj:
                objects.append(completed_obj)
        
        return {"data": objects}

    def _complete_truncated_object(self, truncated_text: str) -> Dict:
        """Attempt to complete a truncated JSON object."""
        import re
        
        # Count opening and closing braces
        open_braces = truncated_text.count('{')
        close_braces = truncated_text.count('}')
        
        # Add missing closing braces
        missing_braces = open_braces - close_braces
        if missing_braces > 0:
            truncated_text += '}' * missing_braces
        
        # Fix common truncation issues
        # Remove trailing commas
        truncated_text = re.sub(r',\s*$', '', truncated_text)
        
        # If the object ends with a comma, remove it
        truncated_text = re.sub(r',\s*}', '}', truncated_text)
        
        # Try to parse the completed object
        try:
            return json.loads(truncated_text)
        except json.JSONDecodeError:
            # Try more aggressive fixes
            return self._aggressive_fix_truncated_object(truncated_text)

    def _aggressive_fix_truncated_object(self, text: str) -> Dict:
        """More aggressive fixing for severely truncated objects."""
        import re
        
        # If the text doesn't start with {, try to find the start
        if not text.startswith('{'):
            start_idx = text.find('{')
            if start_idx != -1:
                text = text[start_idx:]
        
        # If the text doesn't end with }, try to find the end
        if not text.endswith('}'):
            end_idx = text.rfind('}')
            if end_idx != -1:
                text = text[:end_idx + 1]
        
        # Remove any trailing incomplete property
        # Look for patterns like "property": "incomplete value
        incomplete_pattern = r'"[^"]*"\s*:\s*"[^"]*$'
        text = re.sub(incomplete_pattern, '', text)
        
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*$', '', text)
        
        # Ensure proper object structure
        if not text.startswith('{'):
            text = '{' + text
        if not text.endswith('}'):
            text = text + '}'
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not fix truncated object: {text[:100]}...")
            return None

    def _try_regex_extract(self, json_text: str) -> Dict:
        """Try extracting data using regex patterns."""
        import re
        
        # Multiple regex patterns to try
        patterns = [
            r'"data"\s*:\s*\[(.*?)\]',
            r'\[(.*?)\]',  # Just find any array
            r'\{[^{}]*"data"[^{}]*\[(.*?)\][^{}]*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, json_text, re.DOTALL)
            if matches:
                try:
                    # Try to parse the largest match
                    largest_match = max(matches, key=len)
                    data_array = f"[{largest_match}]"
                    data_items = json.loads(data_array)
                    return {"data": data_items}
                except json.JSONDecodeError:
                    continue
        
        return {"data": []}

    def _try_manual_construct(self, json_text: str) -> Dict:
        """Try to manually construct valid JSON from fragments."""
        import re
        
        # Extract all potential objects
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        objects = re.findall(object_pattern, json_text)
        
        valid_objects = []
        for obj_text in objects:
            try:
                obj = json.loads(obj_text)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                # Try to fix this object
                fixed_obj = self._fix_object_json(obj_text)
                if fixed_obj:
                    valid_objects.append(fixed_obj)
        
        return {"data": valid_objects}

    def _fix_object_json(self, obj_text: str) -> Dict:
        """Fix common JSON issues in a single object."""
        import re
        
        # Remove trailing commas
        obj_text = re.sub(r',(\s*[}\]])', r'\1', obj_text)
        
        # Fix missing quotes around property names
        obj_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', obj_text)
        
        # Fix unescaped quotes in string values
        obj_text = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\2\3"', obj_text)
        
        # Remove control characters
        obj_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', obj_text)
        
        try:
            return json.loads(obj_text)
        except json.JSONDecodeError:
            return None

    def _clean_json_text(self, text: str) -> str:
        """
        Clean JSON text by removing control characters and fixing common issues.
        """
        import re
        
        # Remove control characters except newlines and tabs
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix common LLM output issues
        cleaned = re.sub(r'\\n', ' ', cleaned)  # Replace literal \n with space
        cleaned = re.sub(r'\\t', ' ', cleaned)  # Replace literal \t with space
        cleaned = re.sub(r'\\r', ' ', cleaned)  # Replace literal \r with space
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common quote issues
        cleaned = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', cleaned)
        
        return cleaned.strip()

    def _fix_common_json_issues(self, text: str) -> str:
        """
        Fix common JSON issues that LLMs often produce.
        """
        import re
        
        # Fix trailing commas in arrays and objects
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix missing quotes around property names
        text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)
        
        # Fix unescaped quotes in string values
        text = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\2\3"', text)
        
        # Ensure proper array structure
        if '"data"' not in text:
            # Try to wrap the content in a data array
            if text.startswith('[') and text.endswith(']'):
                text = f'{{"data": {text}}}'
        
        return text

    async def _run_generation(
        self, input_data: List[Dict], count: int
    ) -> List[Dict]:
        async with PerformanceTracker("generation", count=count, method="single"):
            await self.initialize()
            llm_instance = self._get_available_model()
            return await self._run_generation_with_model(llm_instance, input_data, count)

    async def generate_mock_data_batch(
        self, input_data: List[Dict], total_count: int, batch_size: int = None
    ) -> List[Dict]:
        """
        Generate data in parallel batches for better performance.
        This is the main method for handling large counts efficiently.
        """
        async with PerformanceTracker("batch", batch_size=total_count):
            await self.initialize()
            
            if batch_size is None:
                batch_size = settings.LLM_BATCH_SIZE
            
            if total_count <= batch_size:
                # For small counts, use single generation
                return await self._run_generation(input_data, total_count)
            
            # Try with the specified batch size first
            result = await self._try_batch_generation(input_data, total_count, batch_size)
            
            # If that fails, try with smaller batch sizes
            if not result or len(result) < total_count * 0.8:  # Less than 80% success
                logger.warning(f"Batch generation with size {batch_size} failed, trying smaller batches")
                
                smaller_batch_sizes = [15, 10, 5]
                for smaller_batch in smaller_batch_sizes:
                    if smaller_batch >= batch_size:
                        continue
                    
                    logger.info(f"Retrying with batch size {smaller_batch}")
                    fallback_result = await self._try_batch_generation(input_data, total_count, smaller_batch)
                    
                    if fallback_result and len(fallback_result) >= total_count * 0.8:
                        logger.info(f"Successfully generated {len(fallback_result)} items with batch size {smaller_batch}")
                        return fallback_result[:total_count]
            
            return result[:total_count] if result else []

    async def _try_batch_generation(
        self, input_data: List[Dict], total_count: int, batch_size: int
    ) -> List[Dict]:
        """Try batch generation with a specific batch size."""
        # Calculate number of batches needed
        num_batches = (total_count + batch_size - 1) // batch_size
        remaining_count = total_count
        
        logger.info(f"Generating {total_count} items in {num_batches} batches of {batch_size}")
        
        # Create tasks for parallel execution
        tasks = []
        for i in range(num_batches):
            current_batch_size = min(batch_size, remaining_count)
            remaining_count -= current_batch_size
            
            # Get a model instance for this batch
            llm_instance = self._get_available_model()
            
            # Create task for this batch
            task = self._run_generation_with_model(llm_instance, input_data, current_batch_size)
            tasks.append(task)
        
        # Execute all batches in parallel with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and handle any errors
        all_data = []
        failed_batches = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i+1} failed: {result}")
                failed_batches.append(i)
            else:
                all_data.extend(result)
        
        # Retry failed batches with exponential backoff
        if failed_batches:
            logger.warning(f"Retrying {len(failed_batches)} failed batches")
            for batch_idx in failed_batches:
                retry_success = await self._retry_failed_batch(
                    input_data, batch_size, all_data, batch_idx
                )
                if not retry_success:
                    logger.error(f"Batch {batch_idx + 1} failed after retry")
        
        return all_data

    async def _retry_failed_batch(
        self, input_data: List[Dict], batch_size: int, all_data: List[Dict], batch_idx: int
    ) -> bool:
        """Retry a failed batch with exponential backoff."""
        max_retries = settings.LLM_RETRY_ATTEMPTS
        base_delay = 1.0
        
        for retry in range(max_retries):
            try:
                delay = base_delay * (2 ** retry)
                logger.info(f"Retrying batch {batch_idx + 1}, attempt {retry + 1}/{max_retries}")
                
                await asyncio.sleep(delay)
                
                llm_instance = self._get_available_model()
                result = await self._run_generation_with_model(llm_instance, input_data, batch_size)
                
                if result:
                    all_data.extend(result)
                    logger.info(f"Batch {batch_idx + 1} succeeded on retry {retry + 1}")
                    return True
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} retry {retry + 1} failed: {e}")
        
        return False

    async def generate_mock_data(
        self, input_data: List[Dict], count: int
    ) -> List[Dict]:
        """
        Main entry point for data generation.
        Automatically chooses between single generation and batch generation based on count.
        """
        # Use batch generation for counts > threshold for better performance
        if count > settings.LLM_BATCH_THRESHOLD:
            return await self._adaptive_batch_generation(input_data, count)
        else:
            return await self._run_generation(input_data, count)

    async def _adaptive_batch_generation(self, input_data: List[Dict], total_count: int) -> List[Dict]:
        """Adaptive batch generation that adjusts batch size based on success rate."""
        batch_sizes = [settings.LLM_BATCH_SIZE, 15, 10, 5]
        
        for batch_size in batch_sizes:
            logger.info(f"Trying batch generation with size {batch_size}")
            
            try:
                result = await self.generate_mock_data_batch(input_data, total_count, batch_size)
                
                # Check if we got a reasonable number of items
                success_rate = len(result) / total_count if total_count > 0 else 0
                
                if success_rate >= 0.8:  # 80% success rate threshold
                    logger.info(f"Batch generation successful with size {batch_size}: {len(result)}/{total_count} items")
                    return result
                else:
                    logger.warning(f"Batch generation with size {batch_size} only achieved {success_rate:.1%} success rate")
                    
            except Exception as e:
                logger.error(f"Batch generation with size {batch_size} failed: {e}")
                continue
        
        # If all batch sizes fail, try single generation for smaller counts
        logger.warning("All batch sizes failed, trying single generation")
        if total_count <= 10:
            return await self._run_generation(input_data, total_count)
        else:
            # Try generating in smaller chunks
            chunks = []
            remaining = total_count
            
            while remaining > 0:
                chunk_size = min(5, remaining)
                try:
                    chunk = await self._run_generation(input_data, chunk_size)
                    chunks.extend(chunk)
                    remaining -= len(chunk)
                except Exception as e:
                    logger.error(f"Chunk generation failed: {e}")
                    break
            
            return chunks[:total_count]

    async def shutdown(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        self._model_pool.clear()
        logger.info("LLM service shutdown complete")


llm_service = LocalLLMService()
