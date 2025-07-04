import asyncio
import functools
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.config.settings import settings
from app.prompts.prompts import create_finetuned_prompt_content
from app.utils.app_exceptions import LLMGenerationError
from app.utils.logging_config import get_logger
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
    _llm = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _is_image_key(self, key: str) -> bool:
        """Determines if a key likely refers to an image."""
        key_lower = key.lower()
        image_key_suffixes = [
            "image",
            "img",
            "picture",
            "photo",
            "pic",
            "avatar",
            "thumbnail",
            "logo",
            "image_url",
            "img_url",
            "photo_url",
            "avatar_url",
            "logo_url",
            "profile_pic",
        ]
        return any(key_lower.endswith(suffix) for suffix in image_key_suffixes)

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
        primitive_rules = [
            r'string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws',
            r'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws',
            'boolean ::= ("true" | "false") ws',
            'null ::= "null" ws',
            "ws ::= [ \\t\\n]*",
        ]

        # Combine the root, dynamically generated rules, and primitives into the final grammar.
        # The dynamic rules are placed in the middle, as they may depend on primitives
        # and are required by the root.
        final_grammar = "\n".join(
            root_and_data_array_rules + list(rules.values()) + primitive_rules
        )
        logger.debug(f"Generated GBNF Grammar:\n{final_grammar}")
        return final_grammar

    async def initialize(self):
        if self._llm is None:
            logger.info(
                "Initializing and loading GGUF model via llama-cpp-python..."
            )
            try:
                model_path = settings.GGUF_MODEL_PATH
                if not os.path.exists(model_path):
                    raise LLMGenerationError(
                        f"GGUF model not found at path: {model_path}"
                    )

                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=8192,
                    n_gpu_layers=-1,
                    n_batch=512,
                    verbose=True,
                    chat_format="chatml",
                )
                logger.info("GGUF model loaded successfully.")
            except Exception as e:
                logger.error(
                    "Failed to load GGUF model", error=str(e), exc_info=True
                )
                raise LLMGenerationError(
                    "Could not load the GGUF language model."
                )

    async def _run_generation(
        self, input_data: List[Dict], count: int
    ) -> List[Dict]:
        await self.initialize()

        prompt_content = create_finetuned_prompt_content(input_data, count)

        # This system prompt must match the one used during fine-tuning
        # to properly activate the model's specialized knowledge.
        system_prompt = (
            "You are an expert assistant for generating synthetic data. "
            "Analyze the user's request and generate a detailed, high-quality synthetic dataset entry."
        )

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
                logger.info(
                    "Successfully created GBNF grammar from sample data."
                )
            except Exception as e:
                logger.warning(
                    f"Could not create GBNF grammar: {e}", exc_info=True
                )

        max_tokens = -1
        logger.info(
            f"Invoking GGUF model via Llama.cpp with GBNF grammar. Generating {count} items.",
            max_tokens=max_tokens,
        )

        try:
            loop = asyncio.get_running_loop()

            llm_call = functools.partial(
                self._llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.9,  # Increased temperature for more creativity
                top_p=0.95,
                stop=["<|im_end|>"],
                grammar=grammar,
            )

            output = await loop.run_in_executor(None, llm_call)

            generated_text = output["choices"][0]["message"]["content"].strip()

            try:
                parsed_output = json.loads(generated_text)
                return parsed_output.get("data", [])
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse JSON from grammar-constrained output",
                    error=str(e),
                    content=generated_text,
                )
                return []

        except Exception as e:
            logger.error(
                "Error during Llama.cpp generation", error=str(e), exc_info=True
            )
            raise LLMGenerationError(
                f"Failed to generate mock data with GGUF model. Error: {e}"
            )

    async def generate_mock_data(
        self, input_data: List[Dict], count: int
    ) -> List[Dict]:
        return await self._run_generation(input_data, count)


llm_service = LocalLLMService()
