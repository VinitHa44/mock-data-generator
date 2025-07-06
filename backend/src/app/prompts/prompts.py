import json
import random
import threading
from typing import Optional
from app.config.settings import settings

# Five carefully curated instructions that trigger different model activation patterns
# while maintaining the same semantic goal and format requirements
INSTRUCTION_VARIATIONS = [
    "Generate {count} NEW entries that follow the same structure as the examples but are completely different content. DO NOT copy or repeat any of the input examples. For image fields, use 'img_keyword+keyword+keyword' format with plus signs."
    "Generate {count} unique entries preserving the JSON schema. Do not include input examples. Convert image URLs to 'img_keyword+keyword+keyword' format.",
    "Create {count} new, unique mock records matching field types. Avoid repeating inputs. Image URLs must be 'img_keyword+keyword+keyword'.",
    "Produce {count} unique objects maintaining the sample structure. Exclude input items. All image URLs must be converted to 'img_keyword+keyword+keyword'.",
    "Add {count} unique items consistent with the sample's format. Do not duplicate from input. Image URLs must be 'img_keyword+keyword+keyword'.",
    "Generate {count} more unique entries like the ones above. Do not repeat input. Transform image URLs into 'img_keyword+keyword+keyword' format."
]

# Thread-safe counter for parallel instruction selection
_parallel_counter = 0
_parallel_lock = threading.Lock()

def get_instruction_for_context(count: int, context: str = "default", previous_instruction_index: Optional[int] = None) -> tuple[str, int]:
    """
    Returns an instruction based on the generation context to introduce variation
    and prevent repetitive outputs from the instruction-tuned model.
    
    The first instruction (index 0) is the primary/best instruction used for initial generations.
    Variations are only used for retries, parallel processing, and cache scenarios.
    
    Args:
        count: Number of items to generate
        context: Generation context ("retry", "parallel", "cache", "default")
        previous_instruction_index: Index of previously used instruction (for retry scenarios)
    
    Returns:
        Tuple of (formatted instruction string, instruction index)
    """
    global _parallel_counter
    
    if context == "retry" and previous_instruction_index is not None:
        # For retries, explicitly avoid the previously used instruction
        available_indices = [i for i in range(len(INSTRUCTION_VARIATIONS)) if i != previous_instruction_index]
        selected_index = random.choice(available_indices)
    elif context == "parallel":
        # For parallel generation, use a deterministic counter to ensure different instructions
        # This ensures each parallel worker gets a different instruction
        with _parallel_lock:
            selected_index = _parallel_counter % len(INSTRUCTION_VARIATIONS)
            _parallel_counter += 1
    elif context == "cache":
        # For cache scenarios, use random selection to avoid potential conflicts
        # with cached data that was generated using the primary instruction
        selected_index = random.randint(1, len(INSTRUCTION_VARIATIONS) - 1)  # Exclude index 0
    else:
        # For default context, always use the first (best) instruction
        selected_index = 0
    
    return INSTRUCTION_VARIATIONS[selected_index].format(count=count), selected_index

def create_finetuned_prompt_content(input_data: list[dict], count: int, context: str = "default", previous_instruction_index: Optional[int] = None) -> tuple[str, int]:
    """
    Creates a detailed and explicit prompt for the instruction-finetuned model
    to guide it toward generating high-quality, unique synthetic data.
    This content will be placed inside the 'user' role of a chat template.

    Args:
        input_data: The user's input JSON data (as a Python object).
        count: The number of mock data objects to generate.
        context: Generation context for instruction variation
        previous_instruction_index: Index of previously used instruction

    Returns:
        A tuple containing the formatted prompt content and the instruction index used
    """
    # Get context-appropriate instruction to introduce variation
    instruction, instruction_index = get_instruction_for_context(count, context, previous_instruction_index)

    # Convert the user's input data to a pretty JSON string
    input_text = json.dumps(input_data, indent=2)

    # Assemble the prompt content exactly as it was during training
    prompt_content = f"{instruction}\n\n{input_text}"

    return prompt_content, instruction_index


def get_system_prompt() -> str:
    """
    Returns the system prompt for the LLM.
    This prompt must match the one used during fine-tuning.
    """
    return settings.LLM_SYSTEM_PROMPT


def get_grammar_rules() -> dict:
    """
    Returns the base grammar rules for JSON generation.
    """
    return {
        'string': r'string ::= "\"" ([^"\\\x00-\x1F\x7F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws',
        'number': r'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws',
        'boolean': 'boolean ::= ("true" | "false" | "1" | "0") ws',
        'strict_boolean': 'strict-boolean ::= ("true" | "false") ws',
        'null': 'null ::= "null" ws',
        'whitespace': "ws ::= [ \\t\\n\\r]*",
    }
