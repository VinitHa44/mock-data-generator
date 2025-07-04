import json


def create_finetuned_prompt_content(input_data: list[dict], count: int) -> str:
    """
    Creates a detailed and explicit prompt for the instruction-finetuned model
    to guide it toward generating high-quality, unique synthetic data.
    This content will be placed inside the 'user' role of a chat template.

    Args:
        input_data: The user's input JSON data (as a Python object).
        count: The number of mock data objects to generate.

    Returns:
        A string containing the formatted prompt content.
    """
    # Be very specific about generating NEW data that's different from examples
    instruction = f"Generate {count} NEW entries that follow the same structure as the examples but are completely different content. DO NOT copy or repeat any of the input examples. For image fields, use 'img_keyword+keyword+keyword' format with plus signs."

    # Convert the user's input data to a pretty JSON string
    input_text = json.dumps(input_data, indent=2)

    # Assemble the prompt content exactly as it was during training
    prompt_content = f"{instruction}\n\n" f"{input_text}"

    return prompt_content
