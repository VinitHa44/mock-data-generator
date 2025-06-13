import json

def create_finetuned_prompt_content(input_data: list[dict], count: int) -> str:
    """
    Creates the prompt content for the instruction-finetuned model.
    This content will be placed inside the 'user' role of a chat template.

    Args:
        input_data: The user's input JSON data (as a Python object).
        count: The number of mock data objects to generate.

    Returns:
        A string containing the formatted prompt content.
    """
    # 1. Use the single, specific instruction for generating unique objects.
    instruction = (
        "For the {n} objects you generate, it is critical that each one is unique and does not repeat or copy "
        "any of the other objects in the output. Prioritize creativity and variation in your response."
    ).format(n=count)

    # 2. Convert the user's input data to a pretty JSON string
    input_text = json.dumps(input_data, indent=4)

    # 3. Assemble the prompt content. The chat template tokens (e.g., <|im_start|>)
    # will be added by the Llama.cpp library based on the `chat_format`.
    prompt_content = (
        f"{instruction}\n\n"
        f"{input_text}"
    )

    return prompt_content