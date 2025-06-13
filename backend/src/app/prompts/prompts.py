# GENERATION_PROMPT = """Your task is to generate a JSON array of mock data based on an example. You must follow all rules perfectly.

# ---

# **CRITICAL NON-NEGOTIABLE RULE: IMAGE URLS**

# This is the most important instruction. You are REQUIRED to find any field in the user's example that contains a URL pointing to an image. In the data you generate, you MUST replace that URL with a string of 1 to 5 descriptive keywords separated by plus signs (+).

# *   **CORRECT:** `"avatar": "woman+smiling+professional"`
# *   **INCORRECT:** `"avatar": "https://example.com/some/fake/image.jpg"`

# This is not optional. You must perform this replacement.

# ---

# **OTHER RULES:**
# 1.  Generate exactly `{count}` new, unique objects.
# 2.  Do NOT copy the provided input examples.
# 3.  The output must be a single, valid JSON array with no extra text.

# ---

# **EXAMPLE OF YOUR TASK:**

# **USER INPUT:**
# ```json
# [
#     {{
#         "id": 1,
#         "name": "John Doe",
#         "email": "john.doe@example.com",
#         "avatar": "https://example.com/avatars/john.jpg"
#     }}
# ]
# ```

# **YOUR REQUIRED OUTPUT (for count=2):**
# ```json
# [
#     {{
#         "id": 2,
#         "name": "Jane Smith",
#         "email": "jane.smith@example.com",
#         "avatar": "woman+smiling+professional"
#     }},
#     {{
#         "id": 3,
#         "name": "Peter Jones",
#         "email": "peter.jones@example.com",
#         "avatar": "man+outdoors+hiking"
#     }}
# ]
# ```

# ---

# **YOUR ACTUAL TASK:**

# **USER INPUT:**
# {input_text}

# **YOUR REQUIRED OUTPUT (for count={count}):**
# """

GENERATION_PROMPT = """Your task is to generate a JSON array of mock data based on the provided input data. You must follow all rules perfectly.

---

**CRITICAL NON-NEGOTIABLE RULE: IMAGE URLS**

This is the most important instruction. You are REQUIRED to find any field in the input data that contains a URL pointing to an image. In the data you generate, you MUST replace that URL with a string of 1 to 5 descriptive keywords separated by plus signs (+).

*   **CORRECT:** `"avatar": "woman+smiling+professional"`
*   **INCORRECT:** `"avatar": "https://example.com/some/fake/image.jpg"`

This is not optional. You must perform this replacement.

---

**OTHER RULES:**
1.  Generate exactly `{count}` new, unique objects.
2.  Do NOT copy the provided input data.
3.  The output must be a single, valid JSON array with no extra text.
4.  Use the structure and field types from the input data, but generate new, unique values.
5.  Maintain the same data types for each field as shown in the input.

---

**YOUR TASK:**

**INPUT DATA:**
{input_text}

**REQUIRED OUTPUT:**
Generate exactly `{count}` new objects following the same structure as the input data, but with new, unique values. Replace any image URLs with descriptive keywords.
"""

FINE_TUNING_INSTRUCTIONS = [
    "Generate {n} entries preserving the JSON schema.",
    "Create {n} mock records that match the field types.",
    "Produce {n} objects maintaining the sample structure.",
    "Add {n} new items adhering to the existing format.",
    "Extend this dataset with {n} entries following the schema.",
    "Make {n} synthetic records that respect the key hierarchy.",
    "Construct {n} more JSON objects with the same layout.",
    "Build {n} additional data points preserving value types.",
    "Supply {n} new records consistent with these fields.",
    "Fabricate {n} examples that mirror this object template.",
    "Deliver {n} items maintaining the relationships shown.",
    "Produce {n} records that adhere to the defined schema.",
    "Generate {n} new entries preserving field order and types.",
    "Create {n} supplementary JSON objects that follow this model.",
    "Add {n} mock items consistent with the sample's format.",
    "Generate {n} more entries like the ones above.",
    "Create {n} additional examples similar to these.",
    "Add {n} more items that follow this pattern.",
    "Make {n} new samples resembling the input.",
    "Extend the set with {n} extra data points.",
    "Fill in {n} further objects based on these seeds.",
    "Append {n} mock entries of the same kind.",
    "Build {n} further records matching this style.",
    "Produce {n} more similar entries.",
    "Add {n} new examples continuing the pattern.",
    "Generate {n} more.",
    "Add {n} entries.",
    "Create {n}.",
    "Produce {n} new records.",
    "Give me {n} additional items.",
    "Let's have {n} more like these.",
    "Spin up {n} extra mock records.",
    "I need {n} new entries like that.",
    "Drop {n} more samples into this list.",
    "Can you come up with {n} additional items?"
]

# This template is structured to be maximally effective with instruction-tuned models.
USER_PROMPT_TEMPLATE = """{instruction}

**CRITICAL RULES:**
1.  **NO COPYING:** Your response must contain entirely new, creative, and diverse data. Do NOT repeat the values from the input data below.
2.  **IMAGE KEYWORDS:** For any JSON key that contains "image", "avatar", or "picture", the value MUST be a string of 1-5 descriptive keywords, separated by plus signs (+). It MUST NOT be a URL.
    -   CORRECT: "avatar": "woman+smiling+professional"
    -   INCORRECT: "avatar": "https://example.com/image.jpg"

**INPUT DATA:**
{input_text}

Your JSON array of {count} new objects:
"""