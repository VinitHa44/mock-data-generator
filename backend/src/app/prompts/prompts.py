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