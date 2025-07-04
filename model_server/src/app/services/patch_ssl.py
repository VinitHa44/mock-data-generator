#!/usr/bin/env python3
"""
Patch HuggingFace library to disable SSL verification for local development.
WARNING: This is for development purposes only! Don't use in production!
"""

import os
import ssl

# Force SSL context to be unverified
ssl._create_default_https_context = ssl._create_unverified_context
print("🔒 SSL verification disabled for local development")

# Environment variables that can help with SSL issues
os.environ["CURL_CA_BUNDLE"] = ""  # Disable cURL certificate verification
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

# Try to patch huggingface_hub directly
try:
    import requests
    import urllib3

    # Monkey patch requests to disable SSL verification
    old_send = requests.adapters.HTTPAdapter.send

    def new_send(*args, **kwargs):
        kwargs["verify"] = False
        return old_send(*args, **kwargs)

    requests.adapters.HTTPAdapter.send = new_send

    # Patch urllib3 to disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Try to patch huggingface_hub's http_backoff function
    try:
        from huggingface_hub.utils._http import http_backoff

        # Store the original function
        original_http_backoff = http_backoff

        # Create a patched version
        def patched_http_backoff(*args, **kwargs):
            if "verify" in kwargs:
                kwargs["verify"] = False
            return original_http_backoff(*args, **kwargs)

        # Replace the function in the module
        import huggingface_hub.utils._http

        huggingface_hub.utils._http.http_backoff = patched_http_backoff

        print("✅ HuggingFace HTTP client patched successfully")
    except Exception as e:
        print(f"⚠️ Couldn't patch HuggingFace HTTP client: {e}")

except Exception as e:
    print(f"⚠️ Error patching requests: {e}")
