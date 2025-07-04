import hashlib
import json
from typing import Any, Dict, List


def compute_deterministic_hash(json_obj: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash for a JSON object.
    Keys are sorted to ensure consistency.
    """
    # Serialize the dict with sorted keys to ensure the hash is deterministic
    serialized_obj = json.dumps(
        json_obj, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized_obj).hexdigest()


def compute_hashes_for_payload(payload: List[Dict[str, Any]]) -> List[str]:
    """
    Computes deterministic hashes for a list of JSON objects and sorts them
    to ensure the overall payload signature is order-independent.
    """
    hashes = [compute_deterministic_hash(obj) for obj in payload]
    return sorted(hashes)
