import re
from typing import Any, Dict, List, Tuple
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class IdProcessingService:
    def __init__(self):
        self.id_patterns: Dict[str, Dict] = {}
        self.id_counters: Dict[str, int] = {}
        self.global_used_ids: set = set()

    def _is_id_field(self, key: str) -> bool:
        """Check if a field is an ID field."""
        return key.lower() == "id" or key.lower().endswith("_id")

    def _discover_id_patterns(self, original_examples: List[Dict]) -> None:
        """Analyze original examples to learn ID patterns."""
        id_examples_by_path: Dict[str, List] = {}

        def discover_recursive(obj, path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if self._is_id_field(key):
                        current_path = f"{path}.{key}"
                        id_examples_by_path.setdefault(current_path, []).append(value)
                    discover_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    discover_recursive(item, path)

        for example in original_examples:
            discover_recursive(example, "")

        # Analyze patterns
        for path, examples in id_examples_by_path.items():
            pattern = self._analyze_id_pattern(examples)
            self.id_patterns[path] = pattern

            # Initialize counters
            if pattern["type"] in ["integer", "digit_string"]:
                used_nums = {int(e) for e in examples}
                self.id_counters[path] = max(used_nums) if used_nums else 0
                self.global_used_ids.update(str(e) for e in examples)
            elif pattern["type"] == "prefix_numeric":
                self.id_counters[path] = max(pattern["used_numbers"]) if pattern["used_numbers"] else 0
                self.global_used_ids.update(
                    f"{pattern['prefix']}{pattern['separator']}{num}"
                    for num in pattern["used_numbers"]
                )
            else:
                self.global_used_ids.update(str(e) for e in examples)

    def _analyze_id_pattern(self, id_samples: List[Any]) -> Dict[str, Any]:
        """Analyze ID samples to determine pattern."""
        if not id_samples:
            return {"type": "generic"}

        if all(isinstance(sample, int) for sample in id_samples):
            return {"type": "integer"}

        str_samples = [str(sample) for sample in id_samples]

        if all(s.isdigit() for s in str_samples):
            return {"type": "digit_string"}

        # Regex for patterns like "user-123", "item_45", "order:789"
        pattern_regexes: List[Tuple[str, str]] = [
            (r"^([a-zA-Z_]+)(\d+)$", ""),
            (r"^([a-zA-Z_]+)-(\d+)$", "-"),
            (r"^([a-zA-Z_]+):(\d+)$", ":"),
            (r"^([a-zA-Z_]+)_(\d+)$", "_"),
            (r"^([a-zA-Z_]+)\.(\d+)$", "."),
        ]

        for pattern_regex, separator in pattern_regexes:
            prefix_matches = {}
            for id_str in str_samples:
                match = re.match(pattern_regex, id_str)
                if match:
                    prefix = match.group(1)
                    num = int(match.group(2))
                    prefix_matches.setdefault(prefix, []).append(num)

            if prefix_matches:
                most_common_prefix = max(prefix_matches, key=lambda p: len(prefix_matches[p]))
                return {
                    "type": "prefix_numeric",
                    "prefix": most_common_prefix,
                    "separator": separator,
                    "used_numbers": set(prefix_matches[most_common_prefix]),
                }

        return {"type": "generic"}

    def _generate_id(self, path: str) -> Any:
        """Generate a new ID based on the learned pattern."""
        pattern = self.id_patterns.get(path, {"type": "generic"})

        if pattern["type"] in ["integer", "digit_string"]:
            counter = self.id_counters.get(path, 0)
            while True:
                counter += 1
                new_id_str = str(counter)
                if new_id_str not in self.global_used_ids:
                    self.id_counters[path] = counter
                    self.global_used_ids.add(new_id_str)
                    return int(new_id_str) if pattern["type"] == "integer" else new_id_str

        elif pattern["type"] == "prefix_numeric":
            counter = self.id_counters.get(path, 0)
            prefix = pattern["prefix"]
            separator = pattern["separator"]
            while True:
                counter += 1
                new_id = f"{prefix}{separator}{counter}"
                if new_id not in self.global_used_ids:
                    self.id_counters[path] = counter
                    self.global_used_ids.add(new_id)
                    return new_id

        # Fallback for generic IDs
        import uuid
        while True:
            new_id = f"id_{uuid.uuid4().hex[:8]}"
            if new_id not in self.global_used_ids:
                self.global_used_ids.add(new_id)
                return new_id

    def remove_ids_from_input(self, input_examples: List[Dict]) -> List[Dict]:
        """Remove ID fields from input examples before sending to LLM."""
        logger.info("Removing ID fields from input examples")
        
        def remove_ids_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: remove_ids_recursive(value)
                    for key, value in obj.items()
                    if not self._is_id_field(key)
                }
            elif isinstance(obj, list):
                return [remove_ids_recursive(item) for item in obj]
            else:
                return obj

        cleaned_examples = [remove_ids_recursive(example) for example in input_examples]
        logger.info(f"Removed ID fields from {len(input_examples)} input examples")
        return cleaned_examples

    def generate_ids_for_count(self, count: int, original_examples: List[Dict]) -> List[Dict]:
        """Generate enriched IDs for the specified count based on original examples."""
        logger.info(f"Generating {count} enriched IDs based on original examples")
        
        # Discover patterns from original examples
        self._discover_id_patterns(original_examples)
        
        # Generate ID objects for the count
        id_objects = []
        for i in range(count):
            id_obj = {}
            
            # Generate IDs for each pattern path
            for path in self.id_patterns.keys():
                # Convert path to nested structure
                keys = path.split('.')[1:]  # Remove empty first element
                current = id_obj
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = self._generate_id(path)
            
            id_objects.append(id_obj)
        
        logger.info(f"Generated {len(id_objects)} ID objects")
        return id_objects

    def combine_llm_response_with_ids(self, llm_response: List[Dict], id_objects: List[Dict]) -> List[Dict]:
        """Combine LLM response with generated IDs."""
        logger.info(f"Combining LLM response ({len(llm_response)} items) with IDs ({len(id_objects)} items)")
        
        if len(llm_response) != len(id_objects):
            logger.warning(f"Mismatch in counts: LLM response has {len(llm_response)} items, IDs has {len(id_objects)} items")
            # Use the smaller count
            min_count = min(len(llm_response), len(id_objects))
            llm_response = llm_response[:min_count]
            id_objects = id_objects[:min_count]
        
        def merge_recursive(llm_obj, id_obj):
            """Recursively merge LLM object with ID object."""
            if isinstance(llm_obj, dict) and isinstance(id_obj, dict):
                result = llm_obj.copy()
                for key, value in id_obj.items():
                    if key in result:
                        # If both have the key, merge recursively
                        result[key] = merge_recursive(result[key], value)
                    else:
                        # If only ID object has the key, add it
                        result[key] = value
                return result
            else:
                # If one is not a dict, return the LLM object (preserve LLM data)
                return llm_obj
        
        combined_data = []
        for llm_item, id_item in zip(llm_response, id_objects):
            combined_item = merge_recursive(llm_item, id_item)
            combined_data.append(combined_item)
        
        logger.info(f"Successfully combined {len(combined_data)} items")
        return combined_data


# Singleton instance
id_processing_service = IdProcessingService() 