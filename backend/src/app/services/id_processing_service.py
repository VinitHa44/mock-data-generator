import re
import uuid
import json
from typing import Any, Dict, List, Set, Tuple
from app.config.settings import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class IdPatternRecognizer:
    @staticmethod
    def analyze_ids(id_samples: List[Any]) -> Dict[str, Any]:
        """
        Analyzes a list of ID samples to determine their pattern.
        e.g., integer, string-prefixed, etc.
        """
        if not id_samples:
            return {"type": "generic"}

        if all(isinstance(sample, int) for sample in id_samples):
            return {"type": "integer"}

        str_samples = [str(sample) for sample in id_samples]

        if all(s.isdigit() for s in str_samples):
            return {"type": "digit_string"}

        # Regex for patterns like "user-123", "item_45", "order:789"
        pattern_regexes: List[Tuple[str, str]] = settings.ID_PATTERN_REGEXES

        # Check for prefix-based numeric patterns
        for pattern_regex, separator in pattern_regexes:
            prefix_matches = {}
            for id_str in str_samples:
                match = re.match(pattern_regex, id_str)
                if match:
                    prefix = match.group(1)
                    # Handle both numeric and alphanumeric patterns
                    if match.group(2).isdigit():
                        # Simple numeric pattern
                        num = int(match.group(2))
                        prefix_matches.setdefault(prefix, []).append(num)
                    else:
                        # Complex alphanumeric pattern
                        alphanumeric_part = match.group(2)
                        prefix_matches.setdefault(prefix, []).append(alphanumeric_part)

            if prefix_matches:
                # Find the most common prefix
                most_common_prefix = max(
                    prefix_matches, key=lambda p: len(prefix_matches[p])
                )
                
                # Check if all values are numeric or alphanumeric
                values = prefix_matches[most_common_prefix]
                if all(isinstance(v, int) for v in values):
                    # All numeric values
                    return {
                        "type": "prefix_numeric",
                        "prefix": most_common_prefix,
                        "separator": separator,
                        "used_numbers": set(values),
                    }
                else:
                    # Mixed or alphanumeric values
                    return {
                        "type": "prefix_alphanumeric",
                        "prefix": most_common_prefix,
                        "separator": separator,
                        "used_values": set(str(v) for v in values),
                        "pattern": pattern_regex,
                    }

        # If no other pattern matches, treat as generic strings
        return {"type": "generic"}


class IdProcessingService:
    def __init__(self):
        self.id_patterns: Dict[str, Dict] = {}
        self.id_counters: Dict[str, int] = {}
        self.global_used_ids: Set[str] = set()

    def _is_id_field(self, key: str) -> bool:
        """Check if a field is an ID field based on settings."""
        return any(key.lower().endswith(pattern) for pattern in settings.ID_KEY_PATTERNS)

    def _discover_id_patterns(self, original_examples: List[Dict], existing_data: List[Dict] = None) -> None:
        """Analyze original examples to learn ID patterns and account for existing IDs."""
        id_examples_by_path: Dict[str, List] = {}

        def discover_recursive(obj, path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if self._is_id_field(key):
                        id_examples_by_path.setdefault(current_path, []).append(value)
                    # Recurse with the correct, new path
                    discover_recursive(value, current_path)
            elif isinstance(obj, list):
                for item in obj:
                    # When recursing into a list, the path doesn't change
                    discover_recursive(item, path)

        for example in original_examples:
            discover_recursive(example, "")
            
        if existing_data:
            for item in existing_data:
                # We don't need to learn patterns from existing data, just collect the IDs
                # to prevent collisions.
                def extract_ids_recursive(obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if self._is_id_field(key):
                                self.global_used_ids.add(str(value))
                            extract_ids_recursive(value)
                    elif isinstance(obj, list):
                        for sub_item in obj:
                            extract_ids_recursive(sub_item)
                extract_ids_recursive(item)

        # Analyze patterns
        for path, examples in id_examples_by_path.items():
            pattern = IdPatternRecognizer.analyze_ids(examples)
            self.id_patterns[path] = pattern

            # Initialize counters and track used IDs from examples
            if pattern["type"] in ["integer", "digit_string"]:
                used_nums = {int(e) for e in examples}
                self.id_counters[path] = max(used_nums) if used_nums else 0
                self.global_used_ids.update(str(e) for e in examples)
            elif pattern["type"] == "prefix_numeric":
                self.id_counters[path] = (
                    max(pattern["used_numbers"])
                    if pattern["used_numbers"]
                    else 0
                )
                self.global_used_ids.update(
                    f"{pattern['prefix']}{pattern['separator']}{num}"
                    for num in pattern["used_numbers"]
                )
            elif pattern["type"] == "prefix_alphanumeric":
                self.id_counters[path] = 0  # Assuming no specific counter needed for alphanumeric
                self.global_used_ids.update(
                    f"{pattern['prefix']}{pattern['separator']}{value}"
                    for value in pattern["used_values"]
                )
            else:  # generic
                self.global_used_ids.update(str(e) for e in examples)

    def _generate_id(self, path: str) -> Any:
        """Generate a new, unique ID based on the learned pattern for a given path."""
        pattern = self.id_patterns.get(path, {"type": "generic"})

        if pattern["type"] in ["integer", "digit_string"]:
            counter = self.id_counters.get(path, 0)
            while True:
                counter += 1
                new_id_str = str(counter)
                if new_id_str not in self.global_used_ids:
                    self.id_counters[path] = counter
                    self.global_used_ids.add(new_id_str)
                    return (
                        int(new_id_str)
                        if pattern["type"] == "integer"
                        else new_id_str
                    )

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

        elif pattern["type"] == "prefix_alphanumeric":
            prefix = pattern["prefix"]
            separator = pattern["separator"]
            pattern_regex = pattern.get("pattern", "")
            
            # Generate based on pattern type
            if "\\d{4}-\\d{2}-\\d{2}-\\d+" in pattern_regex:
                # Date-based pattern: YYYY-MM-DD-NNN
                from datetime import datetime, timedelta
                base_date = datetime.now()
                counter = self.id_counters.get(path, 0)
                while True:
                    counter += 1
                    date_str = (base_date + timedelta(days=counter-1)).strftime("%Y-%m-%d")
                    new_id = f"{prefix}{separator}{date_str}-{counter:03d}"
                    if new_id not in self.global_used_ids:
                        self.id_counters[path] = counter
                        self.global_used_ids.add(new_id)
                        return new_id
                        
            elif "\\d{4}-Q\\d-\\d+" in pattern_regex:
                # Quarter-based pattern: YYYY-QN-NNN
                from datetime import datetime
                current_year = datetime.now().year
                counter = self.id_counters.get(path, 0)
                while True:
                    counter += 1
                    quarter = ((counter - 1) // 100) + 1
                    quarter_num = counter % 100 if counter % 100 != 0 else 100
                    new_id = f"{prefix}{separator}{current_year}-Q{quarter}-{quarter_num:03d}"
                    if new_id not in self.global_used_ids:
                        self.id_counters[path] = counter
                        self.global_used_ids.add(new_id)
                        return new_id
                        
            elif "\\d+-[a-zA-Z0-9]+" in pattern_regex:
                # Revision pattern: NNN-REV
                counter = self.id_counters.get(path, 0)
                while True:
                    counter += 1
                    import random
                    import string
                    rev_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
                    new_id = f"{prefix}{separator}{counter:03d}-{rev_suffix}"
                    if new_id not in self.global_used_ids:
                        self.id_counters[path] = counter
                        self.global_used_ids.add(new_id)
                        return new_id
                        
            else:
                # Generic alphanumeric pattern
                counter = self.id_counters.get(path, 0)
                while True:
                    counter += 1
                    # Try to maintain some alphanumeric complexity
                    import random
                    import string
                    if random.random() < 0.3:  # 30% chance of adding letters
                        alphanumeric_part = f"{counter:03d}-{random.choice(string.ascii_uppercase)}"
                    else:
                        alphanumeric_part = f"{counter:03d}"
                    
                    new_id = f"{prefix}{separator}{alphanumeric_part}"
                if new_id not in self.global_used_ids:
                    self.id_counters[path] = counter
                    self.global_used_ids.add(new_id)
                    return new_id

        # Fallback for generic IDs
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

    def generate_ids_for_count(self, count: int, original_examples: List[Dict], existing_data: List[Dict] = None) -> List[Dict]:
        """Generate enriched IDs for the specified count based on original examples."""
        # Reset state for each call to ensure thread-safety in concurrent environments
        self.__init__()
        
        logger.info(f"Generating {count} enriched IDs based on original examples")
        
        # Discover patterns from original examples and account for existing IDs
        self._discover_id_patterns(original_examples, existing_data=existing_data)
        
        # Generate ID objects for the count
        id_objects = []
        for _ in range(count):
            id_obj = {}
            
            # Generate IDs for each pattern path
            for path in self.id_patterns.keys():
                # Convert path to nested structure
                keys = path.split('.')
                current = id_obj
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = self._generate_id(path)
            
            id_objects.append(id_obj)
        
        logger.info(f"Generated {len(id_objects)} ID objects")
        return id_objects

    def combine_llm_response_with_ids(self, llm_response: List[Dict], id_objects: List[Dict], schema_template: Dict) -> List[Dict]:
        """Combine LLM response with generated IDs, preserving the field order from the schema_template."""
        logger.info(f"Combining LLM response ({len(llm_response)} items) with IDs ({len(id_objects)} items) while preserving field order")
        
        if len(llm_response) != len(id_objects):
            logger.warning(f"Mismatch in counts: LLM response has {len(llm_response)} items, IDs has {len(id_objects)} items")
            # Use the smaller count
            min_count = min(len(llm_response), len(id_objects))
            llm_response = llm_response[:min_count]
            id_objects = id_objects[:min_count]
        
        def merge_with_schema(llm_obj: Dict, id_obj: Dict, template: Dict) -> Dict:
            """
            Builds a new dictionary by following the key order of the template.
            """
            new_obj = {}
            if not isinstance(template, dict):
                return llm_obj
        
            for key, template_value in template.items():
                if self._is_id_field(key):
                    # If it's an ID field, take the value from the id_obj if it exists
                    if id_obj and key in id_obj:
                        new_obj[key] = id_obj[key]
                elif key in llm_obj:
                    # If it's a non-ID field, take it from the llm_obj
                    llm_value = llm_obj[key]
                    # If the field is a nested object, recurse to preserve sub-ordering
                    if isinstance(template_value, dict) and isinstance(llm_value, dict):
                        # Pass the corresponding sub-dictionary from id_obj if it exists
                        id_sub_obj = id_obj.get(key) if id_obj else None
                        new_obj[key] = merge_with_schema(llm_value, id_sub_obj, template_value)
                    else:
                        # Otherwise, just take the value from the LLM
                        new_obj[key] = llm_value
            return new_obj

        combined_data = []
        for llm_item, id_item in zip(llm_response, id_objects):
            combined_item = merge_with_schema(llm_item, id_item, schema_template)
            combined_data.append(combined_item)
        
        logger.info(f"Successfully combined {len(combined_data)} items, preserving field order")
        return combined_data


# Singleton instance
id_processing_service = IdProcessingService() 