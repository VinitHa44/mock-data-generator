import re
import uuid
import random
from typing import List, Dict, Any, Set, Tuple

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
        pattern_regexes: List[Tuple[str, str]] = [
            (r'^([a-zA-Z_]+)(\d+)$', ''),
            (r'^([a-zA-Z_]+)-(\d+)$', '-'),
            (r'^([a-zA-Z_]+):(\d+)$', ':'),
            (r'^([a-zA-Z_]+)_(\d+)$', '_'),
            (r'^([a-zA-Z_]+)\.(\d+)$', '.'),
        ]

        # Check for prefix-based numeric patterns
        for pattern_regex, separator in pattern_regexes:
            prefix_matches = {}
            for id_str in str_samples:
                match = re.match(pattern_regex, id_str)
                if match:
                    prefix = match.group(1)
                    num = int(match.group(2))
                    prefix_matches.setdefault(prefix, []).append(num)
            
            if prefix_matches:
                # Find the most common prefix
                most_common_prefix = max(prefix_matches, key=lambda p: len(prefix_matches[p]))
                return {
                    "type": "prefix_numeric",
                    "prefix": most_common_prefix,
                    "separator": separator,
                    "used_numbers": set(prefix_matches[most_common_prefix])
                }

        # If no other pattern matches, treat as generic strings
        return {"type": "generic"}


class IdGenerationService:
    def __init__(self):
        self.id_patterns: Dict[str, Dict] = {}
        self.id_counters: Dict[str, int] = {}
        self.global_used_ids: Set[str] = set()

    def _discover_and_learn_patterns(self, original_examples: List[Dict]):
        """Traverse original examples to find ID fields and learn their patterns."""
        id_examples_by_path: Dict[str, List] = {}

        def discover_recursive(obj, path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check if the key is 'id' or ends with '_id'
                    if key.lower() == 'id' or key.lower().endswith('_id'):
                        current_path = f"{path}.{key}"
                        id_examples_by_path.setdefault(current_path, []).append(value)
                    discover_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    discover_recursive(item, path)

        for example in original_examples:
            discover_recursive(example, "")

        # Analyze collected examples to determine patterns
        for path, examples in id_examples_by_path.items():
            pattern = IdPatternRecognizer.analyze_ids(examples)
            self.id_patterns[path] = pattern
            
            # Initialize counters and track used IDs from examples
            if pattern['type'] in ['integer', 'digit_string']:
                used_nums = {int(e) for e in examples}
                self.id_counters[path] = max(used_nums) if used_nums else 0
                self.global_used_ids.update(str(e) for e in examples)
            elif pattern['type'] == 'prefix_numeric':
                self.id_counters[path] = max(pattern['used_numbers']) if pattern['used_numbers'] else 0
                self.global_used_ids.update(f"{pattern['prefix']}{pattern['separator']}{num}" for num in pattern['used_numbers'])
            else: # generic
                self.global_used_ids.update(str(e) for e in examples)

    def _generate_new_id(self, path: str) -> Any:
        """Generate a new, unique ID based on the learned pattern for a given path."""
        pattern = self.id_patterns.get(path, {"type": "generic"})
        
        if pattern['type'] in ['integer', 'digit_string']:
            counter = self.id_counters.get(path, 0)
            while True:
                counter += 1
                new_id_str = str(counter)
                if new_id_str not in self.global_used_ids:
                    self.id_counters[path] = counter
                    self.global_used_ids.add(new_id_str)
                    return int(new_id_str) if pattern['type'] == 'integer' else new_id_str
        
        elif pattern['type'] == 'prefix_numeric':
            counter = self.id_counters.get(path, 0)
            prefix = pattern['prefix']
            separator = pattern['separator']
            while True:
                counter += 1
                new_id = f"{prefix}{separator}{counter}"
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

    def _replace_ids_recursive(self, obj: Any, path: str = "") -> Any:
        """Traverse the generated data and replace IDs with unique, pattern-preserved ones."""
        if isinstance(obj, dict):
            return {
                key: (self._generate_new_id(f"{path}.{key}") 
                      if (key.lower() == 'id' or key.lower().endswith('_id')) and f"{path}.{key}" in self.id_patterns
                      else self._replace_ids_recursive(value, f"{path}.{key}"))
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._replace_ids_recursive(item, path) for item in obj]
        else:
            return obj

    def process_and_replace_ids(self, generated_data: List[Dict], original_examples: List[Dict]) -> List[Dict]:
        """
        Main method to orchestrate the ID replacement process.
        """
        self.__init__() # Reset state for each call
        self._discover_and_learn_patterns(original_examples)
        
        processed_data = []
        for item in generated_data:
            processed_data.append(self._replace_ids_recursive(item))
            
        return processed_data 