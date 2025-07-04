from typing import Any, Dict, List
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageKeyAnalyzerService:
    def __init__(self):
        self.image_extensions = [
            ".jpg",
            ".jpeg", 
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".svg",
        ]

    def _analyze_input_schema(self, input_data: Any, path: str = "") -> Dict[str, Any]:
        """
        Recursively analyzes input schema to find keys with HTTP/HTTPS values.
        Returns dict of {key_path: value} for keys with HTTP/HTTPS values.
        """
        http_keys = {}
        
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if value contains HTTP/HTTPS
                if isinstance(value, str) and (value.lower().startswith(("http://", "https://"))):
                    http_keys[current_path] = value
                
                # Recursively analyze nested structures
                if isinstance(value, (dict, list)):
                    nested_keys = self._analyze_input_schema(value, current_path)
                    http_keys.update(nested_keys)
                    
        elif isinstance(input_data, list):
            for i, item in enumerate(input_data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                nested_keys = self._analyze_input_schema(item, current_path)
                http_keys.update(nested_keys)
                
        return http_keys

    def _analyze_output_data(self, output_data: Any, path: str = "") -> Dict[str, Any]:
        """
        Recursively analyzes output data to find keys with img_ or image extensions.
        Returns dict of {key_path: value} for keys with img_ or image extensions.
        """
        img_keys = {}
        
        if isinstance(output_data, dict):
            for key, value in output_data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if value contains img_ or has image extensions
                if isinstance(value, str):
                    value_lower = value.lower()
                    if (value_lower.startswith("img_") or 
                        any(value_lower.endswith(ext) for ext in self.image_extensions)):
                        img_keys[current_path] = value
                
                # Recursively analyze nested structures
                if isinstance(value, (dict, list)):
                    nested_values = self._analyze_output_data(value, current_path)
                    img_keys.update(nested_values)
                    
        elif isinstance(output_data, list):
            for i, item in enumerate(output_data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                nested_values = self._analyze_output_data(item, current_path)
                img_keys.update(nested_values)
                
        return img_keys

    def identify_image_keys(self, input_data: Any, output_data: Any) -> List[str]:
        """
        Identifies which keys should be enriched with images by analyzing both input and output values.
        
        Logic:
        1. Input: Find keys with HTTP/HTTPS values
        2. Output: Find keys with img_ or image extensions
        3. Intersection: Keys that appear in both are confirmed image keys
        
        Returns a list of key paths that should be enriched.
        """
        logger.info("Starting image key analysis based on values...")
        
        # Step 1: Analyze input to find keys with HTTP/HTTPS values
        http_keys = self._analyze_input_schema(input_data)
        logger.info(f"Found {len(http_keys)} keys with HTTP/HTTPS values in input")
        for key, value in http_keys.items():
            logger.info(f"Input HTTP key: {key} = {value}")
        
        # Step 2: Analyze output to find keys with img_ or image extensions
        img_keys = self._analyze_output_data(output_data)
        logger.info(f"Found {len(img_keys)} keys with img_ or image extensions in output")
        for key, value in img_keys.items():
            logger.info(f"Output img key: {key} = {value}")
        
        # Step 3: Find intersection - keys that appear in both
        confirmed_image_keys = []
        for key_path in http_keys.keys():
            if key_path in img_keys:
                input_value = http_keys[key_path]
                output_value = img_keys[key_path]
                logger.info(f"Confirmed image key: {key_path}")
                logger.info(f"  Input value: {input_value}")
                logger.info(f"  Output value: {output_value}")
                confirmed_image_keys.append(key_path)
            else:
                logger.info(f"Excluded key (no matching output): {key_path}")
        
        logger.info(f"Final confirmed image keys: {len(confirmed_image_keys)}")
        return confirmed_image_keys

# Singleton instance
image_key_analyzer_service = ImageKeyAnalyzerService() 