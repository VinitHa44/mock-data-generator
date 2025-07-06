import asyncio
import json
import uuid
from typing import Any, Dict, List, Tuple

from app.config.settings import settings
from app.repositories.log_repository import log_repository
from app.services.cache_service import cache_service
from app.services.id_processing_service import id_processing_service
from app.services.image_enrichment_service import image_enrichment_service
from app.services.image_key_analyzer_service import image_key_analyzer_service
from app.services.llm_service import llm_service
from app.usecases.generation_helpers import compute_hashes_for_payload
from app.utils.app_exceptions import LLMGenerationError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class GenerateMockDataUsecase:

    def __init__(self):
        pass

    def _save_intermediate_step(
        self, data: Any, file_prefix: str, request_id: str
    ):
        """Saves intermediate data to a file for debugging."""
        path = (
            f"{settings.INTERMEDIATE_SAVE_PATH}/{file_prefix}_{request_id}.json"
        )
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\033[96m🔍 INTERMEDIATE STEP SAVED: {path}\033[0m")
            logger.info("Saved intermediate step", path=path)
        except Exception as e:
            logger.error(
                "Failed to save intermediate step", path=path, error=str(e)
            )

    def _is_structurally_valid(self, reference_obj: Any, target_obj: Any) -> bool:
        """Recursively checks if target_obj has the same structure (keys and value types) as reference_obj."""
        if type(reference_obj) != type(target_obj):
            return False

        if isinstance(reference_obj, dict):
            if set(reference_obj.keys()) != set(target_obj.keys()):
                return False
            return all(self._is_structurally_valid(reference_obj[k], target_obj[k]) for k in reference_obj)

        if isinstance(reference_obj, list):
            if not reference_obj:
                return not target_obj
            if not target_obj:
                return False
            return self._is_structurally_valid(reference_obj[0], target_obj[0])
        
        return True

    def _validate_and_filter_by_schema(
        self,
        generated_data: List[Dict],
        reference_schema: Dict
    ) -> List[Dict]:
        """
        Validates that each object in generated_data conforms to the schema of reference_schema.
        """
        if not generated_data or not reference_schema:
            return generated_data

        valid_items = []
        for item in generated_data:
            if self._is_structurally_valid(reference_schema, item):
                valid_items.append(item)
            else:
                logger.warning(
                    "Discarding object due to schema mismatch with input example.",
                    item=item
                )

        if len(valid_items) < len(generated_data):
            logger.info(
                "Filtered out objects that did not conform to the input schema.",
                original_count=len(generated_data),
                filtered_count=len(valid_items)
            )

        return valid_items

    def _filter_copied_data(self, generated_data: List[Dict], input_examples: List[Dict]) -> List[Dict]:
        """
        Filters out copied or very similar data from the generated list.
        Returns a list of items that are considered original.
        """
        if not generated_data or not input_examples:
            return generated_data
            
        original_items = []
        for gen_item in generated_data:
            is_copy = False
            # Check for exact matches
            if gen_item in input_examples:
                is_copy = True
            else:
                # If no exact match, check for partial copying
                for input_item in input_examples:
                    # Check if key string or list fields are identical
                    for key, value in gen_item.items():
                        if isinstance(value, (str, list)) and key in input_item and value == input_item[key]:
                            is_copy = True
                            break
                    if is_copy:
                        break
            
            if not is_copy:
                original_items.append(gen_item)

        if len(original_items) < len(generated_data):
            logger.warning(
                "Filtered out copied or very similar data to input examples.",
                original_count=len(generated_data),
                filtered_count=len(original_items)
            )
            
        return original_items

    async def _generate_and_validate_data(
        self, 
        input_examples: List[Dict], 
        count: int, 
        request_id: str,
        enable_moderation: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        context: str = "default"
    ) -> List[Dict]:
        """
        Generate and validate mock data with retry logic and instruction variation.
        """
        total_generated_data = []
        attempts = 0
        max_attempts = settings.MAX_GENERATION_ATTEMPTS
        previous_instruction_index = None

        while len(total_generated_data) < count and attempts < max_attempts:
            remaining_count = count - len(total_generated_data)

            logger.info(
                "Attempting to generate mock data",
                attempt=attempts + 1,
                items_to_generate=remaining_count,
            )

            # Determine context for this attempt
            attempt_context = "retry" if attempts > 0 else context

            # Use batch generation for better performance with large counts
            generated_data = await llm_service.generate_mock_data(
                input_examples, 
                remaining_count,
                enable_moderation,
                temperature,
                max_tokens,
                top_p,
                context=attempt_context,
                previous_instruction_index=previous_instruction_index
            )
            self._save_intermediate_step(
                generated_data, f"llm_response_attempt_{attempts+1}", request_id
            )

            if not generated_data:
                logger.warning(
                    "LLM returned no data on this attempt.",
                    attempt=attempts + 1,
                )
                attempts += 1
                continue

            # **Enhanced Validation Step**: Filter out items that are copies of the input
            original_data = self._filter_copied_data(generated_data, input_examples)
            
            if not original_data and generated_data:
                # If all items were filtered, it's a sign of a major copying issue, treat as a failed attempt.
                logger.warning(
                    "LLM returned only copied data. Discarding and retrying.",
                    attempt=attempts + 1,
                )
                attempts += 1
                continue

            total_generated_data.extend(original_data)
            logger.info(
                "Data generation attempt complete.",
                items_generated_this_attempt=len(original_data),
                total_items_so_far=len(total_generated_data),
            )
            attempts += 1

        if len(total_generated_data) < count:
            logger.error(
                "LLM failed to generate the required number of items.",
                required=count,
                generated=len(total_generated_data),
                attempts=max_attempts,
            )
            raise LLMGenerationError(
                f"LLM failed to generate the required number of items after {max_attempts} attempts."
            )

        # Ensure we don't return more than requested
        return total_generated_data[:count]

    async def _generate_ids_async(self, count: int, original_examples: List[Dict], existing_data: List[Dict] = None) -> List[Dict]:
        """Generate IDs asynchronously for the specified count."""
        # Run ID generation in a thread pool since it's CPU-bound
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            id_processing_service.generate_ids_for_count, 
            count, 
            original_examples,
            existing_data
        )

    async def execute(
        self, 
        input_examples: List[Dict], 
        count: int,
        enable_moderation: bool = True,
        temperature: float = 0.85,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        cache_expiration: bool = False
    ) -> Tuple[Dict, bool, dict]:
        request_id = str(uuid.uuid4())
        logger.info(
            "Starting mock data generation use case with parallel ID enrichment",
            request_id=request_id,
            requested_count=count,
        )

        # 1. Check cache using the new partial cache hit method
        object_hashes = compute_hashes_for_payload(input_examples)
        cached_data, remaining_count, group_id = await cache_service.get_partial_cache_hit(
            object_hashes, count
        )

        if cached_data and remaining_count == 0:
            # Full cache hit - return all data from cache
            logger.info(
                "Full response served from cache.", request_id=request_id
            )
            validated_data = self._validate_and_filter_by_schema(cached_data, input_examples[0])
            cache_info = {
                "cachedCount": len(validated_data),
                "generatedCount": 0,
                "totalCount": len(validated_data),
                "cacheHitType": "full"
            }
            return (validated_data, True, cache_info)
        
        elif cached_data and remaining_count > 0:
            # Partial cache hit - we have some data, need to generate the rest
            logger.info(
                f"Partial cache hit. Returning {len(cached_data)} cached items and generating {remaining_count} more.",
                request_id=request_id,
            )
            
            # Generate only the remaining count needed, aware of existing data
            # 2a. Remove IDs from input examples before sending to LLM
            cleaned_input_examples = id_processing_service.remove_ids_from_input(input_examples)
            
            # 2b. Generate IDs in parallel with LLM generation, avoiding collisions with cached data
            llm_task = self._generate_and_validate_data(cleaned_input_examples, remaining_count, request_id, enable_moderation, temperature, max_tokens, top_p, context="cache")
            id_task = asyncio.create_task(self._generate_ids_async(remaining_count, input_examples, existing_data=cached_data))
            
            # Wait for both tasks to complete
            raw_additional_data, id_objects = await asyncio.gather(llm_task, id_task)

            # 2c. Combine LLM response with generated IDs, preserving field order
            additional_data = id_processing_service.combine_llm_response_with_ids(
                raw_additional_data, id_objects, input_examples[0]
            )

            # Enrich the additional data
            all_data_for_analysis = cached_data + additional_data
            confirmed_image_keys = image_key_analyzer_service.identify_image_keys(
                input_examples, all_data_for_analysis
            )
            enriched_additional_data = await image_enrichment_service.enrich_mock_data(
                additional_data, confirmed_keys=confirmed_image_keys
            )
            
            # Combine cached and new data
            final_mock_data = cached_data + enriched_additional_data
            
            # Validate the final combined list against the input schema
            validated_data = self._validate_and_filter_by_schema(final_mock_data, input_examples[0])

            # Update the existing cache group with the new combined data
            if group_id:
                await cache_service.update_existing_group_cache(
                    group_id, object_hashes, validated_data, cache_expiration
                )
            else:
                # Fallback: create new group if no group_id was returned
                await cache_service.create_new_group_cache(
                    object_hashes, validated_data, cache_expiration
                )
            
            # Log token usage for the additional generation only
            input_tokens = len(json.dumps(input_examples)) // 2
            output_tokens = len(json.dumps(additional_data)) // 2
            await log_repository.log_token_usage(
                request_id, input_tokens, output_tokens, "LocalSmolLM"
            )
            
            # Recalculate counts based on validated data
            # Note: This is an approximation as we can't perfectly distinguish validated cached vs generated items.
            final_cached_count = min(len(cached_data), len(validated_data))
            final_generated_count = len(validated_data) - final_cached_count

            cache_info = {
                "cachedCount": final_cached_count,
                "generatedCount": final_generated_count,
                "totalCount": len(validated_data),
                "cacheHitType": "partial"
            }
            return (validated_data, False, cache_info)

        # 2. No cache hit - generate everything from scratch
        logger.info(
            "No cache hit. Starting parallel ID enrichment for new generation.",
            request_id=request_id,
        )
        
        # 2a. Remove IDs from input examples before sending to LLM
        cleaned_input_examples = id_processing_service.remove_ids_from_input(input_examples)
        
        # 2b. Generate IDs in parallel with LLM generation
        logger.info(
            "Starting parallel ID generation and LLM data generation.",
            request_id=request_id,
        )
        
        # Create tasks for parallel execution
        llm_task = self._generate_and_validate_data(cleaned_input_examples, count, request_id, enable_moderation, temperature, max_tokens, top_p, context="default")
        id_task = asyncio.create_task(self._generate_ids_async(count, input_examples))
        
        # Wait for both tasks to complete
        raw_mock_data, id_objects = await asyncio.gather(llm_task, id_task)
        
        # 2c. Combine LLM response with generated IDs, preserving field order
        processed_data = id_processing_service.combine_llm_response_with_ids(
            raw_mock_data, id_objects, input_examples[0]
        )

        # 3. Analyze and identify image keys
        if not processed_data:
            raise LLMGenerationError(
                "Failed to generate any data from the LLM."
            )

        logger.info(
            "Starting image key analysis.",
            request_id=request_id,
        )
        
        # Analyze input examples and processed data to identify which keys should be enriched
        confirmed_image_keys = image_key_analyzer_service.identify_image_keys(
            input_examples, processed_data
        )
        
        logger.info(
            f"Identified {len(confirmed_image_keys)} confirmed image keys for enrichment \n\n {confirmed_image_keys}.",
            request_id=request_id,
        )

        # 4. Enrich Data (Image URLs) - only for confirmed keys
        logger.info(
            "Starting image URL enrichment post-processing.",
            request_id=request_id,
        )
        final_mock_data = await image_enrichment_service.enrich_mock_data(
            processed_data, confirmed_keys=confirmed_image_keys
        )

        # ** NEW: Validate final data against input schema **
        validated_data = self._validate_and_filter_by_schema(final_mock_data, input_examples[0])

        self._save_intermediate_step(
            validated_data, "final_response", request_id
        )

        # 5. Cache the new data
        await cache_service.create_new_group_cache(
            object_hashes, validated_data, cache_expiration
        )

        # 6. Log token usage (simplified)
        input_tokens = len(json.dumps(input_examples)) // 2
        output_tokens = len(json.dumps(validated_data)) // 2
        await log_repository.log_token_usage(
            request_id, input_tokens, output_tokens, "LocalSmolLM"
        )

        cache_info = {
            "cachedCount": 0,
            "generatedCount": len(validated_data),
            "totalCount": len(validated_data),
            "cacheHitType": "none"
        }
        return (validated_data, False, cache_info)


generate_mock_data_usecase = GenerateMockDataUsecase()