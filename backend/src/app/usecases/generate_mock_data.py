import asyncio
import json
import uuid
from typing import Any, Dict, List, Tuple

from app.config.settings import settings
from app.repositories.log_repository import log_repository
from app.services.cache_service import cache_service
from app.services.id_generation_service import IdGenerationService
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
        self.id_generation_service = IdGenerationService()

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

    def _detect_copying(self, generated_data: List[Dict], input_examples: List[Dict]) -> bool:
        """
        Enhanced detection of copying behavior from LLM.
        Checks for exact matches, partial copying, and very similar content.
        """
        if not generated_data or not input_examples:
            return False
            
        # Check for exact matches first
        for gen_item in generated_data:
            if gen_item in input_examples:
                logger.debug("Found exact copy of input example")
                return True
        
        # Check for partial copying (same values in key fields)
        for gen_item in generated_data:
            for input_item in input_examples:
                # Check if key string fields are identical
                for key, value in gen_item.items():
                    if isinstance(value, str) and key in input_item:
                        if value == input_item[key]:
                            logger.debug(f"Found copied value '{value}' in field '{key}'")
                            return True
                
                # Check if lists/arrays are identical
                for key, value in gen_item.items():
                    if isinstance(value, list) and key in input_item:
                        if value == input_item[key]:
                            logger.debug(f"Found copied list '{value}' in field '{key}'")
                            return True
        
        # Check if all generated items are very similar to input examples
        # (e.g., same structure but with minor variations)
        if len(generated_data) == len(input_examples):
            similar_count = 0
            for gen_item in generated_data:
                for input_item in input_examples:
                    # Check if most fields match
                    matching_fields = 0
                    total_fields = 0
                    for key in gen_item.keys():
                        if key in input_item:
                            total_fields += 1
                            if gen_item[key] == input_item[key]:
                                matching_fields += 1
                    
                    if total_fields > 0 and matching_fields / total_fields > 0.8:
                        similar_count += 1
                        break
            
            if similar_count >= len(generated_data) * 0.8:
                logger.debug("Found mostly similar items to input examples")
                return True
        
        return False

    async def _generate_and_validate_data(
        self, input_examples: List[Dict], count: int, request_id: str
    ) -> List[Dict]:
        """
        Generates data and ensures the count is correct, using an intelligent retry loop
        as a fallback if the LLM doesn't return the exact count requested.
        """
        total_generated_data = []
        attempts = 0
        max_attempts = settings.MAX_GENERATION_ATTEMPTS

        while len(total_generated_data) < count and attempts < max_attempts:
            remaining_count = count - len(total_generated_data)

            logger.info(
                "Attempting to generate mock data",
                attempt=attempts + 1,
                items_to_generate=remaining_count,
            )

            # Use batch generation for better performance with large counts
            generated_data = await llm_service.generate_mock_data(
                input_examples, remaining_count
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

            # **Enhanced Validation Step**: Check if the LLM is copying the input
            is_copy = self._detect_copying(generated_data, input_examples)
            
            if is_copy:
                logger.warning(
                    "LLM returned copied or very similar data to input examples. Discarding and retrying.",
                    attempt=attempts + 1,
                )
                attempts += 1
                continue

            total_generated_data.extend(generated_data)
            logger.info(
                "Data generation attempt complete.",
                items_generated_this_attempt=len(generated_data),
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

    async def _generate_ids_async(self, count: int, original_examples: List[Dict]) -> List[Dict]:
        """Generate IDs asynchronously for the specified count."""
        # Run ID generation in a thread pool since it's CPU-bound
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            id_processing_service.generate_ids_for_count, 
            count, 
            original_examples
        )

    async def execute(
        self, input_examples: List[Dict], count: int
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
            cache_info = {
                "cachedCount": len(cached_data),
                "generatedCount": 0,
                "totalCount": len(cached_data),
                "cacheHitType": "full"
            }
            return (cached_data, True, cache_info)
        
        elif cached_data and remaining_count > 0:
            # Partial cache hit - we have some data, need to generate the rest
            logger.info(
                f"Partial cache hit. Returning {len(cached_data)} cached items and generating {remaining_count} more.",
                request_id=request_id,
            )
            
            # Generate only the remaining count needed
            additional_data = await self._generate_remaining_data(
                input_examples, remaining_count, request_id, cached_data
            )
            
            # Combine cached and new data
            final_mock_data = cached_data + additional_data
            
            # Update the existing cache group with the new combined data
            if group_id:
                await cache_service.update_existing_group_cache(
                    group_id, object_hashes, final_mock_data
                )
            else:
                # Fallback: create new group if no group_id was returned
                await cache_service.create_new_group_cache(
                    object_hashes, final_mock_data
                )
            
            # Log token usage for the additional generation only
            input_tokens = len(json.dumps(input_examples)) // 2
            output_tokens = len(json.dumps(additional_data)) // 2
            await log_repository.log_token_usage(
                request_id, input_tokens, output_tokens, "LocalSmolLM"
            )
            
            cache_info = {
                "cachedCount": len(cached_data),
                "generatedCount": len(additional_data),
                "totalCount": len(final_mock_data),
                "cacheHitType": "partial"
            }
            return (final_mock_data, False, cache_info)  # False because we generated some new data

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
        llm_task = self._generate_and_validate_data(cleaned_input_examples, count, request_id)
        id_task = asyncio.create_task(self._generate_ids_async(count, input_examples))
        
        # Wait for both tasks to complete
        raw_mock_data, id_objects = await asyncio.gather(llm_task, id_task)
        
        # 2c. Combine LLM response with generated IDs
        processed_data = id_processing_service.combine_llm_response_with_ids(
            raw_mock_data, id_objects
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

        self._save_intermediate_step(
            final_mock_data, "final_response", request_id
        )

        # 5. Cache the new data
        await cache_service.create_new_group_cache(
            object_hashes, final_mock_data
        )

        # 6. Log token usage (simplified)
        input_tokens = len(json.dumps(input_examples)) // 2
        output_tokens = len(json.dumps(final_mock_data)) // 2
        await log_repository.log_token_usage(
            request_id, input_tokens, output_tokens, "LocalSmolLM"
        )

        cache_info = {
            "cachedCount": 0,
            "generatedCount": len(final_mock_data),
            "totalCount": len(final_mock_data),
            "cacheHitType": "none"
        }
        return (final_mock_data, False, cache_info)

    async def _generate_remaining_data(
        self, input_examples: List[Dict], remaining_count: int, request_id: str, existing_data: List[Dict]
    ) -> List[Dict]:
        """
        Generate only the remaining count of data items needed.
        This method is optimized for partial cache hits.
        """
        logger.info(
            f"Generating remaining {remaining_count} items for partial cache hit",
            request_id=request_id,
        )
        
        # Remove IDs from input examples before sending to LLM
        cleaned_input_examples = id_processing_service.remove_ids_from_input(input_examples)
        
        # Generate IDs in parallel with LLM generation
        llm_task = self._generate_and_validate_data(cleaned_input_examples, remaining_count, request_id)
        id_task = asyncio.create_task(self._generate_ids_async(remaining_count, input_examples))
        
        # Wait for both tasks to complete
        raw_additional_data, id_objects = await asyncio.gather(llm_task, id_task)
        
        logger.info(
            "Combining additional LLM response with generated IDs",
            request_id=request_id,
        )
        
        processed_additional_data = id_processing_service.combine_llm_response_with_ids(
            raw_additional_data, id_objects
        )

        # Analyze and identify image keys (using existing data as reference)
        if not processed_additional_data:
            raise LLMGenerationError(
                "Failed to generate any additional data from the LLM."
            )

        logger.info(
            "Starting image key analysis for additional data.",
            request_id=request_id,
        )
        
        # Use existing data to help identify image keys
        all_data_for_analysis = existing_data + processed_additional_data
        confirmed_image_keys = image_key_analyzer_service.identify_image_keys(
            input_examples, all_data_for_analysis
        )
        
        logger.info(
            f"Identified {len(confirmed_image_keys)} confirmed image keys for additional data enrichment.",
            request_id=request_id,
        )

        # Enrich the additional data with image URLs
        logger.info(
            "Starting image URL enrichment for additional data.",
            request_id=request_id,
        )
        enriched_additional_data = await image_enrichment_service.enrich_mock_data(
            processed_additional_data, confirmed_keys=confirmed_image_keys
        )

        self._save_intermediate_step(
            enriched_additional_data, f"additional_data_{request_id}", request_id
        )

        return enriched_additional_data


generate_mock_data_usecase = GenerateMockDataUsecase()