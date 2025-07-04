import json
import uuid
from typing import Any, Dict, List, Tuple

from app.config.settings import settings
from app.repositories.log_repository import log_repository
from app.services.cache_service import cache_service
from app.services.id_generation_service import IdGenerationService
from app.services.image_enrichment_service import image_enrichment_service
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

    async def _generate_and_validate_data(
        self, input_examples: List[Dict], count: int, request_id: str
    ) -> List[Dict]:
        """
        Generates data and ensures the count is correct, using an intelligent retry loop
        as a fallback if the LLM doesn't return the exact count requested.
        """
        total_generated_data = []
        attempts = 0
        max_attempts = 3

        while len(total_generated_data) < count and attempts < max_attempts:
            remaining_count = count - len(total_generated_data)

            logger.info(
                "Attempting to generate mock data",
                attempt=attempts + 1,
                items_to_generate=remaining_count,
            )

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

            # **New Validation Step**: Check if the LLM is just copying the input
            is_copy = True
            for gen_item in generated_data:
                if gen_item not in input_examples:
                    is_copy = False
                    break

            if is_copy:
                logger.warning(
                    "LLM returned a copy of the input examples. Discarding and retrying.",
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

    async def execute(
        self, input_examples: List[Dict], count: int
    ) -> Tuple[Dict, bool]:
        request_id = str(uuid.uuid4())
        logger.info(
            "Starting mock data generation use case",
            request_id=request_id,
            requested_count=count,
        )

        # 1. Check cache using the new efficient method
        object_hashes = compute_hashes_for_payload(input_examples)
        cached_data = await cache_service.get_data_if_cache_hit(
            object_hashes, count
        )

        if cached_data:
            logger.info(
                "Full response served from cache.", request_id=request_id
            )
            return (cached_data[:count], True)

        # 2. Generate Data if cache missed
        logger.info(
            "Cache miss or insufficient data. Generating new mock data.",
            request_id=request_id,
        )

        raw_mock_data = await self._generate_and_validate_data(
            input_examples, count, request_id
        )

        # 3. Post-process to ensure unique and pattern-preserved IDs
        processed_data = self.id_generation_service.process_and_replace_ids(
            raw_mock_data, input_examples
        )

        # 4. Enrich Data (Image URLs)
        if not processed_data:
            raise LLMGenerationError(
                "Failed to generate any data from the LLM."
            )

        logger.info(
            "Starting image URL enrichment post-processing.",
            request_id=request_id,
        )
        final_mock_data = await image_enrichment_service.enrich_mock_data(
            processed_data
        )

        self._save_intermediate_step(
            final_mock_data, "final_response", request_id
        )

        # 5. Cache the new data
        await cache_service.create_new_group_cache(
            object_hashes, final_mock_data
        )

        # 6. Log token usage (simplified)
        # A real implementation would get exact token counts from the tokenizer
        input_tokens = len(json.dumps(input_examples)) // 2
        output_tokens = len(json.dumps(final_mock_data)) // 2
        await log_repository.log_token_usage(
            request_id, input_tokens, output_tokens, "LocalSmolLM"
        )

        return (final_mock_data, False)


generate_mock_data_usecase = GenerateMockDataUsecase()
