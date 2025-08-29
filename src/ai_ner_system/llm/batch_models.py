"""Batch processing data models for LLM clients."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class BatchStatus(Enum):
    """Enumeration of batch processing statuses.

    Based on Anthropic's Message Batches API documentation:
    - in_progress: The batch is currently being processed
    - ended: The batch has completed processing (success or failure)
    - canceling: The batch is being canceled
    """
    IN_PROGRESS = 'in_progress'
    ENDED = 'ended'
    CANCELING = 'canceling'

@dataclass
class BatchRequest:
    """Represents a single request in a batch using Claude Batches API.

    Attributes:
        custom_id: Unique identifier for this request within the batch.
        prompt: The input prompt text to process.
        max_tokens: Maximum number of tokens in the response.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
    """
    custom_id: str
    prompt: str
    max_tokens: int = 20000
    temperature: float = 0.0

@dataclass
class BatchResponse:
    """Represents a response from batch processing using Claude Batches API.

    Attributes:
        custom_id: The unique identifier from the original request.
        response_text: The generated response text.
        success: Whether the request was processed successfully.
        error_message: Error message if success is False.
    """
    custom_id: str
    response_text: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class BatchProgress:
    """Represents progress information for a batch job.

    Attributes:
        batch_id: The unique identifier for the batch job.
        status: Current status of the batch.
        elapsed_time: Time elapsed since batch creation (seconds).
        # estimated_remaining: Estimated time remaining (seconds, if available).
        request_counts: Dictionary with request counts by status.
        created_at: When the batch was created.
        expires_at: When the batch will expire (24 hours from creation).
    """
    batch_id: str
    status: BatchStatus
    elapsed_time: float
    request_counts: Dict[str, int]
    created_at: str
    expires_at: str