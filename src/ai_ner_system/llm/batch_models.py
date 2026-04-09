"""Batch processing data models for LLM clients.

This module defines the shared batch-processing data model that the rest of the LLM
and async processing stack uses. It is the shared batch data contract for the LLM layer.
A rough dependency:

batch_models.py
    -> base_client.py
    -> claude_client.py
    -> processing/processor.py
    -> pipeline/async_processor.py
"""

from dataclasses import dataclass
from enum import Enum


class BatchStatus(Enum):
    """Enumeration of batch job state.

    Based on Anthropic's Message Batches API documentation:
    - in_progress: The batch is currently being processed
    - ended: The batch has completed processing (success or failure)
    - canceling: The batch is being canceled
    """

    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    CANCELING = "canceling"


@dataclass
class BatchRequest:
    """Represents one single request in a batch using Claude/Anthropic Batches API.

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

    def __post_init__(self) -> None:
        """Validate request parameters after initialization."""
        if not self.custom_id.strip():
            raise ValueError("custom_id cannot be empty")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")


@dataclass
class BatchResponse:
    """Represents one response from batch processing using Claude/Anthropic Batches API.

    Attributes:
        custom_id: The unique identifier from the original request.
        response_text: The generated response text.
        success: Whether the request was processed successfully.
        error_message: Error message if success is False.
    """

    custom_id: str
    response_text: str
    success: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate response after initialization."""
        if not self.custom_id:
            raise ValueError("custom_id cannot be empty")
        if self.success and not self.response_text.strip():
            raise ValueError(
                "Successful response cannot have empty response_text",
            )
        if not self.success and not self.error_message:
            raise ValueError("Failed response must have error_message")


@dataclass
class BatchProgress:
    """Represents progress information snapshot.

    It is used for a batch job progress to be monitored during async batch processing.

    Attributes:
        batch_num: The batch number for tracking multiple batches.
        batch_id: The unique identifier for the batch job.
        status: Current status of the batch.
        elapsed_time: Time elapsed since batch creation (seconds).
        request_counts: Dictionary with request counts by status.
        created_at: When the batch was created.
        expires_at: When the batch will expire (24 hours from creation).
    """

    batch_num: int
    batch_id: str
    status: BatchStatus
    elapsed_time: float
    request_counts: dict[str, int]
    created_at: str
    expires_at: str

    def __post_init__(self) -> None:
        """Validate progress information after initialization."""
        if not self.batch_id.strip():
            raise ValueError("batch_id cannot be empty")
        if self.elapsed_time < 0:
            raise ValueError("elapsed_time cannot be negative")
