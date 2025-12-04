"""Ingest route for message ingestion."""

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from mindcore.utils.logger import get_logger
from mindcore.utils.security import get_rate_limiter


logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestMessageRequest(BaseModel):
    """Request model for message ingestion."""

    user_id: str = Field(..., description="User identifier")
    thread_id: str = Field(..., description="Thread identifier")
    session_id: str = Field(..., description="Session identifier")
    role: str = Field(..., description="Message role (user, assistant, system, tool)")
    text: str = Field(..., description="Message text content")
    message_id: str | None = Field(
        None, description="Optional message ID (auto-generated if not provided)"
    )


class IngestMessageResponse(BaseModel):
    """Response model for message ingestion."""

    success: bool
    message_id: str
    message: str


async def check_rate_limit(
    request: Request, x_user_id: str | None = Header(None, alias="X-User-ID")
) -> str:
    """Dependency to check rate limit.

    Uses X-User-ID header if provided, otherwise uses client IP.

    Args:
        request: FastAPI request object.
        x_user_id: Optional user ID from header.

    Returns:
        The identifier used for rate limiting.

    Raises:
        HTTPException: If rate limit exceeded.
    """
    rate_limiter = get_rate_limiter()

    # Use user ID from header, or fall back to client IP
    identifier = x_user_id or request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(identifier):
        remaining = rate_limiter.get_remaining(identifier)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests.",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )

    return identifier


@router.post("", response_model=IngestMessageResponse)
async def ingest_message(
    request: IngestMessageRequest, rate_limit_id: str = Depends(check_rate_limit)
):
    """Ingest a new message for enrichment and storage.

    This endpoint:
    1. Receives a message
    2. Enriches it with metadata using the Enrichment Agent
    3. Stores it in PostgreSQL
    4. Caches it in memory

    Args:
        request: IngestMessageRequest with message details.
        rate_limit_id: Rate limit identifier (from dependency).

    Returns:
        IngestMessageResponse with success status and message_id.
    """
    from mindcore import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        # Convert request to dict
        message_dict = request.model_dump()

        # Ingest message
        message = mindcore.ingest_message(message_dict)

        return IngestMessageResponse(
            success=True,
            message_id=message.message_id,
            message="Message ingested and enriched successfully",
        )

    except ValueError as e:
        # Validation errors return 400
        logger.warning(f"Validation error during ingestion: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to ingest message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=dict[str, Any])
async def ingest_batch(
    messages: list[IngestMessageRequest], rate_limit_id: str = Depends(check_rate_limit)
):
    """Ingest multiple messages in batch.

    Args:
        messages: List of IngestMessageRequest objects.
        rate_limit_id: Rate limit identifier (from dependency).

    Returns:
        Batch ingestion result.
    """
    from mindcore import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        message_dicts = [msg.model_dump() for msg in messages]
        ingested = []
        failed = []

        for i, msg_dict in enumerate(message_dicts):
            try:
                message = mindcore.ingest_message(msg_dict)
                ingested.append(message.message_id)
            except Exception as e:
                logger.exception(f"Failed to ingest message {i} in batch: {e}")
                failed.append({"index": i, "error": str(e)})
                continue

        return {
            "success": len(failed) == 0,
            "total": len(messages),
            "ingested": len(ingested),
            "failed": len(failed),
            "message_ids": ingested,
            "errors": failed if failed else None,
        }

    except Exception as e:
        logger.exception(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
