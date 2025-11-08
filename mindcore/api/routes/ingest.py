"""
Ingest route for message ingestion.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestMessageRequest(BaseModel):
    """Request model for message ingestion."""
    user_id: str = Field(..., description="User identifier")
    thread_id: str = Field(..., description="Thread identifier")
    session_id: str = Field(..., description="Session identifier")
    role: str = Field(..., description="Message role (user, assistant, system, tool)")
    text: str = Field(..., description="Message text content")
    message_id: Optional[str] = Field(None, description="Optional message ID (auto-generated if not provided)")


class IngestMessageResponse(BaseModel):
    """Response model for message ingestion."""
    success: bool
    message_id: str
    message: str


@router.post("", response_model=IngestMessageResponse)
async def ingest_message(request: IngestMessageRequest):
    """
    Ingest a new message for enrichment and storage.

    This endpoint:
    1. Receives a message
    2. Enriches it with metadata using the Enrichment Agent
    3. Stores it in PostgreSQL
    4. Caches it in memory

    Args:
        request: IngestMessageRequest with message details.

    Returns:
        IngestMessageResponse with success status and message_id.
    """
    from ... import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        # Convert request to dict
        message_dict = request.dict()

        # Ingest message
        message = mindcore.ingest_message(message_dict)

        return IngestMessageResponse(
            success=True,
            message_id=message.message_id,
            message=f"Message ingested and enriched successfully"
        )

    except Exception as e:
        logger.error(f"Failed to ingest message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, Any])
async def ingest_batch(messages: list[IngestMessageRequest]):
    """
    Ingest multiple messages in batch.

    Args:
        messages: List of IngestMessageRequest objects.

    Returns:
        Batch ingestion result.
    """
    from ... import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        message_dicts = [msg.dict() for msg in messages]
        ingested = []

        for msg_dict in message_dicts:
            try:
                message = mindcore.ingest_message(msg_dict)
                ingested.append(message.message_id)
            except Exception as e:
                logger.error(f"Failed to ingest message in batch: {e}")
                continue

        return {
            "success": True,
            "total": len(messages),
            "ingested": len(ingested),
            "message_ids": ingested
        }

    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
