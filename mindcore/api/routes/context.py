"""
Context route for context assembly.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Header, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from ...utils.logger import get_logger
from ...utils.security import get_rate_limiter

logger = get_logger(__name__)

router = APIRouter(prefix="/context", tags=["context"])


async def check_rate_limit(
    request: Request, x_user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> str:
    """
    Dependency to check rate limit.

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
            detail=f"Rate limit exceeded. Please wait before making more requests.",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )

    return identifier


class ContextRequest(BaseModel):
    """Request model for context assembly."""

    user_id: str = Field(..., description="User identifier")
    thread_id: str = Field(..., description="Thread identifier")
    query: str = Field(..., description="Query or topic for context assembly")
    max_messages: Optional[int] = Field(50, description="Maximum messages to consider")


class ContextResponse(BaseModel):
    """Response model for context assembly."""

    success: bool
    assembled_context: str
    key_points: List[str]
    relevant_message_ids: List[str]
    metadata: Dict[str, Any]


@router.post("", response_model=ContextResponse)
async def get_context(request: ContextRequest, rate_limit_id: str = Depends(check_rate_limit)):
    """
    Assemble relevant context for a query.

    This endpoint:
    1. Retrieves recent messages from cache and database
    2. Uses the Context Assembler Agent to analyze and summarize
    3. Returns structured context ready for LLM prompt injection

    Args:
        request: ContextRequest with user/thread and query.
        rate_limit_id: Rate limit identifier (from dependency).

    Returns:
        ContextResponse with assembled context and metadata.
    """
    from ... import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        # Get context
        context = mindcore.get_context(
            user_id=request.user_id,
            thread_id=request.thread_id,
            query=request.query,
            max_messages=request.max_messages,
        )

        return ContextResponse(
            success=True,
            assembled_context=context.assembled_context,
            key_points=context.key_points,
            relevant_message_ids=context.relevant_message_ids,
            metadata=context.metadata,
        )

    except ValueError as e:
        logger.warning(f"Validation error during context retrieval: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context assembly failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{user_id}/{thread_id}", response_model=ContextResponse)
async def get_context_query_param(
    user_id: str,
    thread_id: str,
    query: str = Query(..., description="Query or topic for context assembly"),
    max_messages: int = Query(50, description="Maximum messages to consider"),
    rate_limit_id: str = Depends(check_rate_limit),
):
    """
    Get context using query parameters (alternative GET endpoint).

    Args:
        user_id: User identifier.
        thread_id: Thread identifier.
        query: Query string.
        max_messages: Maximum messages to consider.
        rate_limit_id: Rate limit identifier (from dependency).

    Returns:
        ContextResponse with assembled context.
    """
    from ... import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        # Get context
        context = mindcore.get_context(
            user_id=user_id, thread_id=thread_id, query=query, max_messages=max_messages
        )

        return ContextResponse(
            success=True,
            assembled_context=context.assembled_context,
            key_points=context.key_points,
            relevant_message_ids=context.relevant_message_ids,
            metadata=context.metadata,
        )

    except ValueError as e:
        logger.warning(f"Validation error during context retrieval: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context assembly failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{user_id}/{thread_id}/formatted")
async def get_formatted_context(
    user_id: str,
    thread_id: str,
    query: str = Query(..., description="Query or topic for context assembly"),
    max_messages: int = Query(50, description="Maximum messages to consider"),
    rate_limit_id: str = Depends(check_rate_limit),
):
    """
    Get context pre-formatted for prompt injection.

    Returns context as a plain text string ready to be inserted into an LLM prompt.

    Args:
        user_id: User identifier.
        thread_id: Thread identifier.
        query: Query string.
        max_messages: Maximum messages to consider.
        rate_limit_id: Rate limit identifier (from dependency).

    Returns:
        Plain text formatted context.
    """
    from ... import get_mindcore_instance

    try:
        mindcore = get_mindcore_instance()

        # Get context
        context = mindcore.get_context(
            user_id=user_id, thread_id=thread_id, query=query, max_messages=max_messages
        )

        # Format for prompt
        from ...utils.helper import format_context_for_prompt

        formatted = format_context_for_prompt(context.to_dict())

        return {"formatted_context": formatted}

    except ValueError as e:
        logger.warning(f"Validation error during context formatting: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context formatting failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
