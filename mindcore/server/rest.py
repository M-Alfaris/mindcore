"""REST API Server for Mindcore v2.

Provides HTTP endpoints for memory operations.
Can be used standalone or alongside MCP.
"""

import logging
from typing import TYPE_CHECKING


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mindcore.access import AccessController
    from mindcore.clst import CLST
    from mindcore.flr import FLR
    from mindcore.vocabulary import VocabularySchema


def create_app(
    flr: "FLR",
    clst: "CLST | None" = None,
    vocabulary: "VocabularySchema | None" = None,
    access_controller: "AccessController | None" = None,
    rate_limit: str | None = "100/minute",
):
    """Create FastAPI application for Mindcore REST API.

    Args:
        flr: FLR instance for fast recall
        clst: Optional CLST instance for long-term storage.
              If not provided, a default CLST will be created using FLR's storage.
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller
        rate_limit: Rate limit string (e.g., "100/minute"). Set to None to disable.

    Returns:
        FastAPI application
    """
    # Create CLST from FLR storage if not provided
    if clst is None:
        from mindcore import clst as clst_module

        clst = clst_module.CLST(storage=flr.storage, vocabulary=vocabulary)

    try:
        from fastapi import FastAPI, Header, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI required for REST server. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Mindcore API",
        description="Memory layer for AI agents - FLR & CLST protocols",
        version="2.0.0",
    )

    # CORS middleware
    # Note: When allow_credentials=True, allow_origins cannot be ["*"]
    # Using allow_credentials=False for security with wildcard origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    rate_limiter = None
    if rate_limit:
        try:
            from mindcore.enterprise.rate_limiting import RateLimiter, RateLimitExceededError

            rate_limiter = RateLimiter(limit=rate_limit)
            logger.info("Rate limiting enabled: %s", rate_limit)
        except ImportError:
            logger.warning(
                "Rate limiting requested but 'limits' library not installed. "
                "Install with: pip install limits"
            )

    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if rate_limiter is None:
                return await call_next(request)

            # Use client IP or X-Forwarded-For as identifier
            client_ip = request.client.host if request.client else "unknown"
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()

            # Check rate limit
            operation = request.url.path.split("/")[-1] or "default"
            if not rate_limiter.is_allowed(client_ip, operation):
                logger.warning("Rate limit exceeded for %s on %s", client_ip, operation)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                    headers={"Retry-After": "60"},
                )

            return await call_next(request)

    app.add_middleware(RateLimitMiddleware)

    # Pydantic models
    class StoreMemoryRequest(BaseModel):
        content: str
        memory_type: str = "episodic"
        user_id: str
        topics: list[str] = Field(default_factory=list)
        categories: list[str] = Field(default_factory=list)
        sentiment: str = "neutral"
        importance: float = 0.5
        entities: list[str] = Field(default_factory=list)
        access_level: str = "private"

    class SearchRequest(BaseModel):
        query: str | None = None
        user_id: str
        topics: list[str] | None = None
        categories: list[str] | None = None
        memory_types: list[str] | None = None
        limit: int = 10

    class RecallRequest(BaseModel):
        query: str
        user_id: str
        attention_hints: list[str] | None = None
        memory_types: list[str] | None = None
        limit: int = 5

    class ReinforceRequest(BaseModel):
        memory_id: str
        signal: float = Field(ge=-1, le=1)

    class RegisterAgentRequest(BaseModel):
        agent_id: str
        name: str
        description: str = ""
        teams: list[str] = Field(default_factory=list)

    # Helper to get agent ID from header
    def get_agent_id(x_agent_id: str | None = Header(None)) -> str | None:
        return x_agent_id

    # Routes
    @app.get("/")
    async def root():
        return {
            "name": "Mindcore API",
            "version": "2.0.0",
            "protocols": ["FLR", "CLST"],
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "flr": flr.get_stats(),
            "clst": clst.get_stats(),
        }

    @app.get("/vocabulary")
    async def get_vocabulary():
        if not vocabulary:
            return {"error": "No vocabulary configured"}
        return vocabulary.to_dict()

    @app.get("/vocabulary/schema")
    async def get_vocabulary_schema():
        if not vocabulary:
            return {"error": "No vocabulary configured"}
        return vocabulary.to_json_schema()

    # Memory operations
    @app.post("/memories")
    async def store_memory(
        data: StoreMemoryRequest,
        x_agent_id: str | None = Header(None),
    ):
        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content=data.content,
            memory_type=data.memory_type,
            user_id=data.user_id,
            agent_id=x_agent_id,
            topics=data.topics,
            categories=data.categories,
            sentiment=data.sentiment,
            importance=data.importance,
            entities=data.entities,
            access_level=data.access_level,
            vocabulary_version=vocabulary.version if vocabulary else "1.0.0",
        )

        try:
            memory_id = clst.store(memory)
            return {"memory_id": memory_id, "success": True}
        except ValueError as e:
            logger.warning("Memory validation failed: %s", e)
            raise HTTPException(
                status_code=422, detail="Invalid memory data. Check content and metadata."
            )

    @app.get("/memories/{memory_id}")
    async def get_memory(
        memory_id: str,
        x_agent_id: str | None = Header(None),
    ):
        memory = clst.retrieve(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Access control check
        if access_controller and x_agent_id:
            from mindcore.access import Permission

            decision = access_controller.can_access(
                agent_id=x_agent_id,
                memory_access_level=memory.access_level,
                memory_agent_id=memory.agent_id,
                permission=Permission.READ,
                memory_id=memory_id,
            )
            if not decision.allowed:
                raise HTTPException(status_code=403, detail=decision.reason)

        return memory.to_dict()

    @app.delete("/memories/{memory_id}")
    async def delete_memory(
        memory_id: str,
        x_agent_id: str | None = Header(None),
    ):
        memory = clst.retrieve(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Access control check
        if access_controller and x_agent_id:
            from mindcore.access import Permission

            decision = access_controller.can_access(
                agent_id=x_agent_id,
                memory_access_level=memory.access_level,
                memory_agent_id=memory.agent_id,
                permission=Permission.DELETE,
                memory_id=memory_id,
            )
            if not decision.allowed:
                raise HTTPException(status_code=403, detail=decision.reason)

        success = clst.delete(memory_id)
        return {"success": success}

    @app.post("/memories/search")
    async def search_memories(
        data: SearchRequest,
        x_agent_id: str | None = Header(None),
    ):
        memories = clst.search(
            query=data.query,
            user_id=data.user_id,
            agent_id=x_agent_id,
            topics=data.topics,
            categories=data.categories,
            memory_types=data.memory_types,
            limit=data.limit,
        )

        # Filter by access control
        if access_controller and x_agent_id:
            from mindcore.access import Permission

            memories = access_controller.filter_accessible_memories(
                x_agent_id, memories, Permission.READ
            )

        return {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    # FLR operations
    @app.post("/recall")
    async def recall(
        data: RecallRequest,
        x_agent_id: str | None = Header(None),
    ):
        result = flr.query(
            query=data.query,
            user_id=data.user_id,
            agent_id=x_agent_id,
            attention_hints=data.attention_hints,
            memory_types=data.memory_types,
            limit=data.limit,
        )

        return {
            "memories": [m.to_dict() for m in result.memories],
            "scores": result.scores,
            "attention_focus": result.attention_focus,
            "suggested_memory_types": result.suggested_memory_types,
            "latency_ms": result.query_latency_ms,
        }

    @app.post("/reinforce")
    async def reinforce(data: ReinforceRequest):
        flr.reinforce(data.memory_id, data.signal)
        return {"success": True}

    # Agent management (if access controller is configured)
    @app.post("/agents")
    async def register_agent(data: RegisterAgentRequest):
        if not access_controller:
            raise HTTPException(status_code=400, detail="Access control not configured")

        try:
            profile = access_controller.register_agent(
                agent_id=data.agent_id,
                name=data.name,
                description=data.description,
                teams=data.teams,
            )
            return profile.to_dict()
        except ValueError as e:
            logger.warning("Agent registration failed for %s: %s", data.agent_id, e)
            raise HTTPException(
                status_code=400, detail="Agent registration failed. Check agent configuration."
            )

    @app.get("/agents")
    async def list_agents():
        if not access_controller:
            return {"agents": []}

        agents = access_controller.list_agents()
        return {"agents": [a.to_dict() for a in agents]}

    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str):
        if not access_controller:
            raise HTTPException(status_code=400, detail="Access control not configured")

        profile = access_controller.get_agent(agent_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Agent not found")

        return profile.to_dict()

    @app.delete("/agents/{agent_id}")
    async def unregister_agent(agent_id: str):
        if not access_controller:
            raise HTTPException(status_code=400, detail="Access control not configured")

        success = access_controller.unregister_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"success": True}

    # CLST operations
    @app.post("/compress")
    async def compress_memories(
        user_id: str = Query(...),
        older_than_days: int = Query(30),
        strategy: str = Query("summarize"),
    ):
        from datetime import timedelta

        from mindcore.clst import CompressionStrategy

        try:
            strategy_enum = CompressionStrategy(strategy)
        except ValueError:
            strategy_enum = CompressionStrategy.SUMMARIZE

        result = clst.compress(
            user_id=user_id,
            older_than=timedelta(days=older_than_days),
            strategy=strategy_enum,
        )

        return {
            "original_count": result.original_count,
            "compressed_count": result.compressed_count,
            "compression_ratio": result.compression_ratio,
            "latency_ms": result.latency_ms,
        }

    @app.post("/sync")
    async def sync_memories(
        source_agent: str = Query(...),
        target_agent: str = Query(...),
        user_id: str = Query(...),
    ):
        result = clst.sync(
            source_agent=source_agent,
            target_agent=target_agent,
            user_id=user_id,
        )

        return {
            "memories_transferred": result.memories_transferred,
            "conflicts_resolved": result.conflicts_resolved,
            "errors": result.errors,
            "latency_ms": result.latency_ms,
        }

    return app


def run_server(
    flr: "FLR",
    clst: "CLST",
    vocabulary: "VocabularySchema | None" = None,
    access_controller: "AccessController | None" = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    rate_limit: str | None = "100/minute",
):
    """Run the REST API server.

    Args:
        flr: FLR instance
        clst: CLST instance
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller
        host: Host to bind to
        port: Port to bind to
        rate_limit: Rate limit string (e.g., "100/minute"). Set to None to disable.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required to run server. Install with: pip install uvicorn")

    app = create_app(flr, clst, vocabulary, access_controller, rate_limit)
    uvicorn.run(app, host=host, port=port)
