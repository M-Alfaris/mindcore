"""REST API Server for Mindcore v2.

Provides HTTP endpoints for memory operations.
Can be used standalone or alongside MCP.

Security Features:
    - Auth hook: Pass custom auth_dependency for authentication
    - Rate limiting: Uses slowapi with configurable limits
    - CORS: Configurable origins

Example with custom auth:
    from fastapi import Depends, HTTPException
    from fastapi.security import APIKeyHeader

    api_key_header = APIKeyHeader(name="X-API-Key")

    async def verify_api_key(api_key: str = Depends(api_key_header)):
        if api_key != "secret":
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key

    app = create_app(flr, clst, auth_dependency=verify_api_key)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..flr import FLR
    from ..clst import CLST
    from ..vocabulary import VocabularySchema
    from ..access import AccessController


def create_app(
    flr: FLR,
    clst: CLST,
    vocabulary: VocabularySchema | None = None,
    access_controller: AccessController | None = None,
    # Security options
    auth_dependency: Callable | None = None,
    rate_limit: str = "100/minute",
    cors_origins: list[str] | None = None,
):
    """Create FastAPI application for Mindcore REST API.

    Args:
        flr: FLR instance for fast recall
        clst: CLST instance for long-term storage
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller
        auth_dependency: Optional FastAPI dependency for authentication.
            Will be applied to all routes. Example: OAuth2, API key, JWT validator.
        rate_limit: Rate limit string (e.g., "100/minute", "1000/hour").
            Uses slowapi format. Set to None to disable.
        cors_origins: Allowed CORS origins. Defaults to ["*"] (allow all).

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException, Header, Query, Depends, Request
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI required for REST server. Install with: pip install fastapi uvicorn"
        )

    # Optional rate limiting with slowapi
    limiter = None
    if rate_limit:
        try:
            from slowapi import Limiter
            from slowapi.util import get_remote_address
            from slowapi.errors import RateLimitExceeded
            from slowapi.middleware import SlowAPIMiddleware

            limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit])
        except ImportError:
            pass  # slowapi not installed, skip rate limiting

    app = FastAPI(
        title="Mindcore API",
        description="Memory layer for AI agents - FLR & CLST protocols",
        version="2.0.0",
    )

    # Add rate limiter to app state and middleware
    if limiter:
        app.state.limiter = limiter
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Build dependencies list - auth is applied globally if provided
    dependencies = []
    if auth_dependency:
        dependencies.append(Depends(auth_dependency))

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

    # Rate limit decorator helper
    def rate_limited(func):
        """Apply rate limit if limiter is configured."""
        if limiter:
            return limiter.limit(rate_limit)(func)
        return func

    # Routes - public endpoints (no auth required)
    @app.get("/")
    async def root():
        return {
            "name": "Mindcore API",
            "version": "2.0.0",
            "protocols": ["FLR", "CLST"],
            "rate_limit": rate_limit if limiter else None,
            "auth_enabled": auth_dependency is not None,
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

    # Memory operations - protected routes with rate limiting
    @app.post("/memories", dependencies=dependencies)
    @rate_limited
    async def store_memory(
        request: Request,
        body: StoreMemoryRequest,
        x_agent_id: str | None = Header(None),
    ):
        from ..flr import Memory

        memory = Memory(
            memory_id="",
            content=body.content,
            memory_type=body.memory_type,
            user_id=body.user_id,
            agent_id=x_agent_id,
            topics=body.topics,
            categories=body.categories,
            sentiment=body.sentiment,
            importance=body.importance,
            entities=body.entities,
            access_level=body.access_level,
            vocabulary_version=vocabulary.version if vocabulary else "1.0.0",
        )

        memory_id = clst.store(memory)
        return {"memory_id": memory_id, "success": True}

    @app.get("/memories/{memory_id}", dependencies=dependencies)
    @rate_limited
    async def get_memory(
        request: Request,
        memory_id: str,
        x_agent_id: str | None = Header(None),
    ):
        memory = clst.retrieve(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Access control check
        if access_controller and x_agent_id:
            from ..access import Permission
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

    @app.delete("/memories/{memory_id}", dependencies=dependencies)
    @rate_limited
    async def delete_memory(
        request: Request,
        memory_id: str,
        x_agent_id: str | None = Header(None),
    ):
        memory = clst.retrieve(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Access control check
        if access_controller and x_agent_id:
            from ..access import Permission
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

    @app.post("/memories/search", dependencies=dependencies)
    @rate_limited
    async def search_memories(
        request: Request,
        body: SearchRequest,
        x_agent_id: str | None = Header(None),
    ):
        memories = clst.search(
            query=body.query,
            user_id=body.user_id,
            agent_id=x_agent_id,
            topics=body.topics,
            categories=body.categories,
            memory_types=body.memory_types,
            limit=body.limit,
        )

        # Filter by access control
        if access_controller and x_agent_id:
            from ..access import Permission
            memories = access_controller.filter_accessible_memories(
                x_agent_id, memories, Permission.READ
            )

        return {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    # FLR operations
    @app.post("/recall", dependencies=dependencies)
    @rate_limited
    async def recall(
        request: Request,
        body: RecallRequest,
        x_agent_id: str | None = Header(None),
    ):
        result = flr.query(
            query=body.query,
            user_id=body.user_id,
            agent_id=x_agent_id,
            attention_hints=body.attention_hints,
            memory_types=body.memory_types,
            limit=body.limit,
        )

        return {
            "memories": [m.to_dict() for m in result.memories],
            "scores": result.scores,
            "attention_focus": result.attention_focus,
            "suggested_memory_types": result.suggested_memory_types,
            "latency_ms": result.query_latency_ms,
        }

    @app.post("/reinforce", dependencies=dependencies)
    @rate_limited
    async def reinforce(request: Request, body: ReinforceRequest):
        flr.reinforce(body.memory_id, body.signal)
        return {"success": True}

    # Agent management (if access controller is configured)
    @app.post("/agents", dependencies=dependencies)
    @rate_limited
    async def register_agent(request: Request, body: RegisterAgentRequest):
        if not access_controller:
            raise HTTPException(
                status_code=400,
                detail="Access control not configured"
            )

        try:
            profile = access_controller.register_agent(
                agent_id=body.agent_id,
                name=body.name,
                description=body.description,
                teams=body.teams,
            )
            return profile.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/agents", dependencies=dependencies)
    @rate_limited
    async def list_agents(request: Request):
        if not access_controller:
            return {"agents": []}

        agents = access_controller.list_agents()
        return {"agents": [a.to_dict() for a in agents]}

    @app.get("/agents/{agent_id}", dependencies=dependencies)
    @rate_limited
    async def get_agent(request: Request, agent_id: str):
        if not access_controller:
            raise HTTPException(
                status_code=400,
                detail="Access control not configured"
            )

        profile = access_controller.get_agent(agent_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Agent not found")

        return profile.to_dict()

    @app.delete("/agents/{agent_id}", dependencies=dependencies)
    @rate_limited
    async def unregister_agent(request: Request, agent_id: str):
        if not access_controller:
            raise HTTPException(
                status_code=400,
                detail="Access control not configured"
            )

        success = access_controller.unregister_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"success": True}

    # CLST operations
    @app.post("/compress", dependencies=dependencies)
    @rate_limited
    async def compress_memories(
        request: Request,
        user_id: str = Query(...),
        older_than_days: int = Query(30),
        strategy: str = Query("summarize"),
    ):
        from datetime import timedelta
        from ..clst import CompressionStrategy

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

    @app.post("/sync", dependencies=dependencies)
    @rate_limited
    async def sync_memories(
        request: Request,
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
    flr: FLR,
    clst: CLST,
    vocabulary: VocabularySchema | None = None,
    access_controller: AccessController | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    # Security options
    auth_dependency: Callable | None = None,
    rate_limit: str = "100/minute",
    cors_origins: list[str] | None = None,
):
    """Run the REST API server.

    Args:
        flr: FLR instance
        clst: CLST instance
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller
        host: Host to bind to
        port: Port to bind to
        auth_dependency: Optional auth dependency (see create_app)
        rate_limit: Rate limit string (default: "100/minute")
        cors_origins: Allowed CORS origins
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn required to run server. Install with: pip install uvicorn"
        )

    app = create_app(
        flr,
        clst,
        vocabulary,
        access_controller,
        auth_dependency=auth_dependency,
        rate_limit=rate_limit,
        cors_origins=cors_origins,
    )
    uvicorn.run(app, host=host, port=port)


# Standalone app factory for uvicorn
def create_standalone_app():
    """Create a standalone FastAPI app using environment configuration.

    This is used when running the server via uvicorn directly:
        uvicorn mindcore.v2.server.rest:app --host 0.0.0.0 --port 8000

    Environment variables:
        MINDCORE_DB_URL: Database connection string (default: sqlite:///mindcore.db)
        MINDCORE_MULTI_AGENT: Enable multi-agent mode (default: false)
        MINDCORE_RATE_LIMIT: Rate limit string (default: 100/minute)
        MINDCORE_CORS_ORIGINS: Comma-separated CORS origins (default: *)
    """
    import os

    from ..mindcore import Mindcore

    db_url = os.environ.get("MINDCORE_DB_URL", "sqlite:///mindcore.db")
    multi_agent = os.environ.get("MINDCORE_MULTI_AGENT", "false").lower() == "true"
    rate_limit = os.environ.get("MINDCORE_RATE_LIMIT", "100/minute")
    cors_origins_str = os.environ.get("MINDCORE_CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_str.split(",")]

    mindcore = Mindcore(storage=db_url, enable_multi_agent=multi_agent)

    return create_app(
        flr=mindcore._flr,
        clst=mindcore._clst,
        vocabulary=mindcore._vocabulary,
        access_controller=mindcore._access_controller if multi_agent else None,
        rate_limit=rate_limit,
        cors_origins=cors_origins,
    )


# Create app instance for uvicorn
app = create_standalone_app()
