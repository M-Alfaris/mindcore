"""REST API Server for Mindcore v2.

Provides HTTP endpoints for memory operations.
Can be used standalone or alongside MCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
):
    """Create FastAPI application for Mindcore REST API.

    Args:
        flr: FLR instance for fast recall
        clst: CLST instance for long-term storage
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException, Header, Query, Body
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        request: StoreMemoryRequest,
        x_agent_id: str | None = Header(None),
    ):
        from ..flr import Memory

        memory = Memory(
            memory_id="",
            content=request.content,
            memory_type=request.memory_type,
            user_id=request.user_id,
            agent_id=x_agent_id,
            topics=request.topics,
            categories=request.categories,
            sentiment=request.sentiment,
            importance=request.importance,
            entities=request.entities,
            access_level=request.access_level,
            vocabulary_version=vocabulary.version if vocabulary else "1.0.0",
        )

        memory_id = clst.store(memory)
        return {"memory_id": memory_id, "success": True}

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

    @app.post("/memories/search")
    async def search_memories(
        request: SearchRequest,
        x_agent_id: str | None = Header(None),
    ):
        memories = clst.search(
            query=request.query,
            user_id=request.user_id,
            agent_id=x_agent_id,
            topics=request.topics,
            categories=request.categories,
            memory_types=request.memory_types,
            limit=request.limit,
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
    @app.post("/recall")
    async def recall(
        request: RecallRequest,
        x_agent_id: str | None = Header(None),
    ):
        result = flr.query(
            query=request.query,
            user_id=request.user_id,
            agent_id=x_agent_id,
            attention_hints=request.attention_hints,
            memory_types=request.memory_types,
            limit=request.limit,
        )

        return {
            "memories": [m.to_dict() for m in result.memories],
            "scores": result.scores,
            "attention_focus": result.attention_focus,
            "suggested_memory_types": result.suggested_memory_types,
            "latency_ms": result.query_latency_ms,
        }

    @app.post("/reinforce")
    async def reinforce(request: ReinforceRequest):
        flr.reinforce(request.memory_id, request.signal)
        return {"success": True}

    # Agent management (if access controller is configured)
    @app.post("/agents")
    async def register_agent(request: RegisterAgentRequest):
        if not access_controller:
            raise HTTPException(
                status_code=400,
                detail="Access control not configured"
            )

        try:
            profile = access_controller.register_agent(
                agent_id=request.agent_id,
                name=request.name,
                description=request.description,
                teams=request.teams,
            )
            return profile.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/agents")
    async def list_agents():
        if not access_controller:
            return {"agents": []}

        agents = access_controller.list_agents()
        return {"agents": [a.to_dict() for a in agents]}

    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str):
        if not access_controller:
            raise HTTPException(
                status_code=400,
                detail="Access control not configured"
            )

        profile = access_controller.get_agent(agent_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Agent not found")

        return profile.to_dict()

    @app.delete("/agents/{agent_id}")
    async def unregister_agent(agent_id: str):
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
    @app.post("/compress")
    async def compress_memories(
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
    flr: FLR,
    clst: CLST,
    vocabulary: VocabularySchema | None = None,
    access_controller: AccessController | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Run the REST API server.

    Args:
        flr: FLR instance
        clst: CLST instance
        vocabulary: Optional vocabulary schema
        access_controller: Optional access controller
        host: Host to bind to
        port: Port to bind to
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn required to run server. Install with: pip install uvicorn"
        )

    app = create_app(flr, clst, vocabulary, access_controller)
    uvicorn.run(app, host=host, port=port)
