"""Mindcore Proxy Server.

HTTP proxy server that intercepts Claude Code sessions to
capture reasoning and inject relevant memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..mindcore import Mindcore

from .session import Session, SessionManager, SessionState
from .extractor import ReasoningExtractor
from .injector import MemoryInjector, InjectionStrategy


@dataclass
class ProxyConfig:
    """Configuration for the Mindcore Proxy.

    Attributes:
        host: Host to bind to
        port: Port to listen on
        storage: Storage connection string
        enable_injection: Whether to inject memories
        enable_extraction: Whether to extract reasoning
        max_memories: Max memories to inject per request
        injection_strategy: How to inject memories
        min_confidence: Minimum confidence for extraction
        auto_store: Automatically store extracted reasoning
    """

    host: str = "127.0.0.1"
    port: int = 8080
    storage: str = "sqlite:///mindcore.db"
    enable_injection: bool = True
    enable_extraction: bool = True
    max_memories: int = 5
    injection_strategy: InjectionStrategy = InjectionStrategy.SYSTEM
    min_confidence: float = 0.6
    auto_store: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


class MindcoreProxy:
    """HTTP Proxy for Claude Code integration.

    Intercepts requests to capture reasoning from sessions
    and inject relevant memories from FLR.

    Example:
        proxy = MindcoreProxy(
            storage="sqlite:///mindcore.db",
            port=8080,
        )
        proxy.start()
    """

    def __init__(
        self,
        storage: str = "sqlite:///mindcore.db",
        port: int = 8080,
        config: ProxyConfig | None = None,
        mindcore: Mindcore | None = None,
    ) -> None:
        """Initialize the proxy.

        Args:
            storage: Storage connection string
            port: Port to listen on
            config: Optional full configuration
            mindcore: Optional existing Mindcore instance
        """
        self.config = config or ProxyConfig(storage=storage, port=port)

        # Initialize Mindcore if not provided
        if mindcore is not None:
            self._mindcore = mindcore
            self._owns_mindcore = False
        else:
            from ..mindcore import Mindcore
            self._mindcore = Mindcore(storage=self.config.storage)
            self._owns_mindcore = True

        # Initialize components
        self.session_manager = SessionManager()
        self.extractor = ReasoningExtractor(
            min_confidence=self.config.min_confidence,
        )
        self.injector = MemoryInjector(
            flr=self._mindcore._flr,
            strategy=self.config.injection_strategy,
            max_memories=self.config.max_memories,
        )

        self._app = None
        self._server = None

    @property
    def mindcore(self) -> Mindcore:
        """Get the Mindcore instance."""
        return self._mindcore

    def create_app(self) -> Any:
        """Create the FastAPI application.

        Returns:
            FastAPI application instance
        """
        try:
            from fastapi import FastAPI, Request, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "FastAPI required for proxy server. "
                "Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="Mindcore Proxy",
            description="Claude Code integration proxy with memory injection",
            version="1.0.0",
        )

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class SessionRequest(BaseModel):
            session_id: str
            user_id: str
            messages: list[dict[str, Any]] = []

        class MessageRequest(BaseModel):
            session_id: str
            role: str
            content: str

        @app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "mindcore-proxy"}

        @app.post("/session/start")
        def start_session(request: SessionRequest):
            """Start or resume a session."""
            session = self.session_manager.get_or_create(
                session_id=request.session_id,
                user_id=request.user_id,
            )
            session.activate()

            result = {"session_id": session.session_id, "status": "active"}

            # Inject memories if enabled
            if self.config.enable_injection and request.messages:
                injection = self.injector.inject_for_messages(
                    messages=request.messages,
                    user_id=request.user_id,
                )
                if injection.formatted_content:
                    result["injected_memories"] = injection.injected_count
                    result["memory_content"] = injection.formatted_content

            return result

        @app.post("/session/message")
        def add_message(request: MessageRequest):
            """Add a message to session and extract reasoning."""
            session = self.session_manager.get(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            session.add_message(request.role, request.content)

            result = {"status": "added"}

            # Extract reasoning if enabled
            if self.config.enable_extraction and request.role == "assistant":
                reasoning_list = self.extractor.extract_from_message(
                    content=request.content,
                    role=request.role,
                )

                if reasoning_list and self.config.auto_store:
                    for reasoning in reasoning_list:
                        memory_dict = reasoning.to_memory_dict(session.user_id)
                        self._mindcore.store(**memory_dict)

                    result["extracted_reasoning"] = len(reasoning_list)

            return result

        @app.post("/session/end")
        def end_session(session_id: str):
            """End a session and extract all reasoning."""
            session = self.session_manager.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            # Extract reasoning from entire session
            if self.config.enable_extraction:
                all_reasoning = self.extractor.extract_from_session(session.messages)

                if all_reasoning and self.config.auto_store:
                    for reasoning in all_reasoning:
                        memory_dict = reasoning.to_memory_dict(session.user_id)
                        self._mindcore.store(**memory_dict)

            session.complete()

            return {
                "session_id": session_id,
                "status": "completed",
                "message_count": len(session.messages),
            }

        @app.get("/session/{session_id}")
        def get_session(session_id: str):
            """Get session details."""
            session = self.session_manager.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return session.to_dict()

        @app.delete("/session/{session_id}")
        def delete_session(session_id: str):
            """Delete a session."""
            if self.session_manager.delete(session_id):
                return {"status": "deleted"}
            raise HTTPException(status_code=404, detail="Session not found")

        @app.post("/inject")
        def inject_memories(request: SessionRequest):
            """Inject memories for a query."""
            if not request.messages:
                return {"injected": 0}

            injection = self.injector.inject_for_messages(
                messages=request.messages,
                user_id=request.user_id,
            )

            return {
                "injected": injection.injected_count,
                "content": injection.formatted_content,
                "tokens": injection.total_tokens,
            }

        self._app = app
        return app

    def start(self, block: bool = True) -> None:
        """Start the proxy server.

        Args:
            block: Whether to block until shutdown
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "Uvicorn required for proxy server. "
                "Install with: pip install uvicorn"
            )

        if self._app is None:
            self.create_app()

        uvicorn.run(
            self._app,
            host=self.config.host,
            port=self.config.port,
        )

    def stop(self) -> None:
        """Stop the proxy server."""
        if self._owns_mindcore and self._mindcore:
            self._mindcore.close()

    def __enter__(self) -> MindcoreProxy:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
