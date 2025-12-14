"""Cross-Agent Memory Layer.

Enables memory sharing, synchronization, and routing between multiple AI agents.

Features:
- Agent Registry: Register and manage agents with capabilities
- Memory Sharing: Share memories based on access levels (private, team, shared, global)
- Cross-Agent Sync: Synchronize memories between agents
- Attention Routing: Route queries to relevant agent memories
- Team Management: Organize agents into teams with shared access

Example:
    from mindcore.v2.cross_agent import CrossAgentLayer

    # Initialize
    layer = CrossAgentLayer(storage=storage)

    # Register agents
    layer.register_agent(
        agent_id="support_bot",
        name="Support Agent",
        capabilities=["customer_support", "billing"],
        teams=["customer_service"],
    )

    layer.register_agent(
        agent_id="sales_bot",
        name="Sales Agent",
        capabilities=["sales", "product_info"],
        teams=["customer_service"],
    )

    # Store memory with team access
    layer.store(
        memory=memory,
        agent_id="support_bot",
        access_level="team",
    )

    # Query across agents
    result = layer.query(
        query="customer preferences",
        user_id="user123",
        requesting_agent="sales_bot",
    )
"""

from .registry import (
    Agent,
    AgentCapability,
    AgentRegistry,
    AgentStatus,
    Team,
)
from .sharing import (
    CrossAgentMemory,
    ShareResult,
    SyncDirection,
    SyncResult,
)
from .routing import (
    AttentionRouter,
    RouteResult,
    RoutingStrategy,
)
from .layer import CrossAgentLayer


__all__ = [
    # Registry
    "Agent",
    "AgentCapability",
    "AgentRegistry",
    "AgentStatus",
    "Team",
    # Sharing
    "CrossAgentMemory",
    "ShareResult",
    "SyncDirection",
    "SyncResult",
    # Routing
    "AttentionRouter",
    "RouteResult",
    "RoutingStrategy",
    # Main
    "CrossAgentLayer",
]
