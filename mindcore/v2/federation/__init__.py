"""Multi-Agent Memory Federation for MindCore.

This module provides enterprise-grade memory federation for organizations
with multiple AI agents. Key features:

1. **Isolated FLR per Agent**: Each agent maintains its own hot memory
2. **Shared CLST with Access Control**: Centralized cold storage with
   namespace-based isolation and RBAC
3. **Unified SVL**: Single vocabulary with scoped feedback aggregation
4. **Cross-Agent Signal Propagation**: Reinforcement signals flow across
   agents with trust weighting

Architecture:
```
                    ┌─────────────────────────────────────┐
                    │           Organization SVL           │
                    │   (Shared Vocabulary + Feedback)     │
                    └─────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Dept A   │               │  Dept B   │               │  Dept C   │
    │ Namespace │               │ Namespace │               │ Namespace │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Team 1   │               │  Team 2   │               │  Team 3   │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
   ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
   │      │      │             │      │      │             │      │      │
┌──▼──┐┌──▼──┐┌──▼──┐       ┌──▼──┐┌──▼──┐┌──▼──┐       ┌──▼──┐┌──▼──┐┌──▼──┐
│FLR 1││FLR 2││FLR 3│       │FLR 4││FLR 5││FLR 6│       │FLR 7││FLR 8││FLR 9│
│Agent││Agent││Agent│       │Agent││Agent││Agent│       │Agent││Agent││Agent│
└─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────────┐
                    │         Federated CLST                │
                    │  (Shared Storage + Access Control)    │
                    └───────────────────────────────────────┘
```

Access Levels:
- PRIVATE: Only the specific agent
- AGENT: Shared with same agent type
- TEAM: Shared within team
- DEPARTMENT: Shared within department
- AD_HOC_GROUP: Custom group membership
- ORGANIZATION: Visible to all agents in org

Example:
    from mindcore.v2.federation import (
        AgentMemoryBridge,
        FederatedCLST,
        FederatedSVL,
        AccessLevel,
        MemoryNamespace,
    )

    # Create org-level components
    svl = FederatedSVL(org_id="acme-corp")
    clst = FederatedCLST(org_id="acme-corp", storage=vector_store)

    # Create namespace for support team
    support_ns = MemoryNamespace(
        org_id="acme-corp",
        department="customer-success",
        team="support-tier-1",
    )

    # Create agent with its own FLR but connected to shared CLST/SVL
    agent = AgentMemoryBridge(
        agent_id="support-agent-001",
        namespace=support_ns,
        federated_clst=clst,
        federated_svl=svl,
    )

    # Store memory with access control
    agent.store(
        content="Customer prefers email contact",
        user_id="customer-123",
        access_level=AccessLevel.TEAM,  # Visible to support-tier-1
    )

    # Reinforcement signals propagate to other agents
    agent.reinforce("memory-id", signal=0.8)  # Other agents benefit
"""

from .access_control import (
    AccessLevel,
    AccessPolicy,
    AccessScope,
    MemoryACL,
)
from .namespace import (
    MemoryNamespace,
    NamespaceHierarchy,
)
from .federated_clst import (
    FederatedCLST,
    FederatedMemory,
    NamespacedQuery,
)
from .federated_svl import (
    AggregatedFeedback,
    FederatedSVL,
    ScopedFeedback,
)
from .signal_aggregator import (
    CrossAgentSignalAggregator,
    SignalWeight,
    TrustPolicy,
)
from .agent_bridge import (
    AgentMemoryBridge,
    AgentIdentity,
    LocalMemory,
)
from .config import (
    Federation,
    FederationConfig,
    create_agent,
    create_federation,
    quick_setup,
)


__all__ = [
    # Access Control
    "AccessLevel",
    "AccessPolicy",
    "AccessScope",
    "MemoryACL",
    # Namespace
    "MemoryNamespace",
    "NamespaceHierarchy",
    # Federated Components
    "FederatedCLST",
    "FederatedMemory",
    "NamespacedQuery",
    "FederatedSVL",
    "AggregatedFeedback",
    "ScopedFeedback",
    # Signal Aggregation
    "CrossAgentSignalAggregator",
    "SignalWeight",
    "TrustPolicy",
    # Agent Bridge
    "AgentMemoryBridge",
    "AgentIdentity",
    "LocalMemory",
    # Configuration
    "Federation",
    "FederationConfig",
    "create_agent",
    "create_federation",
    "quick_setup",
]
