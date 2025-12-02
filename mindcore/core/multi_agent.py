"""
Multi-Agent Memory Configuration.

Provides configuration and utilities for shared memory across multiple AI agents.
Supports various memory sharing modes:
- Isolated: Each agent has completely separate memory
- Shared: Agents can share memory with specified groups
- Public: All agents can access all memory
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemorySharingMode(str, Enum):
    """Memory sharing modes for multi-agent deployments."""

    DISABLED = "disabled"  # Single-agent mode (no agent_id required)
    ISOLATED = "isolated"  # Each agent has private memory only
    SHARED = "shared"  # Agents can share with specific groups
    PUBLIC = "public"  # All agents share all memory


class AgentVisibility(str, Enum):
    """Visibility levels for agent-created content."""

    PRIVATE = "private"  # Only owning agent can access
    SHARED = "shared"  # Agents in same sharing groups can access
    PUBLIC = "public"  # All agents can access


@dataclass
class AgentProfile:
    """Profile for a registered agent."""

    agent_id: str
    name: str
    description: Optional[str] = None
    sharing_groups: List[str] = field(default_factory=list)
    default_visibility: AgentVisibility = AgentVisibility.PRIVATE
    can_read_public: bool = True
    can_write_public: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_access(self, visibility: str, owner_groups: List[str]) -> bool:
        """Check if this agent can access content with given visibility."""
        if visibility == "public":
            return self.can_read_public
        if visibility == "private":
            return False  # Only owner can access private
        if visibility == "shared":
            # Check for group overlap
            return bool(set(self.sharing_groups) & set(owner_groups))
        return False


@dataclass
class MultiAgentConfig:
    """
    Configuration for multi-agent memory sharing.

    Example:
        >>> config = MultiAgentConfig(
        ...     enabled=True,
        ...     mode=MemorySharingMode.SHARED,
        ...     require_agent_id=True,
        ...     default_visibility=AgentVisibility.PRIVATE
        ... )
    """

    # Core settings
    enabled: bool = False
    mode: MemorySharingMode = MemorySharingMode.DISABLED
    require_agent_id: bool = False  # If True, agent_id is required for all operations

    # Default behavior
    default_visibility: AgentVisibility = AgentVisibility.PRIVATE
    default_sharing_groups: List[str] = field(default_factory=list)

    # Access control
    allow_cross_agent_context: bool = (
        True  # Allow agents to see other agents' shared/public content
    )
    allow_anonymous_read: bool = False  # Allow reading without agent_id (single-agent compat)
    allow_anonymous_write: bool = False  # Allow writing without agent_id

    # Validation
    registered_agents_only: bool = False  # Only allow registered agent_ids
    max_sharing_groups: int = 10  # Maximum sharing groups per message

    def validate(self) -> List[str]:
        """Validate configuration, return list of issues."""
        issues = []

        if self.enabled and self.mode == MemorySharingMode.DISABLED:
            issues.append("Multi-agent is enabled but mode is DISABLED")

        if self.require_agent_id and self.allow_anonymous_write:
            issues.append("Cannot require agent_id while allowing anonymous writes")

        if (
            self.mode == MemorySharingMode.PUBLIC
            and self.default_visibility == AgentVisibility.PRIVATE
        ):
            issues.append("Public mode typically uses shared/public default visibility")

        return issues


class MultiAgentManager:
    """
    Manages multi-agent memory sharing.

    Handles agent registration, access control, and visibility rules.

    Example:
        >>> manager = MultiAgentManager(config)
        >>>
        >>> # Register agents
        >>> manager.register_agent("agent1", "Customer Support", groups=["support"])
        >>> manager.register_agent("agent2", "Sales Assistant", groups=["sales"])
        >>>
        >>> # Check access
        >>> if manager.can_access("agent2", message):
        ...     # Agent can access this message
    """

    def __init__(self, config: Optional[MultiAgentConfig] = None):
        """
        Initialize multi-agent manager.

        Args:
            config: Multi-agent configuration (uses defaults if None)
        """
        self.config = config or MultiAgentConfig()
        self._lock = threading.Lock()
        self._agents: Dict[str, AgentProfile] = {}
        self._sharing_groups: Dict[str, Set[str]] = {}  # group -> agent_ids

    @property
    def is_enabled(self) -> bool:
        """Check if multi-agent mode is enabled."""
        return self.config.enabled and self.config.mode != MemorySharingMode.DISABLED

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: Optional[str] = None,
        sharing_groups: Optional[List[str]] = None,
        default_visibility: Optional[AgentVisibility] = None,
        can_read_public: bool = True,
        can_write_public: bool = False,
        **metadata,
    ) -> AgentProfile:
        """
        Register an agent for multi-agent memory access.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            description: Optional agent description
            sharing_groups: Groups this agent belongs to
            default_visibility: Default visibility for content created by this agent
            can_read_public: Whether agent can read public content
            can_write_public: Whether agent can create public content
            **metadata: Additional agent metadata

        Returns:
            AgentProfile for the registered agent
        """
        groups = sharing_groups or self.config.default_sharing_groups.copy()
        visibility = default_visibility or self.config.default_visibility

        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            description=description,
            sharing_groups=groups,
            default_visibility=visibility,
            can_read_public=can_read_public,
            can_write_public=can_write_public,
            metadata=metadata,
        )

        with self._lock:
            self._agents[agent_id] = profile
            for group in groups:
                if group not in self._sharing_groups:
                    self._sharing_groups[group] = set()
                self._sharing_groups[group].add(agent_id)

        logger.info(f"Registered agent '{name}' ({agent_id}) with groups: {groups}")
        return profile

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        with self._lock:
            if agent_id not in self._agents:
                return False

            profile = self._agents.pop(agent_id)
            for group in profile.sharing_groups:
                if group in self._sharing_groups:
                    self._sharing_groups[group].discard(agent_id)

        logger.info(f"Unregistered agent {agent_id}")
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentProfile]:
        """List all registered agents."""
        return list(self._agents.values())

    def validate_agent_id(self, agent_id: Optional[str], for_write: bool = False) -> tuple:
        """
        Validate an agent_id for an operation.

        Args:
            agent_id: Agent ID to validate (can be None)
            for_write: True if this is a write operation

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        if not self.is_enabled:
            return True, None

        # Check if agent_id is required
        if agent_id is None:
            if for_write and not self.config.allow_anonymous_write:
                return False, "agent_id is required for write operations in multi-agent mode"
            if not for_write and not self.config.allow_anonymous_read:
                return False, "agent_id is required for read operations in multi-agent mode"
            if self.config.require_agent_id:
                return False, "agent_id is required (require_agent_id=True)"
            return True, None

        # Check if agent is registered (if required)
        if self.config.registered_agents_only:
            if agent_id not in self._agents:
                return False, f"Agent '{agent_id}' is not registered"

        return True, None

    def can_access(
        self,
        agent_id: Optional[str],
        owner_agent_id: Optional[str],
        visibility: str,
        sharing_groups: List[str],
    ) -> bool:
        """
        Check if an agent can access content.

        Args:
            agent_id: Agent trying to access (None for anonymous)
            owner_agent_id: Agent that owns the content
            visibility: Content visibility level
            sharing_groups: Content sharing groups

        Returns:
            True if access is allowed
        """
        if not self.is_enabled:
            return True

        # Owner always has access
        if agent_id and agent_id == owner_agent_id:
            return True

        # Public mode - everyone can access everything
        if self.config.mode == MemorySharingMode.PUBLIC:
            return True

        # Isolated mode - only owner has access
        if self.config.mode == MemorySharingMode.ISOLATED:
            return agent_id == owner_agent_id

        # Shared mode - check visibility
        if visibility == "public":
            if agent_id:
                profile = self._agents.get(agent_id)
                return profile.can_read_public if profile else self.config.allow_anonymous_read
            return self.config.allow_anonymous_read

        if visibility == "private":
            return agent_id == owner_agent_id

        if visibility == "shared":
            if not agent_id:
                return False
            profile = self._agents.get(agent_id)
            if not profile:
                return False
            return bool(set(profile.sharing_groups) & set(sharing_groups))

        return False

    def get_access_filter(
        self,
        agent_id: Optional[str],
        include_own: bool = True,
        include_shared: bool = True,
        include_public: bool = True,
    ) -> Dict[str, Any]:
        """
        Get filter criteria for database queries.

        Args:
            agent_id: Agent making the query
            include_own: Include agent's own content
            include_shared: Include shared content from groups
            include_public: Include public content

        Returns:
            Dictionary with filter criteria for database queries
        """
        if not self.is_enabled:
            return {}

        filter_criteria = {
            "multi_agent_enabled": True,
            "agent_id": agent_id,
            "visibility_filters": [],
        }

        if include_own and agent_id:
            filter_criteria["visibility_filters"].append({"type": "own", "agent_id": agent_id})

        if include_shared and agent_id:
            profile = self._agents.get(agent_id)
            if profile and profile.sharing_groups:
                filter_criteria["visibility_filters"].append(
                    {"type": "shared", "groups": profile.sharing_groups}
                )

        if include_public:
            filter_criteria["visibility_filters"].append({"type": "public"})

        return filter_criteria

    def get_default_visibility(self, agent_id: Optional[str]) -> str:
        """Get default visibility for an agent's content."""
        if agent_id:
            profile = self._agents.get(agent_id)
            if profile:
                return profile.default_visibility.value
        return self.config.default_visibility.value

    def get_default_sharing_groups(self, agent_id: Optional[str]) -> List[str]:
        """Get default sharing groups for an agent's content."""
        if agent_id:
            profile = self._agents.get(agent_id)
            if profile:
                return profile.sharing_groups.copy()
        return self.config.default_sharing_groups.copy()

    def get_agents_in_group(self, group: str) -> List[str]:
        """Get all agent IDs in a sharing group."""
        return list(self._sharing_groups.get(group, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get multi-agent statistics."""
        return {
            "enabled": self.is_enabled,
            "mode": self.config.mode.value,
            "registered_agents": len(self._agents),
            "sharing_groups": len(self._sharing_groups),
            "agents": [
                {
                    "agent_id": p.agent_id,
                    "name": p.name,
                    "groups": p.sharing_groups,
                    "default_visibility": p.default_visibility.value,
                }
                for p in self._agents.values()
            ],
        }


# Singleton instance
_manager: Optional[MultiAgentManager] = None
_manager_lock = threading.Lock()


def get_multi_agent_manager(config: Optional[MultiAgentConfig] = None) -> MultiAgentManager:
    """Get or create the singleton multi-agent manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = MultiAgentManager(config)
    return _manager


def reset_multi_agent_manager() -> None:
    """Reset the singleton manager (for testing)."""
    global _manager
    with _manager_lock:
        _manager = None


def configure_multi_agent(
    enabled: bool = True, mode: str = "shared", require_agent_id: bool = True, **kwargs
) -> MultiAgentManager:
    """
    Configure and return multi-agent manager.

    Convenience function for quick setup.

    Args:
        enabled: Enable multi-agent mode
        mode: Sharing mode ("disabled", "isolated", "shared", "public")
        require_agent_id: Require agent_id for all operations
        **kwargs: Additional MultiAgentConfig options

    Returns:
        Configured MultiAgentManager

    Example:
        >>> manager = configure_multi_agent(
        ...     enabled=True,
        ...     mode="shared",
        ...     require_agent_id=True
        ... )
        >>> manager.register_agent("agent1", "Support Bot", groups=["support"])
    """
    reset_multi_agent_manager()

    config = MultiAgentConfig(
        enabled=enabled, mode=MemorySharingMode(mode), require_agent_id=require_agent_id, **kwargs
    )

    return get_multi_agent_manager(config)
