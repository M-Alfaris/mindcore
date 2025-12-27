"""Central Configuration for Federation.

Provides easy-to-use configuration and factory methods for setting up
the multi-agent federation system.

Usage:
    from mindcore.federation.config import (
        FederationConfig,
        create_federation,
        create_agent,
    )

    # Option 1: From YAML/JSON config file
    federation = create_federation("./config/federation.yaml")

    # Option 2: Programmatic configuration
    config = FederationConfig(
        org_id="acme-corp",
        departments=["sales", "support", "engineering"],
        default_access_level="team",
    )
    federation = Federation.from_config(config)

    # Create an agent connected to federation
    agent = create_agent(
        federation=federation,
        agent_id="support-001",
        agent_type="support-bot",
        department="support",
        team="tier-1",
    )

Config File Example (federation.yaml):
    org_id: "acme-corp"

    # Organizational structure
    structure:
      departments:
        sales:
          teams: ["enterprise", "smb", "partnerships"]
        support:
          teams: ["tier-1", "tier-2", "escalations"]
        engineering:
          teams: ["backend", "frontend", "devops"]

    # Default settings
    defaults:
      access_level: "team"
      auto_persist_threshold: 0.7
      propagate_signals: true

    # SVL vocabulary
    vocabulary:
      topics:
        - billing
        - technical
        - sales
        - onboarding
      categories:
        - question
        - complaint
        - feedback
        - request

    # Trust policy
    trust_policy: "namespace_weighted"
    signal_weights:
      same_team_bonus: 0.5
      same_department_bonus: 0.3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .access_control import AccessLevel
from .agent_bridge import AgentIdentity, AgentMemoryBridge
from .federated_clst import FederatedCLST, StorageBackend
from .federated_svl import FederatedSVL
from .namespace import NamespaceHierarchy
from .signal_aggregator import (
    CrossAgentSignalAggregator,
    SignalWeight,
    TrustPolicy,
)


@dataclass
class DepartmentConfig:
    """Configuration for a department."""

    name: str
    teams: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)


@dataclass
class StructureConfig:
    """Organizational structure configuration."""

    departments: dict[str, DepartmentConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructureConfig:
        """Create from dictionary."""
        departments = {}
        for dept_name, dept_data in data.get("departments", {}).items():
            if isinstance(dept_data, dict):
                departments[dept_name] = DepartmentConfig(
                    name=dept_name,
                    teams=dept_data.get("teams", []),
                    labels=set(dept_data.get("labels", [])),
                )
            elif isinstance(dept_data, list):
                # Simple list of teams
                departments[dept_name] = DepartmentConfig(
                    name=dept_name,
                    teams=dept_data,
                )
        return cls(departments=departments)


@dataclass
class DefaultsConfig:
    """Default settings configuration."""

    access_level: AccessLevel = AccessLevel.TEAM
    auto_persist_threshold: float = 0.7
    propagate_signals: bool = True
    max_local_memories: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefaultsConfig:
        """Create from dictionary."""
        access_level_str = data.get("access_level", "team")
        access_level = AccessLevel[access_level_str.upper()]

        return cls(
            access_level=access_level,
            auto_persist_threshold=data.get("auto_persist_threshold", 0.7),
            propagate_signals=data.get("propagate_signals", True),
            max_local_memories=data.get("max_local_memories", 1000),
        )


@dataclass
class VocabularyConfig:
    """Vocabulary configuration."""

    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    schemas: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularyConfig:
        """Create from dictionary."""
        return cls(
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            entities=data.get("entities", []),
            schemas=data.get("schemas", {}),
        )


@dataclass
class SignalConfig:
    """Signal aggregation configuration."""

    trust_policy: TrustPolicy = TrustPolicy.NAMESPACE_WEIGHTED
    same_team_bonus: float = 0.5
    same_department_bonus: float = 0.3
    same_agent_type_bonus: float = 0.2
    decay_half_life_hours: float = 168.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalConfig:
        """Create from dictionary."""
        trust_policy_str = data.get("trust_policy", "namespace_weighted")
        trust_policy = TrustPolicy(trust_policy_str)

        weights = data.get("signal_weights", {})
        return cls(
            trust_policy=trust_policy,
            same_team_bonus=weights.get("same_team_bonus", 0.5),
            same_department_bonus=weights.get("same_department_bonus", 0.3),
            same_agent_type_bonus=weights.get("same_agent_type_bonus", 0.2),
            decay_half_life_hours=weights.get("decay_half_life_hours", 168.0),
        )

    def to_signal_weight(self) -> SignalWeight:
        """Convert to SignalWeight."""
        return SignalWeight(
            same_team_bonus=self.same_team_bonus,
            same_department_bonus=self.same_department_bonus,
            same_agent_type_bonus=self.same_agent_type_bonus,
            decay_half_life_hours=self.decay_half_life_hours,
        )


@dataclass
class FederationConfig:
    """Complete federation configuration.

    This is the main configuration class that brings together
    all components of the federation system.
    """

    org_id: str
    structure: StructureConfig = field(default_factory=StructureConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederationConfig:
        """Create configuration from dictionary."""
        return cls(
            org_id=data["org_id"],
            structure=StructureConfig.from_dict(data.get("structure", {})),
            defaults=DefaultsConfig.from_dict(data.get("defaults", {})),
            vocabulary=VocabularyConfig.from_dict(data.get("vocabulary", {})),
            signals=SignalConfig.from_dict(data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> FederationConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config. Install with: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> FederationConfig:
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> FederationConfig:
        """Load configuration from file (auto-detect format)."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        # Try YAML first, then JSON
        try:
            return cls.from_yaml(path)
        except Exception:
            return cls.from_json(path)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "org_id": self.org_id,
            "structure": {
                "departments": {
                    name: {"teams": dept.teams, "labels": list(dept.labels)}
                    for name, dept in self.structure.departments.items()
                }
            },
            "defaults": {
                "access_level": self.defaults.access_level.name.lower(),
                "auto_persist_threshold": self.defaults.auto_persist_threshold,
                "propagate_signals": self.defaults.propagate_signals,
                "max_local_memories": self.defaults.max_local_memories,
            },
            "vocabulary": {
                "topics": self.vocabulary.topics,
                "categories": self.vocabulary.categories,
                "entities": self.vocabulary.entities,
                "schemas": self.vocabulary.schemas,
            },
            "trust_policy": self.signals.trust_policy.value,
            "signal_weights": {
                "same_team_bonus": self.signals.same_team_bonus,
                "same_department_bonus": self.signals.same_department_bonus,
                "same_agent_type_bonus": self.signals.same_agent_type_bonus,
                "decay_half_life_hours": self.signals.decay_half_life_hours,
            },
        }


@dataclass
class Federation:
    """Main federation container.

    Holds all shared components of the federation system.
    """

    config: FederationConfig
    hierarchy: NamespaceHierarchy
    clst: FederatedCLST | None = None
    svl: FederatedSVL | None = None
    signal_aggregator: CrossAgentSignalAggregator | None = None

    # Registered agents
    agents: dict[str, AgentMemoryBridge] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: FederationConfig,
        storage: StorageBackend | None = None,
    ) -> Federation:
        """Create federation from configuration.

        Args:
            config: Federation configuration
            storage: Optional storage backend for CLST

        Returns:
            Configured Federation instance
        """
        # Build namespace hierarchy
        hierarchy = NamespaceHierarchy(org_id=config.org_id)
        for dept_name, dept_config in config.structure.departments.items():
            hierarchy.register_department(dept_name)
            for team in dept_config.teams:
                hierarchy.register_team(dept_name, team)

        # Create SVL
        svl = FederatedSVL(org_id=config.org_id)
        svl.set_vocabulary(
            topics=config.vocabulary.topics,
            categories=config.vocabulary.categories,
            entities=config.vocabulary.entities,
            schemas=config.vocabulary.schemas,
        )

        # Create signal aggregator
        signal_aggregator = CrossAgentSignalAggregator(
            trust_policy=config.signals.trust_policy,
            weight_config=config.signals.to_signal_weight(),
        )

        # Create CLST if storage provided
        clst = None
        if storage:
            clst = FederatedCLST(
                org_id=config.org_id,
                storage=storage,
            )

        return cls(
            config=config,
            hierarchy=hierarchy,
            clst=clst,
            svl=svl,
            signal_aggregator=signal_aggregator,
        )

    def create_agent(
        self,
        agent_id: str,
        agent_type: str,
        department: str | None = None,
        team: str | None = None,
        display_name: str | None = None,
        capabilities: set[str] | None = None,
        groups: set[str] | None = None,
    ) -> AgentMemoryBridge:
        """Create and register an agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            department: Agent's department
            team: Agent's team
            display_name: Human-readable name
            capabilities: Agent capabilities
            groups: Ad-hoc group memberships

        Returns:
            Configured AgentMemoryBridge
        """
        # Create namespace
        namespace = self.hierarchy.get_namespace(
            department=department,
            team=team,
        )

        # Create identity
        identity = AgentIdentity(
            agent_id=agent_id,
            agent_type=agent_type,
            display_name=display_name,
            capabilities=capabilities or set(),
            groups=groups or set(),
        )

        # Create bridge
        bridge = AgentMemoryBridge(
            identity=identity,
            namespace=namespace,
            federated_clst=self.clst,
            federated_svl=self.svl,
            signal_aggregator=self.signal_aggregator,
            default_access_level=self.config.defaults.access_level,
            auto_persist_threshold=self.config.defaults.auto_persist_threshold,
            max_local_memories=self.config.defaults.max_local_memories,
            propagate_signals=self.config.defaults.propagate_signals,
        )

        # Register
        self.agents[agent_id] = bridge
        return bridge

    def get_agent(self, agent_id: str) -> AgentMemoryBridge | None:
        """Get a registered agent."""
        return self.agents.get(agent_id)

    def get_agents_in_team(
        self,
        department: str,
        team: str,
    ) -> list[AgentMemoryBridge]:
        """Get all agents in a specific team."""
        return [
            agent
            for agent in self.agents.values()
            if agent.namespace.department == department and agent.namespace.team == team
        ]

    def get_agents_in_department(
        self,
        department: str,
    ) -> list[AgentMemoryBridge]:
        """Get all agents in a department."""
        return [agent for agent in self.agents.values() if agent.namespace.department == department]


# =============================================================================
# Factory Functions (Convenience API)
# =============================================================================


def create_federation(
    config: str | Path | FederationConfig,
    storage: StorageBackend | None = None,
) -> Federation:
    """Create a federation from configuration.

    Args:
        config: Config file path or FederationConfig instance
        storage: Optional storage backend

    Returns:
        Configured Federation

    Example:
        # From file
        federation = create_federation("./config/federation.yaml")

        # From config object
        config = FederationConfig(org_id="acme")
        federation = create_federation(config)
    """
    if isinstance(config, (str, Path)):
        config = FederationConfig.from_file(config)

    return Federation.from_config(config, storage)


def create_agent(
    federation: Federation,
    agent_id: str,
    agent_type: str,
    department: str | None = None,
    team: str | None = None,
    **kwargs: Any,
) -> AgentMemoryBridge:
    """Create an agent connected to federation.

    Args:
        federation: Federation instance
        agent_id: Unique agent identifier
        agent_type: Type of agent
        department: Agent's department
        team: Agent's team
        **kwargs: Additional agent configuration

    Returns:
        Configured AgentMemoryBridge

    Example:
        agent = create_agent(
            federation=federation,
            agent_id="support-001",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )
    """
    return federation.create_agent(
        agent_id=agent_id,
        agent_type=agent_type,
        department=department,
        team=team,
        **kwargs,
    )


def quick_setup(
    org_id: str,
    departments: list[str] | dict[str, list[str]] | None = None,
    topics: list[str] | None = None,
    categories: list[str] | None = None,
    storage: StorageBackend | None = None,
) -> Federation:
    """Quick setup for simple federation configurations.

    Args:
        org_id: Organization identifier
        departments: Department names or {dept: [teams]} mapping
        topics: Vocabulary topics
        categories: Vocabulary categories
        storage: Optional storage backend

    Returns:
        Configured Federation

    Example:
        # Simple setup
        federation = quick_setup(
            org_id="startup",
            departments=["engineering", "sales"],
            topics=["bug", "feature", "question"],
        )

        # With teams
        federation = quick_setup(
            org_id="enterprise",
            departments={
                "engineering": ["backend", "frontend"],
                "sales": ["enterprise", "smb"],
            },
        )
    """
    # Build structure config
    structure = StructureConfig()

    if departments:
        if isinstance(departments, list):
            # Simple list of departments
            for dept in departments:
                structure.departments[dept] = DepartmentConfig(name=dept)
        else:
            # Dict mapping departments to teams
            for dept, teams in departments.items():
                structure.departments[dept] = DepartmentConfig(
                    name=dept,
                    teams=teams,
                )

    config = FederationConfig(
        org_id=org_id,
        structure=structure,
        vocabulary=VocabularyConfig(
            topics=topics or [],
            categories=categories or [],
        ),
    )

    return Federation.from_config(config, storage)
