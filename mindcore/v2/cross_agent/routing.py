"""Cross-Agent Attention Routing - Query routing across agents.

Routes queries to the most relevant agent memories based on
capabilities, specializations, and historical performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..flr import Memory, RecallResult
    from ..storage.base import BaseStorage
    from .registry import Agent, AgentRegistry


class RoutingStrategy(str, Enum):
    """Strategy for routing queries across agents."""

    BROADCAST = "broadcast"  # Query all agents
    CAPABILITY_MATCH = "capability_match"  # Route to agents with matching capabilities
    SPECIALIZATION_MATCH = "specialization_match"  # Route to specialized agents
    TEAM_FIRST = "team_first"  # Prioritize team members
    BEST_MATCH = "best_match"  # Use scoring to find best agent
    ROUND_ROBIN = "round_robin"  # Distribute evenly


@dataclass
class AgentScore:
    """Score for an agent's relevance to a query."""

    agent_id: str
    score: float
    reasons: list[str] = field(default_factory=list)

    # Breakdown
    capability_score: float = 0.0
    specialization_score: float = 0.0
    team_score: float = 0.0
    history_score: float = 0.0


@dataclass
class RouteResult:
    """Result of routing a query across agents."""

    query: str
    requesting_agent: str | None
    strategy: RoutingStrategy
    selected_agents: list[str]
    agent_scores: list[AgentScore]
    memories: list[Memory]
    total_memories: int
    routing_latency_ms: float
    routed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "requesting_agent": self.requesting_agent,
            "strategy": self.strategy.value,
            "selected_agents": self.selected_agents,
            "agent_scores": [
                {
                    "agent_id": s.agent_id,
                    "score": s.score,
                    "reasons": s.reasons,
                }
                for s in self.agent_scores
            ],
            "total_memories": self.total_memories,
            "routing_latency_ms": self.routing_latency_ms,
            "routed_at": self.routed_at.isoformat(),
        }


class AttentionRouter:
    """Cross-agent attention routing.

    Routes queries to the most relevant agents based on various strategies,
    then aggregates and ranks memories from those agents.

    Example:
        router = AttentionRouter(storage, registry)

        # Route query with capability matching
        result = router.route(
            query="billing issue",
            user_id="user123",
            requesting_agent="sales_bot",
            strategy=RoutingStrategy.CAPABILITY_MATCH,
            attention_hints=["billing"],
        )

        # Get agent rankings for a query
        rankings = router.rank_agents(
            query="technical support",
            attention_hints=["api", "integration"],
        )
    """

    def __init__(
        self,
        storage: BaseStorage,
        registry: AgentRegistry,
    ) -> None:
        """Initialize attention router.

        Args:
            storage: Storage backend
            registry: Agent registry
        """
        self.storage = storage
        self.registry = registry

        # Round-robin state
        self._round_robin_index = 0

        # Query history for learning
        self._query_history: list[RouteResult] = []

    def route(
        self,
        query: str,
        user_id: str,
        requesting_agent: str | None = None,
        strategy: RoutingStrategy = RoutingStrategy.BEST_MATCH,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        max_agents: int = 5,
        max_memories_per_agent: int = 10,
    ) -> RouteResult:
        """Route a query across agents.

        Args:
            query: Search query
            user_id: User context
            requesting_agent: Agent making the request
            strategy: Routing strategy
            attention_hints: Topics/capabilities to prioritize
            memory_types: Filter by memory types
            max_agents: Maximum agents to query
            max_memories_per_agent: Maximum memories per agent

        Returns:
            RouteResult with memories from selected agents
        """
        start_time = time.time()
        attention_hints = attention_hints or []
        memory_types = memory_types or []

        # Get all active agents
        all_agents = self.registry.list_agents()
        active_agents = [a for a in all_agents if a.is_active()]

        # Score and select agents based on strategy
        agent_scores = self._score_agents(
            agents=active_agents,
            query=query,
            attention_hints=attention_hints,
            requesting_agent=requesting_agent,
            strategy=strategy,
        )

        # Select top agents
        selected = self._select_agents(
            agent_scores=agent_scores,
            strategy=strategy,
            max_agents=max_agents,
            requesting_agent=requesting_agent,
        )

        # Query memories from selected agents
        all_memories = []
        for agent_id in selected:
            # Check access
            if requesting_agent:
                requester = self.registry.get_agent(requesting_agent)
                target = self.registry.get_agent(agent_id)
                if not requester or not target:
                    continue

            # Get accessible memories from this agent
            memories = self._get_agent_memories(
                agent_id=agent_id,
                user_id=user_id,
                query=query,
                requesting_agent=requesting_agent,
                attention_hints=attention_hints,
                memory_types=memory_types,
                limit=max_memories_per_agent,
            )
            all_memories.extend(memories)

        # Deduplicate and sort by relevance
        unique_memories = self._deduplicate_memories(all_memories)

        latency = (time.time() - start_time) * 1000

        result = RouteResult(
            query=query,
            requesting_agent=requesting_agent,
            strategy=strategy,
            selected_agents=selected,
            agent_scores=agent_scores,
            memories=unique_memories,
            total_memories=len(unique_memories),
            routing_latency_ms=latency,
        )

        self._query_history.append(result)
        return result

    def rank_agents(
        self,
        query: str,
        attention_hints: list[str] | None = None,
        requesting_agent: str | None = None,
    ) -> list[AgentScore]:
        """Rank agents by relevance to a query.

        Args:
            query: Search query
            attention_hints: Topics/capabilities to prioritize
            requesting_agent: Agent making the request

        Returns:
            List of AgentScore sorted by score descending
        """
        attention_hints = attention_hints or []

        agents = self.registry.list_agents()
        active_agents = [a for a in agents if a.is_active()]

        scores = self._score_agents(
            agents=active_agents,
            query=query,
            attention_hints=attention_hints,
            requesting_agent=requesting_agent,
            strategy=RoutingStrategy.BEST_MATCH,
        )

        return sorted(scores, key=lambda s: s.score, reverse=True)

    def suggest_agents(
        self,
        attention_hints: list[str],
        limit: int = 3,
    ) -> list[Agent]:
        """Suggest agents based on attention hints.

        Args:
            attention_hints: Topics/capabilities needed
            limit: Maximum agents to suggest

        Returns:
            List of suggested agents
        """
        agents = self.registry.list_agents()
        active_agents = [a for a in agents if a.is_active()]

        scores = []
        for agent in active_agents:
            score = 0.0

            # Check capabilities
            for hint in attention_hints:
                if agent.has_capability(hint):
                    score += 1.0
                if hint in agent.specializations:
                    score += 0.5

            scores.append((agent, score))

        # Sort by score and return top
        scores.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in scores[:limit] if scores]

    def _score_agents(
        self,
        agents: list[Agent],
        query: str,
        attention_hints: list[str],
        requesting_agent: str | None,
        strategy: RoutingStrategy,
    ) -> list[AgentScore]:
        """Score agents for a query."""
        scores = []
        query_words = set(query.lower().split())

        for agent in agents:
            score = AgentScore(agent_id=agent.agent_id, score=0.0)

            # Capability matching
            capability_score = 0.0
            for hint in attention_hints:
                if agent.has_capability(hint):
                    capability_score += 1.0
                    score.reasons.append(f"Has capability: {hint}")
            score.capability_score = capability_score

            # Specialization matching
            spec_score = 0.0
            for hint in attention_hints:
                if hint in agent.specializations:
                    spec_score += 0.8
                    score.reasons.append(f"Specializes in: {hint}")
            # Also check query words against specializations
            for spec in agent.specializations:
                if spec.lower() in query_words:
                    spec_score += 0.5
            score.specialization_score = spec_score

            # Team score (bonus if same team as requester)
            team_score = 0.0
            if requesting_agent:
                requester = self.registry.get_agent(requesting_agent)
                if requester and agent.shares_team_with(requester):
                    team_score = 0.5
                    score.reasons.append("Same team as requester")
            score.team_score = team_score

            # History score (based on past successful queries)
            history_score = self._calculate_history_score(agent.agent_id, attention_hints)
            score.history_score = history_score

            # Calculate final score based on strategy
            if strategy == RoutingStrategy.CAPABILITY_MATCH:
                score.score = capability_score * 2 + spec_score + team_score
            elif strategy == RoutingStrategy.SPECIALIZATION_MATCH:
                score.score = spec_score * 2 + capability_score + team_score
            elif strategy == RoutingStrategy.TEAM_FIRST:
                score.score = team_score * 3 + capability_score + spec_score
            else:  # BEST_MATCH
                score.score = (
                    capability_score * 1.0 +
                    spec_score * 0.8 +
                    team_score * 0.5 +
                    history_score * 0.3
                )

            scores.append(score)

        return scores

    def _select_agents(
        self,
        agent_scores: list[AgentScore],
        strategy: RoutingStrategy,
        max_agents: int,
        requesting_agent: str | None,
    ) -> list[str]:
        """Select agents based on strategy and scores."""
        if strategy == RoutingStrategy.BROADCAST:
            return [s.agent_id for s in agent_scores[:max_agents]]

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate through agents
            all_ids = [s.agent_id for s in agent_scores]
            selected = []
            for i in range(min(max_agents, len(all_ids))):
                idx = (self._round_robin_index + i) % len(all_ids)
                selected.append(all_ids[idx])
            self._round_robin_index = (self._round_robin_index + max_agents) % max(1, len(all_ids))
            return selected

        else:
            # Sort by score and select top
            sorted_scores = sorted(agent_scores, key=lambda s: s.score, reverse=True)

            # Filter out zero scores
            non_zero = [s for s in sorted_scores if s.score > 0]
            if not non_zero:
                # Fallback to top agents by any criteria
                non_zero = sorted_scores

            return [s.agent_id for s in non_zero[:max_agents]]

    def _get_agent_memories(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        requesting_agent: str | None,
        attention_hints: list[str],
        memory_types: list[str],
        limit: int,
    ) -> list[Memory]:
        """Get accessible memories from an agent."""
        # Determine which access levels the requester can see
        access_levels = ["shared", "global"]

        if requesting_agent:
            requester = self.registry.get_agent(requesting_agent)
            target = self.registry.get_agent(agent_id)

            if requester and target:
                if requester.agent_id == target.agent_id:
                    # Own memories - all access levels
                    access_levels = ["private", "team", "shared", "global"]
                elif requester.shares_team_with(target):
                    # Teammate - team and above
                    access_levels = ["team", "shared", "global"]

        memories = self.storage.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            topics=attention_hints if attention_hints else None,
            memory_types=memory_types if memory_types else None,
            access_levels=access_levels,
            limit=limit,
        )

        return memories

    def _deduplicate_memories(self, memories: list[Memory]) -> list[Memory]:
        """Deduplicate memories by ID."""
        seen = set()
        unique = []
        for mem in memories:
            if mem.memory_id not in seen:
                seen.add(mem.memory_id)
                unique.append(mem)
        return unique

    def _calculate_history_score(
        self,
        agent_id: str,
        attention_hints: list[str],
    ) -> float:
        """Calculate score based on historical performance."""
        if not self._query_history:
            return 0.0

        # Look at recent queries with similar attention hints
        relevant_queries = []
        for result in self._query_history[-100:]:  # Last 100 queries
            hint_overlap = len(
                set(attention_hints) & set(result.agent_scores[0].reasons if result.agent_scores else [])
            )
            if hint_overlap > 0:
                relevant_queries.append(result)

        if not relevant_queries:
            return 0.0

        # Calculate success rate for this agent
        successes = sum(
            1 for r in relevant_queries
            if agent_id in r.selected_agents and r.total_memories > 0
        )

        return successes / len(relevant_queries) if relevant_queries else 0.0

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._query_history:
            return {
                "total_queries": 0,
                "avg_agents_per_query": 0,
                "avg_memories_per_query": 0,
                "avg_latency_ms": 0,
                "strategy_usage": {},
            }

        total = len(self._query_history)
        avg_agents = sum(len(r.selected_agents) for r in self._query_history) / total
        avg_memories = sum(r.total_memories for r in self._query_history) / total
        avg_latency = sum(r.routing_latency_ms for r in self._query_history) / total

        strategy_usage = {}
        for r in self._query_history:
            strategy = r.strategy.value
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        return {
            "total_queries": total,
            "avg_agents_per_query": round(avg_agents, 2),
            "avg_memories_per_query": round(avg_memories, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "strategy_usage": strategy_usage,
        }
