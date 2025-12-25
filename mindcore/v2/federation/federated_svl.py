"""Federated Structured Vocabulary Layer (SVL).

Provides unified vocabulary across the organization with scoped
feedback aggregation for metadata quality improvement.

Features:
- Single vocabulary shared across all agents
- Scoped feedback: team-level, department-level, org-level
- Feedback aggregation with hierarchy-aware weighting
- Per-agent feedback tracking for personalization

Architecture:
    ┌─────────────────────────────────────────────┐
    │           Organization Vocabulary           │
    │  (topics, categories, entities, schemas)    │
    └─────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Dept A  │    │ Dept B  │    │ Dept C  │
    │Feedback │    │Feedback │    │Feedback │
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
    Agent-level feedback rolls up to team → dept → org

Example:
    # Create federated SVL
    svl = FederatedSVL(org_id="acme-corp")

    # Get vocabulary (same for all agents)
    vocab = svl.get_vocabulary()

    # Record feedback from specific agent/team
    svl.record_feedback(
        topic="billing",
        was_effective=True,
        scope=agent_scope,
        namespace=team_namespace,
    )

    # Get aggregated feedback for an agent
    # Combines: agent's own + team's + department's + org's
    feedback = svl.get_aggregated_feedback(agent_scope, team_namespace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .access_control import AccessScope
from .namespace import MemoryNamespace


@dataclass
class TopicFeedback:
    """Feedback statistics for a single topic.

    Tracks effectiveness at different scopes.
    """

    topic: str
    total_uses: int = 0
    effective_uses: int = 0
    last_used: datetime | None = None

    @property
    def effectiveness_rate(self) -> float:
        """Calculate effectiveness rate."""
        if self.total_uses == 0:
            return 0.5  # Neutral for unseen topics
        return self.effective_uses / self.total_uses

    def record(self, was_effective: bool) -> None:
        """Record a usage."""
        self.total_uses += 1
        if was_effective:
            self.effective_uses += 1
        self.last_used = datetime.utcnow()


@dataclass
class ScopedFeedback:
    """Feedback scoped to a specific namespace level.

    Attributes:
        namespace: The namespace this feedback applies to
        topic_feedback: Per-topic effectiveness tracking
        category_feedback: Per-category effectiveness tracking
        total_extractions: Total metadata extractions at this scope
    """

    namespace: MemoryNamespace
    topic_feedback: dict[str, TopicFeedback] = field(default_factory=dict)
    category_feedback: dict[str, TopicFeedback] = field(default_factory=dict)
    total_extractions: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def record_topic(self, topic: str, was_effective: bool) -> None:
        """Record topic feedback."""
        if topic not in self.topic_feedback:
            self.topic_feedback[topic] = TopicFeedback(topic=topic)
        self.topic_feedback[topic].record(was_effective)
        self.updated_at = datetime.utcnow()

    def record_category(self, category: str, was_effective: bool) -> None:
        """Record category feedback."""
        if category not in self.category_feedback:
            self.category_feedback[category] = TopicFeedback(topic=category)
        self.category_feedback[category].record(was_effective)
        self.updated_at = datetime.utcnow()

    def get_effective_topics(self, min_uses: int = 3) -> list[tuple[str, float]]:
        """Get topics with high effectiveness."""
        effective = []
        for topic, fb in self.topic_feedback.items():
            if fb.total_uses >= min_uses and fb.effectiveness_rate > 0.6:
                effective.append((topic, fb.effectiveness_rate))
        return sorted(effective, key=lambda x: -x[1])

    def get_ineffective_topics(self, min_uses: int = 3) -> list[tuple[str, float]]:
        """Get topics with low effectiveness."""
        ineffective = []
        for topic, fb in self.topic_feedback.items():
            if fb.total_uses >= min_uses and fb.effectiveness_rate < 0.4:
                ineffective.append((topic, fb.effectiveness_rate))
        return sorted(ineffective, key=lambda x: x[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize feedback."""
        return {
            "namespace": self.namespace.to_dict(),
            "topic_feedback": {
                t: {"total": f.total_uses, "effective": f.effective_uses}
                for t, f in self.topic_feedback.items()
            },
            "category_feedback": {
                c: {"total": f.total_uses, "effective": f.effective_uses}
                for c, f in self.category_feedback.items()
            },
            "total_extractions": self.total_extractions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class AggregatedFeedback:
    """Aggregated feedback from multiple scopes.

    Combines feedback from agent, team, department, and org levels
    with appropriate weighting.
    """

    # Weighted topic effectiveness
    topic_effectiveness: dict[str, float] = field(default_factory=dict)
    topic_confidence: dict[str, float] = field(default_factory=dict)

    # Weighted category effectiveness
    category_effectiveness: dict[str, float] = field(default_factory=dict)
    category_confidence: dict[str, float] = field(default_factory=dict)

    # Recommendations
    preferred_topics: list[str] = field(default_factory=list)
    avoid_topics: list[str] = field(default_factory=list)
    preferred_categories: list[str] = field(default_factory=list)
    avoid_categories: list[str] = field(default_factory=list)

    def to_injection_format(self) -> dict[str, Any]:
        """Convert to format for ContextInjector."""
        return {
            "high_quality_topics": [
                (t, self.topic_effectiveness[t])
                for t in self.preferred_topics
                if t in self.topic_effectiveness
            ],
            "low_quality_topics": [
                (t, self.topic_effectiveness.get(t, 0.3))
                for t in self.avoid_topics
            ],
            "high_quality_categories": [
                (c, self.category_effectiveness[c])
                for c in self.preferred_categories
                if c in self.category_effectiveness
            ],
            "low_quality_categories": [
                (c, self.category_effectiveness.get(c, 0.3))
                for c in self.avoid_categories
            ],
        }


@dataclass
class FederatedSVL:
    """Federated Structured Vocabulary Layer.

    Provides organization-wide vocabulary with scoped feedback.

    Attributes:
        org_id: Organization identifier
        vocabulary: Base vocabulary (topics, categories, schemas)
        feedback_store: Scoped feedback by namespace path
        agent_feedback: Per-agent feedback tracking
    """

    org_id: str
    vocabulary: dict[str, Any] = field(default_factory=dict)
    feedback_store: dict[str, ScopedFeedback] = field(default_factory=dict)
    agent_feedback: dict[str, ScopedFeedback] = field(default_factory=dict)

    # Aggregation weights (can be tuned)
    agent_weight: float = 0.4  # Personal experience
    team_weight: float = 0.3  # Team patterns
    department_weight: float = 0.2  # Department patterns
    org_weight: float = 0.1  # Org-wide defaults

    def get_vocabulary(self) -> dict[str, Any]:
        """Get the organization vocabulary.

        Returns same vocabulary for all agents.
        """
        return self.vocabulary

    def set_vocabulary(
        self,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        entities: list[str] | None = None,
        schemas: dict[str, Any] | None = None,
    ) -> None:
        """Set vocabulary components."""
        if topics is not None:
            self.vocabulary["topics"] = topics
        if categories is not None:
            self.vocabulary["categories"] = categories
        if entities is not None:
            self.vocabulary["entities"] = entities
        if schemas is not None:
            self.vocabulary["schemas"] = schemas

    def record_feedback(
        self,
        topic: str | None = None,
        category: str | None = None,
        was_effective: bool = True,
        scope: AccessScope | None = None,
        namespace: MemoryNamespace | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Record feedback for a topic or category.

        Feedback is stored at the appropriate namespace level
        and rolls up to parent scopes.

        Args:
            topic: Topic to record feedback for
            category: Category to record feedback for
            was_effective: Whether the metadata led to useful retrieval
            scope: Access scope of the agent
            namespace: Namespace context
            agent_id: Specific agent ID for personalization
        """
        if namespace is None:
            namespace = MemoryNamespace(org_id=self.org_id)

        # Record at namespace level
        ns_path = namespace.path
        if ns_path not in self.feedback_store:
            self.feedback_store[ns_path] = ScopedFeedback(namespace=namespace)

        feedback = self.feedback_store[ns_path]
        feedback.total_extractions += 1

        if topic:
            feedback.record_topic(topic, was_effective)
        if category:
            feedback.record_category(category, was_effective)

        # Record at agent level for personalization
        if agent_id:
            if agent_id not in self.agent_feedback:
                self.agent_feedback[agent_id] = ScopedFeedback(namespace=namespace)

            agent_fb = self.agent_feedback[agent_id]
            if topic:
                agent_fb.record_topic(topic, was_effective)
            if category:
                agent_fb.record_category(category, was_effective)

        # Roll up to ancestors
        for ancestor in namespace.get_ancestors():
            anc_path = ancestor.path
            if anc_path not in self.feedback_store:
                self.feedback_store[anc_path] = ScopedFeedback(namespace=ancestor)

            anc_feedback = self.feedback_store[anc_path]
            if topic:
                anc_feedback.record_topic(topic, was_effective)
            if category:
                anc_feedback.record_category(category, was_effective)

    def get_scoped_feedback(
        self,
        namespace: MemoryNamespace,
    ) -> ScopedFeedback | None:
        """Get feedback for a specific namespace."""
        return self.feedback_store.get(namespace.path)

    def get_agent_feedback(
        self,
        agent_id: str,
    ) -> ScopedFeedback | None:
        """Get feedback for a specific agent."""
        return self.agent_feedback.get(agent_id)

    def get_aggregated_feedback(
        self,
        agent_id: str | None = None,
        namespace: MemoryNamespace | None = None,
        min_uses: int = 3,
    ) -> AggregatedFeedback:
        """Get aggregated feedback combining multiple scopes.

        Combines:
        1. Agent's personal feedback (highest weight)
        2. Team-level feedback
        3. Department-level feedback
        4. Organization-level feedback

        Args:
            agent_id: Agent ID for personalization
            namespace: Current namespace context
            min_uses: Minimum uses to consider a topic

        Returns:
            Aggregated feedback with weighted effectiveness
        """
        result = AggregatedFeedback()

        # Collect feedback from each level
        feedbacks: list[tuple[ScopedFeedback, float]] = []

        # Agent level
        if agent_id and agent_id in self.agent_feedback:
            feedbacks.append((self.agent_feedback[agent_id], self.agent_weight))

        # Namespace hierarchy
        if namespace:
            lineage = namespace.get_lineage()
            weights = [self.team_weight, self.department_weight, self.org_weight]

            for i, ns in enumerate(reversed(lineage)):
                if ns.path in self.feedback_store:
                    weight = weights[i] if i < len(weights) else 0.1
                    feedbacks.append((self.feedback_store[ns.path], weight))

        # Aggregate topic effectiveness
        topic_scores: dict[str, list[tuple[float, float]]] = {}  # topic -> [(score, weight)]

        for feedback, weight in feedbacks:
            for topic, fb in feedback.topic_feedback.items():
                if fb.total_uses >= min_uses:
                    if topic not in topic_scores:
                        topic_scores[topic] = []
                    topic_scores[topic].append((fb.effectiveness_rate, weight))

        # Compute weighted averages
        for topic, scores in topic_scores.items():
            total_weight = sum(w for _, w in scores)
            if total_weight > 0:
                weighted_sum = sum(s * w for s, w in scores)
                result.topic_effectiveness[topic] = weighted_sum / total_weight
                result.topic_confidence[topic] = min(1.0, total_weight)

        # Same for categories
        category_scores: dict[str, list[tuple[float, float]]] = {}

        for feedback, weight in feedbacks:
            for category, fb in feedback.category_feedback.items():
                if fb.total_uses >= min_uses:
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append((fb.effectiveness_rate, weight))

        for category, scores in category_scores.items():
            total_weight = sum(w for _, w in scores)
            if total_weight > 0:
                weighted_sum = sum(s * w for s, w in scores)
                result.category_effectiveness[category] = weighted_sum / total_weight
                result.category_confidence[category] = min(1.0, total_weight)

        # Generate recommendations
        result.preferred_topics = [
            t for t, e in result.topic_effectiveness.items()
            if e > 0.6 and result.topic_confidence.get(t, 0) > 0.3
        ]
        result.avoid_topics = [
            t for t, e in result.topic_effectiveness.items()
            if e < 0.4 and result.topic_confidence.get(t, 0) > 0.3
        ]
        result.preferred_categories = [
            c for c, e in result.category_effectiveness.items()
            if e > 0.6 and result.category_confidence.get(c, 0) > 0.3
        ]
        result.avoid_categories = [
            c for c, e in result.category_effectiveness.items()
            if e < 0.4 and result.category_confidence.get(c, 0) > 0.3
        ]

        return result

    def get_feedback_for_extractor(
        self,
        agent_id: str | None = None,
        namespace: MemoryNamespace | None = None,
    ) -> dict[str, Any]:
        """Get feedback in format suitable for MetadataExtractor.

        Compatible with FLR.get_metadata_feedback_for_extractor().

        Args:
            agent_id: Agent ID for personalization
            namespace: Current namespace context

        Returns:
            Dict with high_quality_topics, low_quality_topics, etc.
        """
        aggregated = self.get_aggregated_feedback(agent_id, namespace)
        return aggregated.to_injection_format()

    def to_dict(self) -> dict[str, Any]:
        """Serialize federated SVL."""
        return {
            "org_id": self.org_id,
            "vocabulary": self.vocabulary,
            "feedback_store": {
                path: fb.to_dict() for path, fb in self.feedback_store.items()
            },
            "agent_feedback": {
                agent: fb.to_dict() for agent, fb in self.agent_feedback.items()
            },
        }
