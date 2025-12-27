"""Memory Namespace for Organizational Hierarchy.

Provides hierarchical namespacing for memories across an organization.

Namespace Structure:
    org_id / department / team / agent_id

Example:
    # Support team namespace
    ns = MemoryNamespace(
        org_id="acme-corp",
        department="customer-success",
        team="support-tier-1",
    )

    # Query with namespace filtering
    memories = clst.search(
        query="refund policy",
        namespace=ns,
        include_ancestors=True,  # Also search department-level
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryNamespace:
    """Hierarchical namespace for memory organization.

    Namespaces provide logical isolation and inheritance:
    - Memories stored at team level are visible to team members
    - Department-level memories are visible to all teams in department
    - Organization-level memories are visible to everyone

    Attributes:
        org_id: Organization identifier (required, root of hierarchy)
        department: Department within org
        team: Team within department
        labels: Additional classification labels
        metadata: Custom namespace metadata
    """

    org_id: str
    department: str | None = None
    team: str | None = None
    labels: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate namespace hierarchy."""
        if self.team and not self.department:
            raise ValueError("Team requires department to be set")

    @property
    def path(self) -> str:
        """Get namespace path string.

        Returns:
            Path like "acme-corp/customer-success/support-tier-1"
        """
        parts = [self.org_id]
        if self.department:
            parts.append(self.department)
        if self.team:
            parts.append(self.team)
        return "/".join(parts)

    @property
    def depth(self) -> int:
        """Get namespace depth (1=org, 2=department, 3=team)."""
        if self.team:
            return 3
        if self.department:
            return 2
        return 1

    def is_ancestor_of(self, other: MemoryNamespace) -> bool:
        """Check if this namespace is an ancestor of another.

        Args:
            other: Namespace to check

        Returns:
            True if this is an ancestor (or equal to) other
        """
        if self.org_id != other.org_id:
            return False

        # Org level is ancestor of everything in same org
        if self.department is None:
            return True

        # Department must match
        if self.department != other.department:
            return False

        # Department level is ancestor of teams
        if self.team is None:
            return True

        # Team level: must match exactly
        return self.team == other.team

    def is_descendant_of(self, other: MemoryNamespace) -> bool:
        """Check if this namespace is a descendant of another."""
        return other.is_ancestor_of(self)

    def get_ancestors(self) -> list[MemoryNamespace]:
        """Get all ancestor namespaces (from most specific to org).

        Returns:
            List of ancestor namespaces
        """
        ancestors = []

        # Team -> Department
        if self.team:
            ancestors.append(
                MemoryNamespace(
                    org_id=self.org_id,
                    department=self.department,
                )
            )

        # Department -> Org
        if self.department:
            ancestors.append(
                MemoryNamespace(
                    org_id=self.org_id,
                )
            )

        return ancestors

    def get_lineage(self) -> list[MemoryNamespace]:
        """Get full lineage from org to this namespace.

        Returns:
            List from org-level down to this namespace
        """
        lineage = [MemoryNamespace(org_id=self.org_id)]

        if self.department:
            lineage.append(
                MemoryNamespace(
                    org_id=self.org_id,
                    department=self.department,
                )
            )

        if self.team:
            lineage.append(self)

        return lineage

    def to_filter(self) -> dict[str, Any]:
        """Convert to filter dict for storage queries."""
        filter_dict: dict[str, Any] = {"org_id": self.org_id}
        if self.department:
            filter_dict["department"] = self.department
        if self.team:
            filter_dict["team"] = self.team
        if self.labels:
            filter_dict["labels"] = list(self.labels)
        return filter_dict

    def to_dict(self) -> dict[str, Any]:
        """Serialize namespace."""
        return {
            "org_id": self.org_id,
            "department": self.department,
            "team": self.team,
            "labels": list(self.labels),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryNamespace:
        """Deserialize namespace."""
        return cls(
            org_id=data["org_id"],
            department=data.get("department"),
            team=data.get("team"),
            labels=set(data.get("labels", [])),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_path(cls, path: str) -> MemoryNamespace:
        """Create namespace from path string.

        Args:
            path: Path like "acme-corp/customer-success/support-tier-1"

        Returns:
            MemoryNamespace instance
        """
        parts = path.split("/")
        if not parts:
            raise ValueError("Empty namespace path")

        return cls(
            org_id=parts[0],
            department=parts[1] if len(parts) > 1 else None,
            team=parts[2] if len(parts) > 2 else None,
        )


@dataclass
class NamespaceHierarchy:
    """Manages namespace hierarchy for an organization.

    Provides utilities for:
    - Creating and validating namespaces
    - Finding common ancestors
    - Computing visibility across namespaces

    Example:
        hierarchy = NamespaceHierarchy(org_id="acme-corp")

        # Register structure
        hierarchy.register_department("customer-success")
        hierarchy.register_team("customer-success", "support-tier-1")
        hierarchy.register_team("customer-success", "support-tier-2")

        # Find common ancestor
        ns1 = hierarchy.get_namespace("customer-success", "support-tier-1")
        ns2 = hierarchy.get_namespace("customer-success", "support-tier-2")
        common = hierarchy.common_ancestor(ns1, ns2)
        # -> MemoryNamespace(org_id="acme-corp", department="customer-success")
    """

    org_id: str
    departments: dict[str, set[str]] = field(default_factory=dict)  # dept -> teams
    labels: dict[str, set[str]] = field(default_factory=dict)  # label -> namespaces

    def register_department(self, department: str) -> MemoryNamespace:
        """Register a new department."""
        if department not in self.departments:
            self.departments[department] = set()
        return MemoryNamespace(org_id=self.org_id, department=department)

    def register_team(self, department: str, team: str) -> MemoryNamespace:
        """Register a new team under a department."""
        if department not in self.departments:
            self.departments[department] = set()
        self.departments[department].add(team)
        return MemoryNamespace(
            org_id=self.org_id,
            department=department,
            team=team,
        )

    def get_namespace(
        self,
        department: str | None = None,
        team: str | None = None,
        labels: set[str] | None = None,
    ) -> MemoryNamespace:
        """Get or create a namespace."""
        return MemoryNamespace(
            org_id=self.org_id,
            department=department,
            team=team,
            labels=labels or set(),
        )

    def get_org_namespace(self) -> MemoryNamespace:
        """Get organization-level namespace."""
        return MemoryNamespace(org_id=self.org_id)

    def common_ancestor(
        self,
        ns1: MemoryNamespace,
        ns2: MemoryNamespace,
    ) -> MemoryNamespace | None:
        """Find the common ancestor of two namespaces.

        Args:
            ns1: First namespace
            ns2: Second namespace

        Returns:
            Common ancestor namespace, or None if different orgs
        """
        if ns1.org_id != ns2.org_id:
            return None

        # Same org is always common ancestor
        if ns1.department != ns2.department:
            return MemoryNamespace(org_id=ns1.org_id)

        # Same department
        if ns1.team != ns2.team:
            return MemoryNamespace(
                org_id=ns1.org_id,
                department=ns1.department,
            )

        # Identical namespaces
        return MemoryNamespace(
            org_id=ns1.org_id,
            department=ns1.department,
            team=ns1.team,
        )

    def get_all_descendants(self, namespace: MemoryNamespace) -> list[MemoryNamespace]:
        """Get all descendant namespaces.

        Args:
            namespace: Parent namespace

        Returns:
            List of all descendant namespaces
        """
        if namespace.org_id != self.org_id:
            return []

        descendants = []

        # Org level: all departments and teams
        if namespace.department is None:
            for dept, teams in self.departments.items():
                descendants.append(
                    MemoryNamespace(
                        org_id=self.org_id,
                        department=dept,
                    )
                )
                for team in teams:
                    descendants.append(
                        MemoryNamespace(
                            org_id=self.org_id,
                            department=dept,
                            team=team,
                        )
                    )

        # Department level: all teams in department
        elif namespace.team is None:
            teams = self.departments.get(namespace.department, set())
            for team in teams:
                descendants.append(
                    MemoryNamespace(
                        org_id=self.org_id,
                        department=namespace.department,
                        team=team,
                    )
                )

        # Team level: no descendants
        return descendants

    def get_visible_namespaces(
        self,
        from_namespace: MemoryNamespace,
    ) -> list[MemoryNamespace]:
        """Get all namespaces visible from a given namespace.

        Visibility includes:
        - The namespace itself
        - All ancestor namespaces (org, department)
        - Sibling teams in same department (for collaboration)

        Args:
            from_namespace: The viewing namespace

        Returns:
            List of visible namespaces
        """
        visible = [from_namespace]
        visible.extend(from_namespace.get_ancestors())

        # Add sibling teams for collaboration visibility
        if from_namespace.department:
            teams = self.departments.get(from_namespace.department, set())
            for team in teams:
                if team != from_namespace.team:
                    visible.append(
                        MemoryNamespace(
                            org_id=self.org_id,
                            department=from_namespace.department,
                            team=team,
                        )
                    )

        return visible

    def to_dict(self) -> dict[str, Any]:
        """Serialize hierarchy."""
        return {
            "org_id": self.org_id,
            "departments": {dept: list(teams) for dept, teams in self.departments.items()},
            "labels": {label: list(namespaces) for label, namespaces in self.labels.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NamespaceHierarchy:
        """Deserialize hierarchy."""
        return cls(
            org_id=data["org_id"],
            departments={dept: set(teams) for dept, teams in data.get("departments", {}).items()},
            labels={label: set(ns) for label, ns in data.get("labels", {}).items()},
        )
