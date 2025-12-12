"""Vocabulary Schema - Versioned schema definitions for memory metadata.

The vocabulary defines valid values for topics, categories, memory types,
and other metadata fields. It's used to:
1. Generate JSON Schema for LLM structured output
2. Validate memories before storage
3. Enable migrations between schema versions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import json


class MemoryType(str, Enum):
    """Core memory types."""

    EPISODIC = "episodic"        # Events, conversations, interactions
    SEMANTIC = "semantic"         # Facts, knowledge, learned information
    PROCEDURAL = "procedural"     # Workflows, how-to, processes
    PREFERENCE = "preference"     # User preferences, settings
    ENTITY = "entity"             # People, places, things
    RELATIONSHIP = "relationship" # Connections between entities
    TEMPORAL = "temporal"         # Time-bound info (auto-expires)
    WORKING = "working"           # Current session context (cleared)


class Sentiment(str, Enum):
    """Sentiment values."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class AccessLevel(str, Enum):
    """Memory access levels for multi-agent."""

    PRIVATE = "private"    # Only this agent
    TEAM = "team"          # Agents in same team/group
    SHARED = "shared"      # All agents for this user
    GLOBAL = "global"      # Cross-user (knowledge base)


@dataclass
class FieldSchema:
    """Schema for a custom field."""

    name: str
    field_type: str  # "string", "number", "boolean", "array", "enum"
    required: bool = False
    enum_values: list[str] | None = None
    default: Any = None
    description: str = ""


@dataclass
class Migration:
    """Migration rules between vocabulary versions."""

    from_version: str
    to_version: str

    # Field transformations
    renames: dict[str, str] = field(default_factory=dict)  # old -> new
    merges: dict[str, list[str]] = field(default_factory=dict)  # new -> [old1, old2]
    splits: dict[str, dict[str, str]] = field(default_factory=dict)  # old -> {condition: new}
    deletes: list[str] = field(default_factory=list)

    # New fields with defaults
    added_fields: dict[str, Any] = field(default_factory=dict)  # field -> default

    def apply_to_topics(self, topics: list[str]) -> list[str]:
        """Apply migration to a list of topics."""
        result = []
        for topic in topics:
            # Check deletes
            if topic in self.deletes:
                continue
            # Check renames
            if topic in self.renames:
                result.append(self.renames[topic])
            # Check if topic should be merged into another
            else:
                merged = False
                for new_topic, old_topics in self.merges.items():
                    if topic in old_topics:
                        if new_topic not in result:
                            result.append(new_topic)
                        merged = True
                        break
                if not merged:
                    result.append(topic)
        return list(set(result))

    def apply_to_categories(self, categories: list[str]) -> list[str]:
        """Apply migration to categories (same logic as topics)."""
        return self.apply_to_topics(categories)  # Reuse same logic


@dataclass
class VocabularySchema:
    """Versioned vocabulary schema for memory metadata.

    Example:
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "orders", "support"],
            categories=["inquiry", "complaint", "feedback"],
        )

        # Export for LLM structured output
        json_schema = vocab.to_json_schema()

        # Validate a memory
        is_valid, errors = vocab.validate(memory)
    """

    version: str

    # Core vocabulary lists
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)

    # Memory types (use defaults if not specified)
    memory_types: list[str] = field(default_factory=lambda: [t.value for t in MemoryType])
    sentiments: list[str] = field(default_factory=lambda: [s.value for s in Sentiment])
    access_levels: list[str] = field(default_factory=lambda: [a.value for a in AccessLevel])

    # Custom fields
    custom_fields: list[FieldSchema] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    # Migrations from previous versions
    migrations: dict[str, Migration] = field(default_factory=dict)  # "1.0.0" -> Migration

    def to_json_schema(self, include_response: bool = True) -> dict[str, Any]:
        """Export as JSON Schema for LLM structured output.

        Args:
            include_response: Include response field for main agent output

        Returns:
            JSON Schema dict compatible with OpenAI, Anthropic, Google, etc.
        """
        memory_schema = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store"
                },
                "memory_type": {
                    "type": "string",
                    "enum": self.memory_types,
                    "description": "Type of memory"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "enum": self.topics} if self.topics else {"type": "string"},
                    "description": "Relevant topics"
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": self.categories} if self.categories else {"type": "string"},
                    "description": "Categories"
                },
                "sentiment": {
                    "type": "string",
                    "enum": self.sentiments,
                    "description": "Sentiment of the content"
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Importance score 0-1"
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extracted entities"
                },
                "access_level": {
                    "type": "string",
                    "enum": self.access_levels,
                    "description": "Access level for multi-agent"
                }
            },
            "required": ["content", "memory_type"]
        }

        # Add custom fields
        for custom in self.custom_fields:
            field_def = {"description": custom.description}
            if custom.field_type == "enum" and custom.enum_values:
                field_def["type"] = "string"
                field_def["enum"] = custom.enum_values
            elif custom.field_type == "array":
                field_def["type"] = "array"
                field_def["items"] = {"type": "string"}
            else:
                field_def["type"] = custom.field_type

            if custom.default is not None:
                field_def["default"] = custom.default

            memory_schema["properties"][custom.name] = field_def
            if custom.required:
                memory_schema["required"].append(custom.name)

        # Full schema with response
        if include_response:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Response to the user"
                    },
                    "memories_to_store": {
                        "type": "array",
                        "items": memory_schema,
                        "description": "Memories to store from this interaction"
                    },
                    "memory_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "topics": {"type": "array", "items": {"type": "string"}},
                                "memory_types": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "description": "Queries to search memories"
                    }
                },
                "required": ["response"]
            }

        return memory_schema

    def to_pydantic(self) -> str:
        """Generate Pydantic model code.

        Returns:
            Python code string for Pydantic models
        """
        topics_literal = f"Literal[{', '.join(repr(t) for t in self.topics)}]" if self.topics else "str"
        categories_literal = f"Literal[{', '.join(repr(c) for c in self.categories)}]" if self.categories else "str"

        code = f'''"""Auto-generated Pydantic models from VocabularySchema v{self.version}."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Memory model with vocabulary-constrained fields."""

    content: str
    memory_type: Literal[{', '.join(repr(m) for m in self.memory_types)}]
    topics: list[{topics_literal}] = Field(default_factory=list)
    categories: list[{categories_literal}] = Field(default_factory=list)
    sentiment: Literal[{', '.join(repr(s) for s in self.sentiments)}] = "neutral"
    importance: float = Field(default=0.5, ge=0, le=1)
    entities: list[str] = Field(default_factory=list)
    access_level: Literal[{', '.join(repr(a) for a in self.access_levels)}] = "private"


class AgentResponse(BaseModel):
    """Response model for LLM structured output."""

    response: str
    memories_to_store: list[Memory] = Field(default_factory=list)
'''
        return code

    def to_typescript(self) -> str:
        """Generate TypeScript types.

        Returns:
            TypeScript type definitions
        """
        topics_union = " | ".join(f'"{t}"' for t in self.topics) if self.topics else "string"
        categories_union = " | ".join(f'"{c}"' for c in self.categories) if self.categories else "string"
        memory_types_union = " | ".join(f'"{m}"' for m in self.memory_types)
        sentiments_union = " | ".join(f'"{s}"' for s in self.sentiments)
        access_levels_union = " | ".join(f'"{a}"' for a in self.access_levels)

        return f'''// Auto-generated TypeScript types from VocabularySchema v{self.version}

export type Topic = {topics_union};
export type Category = {categories_union};
export type MemoryType = {memory_types_union};
export type Sentiment = {sentiments_union};
export type AccessLevel = {access_levels_union};

export interface Memory {{
  content: string;
  memory_type: MemoryType;
  topics?: Topic[];
  categories?: Category[];
  sentiment?: Sentiment;
  importance?: number;
  entities?: string[];
  access_level?: AccessLevel;
}}

export interface AgentResponse {{
  response: string;
  memories_to_store?: Memory[];
}}
'''

    def to_prompt_instructions(self) -> str:
        """Generate prompt instructions for vocabulary.

        Returns:
            Human-readable instructions for LLM prompts
        """
        instructions = f"""## Memory Vocabulary (v{self.version})

When storing memories, use ONLY these values:

**TOPICS**: {', '.join(self.topics) if self.topics else '(any)'}

**CATEGORIES**: {', '.join(self.categories) if self.categories else '(any)'}

**MEMORY TYPES**:
- episodic: Events, conversations, interactions
- semantic: Facts, knowledge, learned information
- procedural: Workflows, how-to, processes
- preference: User preferences, settings
- entity: People, places, things
- relationship: Connections between entities
- temporal: Time-bound info (auto-expires)
- working: Current session context (cleared on session end)

**SENTIMENT**: {', '.join(self.sentiments)}

**ACCESS LEVELS**:
- private: Only this agent can access
- team: Agents in same team can access
- shared: All agents for this user can access
- global: Cross-user knowledge base

Always include memory_type and at least one topic when storing memories.
"""
        return instructions

    def validate(self, memory: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a memory against this vocabulary.

        Args:
            memory: Memory dict to validate

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Required fields
        if "content" not in memory:
            errors.append("Missing required field: content")
        if "memory_type" not in memory:
            errors.append("Missing required field: memory_type")
        elif memory["memory_type"] not in self.memory_types:
            errors.append(f"Invalid memory_type: {memory['memory_type']}")

        # Validate topics
        if "topics" in memory and self.topics:
            invalid_topics = [t for t in memory["topics"] if t not in self.topics]
            if invalid_topics:
                errors.append(f"Invalid topics: {invalid_topics}")

        # Validate categories
        if "categories" in memory and self.categories:
            invalid_cats = [c for c in memory["categories"] if c not in self.categories]
            if invalid_cats:
                errors.append(f"Invalid categories: {invalid_cats}")

        # Validate sentiment
        if "sentiment" in memory and memory["sentiment"] not in self.sentiments:
            errors.append(f"Invalid sentiment: {memory['sentiment']}")

        # Validate importance
        if "importance" in memory:
            imp = memory["importance"]
            if not isinstance(imp, (int, float)) or imp < 0 or imp > 1:
                errors.append(f"Invalid importance: {imp} (must be 0-1)")

        # Validate access_level
        if "access_level" in memory and memory["access_level"] not in self.access_levels:
            errors.append(f"Invalid access_level: {memory['access_level']}")

        # Validate custom fields
        for custom in self.custom_fields:
            if custom.required and custom.name not in memory:
                errors.append(f"Missing required custom field: {custom.name}")
            elif custom.name in memory and custom.enum_values:
                if memory[custom.name] not in custom.enum_values:
                    errors.append(f"Invalid {custom.name}: {memory[custom.name]}")

        return len(errors) == 0, errors

    def migrate_memory(self, memory: dict[str, Any], from_version: str) -> dict[str, Any]:
        """Migrate a memory from an older vocabulary version.

        Args:
            memory: Memory dict to migrate
            from_version: Source vocabulary version

        Returns:
            Migrated memory dict
        """
        if from_version == self.version:
            return memory

        if from_version not in self.migrations:
            raise ValueError(f"No migration path from {from_version} to {self.version}")

        migration = self.migrations[from_version]
        result = memory.copy()

        # Apply topic migrations
        if "topics" in result:
            result["topics"] = migration.apply_to_topics(result["topics"])

        # Apply category migrations
        if "categories" in result:
            result["categories"] = migration.apply_to_categories(result["categories"])

        # Apply added fields with defaults
        for field_name, default_value in migration.added_fields.items():
            if field_name not in result:
                result[field_name] = default_value

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "topics": self.topics,
            "categories": self.categories,
            "intents": self.intents,
            "memory_types": self.memory_types,
            "sentiments": self.sentiments,
            "access_levels": self.access_levels,
            "custom_fields": [
                {
                    "name": f.name,
                    "field_type": f.field_type,
                    "required": f.required,
                    "enum_values": f.enum_values,
                    "default": f.default,
                    "description": f.description,
                }
                for f in self.custom_fields
            ],
            "created_at": self.created_at.isoformat(),
            "description": self.description,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularySchema:
        """Create from dictionary."""
        custom_fields = [
            FieldSchema(**f) for f in data.get("custom_fields", [])
        ]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            version=data["version"],
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            intents=data.get("intents", []),
            memory_types=data.get("memory_types", [t.value for t in MemoryType]),
            sentiments=data.get("sentiments", [s.value for s in Sentiment]),
            access_levels=data.get("access_levels", [a.value for a in AccessLevel]),
            custom_fields=custom_fields,
            created_at=created_at or datetime.now(timezone.utc),
            description=data.get("description", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> VocabularySchema:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml(cls, path: str) -> VocabularySchema:
        """Load from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            VocabularySchema instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML support: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)


# Default vocabulary for quick start
DEFAULT_VOCABULARY = VocabularySchema(
    version="1.0.0",
    topics=[
        "greeting", "farewell", "thanks", "help", "feedback",
        "issue", "bug", "error", "problem", "complaint",
        "billing", "payment", "refund", "subscription",
        "feature", "product", "service", "pricing",
        "api", "integration", "setup", "documentation",
        "account", "login", "password", "settings", "profile",
        "order", "shipping", "delivery", "tracking",
    ],
    categories=[
        "support", "billing", "technical", "account",
        "product", "feedback", "general", "urgent",
    ],
    intents=[
        "ask_question", "request_action", "provide_info",
        "express_opinion", "complaint", "greeting",
        "confirmation", "clarification",
    ],
    description="Default Mindcore vocabulary for general use cases",
)
