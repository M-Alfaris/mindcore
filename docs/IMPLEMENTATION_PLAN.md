# Mindcore Enhancement Implementation Plan

## Overview

This plan covers three major feature additions to Mindcore:

1. **Thread/Session Summarization** - Compress old conversations into summaries
2. **User Preferences** - Amendable user settings vs read-only system data
3. **External Connectors** - Read-only links to business systems (orders, billing, etc.)

---

## Phase 1: Thread Summarization

**Goal**: Compress old threads into summaries to reduce storage and speed up retrieval.

### 1.1 New Schema: ThreadSummary

**File**: `mindcore/core/schemas.py`

```python
@dataclass
class ThreadSummary:
    """Compressed summary of a thread/session."""
    summary_id: str
    user_id: str
    thread_id: str
    session_id: Optional[str] = None

    # Summary content
    summary: str                          # LLM-generated summary
    key_facts: List[str] = field(default_factory=list)  # Extractable facts
    topics: List[str] = field(default_factory=list)     # Aggregated topics
    categories: List[str] = field(default_factory=list) # Aggregated categories
    overall_sentiment: str = "neutral"    # Overall sentiment

    # Metadata
    message_count: int = 0                # Original message count
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    summarized_at: Optional[datetime] = None

    # Entities extracted (order IDs, dates, etc.)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    # Example: {"order_ids": ["#12345"], "dates": ["2024-03-15"]}

    # Status
    messages_deleted: bool = False  # Whether raw messages were purged

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 1.2 Database Schema Updates

**File**: `mindcore/core/sqlite_manager.py` and `mindcore/core/async_db.py`

Add new table:

```sql
CREATE TABLE IF NOT EXISTS thread_summaries (
    summary_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    session_id TEXT,
    summary TEXT NOT NULL,
    key_facts TEXT DEFAULT '[]',        -- JSON array
    topics TEXT DEFAULT '[]',           -- JSON array
    categories TEXT DEFAULT '[]',       -- JSON array
    overall_sentiment TEXT DEFAULT 'neutral',
    message_count INTEGER DEFAULT 0,
    first_message_at TIMESTAMP,
    last_message_at TIMESTAMP,
    summarized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    entities TEXT DEFAULT '{}',         -- JSON object
    messages_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_summary_user_thread
    ON thread_summaries(user_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_summary_topics
    ON thread_summaries(topics);  -- For search
```

### 1.3 Summarization Agent

**File**: `mindcore/agents/summarization_agent.py` (new)

```python
class SummarizationAgent(BaseAgent):
    """Compress messages into thread summaries."""

    def summarize_thread(
        self,
        messages: List[Message],
        thread_id: str,
        user_id: str
    ) -> ThreadSummary:
        """
        Generate a summary from a list of messages.

        Extracts:
        - Overall summary (1-3 paragraphs)
        - Key facts (bullet points)
        - Aggregated topics/categories
        - Important entities (order IDs, dates, names)
        - Overall sentiment
        """
        pass
```

**LLM Prompt for Summarization**:
```
Summarize this conversation thread. Extract:
1. A concise summary (2-3 sentences)
2. Key facts mentioned (as bullet points)
3. Main topics discussed
4. Important entities (order IDs, dates, product names, etc.)
5. Overall sentiment (positive/negative/neutral/mixed)

Conversation:
{messages}

Respond in JSON format:
{
    "summary": "...",
    "key_facts": ["...", "..."],
    "topics": ["...", "..."],
    "entities": {"order_ids": [...], "dates": [...], ...},
    "overall_sentiment": "..."
}
```

### 1.4 Database Methods

**Add to SQLiteManager and AsyncSQLiteManager**:

```python
def insert_summary(self, summary: ThreadSummary) -> bool:
    """Insert or update a thread summary."""
    pass

def get_summary(self, user_id: str, thread_id: str) -> Optional[ThreadSummary]:
    """Get summary for a thread."""
    pass

def get_user_summaries(
    self,
    user_id: str,
    topics: Optional[List[str]] = None,
    limit: int = 20
) -> List[ThreadSummary]:
    """Get summaries for a user, optionally filtered by topics."""
    pass

def delete_summarized_messages(
    self,
    thread_id: str,
    keep_last_n: int = 0
) -> int:
    """Delete raw messages for a summarized thread."""
    pass
```

### 1.5 Background Summarization Worker

**File**: `mindcore/workers/summarization_worker.py` (new)

```python
class SummarizationWorker:
    """Background worker to summarize old threads."""

    def __init__(
        self,
        db: SQLiteManager,
        summarization_agent: SummarizationAgent,
        max_age_days: int = 7,           # Summarize threads older than this
        min_messages: int = 5,            # Minimum messages to summarize
        delete_after_summary: bool = False,  # Whether to delete raw messages
        keep_last_n: int = 3              # Keep last N messages even if deleting
    ):
        pass

    async def run_once(self) -> int:
        """Run one summarization cycle. Returns count of threads summarized."""
        pass

    async def run_continuous(self, interval_seconds: int = 3600):
        """Run continuously with interval."""
        pass
```

### 1.6 Integration with SmartContextAgent

Update `SmartContextAgent` to include a new tool:

```python
{
    "type": "function",
    "function": {
        "name": "get_historical_summaries",
        "description": "Get summaries of past conversation threads. Use when user references older conversations.",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topics to filter summaries by"
                },
                "limit": {
                    "type": "integer",
                    "default": 5
                }
            }
        }
    }
}
```

### 1.7 Tasks for Phase 1

- [ ] Add `ThreadSummary` to schemas.py
- [ ] Add thread_summaries table to SQLite and PostgreSQL managers
- [ ] Create SummarizationAgent with LLM prompt
- [ ] Add summary CRUD methods to database managers
- [ ] Create SummarizationWorker for background processing
- [ ] Add `get_historical_summaries` tool to SmartContextAgent
- [ ] Update context assembly to include summaries
- [ ] Add configuration options for summarization rules
- [ ] Write tests for summarization flow

---

## Phase 2: User Preferences

**Goal**: Store amendable user preferences separately from read-only system data.

### 2.1 New Schema: UserPreferences

**File**: `mindcore/core/schemas.py`

```python
@dataclass
class UserPreferences:
    """
    Amendable user preferences.

    These can be updated by user request through the AI agent.
    """
    user_id: str

    # Communication preferences
    language: str = "en"
    timezone: str = "UTC"
    communication_style: str = "balanced"  # formal, casual, technical, balanced

    # Personalization
    interests: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    preferred_name: Optional[str] = None

    # Context hints (user-provided context that should always be included)
    custom_context: Dict[str, Any] = field(default_factory=dict)
    # Example: {"role": "developer", "company": "Acme Inc"}

    # Notification preferences
    notification_topics: List[str] = field(default_factory=list)  # Topics to alert on

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Define which fields are amendable
    AMENDABLE_FIELDS: ClassVar[Set[str]] = {
        "language", "timezone", "communication_style",
        "interests", "goals", "preferred_name",
        "custom_context", "notification_topics"
    }

    def update(self, field: str, value: Any) -> bool:
        """Update a preference field if amendable."""
        if field not in self.AMENDABLE_FIELDS:
            return False
        setattr(self, field, value)
        self.updated_at = datetime.now(timezone.utc)
        return True

    def to_context_string(self) -> str:
        """Format preferences for inclusion in AI context."""
        parts = []
        if self.preferred_name:
            parts.append(f"User prefers to be called: {self.preferred_name}")
        if self.communication_style != "balanced":
            parts.append(f"Communication style: {self.communication_style}")
        if self.custom_context:
            for k, v in self.custom_context.items():
                parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else ""
```

### 2.2 Database Schema

```sql
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    language TEXT DEFAULT 'en',
    timezone TEXT DEFAULT 'UTC',
    communication_style TEXT DEFAULT 'balanced',
    interests TEXT DEFAULT '[]',        -- JSON array
    goals TEXT DEFAULT '[]',            -- JSON array
    preferred_name TEXT,
    custom_context TEXT DEFAULT '{}',   -- JSON object
    notification_topics TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 Preferences Manager

**File**: `mindcore/core/preferences_manager.py` (new)

```python
class PreferencesManager:
    """Manage user preferences."""

    def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating default if not exists."""
        pass

    def update_preference(
        self,
        user_id: str,
        field: str,
        value: Any
    ) -> Tuple[bool, str]:
        """
        Update a user preference.

        Returns:
            (success, message) - message explains why if failed
        """
        pass

    def add_interest(self, user_id: str, interest: str) -> bool:
        pass

    def remove_interest(self, user_id: str, interest: str) -> bool:
        pass

    def set_custom_context(self, user_id: str, key: str, value: Any) -> bool:
        pass
```

### 2.4 Integration with Context Assembly

Update `MindcoreClient.get_context_smart()` to include preferences:

```python
async def get_context_smart(self, user_id: str, thread_id: str, query: str, ...):
    # Get user preferences
    preferences = self.preferences_manager.get_preferences(user_id)

    # Include preferences in additional_context
    pref_context = preferences.to_context_string()
    full_context = f"{pref_context}\n\n{additional_context}" if pref_context else additional_context

    # Continue with context assembly...
```

### 2.5 Preference Update Tool for Agent

Add tool to SmartContextAgent:

```python
{
    "type": "function",
    "function": {
        "name": "update_user_preference",
        "description": "Update a user's preference when they request it. Only works for amendable fields like language, timezone, interests, goals, communication_style.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["language", "timezone", "communication_style", "interests", "goals", "preferred_name"],
                    "description": "The preference field to update"
                },
                "value": {
                    "description": "The new value for the field"
                },
                "action": {
                    "type": "string",
                    "enum": ["set", "add", "remove"],
                    "description": "For list fields: add/remove item. For others: set value."
                }
            },
            "required": ["field", "value"]
        }
    }
}
```

### 2.6 Tasks for Phase 2

- [ ] Add `UserPreferences` to schemas.py
- [ ] Add user_preferences table to database managers
- [ ] Create PreferencesManager class
- [ ] Add preference update tool to SmartContextAgent
- [ ] Integrate preferences into context assembly
- [ ] Add API endpoints for preference management
- [ ] Write tests for preference updates

---

## Phase 3: External Connectors

**Goal**: Read-only access to external business systems (orders, billing, CRM, etc.)

### 3.1 Connector Base Class

**File**: `mindcore/connectors/__init__.py` and `mindcore/connectors/base.py` (new)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ConnectorResult:
    """Result from an external connector lookup."""
    data: Dict[str, Any]           # The fetched data
    source: str                     # Connector name
    fetched_at: datetime           # When data was fetched
    cache_ttl: int = 300           # How long to cache (seconds)
    error: Optional[str] = None    # Error message if failed


class BaseConnector(ABC):
    """
    Base class for external system connectors.

    Connectors provide READ-ONLY access to external systems.
    They map topics to external data sources.
    """

    # Topic(s) this connector handles
    topics: List[str] = []

    # Human-readable name
    name: str = "base_connector"

    # Cache TTL in seconds
    cache_ttl: int = 300

    @abstractmethod
    async def lookup(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> ConnectorResult:
        """
        Fetch data from external system.

        Args:
            user_id: The user making the request
            context: Extracted context (entities, dates, IDs mentioned)

        Returns:
            ConnectorResult with fetched data
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract relevant entities from text for lookup.

        Example for orders: extract order IDs, dates
        Example for billing: extract invoice numbers, amounts
        """
        pass

    def can_handle(self, topics: List[str]) -> bool:
        """Check if this connector handles any of the given topics."""
        return bool(set(self.topics) & set(topics))
```

### 3.2 Connector Registry

**File**: `mindcore/connectors/registry.py` (new)

```python
class ConnectorRegistry:
    """Registry of external data connectors."""

    def __init__(self):
        self._connectors: Dict[str, BaseConnector] = {}
        self._topic_map: Dict[str, BaseConnector] = {}
        self._cache: DiskCacheManager = None  # Optional caching

    def register(self, connector: BaseConnector) -> None:
        """Register a connector."""
        self._connectors[connector.name] = connector
        for topic in connector.topics:
            self._topic_map[topic] = connector

    def get_connector(self, topic: str) -> Optional[BaseConnector]:
        """Get connector for a topic."""
        return self._topic_map.get(topic)

    async def lookup(
        self,
        user_id: str,
        topics: List[str],
        context: Dict[str, Any]
    ) -> List[ConnectorResult]:
        """
        Lookup data from all relevant connectors.

        Args:
            user_id: User identifier
            topics: Topics mentioned in conversation
            context: Extracted entities and context

        Returns:
            List of results from matching connectors
        """
        results = []
        seen_connectors = set()

        for topic in topics:
            connector = self._topic_map.get(topic)
            if connector and connector.name not in seen_connectors:
                seen_connectors.add(connector.name)
                try:
                    result = await connector.lookup(user_id, context)
                    results.append(result)
                except Exception as e:
                    results.append(ConnectorResult(
                        data={},
                        source=connector.name,
                        fetched_at=datetime.now(),
                        error=str(e)
                    ))

        return results
```

### 3.3 Example Connectors

**File**: `mindcore/connectors/orders.py` (new)

```python
class OrdersConnector(BaseConnector):
    """
    Read-only connector for orders system.

    Usage:
        connector = OrdersConnector(
            db_url="postgresql://readonly:pass@orders-db/orders",
            user_id_column="customer_id"
        )
        mindcore.register_connector(connector)
    """

    topics = ["orders", "order", "purchase", "delivery", "shipping"]
    name = "orders"

    def __init__(
        self,
        db_url: str,                    # Read-only connection string
        user_id_column: str = "user_id",  # Column that maps to Mindcore user_id
        table_name: str = "orders"
    ):
        self.db_url = db_url
        self.user_id_column = user_id_column
        self.table_name = table_name
        self._pool = None

    async def lookup(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> ConnectorResult:
        """Fetch orders for user."""
        order_id = context.get("order_id")
        date_from = context.get("date_from")
        date_to = context.get("date_to")

        query = f"""
            SELECT order_id, status, total, items, created_at
            FROM {self.table_name}
            WHERE {self.user_id_column} = $1
        """
        params = [user_id]

        if order_id:
            query += " AND order_id = $2"
            params.append(order_id)

        query += " ORDER BY created_at DESC LIMIT 10"

        # Execute query (read-only)
        orders = await self._execute_query(query, params)

        return ConnectorResult(
            data={"orders": orders, "count": len(orders)},
            source="orders",
            fetched_at=datetime.now()
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract order IDs and dates from text."""
        import re

        entities = {}

        # Extract order IDs (e.g., #12345, ORD-12345)
        order_patterns = [
            r'#(\d{4,})',
            r'order[:\s#]*(\d{4,})',
            r'ORD-?(\d+)',
        ]
        for pattern in order_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["order_id"] = matches[0]
                break

        # Extract dates (basic)
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(last week|yesterday|today)',
        ]
        # ... date extraction logic

        return entities
```

**File**: `mindcore/connectors/billing.py` (new)

```python
class BillingConnector(BaseConnector):
    """Read-only connector for billing/payment system."""

    topics = ["billing", "payment", "invoice", "subscription", "charge"]
    name = "billing"

    # Similar structure to OrdersConnector
```

### 3.4 Integration with SmartContextAgent

Add tool for external data lookup:

```python
{
    "type": "function",
    "function": {
        "name": "lookup_external_data",
        "description": "Fetch data from external systems like orders, billing, etc. Use when user asks about their orders, payments, subscriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topics to look up (orders, billing, etc.)"
                },
                "context": {
                    "type": "object",
                    "description": "Extracted entities like order_id, date range"
                }
            },
            "required": ["topics"]
        }
    }
}
```

### 3.5 Entity Extraction Enhancement

Update `EnrichmentAgent` to extract actionable entities:

```python
# Add to enrichment output
@dataclass
class MessageMetadata:
    # ... existing fields ...

    # Actionable entities (for connector lookups)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    # Example: {"order_ids": ["12345"], "dates": ["2024-03-15"]}
```

### 3.6 Tasks for Phase 3

- [ ] Create `mindcore/connectors/` package
- [ ] Implement BaseConnector ABC
- [ ] Implement ConnectorRegistry
- [ ] Create OrdersConnector example
- [ ] Create BillingConnector example
- [ ] Add `lookup_external_data` tool to SmartContextAgent
- [ ] Enhance entity extraction in EnrichmentAgent
- [ ] Add connector configuration to MindcoreClient
- [ ] Add caching layer for connector results
- [ ] Write documentation for creating custom connectors
- [ ] Write tests for connector flow

---

## Phase 4: Integration & Testing

### 4.1 Update MindcoreClient

```python
class MindcoreClient:
    def __init__(self, config_path: str = "config.yaml"):
        # ... existing init ...

        # New managers
        self.preferences_manager = PreferencesManager(self.db)
        self.connector_registry = ConnectorRegistry()
        self.summarization_agent = None  # Lazy init
        self.summarization_worker = None

    def register_connector(self, connector: BaseConnector) -> None:
        """Register an external data connector."""
        self.connector_registry.register(connector)

    def start_summarization_worker(
        self,
        max_age_days: int = 7,
        interval_seconds: int = 3600
    ) -> None:
        """Start background summarization worker."""
        pass
```

### 4.2 Update AsyncMindcoreClient

Same additions for async client.

### 4.3 Configuration

Add to `config.yaml`:

```yaml
# Summarization settings
summarization:
  enabled: true
  max_age_days: 7
  min_messages: 5
  delete_after_summary: false
  keep_last_n: 3
  run_interval_seconds: 3600

# Preferences settings
preferences:
  enabled: true
  include_in_context: true

# Connector settings
connectors:
  cache_ttl: 300
  max_retries: 2
  timeout_seconds: 10
```

### 4.4 API Endpoints

Add new endpoints:

```python
# POST /preferences/{user_id}
# GET /preferences/{user_id}
# POST /summarize/{thread_id}
# GET /summaries/{user_id}
```

### 4.5 Testing Plan

1. **Unit Tests**
   - ThreadSummary creation and serialization
   - UserPreferences CRUD
   - Connector entity extraction
   - Database operations

2. **Integration Tests**
   - Full summarization flow
   - Preference update through agent
   - Connector lookup through SmartContextAgent

3. **Performance Tests**
   - Summarization with 1000+ messages
   - Concurrent connector lookups
   - Cache hit/miss ratios

---

## File Structure After Implementation

```
mindcore/
├── __init__.py
├── async_client.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── enrichment_agent.py
│   ├── context_assembler_agent.py
│   ├── retrieval_query_agent.py
│   ├── smart_context_agent.py
│   └── summarization_agent.py      # NEW
├── connectors/                      # NEW PACKAGE
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── orders.py                   # Example
│   └── billing.py                  # Example
├── core/
│   ├── __init__.py
│   ├── schemas.py                  # Updated with new schemas
│   ├── config_loader.py
│   ├── db_manager.py
│   ├── sqlite_manager.py           # Updated with new tables/methods
│   ├── async_db.py                 # Updated with new tables/methods
│   ├── cache_manager.py
│   ├── disk_cache_manager.py
│   ├── preferences_manager.py      # NEW
│   └── metrics_manager.py
├── workers/                         # NEW PACKAGE
│   ├── __init__.py
│   └── summarization_worker.py
└── ...
```

---

## Implementation Order

### Sprint 1 (Foundation)
1. Add new schemas (ThreadSummary, UserPreferences)
2. Add database tables and migrations
3. Basic CRUD for summaries and preferences

### Sprint 2 (Summarization)
1. SummarizationAgent implementation
2. SummarizationWorker (background job)
3. Integration with SmartContextAgent
4. Tests

### Sprint 3 (Preferences)
1. PreferencesManager implementation
2. Preference update tool for agent
3. Integration with context assembly
4. API endpoints
5. Tests

### Sprint 4 (Connectors)
1. Connector base class and registry
2. Example connectors (Orders, Billing)
3. Entity extraction enhancements
4. Integration with SmartContextAgent
5. Documentation for custom connectors
6. Tests

### Sprint 5 (Polish)
1. Configuration options
2. API documentation
3. Performance optimization
4. End-to-end testing
5. Migration guide for existing users

---

## Success Metrics

1. **Summarization**
   - 90% reduction in storage for threads > 7 days old
   - Summary retrieval < 50ms
   - Summary quality: manual review of 100 samples

2. **Preferences**
   - 100% of amendable fields updatable via agent
   - 0 updates to read-only fields
   - Preferences included in context within 10ms

3. **Connectors**
   - External data fetch < 500ms (with cache)
   - 95% cache hit rate for repeated queries
   - 0 writes to external systems (read-only enforced)

---

## Notes

- All external connectors are **read-only** - they cannot modify external systems
- Summarization should run during low-traffic periods
- User preferences are **separate** from system-of-record data
- Connectors should fail gracefully - missing data shouldn't break context assembly
