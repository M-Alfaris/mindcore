# Domain Source Examples

This directory contains real working examples for connecting Mindcore topics to your domain data sources (tables, APIs, etc.).

## Overview

When a topic like "orders" is detected in a query, Mindcore can automatically fetch relevant data from your configured sources and inject it into the LLM context.

```
User Query: "What's the status of my order?"
                    ↓
Topic Detected: ["orders"]
                    ↓
Source Triggered: orders_table
                    ↓
Query: SELECT * FROM orders WHERE user_id = :user_id
                    ↓
Data Injected into Context
```

## Examples

### 1. PostgreSQL Tables (`postgres_tables.yaml`)

Connect topics to your PostgreSQL tables:

```yaml
sources:
  - name: orders_source
    type: table
    topic: orders
    connection: ${DATABASE_URL}
    query: |
      SELECT order_id, status, total, created_at
      FROM orders
      WHERE user_id = :user_id
      ORDER BY created_at DESC
      LIMIT 10
    params:
      user_id: user_id
```

### 2. REST APIs (`rest_apis.yaml`)

Connect topics to external APIs:

```yaml
sources:
  - name: weather_api
    type: api
    topic: weather
    url: https://api.weather.com/v1/current
    method: GET
    headers:
      Authorization: Bearer ${WEATHER_API_KEY}
    params:
      location: location
```

### 3. Mixed Sources (`ecommerce_setup.yaml`)

Complete e-commerce setup with tables and APIs:

```yaml
sources:
  # Database tables
  - name: orders
    type: table
    topics: [orders, purchases, transactions]
    connection: ${DATABASE_URL}
    query: SELECT * FROM orders WHERE user_id = :user_id

  - name: products
    type: table
    topics: [products, catalog, inventory]
    connection: ${DATABASE_URL}
    query: SELECT * FROM products WHERE id = ANY(:product_ids)

  # External APIs
  - name: shipping_tracker
    type: api
    topics: [shipping, delivery, tracking]
    url: https://api.shipping.com/track/{tracking_id}
    params:
      tracking_id: tracking_number
```

## Usage

### Python SDK

```python
from mindcore import Mindcore
from mindcore.svl import StructuredValidationLayer, TableSource, APISource

# Create SVL with domain sources
svl = StructuredValidationLayer(domains=["ecommerce"])

# Map "orders" topic to your orders table
svl.map_source("orders", TableSource(
    name="orders_db",
    connection_string="postgresql://localhost/mydb",
    query_template="""
        SELECT order_id, status, total, items, created_at
        FROM orders
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT 10
    """,
    param_mapping={"user_id": "user_id"},
))

# Map "weather" topic to weather API
svl.map_source("weather", APISource(
    name="weather_api",
    url="https://api.weather.com/v1/current",
    method="GET",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    url_params={"location": "city"},
))

# Initialize Mindcore with your SVL
memory = Mindcore(
    storage="postgresql://localhost/mindcore",
    svl=svl,
)

# Now when you recall with topic "orders", data is auto-fetched
result = memory.recall(
    query="What's my order status?",
    user_id="user_123",
    topics=["orders"],  # Triggers orders_db query
)
```

### YAML Configuration

Load sources from YAML file:

```python
from mindcore.svl import load_sources_from_yaml

# Load all sources from config
sources = load_sources_from_yaml("config/domain_sources.yaml")

# Apply to SVL
for source in sources:
    svl.map_source(source.topic, source)
```

### Environment Variables

All examples support environment variable substitution:

```yaml
connection: ${DATABASE_URL}           # Uses DATABASE_URL env var
headers:
  Authorization: Bearer ${API_KEY}    # Uses API_KEY env var
```

## PostgreSQL-Centric Architecture

For production deployments, we recommend keeping the heavy lifting in PostgreSQL:

1. **Use PostgreSQL functions** for complex queries
2. **Use triggers** for automatic data updates
3. **Use materialized views** for frequently accessed data

Example PostgreSQL function:

```sql
CREATE OR REPLACE FUNCTION get_user_context(p_user_id TEXT)
RETURNS TABLE (
    recent_orders JSONB,
    preferences JSONB,
    interaction_summary JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT jsonb_agg(row_to_json(o)) FROM orders o WHERE o.user_id = p_user_id LIMIT 5),
        (SELECT preferences FROM user_preferences WHERE user_id = p_user_id),
        (SELECT jsonb_build_object('total_orders', count(*)) FROM orders WHERE user_id = p_user_id);
END;
$$ LANGUAGE plpgsql;
```

Then map it:

```python
svl.map_source("user_context", TableSource(
    name="user_context_fn",
    connection_string="postgresql://...",
    query_template="SELECT * FROM get_user_context(:user_id)",
    param_mapping={"user_id": "user_id"},
))
```

## Traceability

All source fetches are logged with:

- Source name and type
- Query/URL executed
- Parameters used
- Latency (ms)
- Cache hit/miss
- Success/failure

Enable detailed logging:

```python
import logging
logging.getLogger("mindcore.svl.sources").setLevel(logging.DEBUG)
```
