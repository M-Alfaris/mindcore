"""Product-related data sources.

This module demonstrates class-based sources using TableSource.
Use this pattern when you want to leverage the built-in SQL execution.
"""

from mindcore.svl import TableSource, TriggerCondition
from mindcore.svl.registry import source


@source(term="products", term_type="topic", priority=10)
class ProductCatalog(TableSource):
    """Fetch products from the catalog database.

    This class-based source uses TableSource for SQL execution.
    The @source decorator registers it with the specified term.
    """

    name = "product_catalog"
    description = "Product catalog database"

    # Database connection (set via environment or config)
    connection_string = "sqlite:///products.db"  # Override in production

    # Query template with parameter placeholders
    query_template = """
        SELECT id, name, description, price, category, stock_quantity
        FROM products
        WHERE category = :category OR :category IS NULL
        ORDER BY name
        LIMIT :limit
    """

    # Map context keys to SQL parameters
    param_mapping = {
        "category": "category",
        "limit": "limit",
    }

    # Defaults
    limit = 50
    cache_ttl_seconds = 300
    trigger = TriggerCondition.ON_QUERY


@source(
    term="product_search",
    term_type="topic",
    cache_ttl=60,
    tags=["search", "ecommerce"],
)
async def search_products(context: dict) -> list[dict]:
    """Search products by query string.

    Args:
        context: Contains 'query' (search text), optional 'category', 'limit'

    Returns:
        List of matching products
    """
    query = context.get("query", "")
    category = context.get("category")
    _limit = context.get("limit", 20)  # Used in real implementation

    # Example: Replace with actual search implementation
    # This could use Elasticsearch, PostgreSQL full-text search, etc.
    #
    # results = await search_client.search(
    #     index="products",
    #     query={"multi_match": {"query": query, "fields": ["name", "description"]}},
    #     filter={"term": {"category": category}} if category else None,
    #     size=_limit,
    # )

    return [
        {
            "id": "prod_001",
            "name": f"Product matching '{query}'",
            "category": category or "general",
            "price": 29.99,
            "relevance_score": 0.95,
        }
    ]


@source(
    term="product_recommendations",
    term_type="topic",
    trigger=TriggerCondition.ON_DEMAND,
    cache_ttl=600,
    tags=["ml", "recommendations"],
)
async def get_recommendations(context: dict) -> list[dict]:
    """Get personalized product recommendations.

    Args:
        context: Contains 'user_id', optional 'product_id' for similar items

    Returns:
        List of recommended products
    """
    _user_id = context.get("user_id")  # Used in real implementation
    _product_id = context.get("product_id")  # Used in real implementation

    # Example: Replace with actual ML recommendation service
    # recommendations = await ml_service.get_recommendations(
    #     user_id=_user_id,
    #     seed_product=_product_id,
    #     limit=10,
    # )

    return [
        {
            "id": f"rec_{i}",
            "name": f"Recommended Product {i}",
            "reason": "Based on your purchase history",
            "score": 0.9 - i * 0.05,
        }
        for i in range(5)
    ]
