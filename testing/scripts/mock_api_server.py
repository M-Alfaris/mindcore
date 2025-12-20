#!/usr/bin/env python3
"""Mock API Server for SVL Source Testing.

This server provides mock endpoints for testing SVL external data source integration.
"""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PORT = 8001
DATA_DIR = (
    Path("/app/data") if Path("/app/data").exists() else Path(__file__).parent.parent / "demo_data"
)


# Mock data
PRODUCTS = [
    {"id": "1", "name": "Widget Pro", "category": "electronics", "sku": "WP-001", "price": 99.99},
    {"id": "2", "name": "Gadget Plus", "category": "accessories", "sku": "GP-002", "price": 49.99},
    {"id": "3", "name": "Super Tool", "category": "tools", "sku": "ST-003", "price": 149.99},
    {
        "id": "4",
        "name": "Smart Device",
        "category": "electronics",
        "sku": "SD-004",
        "price": 199.99,
    },
    {"id": "5", "name": "Power Bank", "category": "accessories", "sku": "PB-005", "price": 29.99},
]

KB_ARTICLES = [
    {
        "id": "kb-001",
        "title": "Getting Started Guide",
        "topic": "documentation",
        "keywords": ["setup", "installation", "quickstart"],
        "content": "Welcome to our platform. This guide will help you get started...",
    },
    {
        "id": "kb-002",
        "title": "API Authentication",
        "topic": "api",
        "keywords": ["api", "auth", "tokens", "security"],
        "content": "To authenticate API requests, include your API key in the header...",
    },
    {
        "id": "kb-003",
        "title": "Billing FAQ",
        "topic": "billing",
        "keywords": ["payment", "invoice", "subscription"],
        "content": "Common billing questions and answers...",
    },
    {
        "id": "kb-004",
        "title": "Troubleshooting Common Issues",
        "topic": "support",
        "keywords": ["error", "bug", "help", "fix"],
        "content": "Solutions to frequently encountered problems...",
    },
]

CONTACTS = [
    {
        "id": 1,
        "name": "Acme Corp",
        "email": "contact@acme.com",
        "segment": "enterprise",
        "tier": "gold",
    },
    {
        "id": 2,
        "name": "StartupXYZ",
        "email": "hello@startupxyz.io",
        "segment": "startup",
        "tier": "silver",
    },
    {
        "id": 3,
        "name": "BigCo Inc",
        "email": "sales@bigco.com",
        "segment": "enterprise",
        "tier": "platinum",
    },
    {
        "id": 4,
        "name": "MediumBiz",
        "email": "info@mediumbiz.com",
        "segment": "smb",
        "tier": "bronze",
    },
]


class MockAPIHandler(BaseHTTPRequestHandler):
    """Handler for mock API requests."""

    def _send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, message, status=400):
        """Send error response."""
        self._send_json({"error": message}, status)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Simulate network latency
        time.sleep(0.05)

        # Health check
        if path == "/health":
            self._send_json({"status": "ok", "timestamp": time.time()})
            return

        # Products endpoint
        if path == "/products":
            # Optional filtering by category
            category = query.get("category", [None])[0]
            if category:
                filtered = [p for p in PRODUCTS if p["category"] == category]
                self._send_json(filtered)
            else:
                self._send_json(PRODUCTS)
            return

        # Single product
        if path.startswith("/products/"):
            product_id = path.split("/")[-1]
            product = next((p for p in PRODUCTS if p["id"] == product_id), None)
            if product:
                self._send_json(product)
            else:
                self._send_error("Product not found", 404)
            return

        # Knowledge base articles
        if path == "/kb/articles":
            topic = query.get("topic", [None])[0]
            if topic:
                filtered = [a for a in KB_ARTICLES if a["topic"] == topic]
                self._send_json(filtered)
            else:
                self._send_json(KB_ARTICLES)
            return

        # Single article
        if path.startswith("/kb/articles/"):
            article_id = path.split("/")[-1]
            article = next((a for a in KB_ARTICLES if a["id"] == article_id), None)
            if article:
                self._send_json(article)
            else:
                self._send_error("Article not found", 404)
            return

        # Contacts endpoint
        if path == "/contacts":
            segment = query.get("segment", [None])[0]
            if segment:
                filtered = [c for c in CONTACTS if c["segment"] == segment]
                self._send_json(filtered)
            else:
                self._send_json(CONTACTS)
            return

        # Search endpoint
        if path == "/search":
            q = query.get("q", [""])[0].lower()
            if not q:
                self._send_error("Query parameter 'q' required")
                return

            results = []

            # Search products
            for p in PRODUCTS:
                if q in p["name"].lower() or q in p["category"].lower():
                    results.append({"type": "product", "data": p})

            # Search articles
            for a in KB_ARTICLES:
                if q in a["title"].lower() or q in " ".join(a["keywords"]).lower():
                    results.append({"type": "article", "data": a})

            self._send_json({"query": q, "results": results, "count": len(results)})
            return

        # 404 for unknown paths
        self._send_error(f"Endpoint not found: {path}", 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()

    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[MockAPI] {args[0]}")


def main():
    """Start the mock API server."""
    server = HTTPServer(("0.0.0.0", PORT), MockAPIHandler)
    print(f"Mock API server starting on port {PORT}")
    print("Available endpoints:")
    print("  GET /health          - Health check")
    print("  GET /products        - List products")
    print("  GET /products/:id    - Get product by ID")
    print("  GET /kb/articles     - List KB articles")
    print("  GET /kb/articles/:id - Get article by ID")
    print("  GET /contacts        - List contacts")
    print("  GET /search?q=...    - Search all")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
