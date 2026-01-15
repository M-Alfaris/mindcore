"""MCP Server CLI command for Mindcore.

Starts an MCP (Model Context Protocol) server that exposes Mindcore
as tools for AI agents like Claude, GPT, etc.

Usage:
    mindcore mcp                           # Start with default settings
    mindcore mcp --storage postgresql://   # Use PostgreSQL
    mindcore mcp --port 3000               # Custom port
"""

import json
import sys
from pathlib import Path

import click


class Colors:
    """ANSI color codes."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def styled(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    return "".join(styles) + text + Colors.RESET


@click.command()
@click.option(
    "--storage",
    "-s",
    default="sqlite:///mindcore.db",
    help="Storage connection string (SQLite or PostgreSQL)",
)
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport type: stdio (default) or http",
)
@click.option(
    "--port",
    "-p",
    default=3000,
    type=int,
    help="Port for HTTP transport (default: 3000)",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host for HTTP transport (default: 127.0.0.1)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to mindcore config file",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def mcp_command(
    storage: str,
    transport: str,
    port: int,
    host: str,
    config: str | None,
    debug: bool,
):
    """Start an MCP server for Mindcore.

    The MCP server exposes Mindcore memory operations as tools that can be
    used by AI agents like Claude, GPT, and other MCP-compatible models.

    Examples:

        # Start with stdio transport (default, for Claude Desktop)
        mindcore mcp

        # Start with HTTP transport
        mindcore mcp --transport http --port 3000

        # Use PostgreSQL storage
        mindcore mcp --storage postgresql://user:pass@localhost/mindcore

    Available MCP Tools:

        store_memory      - Store a memory in long-term storage
        search_memories   - Search stored memories
        recall            - Fast recall of relevant memories
        reinforce_memory  - Apply learning signal to memory
        get_schema        - Get vocabulary schema

    Claude Desktop Integration:

        Add to your Claude Desktop config (claude_desktop_config.json):

        {
          "mcpServers": {
            "mindcore": {
              "command": "mindcore",
              "args": ["mcp", "--storage", "sqlite:///path/to/mindcore.db"]
            }
          }
        }
    """
    import logging

    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Log to stderr, keep stdout for MCP protocol
    )
    logger = logging.getLogger("mindcore.mcp")

    # Load config if provided
    if config:
        try:
            import yaml

            with open(config) as f:
                cfg = yaml.safe_load(f)
                storage = cfg.get("storage", {}).get("connection_string", storage)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Initialize Mindcore
    try:
        from mindcore import Mindcore
        from mindcore.server.mcp import MCPServer

        logger.info(f"Initializing Mindcore with storage: {storage}")
        mindcore = Mindcore(storage=storage)

        # Create MCP server
        server = MCPServer(
            flr=mindcore._flr,
            clst=mindcore._clst,
            vocabulary=mindcore._svl,
        )

        logger.info("MCP Server initialized")
        logger.info(f"Available tools: {[t['name'] for t in server.get_tools()]}")

    except Exception as e:
        click.echo(styled(f"  ✗ Failed to initialize Mindcore: {e}", Colors.RED), err=True)
        sys.exit(1)

    if transport == "stdio":
        _run_stdio_server(server, logger)
    else:
        _run_http_server(server, host, port, logger)


def _run_stdio_server(server, logger):
    """Run MCP server with stdio transport."""
    import sys

    logger.info("Starting MCP server with stdio transport")
    logger.info("Waiting for MCP messages on stdin...")

    # Read JSON-RPC messages from stdin, write to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = _handle_mcp_request(server, request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error handling request: {e}")


def _run_http_server(server, host: str, port: int, logger):
    """Run MCP server with HTTP transport."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        click.echo(styled("  ✗ HTTP server not available", Colors.RED), err=True)
        sys.exit(1)

    class MCPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                request = json.loads(body)
                response = _handle_mcp_request(server, request)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                self.send_response(500)
                self.end_headers()

        def log_message(self, format, *args):
            logger.debug(format % args)

    click.echo()
    click.echo(styled("  Mindcore MCP Server", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo()
    click.echo(f"  {styled('Transport:', Colors.BOLD)} HTTP")
    click.echo(f"  {styled('Endpoint:', Colors.BOLD)} http://{host}:{port}")
    click.echo()
    click.echo(styled("  Available Tools:", Colors.BOLD))
    for tool in server.get_tools():
        click.echo(f"    • {tool['name']}: {tool['description']}")
    click.echo()
    click.echo(styled("  Press Ctrl+C to stop", Colors.DIM))
    click.echo()

    httpd = HTTPServer((host, port), MCPHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\n  Shutting down...")
        httpd.shutdown()


def _handle_mcp_request(server, request: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    result = None
    error = None

    try:
        if method == "initialize":
            result = server.get_server_info()

        elif method == "tools/list":
            result = {"tools": server.get_tools()}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = server.call_tool(tool_name, tool_args)

        elif method == "resources/list":
            result = {"resources": server.get_resources()}

        elif method == "resources/read":
            uri = params.get("uri", "")
            result = server.read_resource(uri)

        elif method == "ping":
            result = {}

        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}

    except Exception as e:
        error = {"code": -32603, "message": str(e)}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return response


# Export for CLI registration
__all__ = ["mcp_command"]
