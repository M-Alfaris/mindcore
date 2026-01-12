# Mindcore Testing Suite

Comprehensive testing environment for validating all Mindcore features.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (SQLite only)
python scripts/run_all_tests.py

# Run with PostgreSQL
docker-compose up -d
python scripts/run_all_tests.py --setup-postgres
```

## Directory Structure

```text
testing/
├── requirements.txt          # Test dependencies
├── docker-compose.yml        # PostgreSQL + mock API
├── pytest.ini               # Pytest configuration
├── demo_data/
│   ├── memories.json         # Sample memories
│   ├── vocabularies.json     # Domain vocabularies
│   ├── agents.json           # Multi-agent configs
│   └── svl_sources.json      # External data sources
├── scripts/
│   ├── setup_postgres.py     # DB initialization
│   ├── load_demo_data.py     # Data loader
│   ├── run_all_tests.py      # Test orchestrator
│   └── mock_api_server.py    # Mock API for SVL tests
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_01_storage.py    # Storage backends
│   ├── test_02_single_agent.py
│   ├── test_03_multi_agent.py
│   ├── test_04_flr_clst_flow.py
│   ├── test_05_svl_domains.py
│   ├── test_06_svl_sources.py
│   ├── test_07_rest_api.py
│   ├── test_08_mcp_server.py
│   ├── test_09_rbac.py
│   ├── test_10_auth_errors.py
│   └── test_11_integration.py
└── notebooks/
    └── interactive_demo.ipynb
```

## Test Categories

| Test | What It Validates |
|------|-------------------|
| 01 Storage | SQLite CRUD, PostgreSQL CRUD, migration |
| 02 Single Agent | Store/recall, vocabulary, importance |
| 03 Multi-Agent | Registration, teams, sharing, isolation |
| 04 FLR/CLST | Hot-path caching, cold storage, reinforcement |
| 05 SVL Domains | Vocabulary creation, validation, merging |
| 06 SVL Sources | API/DB ingestion, topic extraction |
| 07 REST API | All endpoints, rate limiting, errors |
| 08 MCP Server | Tool calls, schema, protocol |
| 09 RBAC | Permissions, team access, admin |
| 10 Auth/Errors | Exception handling, validation |
| 11 Integration | Full workflow end-to-end |

## Running Tests

### Run All Tests

```bash
python scripts/run_all_tests.py
```

### Run Specific Test File

```bash
pytest tests/test_01_storage.py -v
```

### Run Tests by Category

```bash
# Only storage tests
pytest tests/test_01_storage.py tests/test_04_flr_clst_flow.py -v

# Only multi-agent tests
pytest tests/test_03_multi_agent.py tests/test_09_rbac.py -v
```

### Skip PostgreSQL Tests

```bash
python scripts/run_all_tests.py --skip-postgres
```

### Run with Coverage

```bash
pytest tests/ --cov=mindcore --cov-report=html
```

## PostgreSQL Setup

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Wait for health check
docker-compose ps

# Run setup script
python scripts/setup_postgres.py
```

## Loading Demo Data

```bash
# Load into SQLite
python scripts/load_demo_data.py

# Load into PostgreSQL with multi-agent
python scripts/load_demo_data.py \
    --storage "postgresql://mindcore:mindcore_test@localhost/mindcore_test" \
    --multi-agent
```

## Success Metrics

| Metric | Target |
|--------|--------|
| All unit tests pass | 100% |
| SQLite operations | < 10ms latency |
| PostgreSQL operations | < 50ms latency |
| FLR cache hit rate | > 80% on repeated queries |
| Memory isolation | 0 cross-user leaks |
| RBAC enforcement | 100% correct decisions |
| REST API response | < 100ms p95 |

## Interactive Demo

Open `notebooks/interactive_demo.ipynb` in Jupyter for an interactive walkthrough of Mindcore features.

```bash
jupyter notebook notebooks/interactive_demo.ipynb
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_HOST | localhost | PostgreSQL host |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| POSTGRES_USER | mindcore | Database user |
| POSTGRES_PASSWORD | mindcore_test | Database password |
| POSTGRES_DB | mindcore_test | Database name |

## Troubleshooting

### PostgreSQL Connection Failed

```bash
# Check if container is running
docker-compose ps

# View logs
docker-compose logs postgres

# Restart
docker-compose restart postgres
```

### Import Errors

```bash
# Ensure mindcore is installed
pip install -e ../.[all]
```

### Test Timeout

```bash
# Increase timeout
pytest tests/ --timeout=600
```
