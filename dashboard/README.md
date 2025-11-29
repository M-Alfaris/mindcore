# Mindcore Dashboard

Modern web dashboard for Mindcore - intelligent memory and context management for AI agents.

## Features

- **Overview** - Stats cards, message trends chart, role distribution
- **Messages** - Browse, search, filter, and manage messages
- **Threads** - View conversation threads and their messages
- **Logs** - Real-time system logs with level filtering
- **Configuration** - Configure LLM models, memory, and cache settings
- **Models** - Visual model selector for cloud and local models

## Tech Stack

- **Vue 3** - Progressive JavaScript framework
- **Vite** - Next-generation frontend tooling
- **Ant Design Vue 4.1.2** - Enterprise-class UI components
- **Chart.js** - Flexible JavaScript charting
- **Pinia** - State management
- **Axios** - HTTP client

## Quick Start

```bash
# Install dependencies
cd dashboard
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Development

The dashboard runs on `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

### Start the backend:
```bash
# From project root
python -m mindcore.api.server
```

### Start the dashboard:
```bash
cd dashboard
npm run dev
```

## Theme

The dashboard uses a light blue-purple color scheme:

- **Primary**: `#7C6EE4` (light blue-purple)
- **Primary Light**: `#9D91ED`
- **Primary Dark**: `#5B4FCF`
- **Background**: `#F0EEFF`

## API Endpoints

The dashboard uses the following API endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/dashboard/stats` | Dashboard statistics |
| `GET /api/dashboard/messages` | Paginated messages |
| `GET /api/dashboard/threads` | Conversation threads |
| `GET /api/dashboard/logs` | System logs |
| `GET /api/dashboard/config` | Configuration |
| `GET /api/dashboard/models` | Available models |

## Screenshots

(Coming soon)

## License

MIT
