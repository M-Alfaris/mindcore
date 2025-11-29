import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  config => config,
  error => Promise.reject(error)
)

// Response interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// Dashboard API
export const dashboardApi = {
  // Stats
  getStats: () => api.get('/dashboard/stats'),
  getMessagesByTime: (days = 7) => api.get(`/dashboard/messages-by-time?days=${days}`),
  getMessagesByRole: () => api.get('/dashboard/messages-by-role'),

  // Messages
  getMessages: (params) => api.get('/dashboard/messages', { params }),
  getMessage: (id) => api.get(`/dashboard/messages/${id}`),
  deleteMessage: (id) => api.delete(`/dashboard/messages/${id}`),
  searchMessages: (query, params) => api.get('/dashboard/messages/search', { params: { q: query, ...params } }),

  // Threads
  getThreads: (params) => api.get('/dashboard/threads', { params }),
  getThread: (threadId) => api.get(`/dashboard/threads/${threadId}`),
  getThreadMessages: (threadId, params) => api.get(`/dashboard/threads/${threadId}/messages`, { params }),

  // Logs
  getLogs: (params) => api.get('/dashboard/logs', { params }),
  clearLogs: () => api.delete('/dashboard/logs'),

  // Configuration
  getConfig: () => api.get('/dashboard/config'),
  updateConfig: (config) => api.put('/dashboard/config', config),
  getSystemStatus: () => api.get('/dashboard/config/status'),
  restartServer: () => api.post('/dashboard/config/restart'),

  // Environment Variables
  getEnvVars: () => api.get('/dashboard/config/env'),
  updateEnvVars: (envVars) => api.put('/dashboard/config/env', envVars),

  // Database Management
  testDatabaseConnection: (dbConfig) => api.post('/dashboard/config/database/test', dbConfig),
  vacuumDatabase: () => api.post('/dashboard/config/database/vacuum'),
  clearOldMetrics: (days) => api.post('/dashboard/config/database/clear-metrics', { days }),
  resetDatabase: () => api.post('/dashboard/config/database/reset'),

  // Models
  getModels: () => api.get('/dashboard/models'),
  getActiveModel: () => api.get('/dashboard/models/active'),
  setActiveModel: (modelId) => api.post('/dashboard/models/active', { model_id: modelId }),
  downloadModel: (modelId) => api.post(`/dashboard/models/${modelId}/download`),

  // Health
  getHealth: () => api.get('/health'),
  getHealthFull: () => api.get('/health/full'),

  // Performance & Observability
  getPerformanceStats: (params) => api.get('/dashboard/performance', { params }),
  getToolStats: () => api.get('/dashboard/tools'),
  getToolCalls: (params) => api.get('/dashboard/tool-calls', { params }),
  getUserPerformance: (params) => api.get('/dashboard/users/performance', { params }),

  // Sessions
  getSessionStats: () => api.get('/dashboard/sessions/stats'),
  getSessions: (params) => api.get('/dashboard/sessions', { params }),
  getSession: (sessionId) => api.get(`/dashboard/sessions/${sessionId}`),
  getSessionMessages: (sessionId, params) => api.get(`/dashboard/sessions/${sessionId}/messages`, { params }),
  deleteSession: (sessionId) => api.delete(`/dashboard/sessions/${sessionId}`)
}

export default api
