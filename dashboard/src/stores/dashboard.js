import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { dashboardApi } from '../api'

export const useDashboardStore = defineStore('dashboard', () => {
  // State
  const stats = ref({
    total_messages: 0,
    today_messages: 0,
    active_users: 0,
    conversations: 0
  })
  const messages = ref([])
  const threads = ref([])
  const logs = ref([])
  const config = ref({
    llm: {
      provider: 'auto',
      model: 'gpt-4o-mini',
      temperature: 0.3,
      max_tokens: 1500
    },
    memory: {
      enabled: true,
      provider: 'llama_cpp'
    }
  })
  const models = ref({
    cloud: [],
    local: []
  })
  const activeModel = ref(null)
  const loading = ref(false)
  const error = ref(null)

  // Actions
  async function fetchStats() {
    try {
      loading.value = true
      const data = await dashboardApi.getStats()
      stats.value = data
    } catch (err) {
      console.error('Failed to fetch stats:', err)
      error.value = err.message
    } finally {
      loading.value = false
    }
  }

  async function fetchMessages(params = {}) {
    try {
      loading.value = true
      const data = await dashboardApi.getMessages(params)
      messages.value = data.messages || data
      return data
    } catch (err) {
      console.error('Failed to fetch messages:', err)
      error.value = err.message
      return { messages: [], total: 0 }
    } finally {
      loading.value = false
    }
  }

  async function deleteMessage(id) {
    try {
      await dashboardApi.deleteMessage(id)
      messages.value = messages.value.filter(m => m.message_id !== id)
      return true
    } catch (err) {
      console.error('Failed to delete message:', err)
      error.value = err.message
      return false
    }
  }

  async function fetchThreads(params = {}) {
    try {
      loading.value = true
      const data = await dashboardApi.getThreads(params)
      threads.value = data.threads || data
      return data
    } catch (err) {
      console.error('Failed to fetch threads:', err)
      error.value = err.message
      return { threads: [], total: 0 }
    } finally {
      loading.value = false
    }
  }

  async function fetchLogs(params = {}) {
    try {
      const data = await dashboardApi.getLogs(params)
      logs.value = data.logs || data
      return data
    } catch (err) {
      console.error('Failed to fetch logs:', err)
      error.value = err.message
      return { logs: [] }
    }
  }

  async function clearLogs() {
    try {
      await dashboardApi.clearLogs()
      logs.value = []
      return true
    } catch (err) {
      console.error('Failed to clear logs:', err)
      error.value = err.message
      return false
    }
  }

  async function fetchConfig() {
    try {
      const data = await dashboardApi.getConfig()
      config.value = data
      return data
    } catch (err) {
      console.error('Failed to fetch config:', err)
      error.value = err.message
      return null
    }
  }

  async function updateConfig(newConfig) {
    try {
      const data = await dashboardApi.updateConfig(newConfig)
      config.value = data
      return true
    } catch (err) {
      console.error('Failed to update config:', err)
      error.value = err.message
      return false
    }
  }

  async function fetchModels() {
    try {
      const data = await dashboardApi.getModels()
      models.value = data
      return data
    } catch (err) {
      console.error('Failed to fetch models:', err)
      error.value = err.message
      return { cloud: [], local: [] }
    }
  }

  async function fetchActiveModel() {
    try {
      const data = await dashboardApi.getActiveModel()
      activeModel.value = data
      return data
    } catch (err) {
      console.error('Failed to fetch active model:', err)
      return null
    }
  }

  async function setActiveModel(modelId) {
    try {
      await dashboardApi.setActiveModel(modelId)
      activeModel.value = modelId
      return true
    } catch (err) {
      console.error('Failed to set active model:', err)
      error.value = err.message
      return false
    }
  }

  return {
    // State
    stats,
    messages,
    threads,
    logs,
    config,
    models,
    activeModel,
    loading,
    error,

    // Actions
    fetchStats,
    fetchMessages,
    deleteMessage,
    fetchThreads,
    fetchLogs,
    clearLogs,
    fetchConfig,
    updateConfig,
    fetchModels,
    fetchActiveModel,
    setActiveModel
  }
})
