<template>
  <div class="logs-page fade-in">
    <!-- Header -->
    <a-card class="dashboard-card filter-card">
      <a-row :gutter="16" align="middle">
        <a-col :xs="12" :sm="8" :md="4">
          <a-select
            v-model:value="levelFilter"
            placeholder="Filter by level"
            allow-clear
            style="width: 100%"
            @change="fetchLogs"
          >
            <a-select-option value="INFO">
              <span class="log-level info">INFO</span>
            </a-select-option>
            <a-select-option value="WARNING">
              <span class="log-level warning">WARNING</span>
            </a-select-option>
            <a-select-option value="ERROR">
              <span class="log-level error">ERROR</span>
            </a-select-option>
            <a-select-option value="DEBUG">
              <span class="log-level debug">DEBUG</span>
            </a-select-option>
          </a-select>
        </a-col>
        <a-col :xs="12" :sm="8" :md="4">
          <a-switch v-model:checked="autoRefresh" /> Auto-refresh
        </a-col>
        <a-col :xs="24" :sm="8" :md="16" style="text-align: right;">
          <a-space>
            <a-button @click="fetchLogs">
              <ReloadOutlined /> Refresh
            </a-button>
            <a-popconfirm
              title="Clear all logs?"
              ok-text="Yes"
              cancel-text="No"
              @confirm="clearLogs"
            >
              <a-button danger>
                <DeleteOutlined /> Clear Logs
              </a-button>
            </a-popconfirm>
          </a-space>
        </a-col>
      </a-row>
    </a-card>

    <!-- Logs List -->
    <a-card class="dashboard-card logs-card">
      <template #title>
        <span>System Logs</span>
        <a-badge
          :count="logs.length"
          :number-style="{ backgroundColor: '#7C6EE4', marginLeft: '8px' }"
        />
      </template>
      <template #extra>
        <a-tag v-if="autoRefresh" color="green">
          <SyncOutlined spin /> Live
        </a-tag>
      </template>

      <div class="logs-container" ref="logsContainer">
        <template v-if="logs.length">
          <div
            v-for="(log, index) in logs"
            :key="index"
            :class="['log-entry', log.level.toLowerCase()]"
          >
            <span class="log-timestamp">{{ formatTime(log.timestamp) }}</span>
            <span :class="['log-level', log.level.toLowerCase()]">{{ log.level }}</span>
            <span class="log-message">{{ log.message }}</span>
            <span v-if="log.logger" class="log-logger">{{ log.logger }}</span>
          </div>
        </template>

        <div v-else class="empty-state">
          <FileTextOutlined class="empty-state-icon" />
          <div class="empty-state-title">No logs available</div>
          <div class="empty-state-desc">System logs will appear here</div>
        </div>
      </div>
    </a-card>

    <!-- Stats -->
    <a-row :gutter="[24, 24]" style="margin-top: 24px;">
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-card-value" style="color: #1890ff;">{{ logStats.info }}</div>
          <div class="stat-card-label">Info</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-card-value" style="color: #faad14;">{{ logStats.warning }}</div>
          <div class="stat-card-label">Warnings</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-card-value" style="color: #ff4d4f;">{{ logStats.error }}</div>
          <div class="stat-card-label">Errors</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-card-value" style="color: #722ed1;">{{ logStats.debug }}</div>
          <div class="stat-card-label">Debug</div>
        </div>
      </a-col>
    </a-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { message } from 'ant-design-vue'
import {
  ReloadOutlined,
  DeleteOutlined,
  SyncOutlined,
  FileTextOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'

const logs = ref([])
const loading = ref(false)
const levelFilter = ref(null)
const autoRefresh = ref(false)
const logsContainer = ref(null)
let refreshInterval = null

const logStats = computed(() => {
  const stats = { info: 0, warning: 0, error: 0, debug: 0 }
  logs.value.forEach(log => {
    const level = log.level.toLowerCase()
    if (stats[level] !== undefined) {
      stats[level]++
    }
  })
  return stats
})

const formatTime = (time) => {
  if (!time) return ''
  return dayjs(time).format('HH:mm:ss.SSS')
}

const fetchLogs = async () => {
  loading.value = true
  try {
    const params = { limit: 200 }
    if (levelFilter.value) {
      params.level = levelFilter.value
    }

    const data = await dashboardApi.getLogs(params)
    logs.value = data.logs || []

    // Scroll to bottom
    await nextTick()
    if (logsContainer.value) {
      logsContainer.value.scrollTop = logsContainer.value.scrollHeight
    }
  } catch (err) {
    console.error('Failed to fetch logs:', err)
  } finally {
    loading.value = false
  }
}

const clearLogs = async () => {
  try {
    await dashboardApi.clearLogs()
    logs.value = []
    message.success('Logs cleared')
  } catch (err) {
    console.error('Failed to clear logs:', err)
    message.error('Failed to clear logs')
  }
}

// Auto-refresh functionality
watch(autoRefresh, (enabled) => {
  if (enabled) {
    refreshInterval = setInterval(fetchLogs, 3000)
  } else {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }
})

onMounted(() => {
  fetchLogs()
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.logs-page {
  padding-bottom: 24px;
}

.filter-card {
  margin-bottom: 24px;
}

.logs-card :deep(.ant-card-head-title) {
  display: flex;
  align-items: center;
}

.logs-container {
  max-height: 500px;
  overflow-y: auto;
  font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
  font-size: 13px;
  background: #1a1a2e;
  border-radius: 8px;
  padding: 16px;
}

.log-entry {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  color: #e0e0e0;
}

.log-entry:last-child {
  border-bottom: none;
}

.log-timestamp {
  color: #6b7280;
  flex-shrink: 0;
  font-size: 12px;
}

.log-level {
  flex-shrink: 0;
  min-width: 60px;
  text-align: center;
}

.log-message {
  flex: 1;
  word-break: break-word;
}

.log-logger {
  color: #6b7280;
  font-size: 11px;
  flex-shrink: 0;
}

.log-entry.info .log-message {
  color: #60a5fa;
}

.log-entry.warning .log-message {
  color: #fbbf24;
}

.log-entry.error .log-message {
  color: #f87171;
}

.log-entry.debug .log-message {
  color: #a78bfa;
}

.stat-card {
  text-align: center;
}

.stat-card .stat-card-value {
  font-size: 24px;
}
</style>
