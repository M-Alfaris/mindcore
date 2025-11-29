<template>
  <div class="sessions-page fade-in">
    <!-- Header Stats -->
    <a-row :gutter="[24, 24]" class="stats-row">
      <a-col :xs="12" :sm="6">
        <div class="mini-stat-card">
          <div class="mini-stat-value">{{ stats.total_sessions }}</div>
          <div class="mini-stat-label">Total Sessions</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="mini-stat-card">
          <div class="mini-stat-value">{{ stats.active_sessions }}</div>
          <div class="mini-stat-label">Active Today</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="mini-stat-card">
          <div class="mini-stat-value">{{ stats.avg_duration }}</div>
          <div class="mini-stat-label">Avg Duration</div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="mini-stat-card">
          <div class="mini-stat-value">{{ stats.avg_messages }}</div>
          <div class="mini-stat-label">Avg Messages/Session</div>
        </div>
      </a-col>
    </a-row>

    <!-- Filters -->
    <a-card class="dashboard-card filter-card">
      <a-row :gutter="16" align="middle">
        <a-col :xs="24" :sm="8" :md="6">
          <a-input-search
            v-model:value="searchQuery"
            placeholder="Search by user ID..."
            allow-clear
            @search="handleSearch"
          />
        </a-col>
        <a-col :xs="12" :sm="8" :md="4">
          <a-date-picker
            v-model:value="dateFilter"
            placeholder="Filter by date"
            style="width: 100%"
            @change="handleFilter"
          />
        </a-col>
        <a-col :xs="12" :sm="8" :md="4">
          <a-button @click="resetFilters">
            <ReloadOutlined /> Reset
          </a-button>
        </a-col>
        <a-col :flex="'auto'" style="text-align: right;">
          <a-button type="primary" @click="exportSessions">
            <DownloadOutlined /> Export CSV
          </a-button>
        </a-col>
      </a-row>
    </a-card>

    <!-- Sessions Table -->
    <a-card class="dashboard-card">
      <a-table
        :columns="columns"
        :data-source="sessions"
        :loading="loading"
        :pagination="pagination"
        :row-key="record => record.session_id"
        @change="handleTableChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'session_id'">
            <a-tooltip :title="record.session_id">
              <code class="id-cell">{{ truncate(record.session_id, 12) }}</code>
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'user_id'">
            <a-avatar size="small" :style="{ backgroundColor: getUserColor(record.user_id) }">
              {{ record.user_id ? record.user_id[0].toUpperCase() : '?' }}
            </a-avatar>
            <span style="margin-left: 8px;">{{ record.user_id || 'anonymous' }}</span>
          </template>

          <template v-else-if="column.key === 'thread_count'">
            <a-badge :count="record.thread_count" :number-style="{ backgroundColor: '#7C6EE4' }" />
          </template>

          <template v-else-if="column.key === 'message_count'">
            <a-progress
              :percent="getProgressPercent(record.message_count)"
              :show-info="false"
              size="small"
              :stroke-color="'#7C6EE4'"
              style="width: 60px; display: inline-block;"
            />
            <span style="margin-left: 8px;">{{ record.message_count }}</span>
          </template>

          <template v-else-if="column.key === 'duration'">
            <ClockCircleOutlined style="margin-right: 4px; color: var(--primary-color);" />
            {{ formatDuration(record.duration_seconds) }}
          </template>

          <template v-else-if="column.key === 'started_at'">
            <a-tooltip :title="formatDateTime(record.started_at)">
              {{ formatTime(record.started_at) }}
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'actions'">
            <a-space>
              <a-button type="text" size="small" @click="viewSession(record)">
                <EyeOutlined />
              </a-button>
              <a-button type="text" size="small" @click="copySessionId(record.session_id)">
                <CopyOutlined />
              </a-button>
            </a-space>
          </template>
        </template>
      </a-table>
    </a-card>

    <!-- Session Detail Modal -->
    <a-modal
      v-model:open="detailModalVisible"
      title="Session Details"
      width="800px"
      :footer="null"
    >
      <template v-if="selectedSession">
        <a-descriptions :column="2" bordered size="small">
          <a-descriptions-item label="Session ID" :span="2">
            <code>{{ selectedSession.session_id }}</code>
          </a-descriptions-item>
          <a-descriptions-item label="User ID">{{ selectedSession.user_id || 'anonymous' }}</a-descriptions-item>
          <a-descriptions-item label="Duration">{{ formatDuration(selectedSession.duration_seconds) }}</a-descriptions-item>
          <a-descriptions-item label="Threads">{{ selectedSession.thread_count }}</a-descriptions-item>
          <a-descriptions-item label="Messages">{{ selectedSession.message_count }}</a-descriptions-item>
          <a-descriptions-item label="Started">{{ formatDateTime(selectedSession.started_at) }}</a-descriptions-item>
          <a-descriptions-item label="Last Activity">{{ formatDateTime(selectedSession.last_activity_at) }}</a-descriptions-item>
        </a-descriptions>

        <a-divider>Session Threads</a-divider>
        <a-list
          :data-source="selectedSession.threads || []"
          size="small"
        >
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  <a @click="goToThread(item.thread_id)">{{ truncate(item.thread_id, 30) }}</a>
                </template>
                <template #description>
                  {{ item.message_count }} messages
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
        </a-list>
      </template>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ReloadOutlined,
  DownloadOutlined,
  EyeOutlined,
  CopyOutlined,
  ClockCircleOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const router = useRouter()

const sessions = ref([])
const loading = ref(false)
const searchQuery = ref('')
const dateFilter = ref(null)
const detailModalVisible = ref(false)
const selectedSession = ref(null)

const stats = ref({
  total_sessions: 0,
  active_sessions: 0,
  avg_duration: '0m',
  avg_messages: 0
})

const pagination = reactive({
  current: 1,
  pageSize: 20,
  total: 0,
  showSizeChanger: true,
  showTotal: (total) => `Total ${total} sessions`
})

const columns = [
  { title: 'Session ID', key: 'session_id', width: 150 },
  { title: 'User', key: 'user_id', width: 150 },
  { title: 'Threads', key: 'thread_count', width: 100, align: 'center' },
  { title: 'Messages', key: 'message_count', width: 150 },
  { title: 'Duration', key: 'duration', width: 120 },
  { title: 'Started', key: 'started_at', width: 120 },
  { title: 'Actions', key: 'actions', width: 100, align: 'center' }
]

const truncate = (text, length) => {
  if (!text) return ''
  return text.length > length ? text.substring(0, length) + '...' : text
}

const formatTime = (time) => {
  if (!time) return '-'
  return dayjs(time).fromNow()
}

const formatDateTime = (time) => {
  if (!time) return '-'
  return dayjs(time).format('YYYY-MM-DD HH:mm:ss')
}

const formatDuration = (seconds) => {
  if (!seconds) return '0m'
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

const getUserColor = (userId) => {
  const colors = ['#7C6EE4', '#1890ff', '#52c41a', '#faad14', '#eb2f96']
  const hash = userId ? userId.charCodeAt(0) % colors.length : 0
  return colors[hash]
}

const getProgressPercent = (count) => {
  const max = sessions.value.length ? Math.max(...sessions.value.map(s => s.message_count)) : 1
  return (count / max) * 100
}

const fetchSessions = async () => {
  loading.value = true
  try {
    // Fetch sessions from the sessions API
    const [sessionsData, statsData] = await Promise.all([
      dashboardApi.getSessions({
        page: pagination.current,
        page_size: pagination.pageSize,
        user_id: searchQuery.value || undefined
      }),
      dashboardApi.getSessionStats()
    ])

    // Calculate durations from timestamps
    sessions.value = (sessionsData.sessions || []).map(s => {
      const start = dayjs(s.started_at)
      const end = dayjs(s.last_activity_at)
      s.duration_seconds = Math.max(0, end.diff(start, 'second'))
      return s
    })

    pagination.total = sessionsData.total || sessions.value.length

    // Set stats from API
    stats.value = {
      total_sessions: statsData.total_sessions || 0,
      active_sessions: statsData.active_today || 0,
      avg_duration: formatDuration((statsData.avg_duration_minutes || 0) * 60),
      avg_messages: statsData.avg_messages_per_session || 0
    }
  } catch (err) {
    console.error('Failed to fetch sessions:', err)
    message.error('Failed to load sessions')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  pagination.current = 1
  fetchSessions()
}

const handleFilter = () => {
  pagination.current = 1
  fetchSessions()
}

const resetFilters = () => {
  searchQuery.value = ''
  dateFilter.value = null
  pagination.current = 1
  fetchSessions()
}

const handleTableChange = (pag) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  fetchSessions()
}

const viewSession = async (session) => {
  try {
    // Fetch detailed session data including threads
    const detail = await dashboardApi.getSession(session.session_id)
    selectedSession.value = {
      ...session,
      ...detail,
      duration_seconds: session.duration_seconds
    }
    detailModalVisible.value = true
  } catch (err) {
    console.error('Failed to fetch session detail:', err)
    // Fallback to basic session data
    selectedSession.value = session
    detailModalVisible.value = true
  }
}

const copySessionId = async (sessionId) => {
  try {
    await navigator.clipboard.writeText(sessionId)
    message.success('Session ID copied')
  } catch {
    message.error('Failed to copy')
  }
}

const goToThread = (threadId) => {
  detailModalVisible.value = false
  router.push(`/threads/${threadId}`)
}

const exportSessions = () => {
  const headers = ['Session ID', 'User ID', 'Threads', 'Messages', 'Duration (s)', 'Started At', 'Last Activity']
  const rows = sessions.value.map(s => [
    s.session_id,
    s.user_id || 'anonymous',
    s.thread_count,
    s.message_count,
    s.duration_seconds,
    s.started_at,
    s.last_activity_at
  ])

  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `sessions_${dayjs().format('YYYY-MM-DD')}.csv`
  a.click()
  URL.revokeObjectURL(url)
  message.success('Sessions exported')
}

onMounted(() => {
  fetchSessions()
})
</script>

<style scoped>
.sessions-page {
  padding-bottom: 24px;
}

.stats-row {
  margin-bottom: 24px;
}

.mini-stat-card {
  background: var(--card-bg);
  padding: 16px;
  border-radius: 8px;
  text-align: center;
  box-shadow: var(--shadow);
}

.mini-stat-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--primary-color);
}

.mini-stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.filter-card {
  margin-bottom: 24px;
}

.id-cell {
  font-size: 11px;
  background: var(--bg-color);
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--primary-color);
}
</style>
