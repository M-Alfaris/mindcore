<template>
  <div class="overview-page fade-in">
    <!-- Stats Cards -->
    <a-row :gutter="[16, 16]" class="stats-row">
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Total Messages</div>
          <div class="stat-value">{{ formatNumber(stats.total_messages) }}</div>
          <div class="stat-change" :class="stats.message_growth >= 0 ? 'positive' : 'negative'">
            {{ stats.message_growth >= 0 ? '+' : '' }}{{ stats.message_growth }}% w/w
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Today's Messages</div>
          <div class="stat-value">{{ formatNumber(stats.today_messages) }}</div>
          <div class="stat-change" :class="stats.today_growth >= 0 ? 'positive' : 'negative'">
            {{ stats.today_growth >= 0 ? '+' : '' }}{{ stats.today_growth || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Active Users</div>
          <div class="stat-value">{{ formatNumber(stats.active_users) }}</div>
          <div class="stat-change" :class="stats.user_growth >= 0 ? 'positive' : 'negative'">
            {{ stats.user_growth >= 0 ? '+' : '' }}{{ stats.user_growth || 0 }}% w/w
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Threads</div>
          <div class="stat-value">{{ formatNumber(stats.conversations) }}</div>
          <div class="stat-change" :class="stats.thread_growth >= 0 ? 'positive' : 'negative'">
            {{ stats.thread_growth >= 0 ? '+' : '' }}{{ stats.thread_growth || 0 }}% w/w
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Avg Response Time</div>
          <div class="stat-value">{{ performanceStats.avg_response_time_ms || 0 }}<span class="stat-unit">ms</span></div>
          <div class="stat-change" :class="performanceStats.latency_change <= 0 ? 'positive' : 'negative'">
            {{ performanceStats.latency_change <= 0 ? '' : '+' }}{{ performanceStats.latency_change || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">P95 Latency</div>
          <div class="stat-value">{{ performanceStats.p95_response_time_ms || 0 }}<span class="stat-unit">ms</span></div>
          <div class="stat-change" :class="performanceStats.p95_change <= 0 ? 'positive' : 'negative'">
            {{ performanceStats.p95_change <= 0 ? '' : '+' }}{{ performanceStats.p95_change || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">LLM Calls</div>
          <div class="stat-value">{{ formatNumber(performanceStats.total_llm_calls || 0) }}</div>
          <div class="stat-change" :class="performanceStats.calls_growth >= 0 ? 'positive' : 'negative'">
            {{ performanceStats.calls_growth >= 0 ? '+' : '' }}{{ performanceStats.calls_growth || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">System Health</div>
          <div class="stat-value" :class="getHealthClass(systemHealth)">{{ systemHealth }}</div>
          <div class="stat-change neutral">Based on latency</div>
        </div>
      </a-col>
    </a-row>

    <!-- Charts Row -->
    <a-row :gutter="[24, 24]" class="charts-row">
      <a-col :xs="24" :lg="16">
        <a-card class="dashboard-card" title="Message Activity">
          <template #extra>
            <a-radio-group v-model:value="timeRange" size="small" button-style="solid">
              <a-radio-button value="7">7D</a-radio-button>
              <a-radio-button value="14">14D</a-radio-button>
              <a-radio-button value="30">30D</a-radio-button>
            </a-radio-group>
          </template>
          <div class="chart-container">
            <Line v-if="lineChartData" :data="lineChartData" :options="lineChartOptions" />
            <a-empty v-else description="No data available" />
          </div>
        </a-card>
      </a-col>
      <a-col :xs="24" :lg="8">
        <a-card class="dashboard-card" title="Messages by Role">
          <div class="chart-container doughnut-container">
            <Doughnut v-if="doughnutChartData" :data="doughnutChartData" :options="doughnutChartOptions" />
            <a-empty v-else description="No data available" />
          </div>
        </a-card>
      </a-col>
    </a-row>

    <!-- Middle Row: Latency Distribution + Top Users -->
    <a-row :gutter="[24, 24]" class="charts-row">
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Latency Distribution">
          <div class="chart-container" style="height: 250px;">
            <Bar v-if="latencyChartData" :data="latencyChartData" :options="barChartOptions" />
            <a-empty v-else description="No latency data" />
          </div>
        </a-card>
      </a-col>
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Top Users">
          <template #extra>
            <a-button type="link" size="small" @click="$router.push('/performance')">
              View Details
            </a-button>
          </template>
          <a-table
            :columns="topUsersColumns"
            :data-source="topUsers"
            :pagination="false"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'user_id'">
                <a-avatar size="small" :style="{ backgroundColor: getUserColor(record.user_id) }">
                  {{ record.user_id ? record.user_id[0].toUpperCase() : '?' }}
                </a-avatar>
                <span style="margin-left: 8px;">{{ truncate(record.user_id, 15) }}</span>
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
            </template>
          </a-table>
        </a-card>
      </a-col>
    </a-row>

    <!-- Recent Activity + System Status -->
    <a-row :gutter="[24, 24]">
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Recent Messages">
          <template #extra>
            <a-button type="link" @click="$router.push('/messages')">View All</a-button>
          </template>
          <a-list
            :data-source="recentMessages"
            :loading="loading"
            item-layout="horizontal"
          >
            <template #renderItem="{ item }">
              <a-list-item>
                <a-list-item-meta>
                  <template #avatar>
                    <a-avatar :style="{ backgroundColor: getRoleColor(item.role) }">
                      {{ item.role[0].toUpperCase() }}
                    </a-avatar>
                  </template>
                  <template #title>
                    <span class="message-preview">{{ truncate(item.raw_text, 60) }}</span>
                  </template>
                  <template #description>
                    <span class="message-meta">
                      <a-tag :class="['role-tag', item.role]">{{ item.role }}</a-tag>
                      <span class="message-time">{{ formatTime(item.created_at) }}</span>
                    </span>
                  </template>
                </a-list-item-meta>
              </a-list-item>
            </template>
            <template #empty>
              <a-empty description="No messages yet" />
            </template>
          </a-list>
        </a-card>
      </a-col>
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="System Status">
          <template #extra>
            <a-badge :status="systemStatus === 'healthy' ? 'success' : 'error'" :text="systemStatus" />
          </template>
          <a-descriptions :column="1" bordered size="small">
            <a-descriptions-item label="LLM Provider">
              <a-tag color="purple">{{ config.llm?.provider || 'Not configured' }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="Active Model">
              <a-tag color="blue">{{ config.llm?.model || 'gpt-4o-mini' }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="Database">
              <a-tag color="green">{{ config.database?.type || 'SQLite' }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="Cache Size">
              {{ config.cache?.max_size || 50 }} messages
            </a-descriptions-item>
            <a-descriptions-item label="Memory">
              <a-switch :checked="config.memory?.enabled" disabled size="small" />
              {{ config.memory?.enabled ? 'Enabled' : 'Disabled' }}
            </a-descriptions-item>
          </a-descriptions>
          <a-row :gutter="12" style="margin-top: 16px;">
            <a-col :span="12">
              <a-button block @click="$router.push('/performance')">
                <ThunderboltOutlined /> Performance
              </a-button>
            </a-col>
            <a-col :span="12">
              <a-button type="primary" block @click="$router.push('/config')">
                <SettingOutlined /> Configure
              </a-button>
            </a-col>
          </a-row>
        </a-card>
      </a-col>
    </a-row>

    <!-- Quick Actions Bar -->
    <a-card class="dashboard-card quick-actions" style="margin-top: 24px;">
      <a-row :gutter="16" align="middle">
        <a-col :flex="'auto'">
          <h4 style="margin: 0;">Quick Actions</h4>
        </a-col>
        <a-col>
          <a-space>
            <a-button @click="$router.push('/messages')">
              <MessageOutlined /> View Messages
            </a-button>
            <a-button @click="$router.push('/threads')">
              <CommentOutlined /> View Threads
            </a-button>
            <a-button @click="$router.push('/logs')">
              <FileTextOutlined /> View Logs
            </a-button>
            <a-button type="primary" @click="$router.push('/models')">
              <RobotOutlined /> Switch Model
            </a-button>
          </a-space>
        </a-col>
      </a-row>
    </a-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { Line, Doughnut, Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import {
  MessageOutlined,
  ThunderboltOutlined,
  CommentOutlined,
  SettingOutlined,
  FileTextOutlined,
  RobotOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const stats = ref({
  total_messages: 0,
  today_messages: 0,
  active_users: 0,
  conversations: 0,
  message_growth: 0,
  total_sessions: 0,
  avg_messages_per_thread: 0
})

const performanceStats = ref({
  avg_response_time_ms: 0,
  p95_response_time_ms: 0,
  total_llm_calls: 0
})

const recentMessages = ref([])
const config = ref({})
const loading = ref(false)
const systemStatus = ref('healthy')
const systemHealth = ref('Good')
const timeRange = ref('7')
const topUsers = ref([])
const latencyDistribution = ref([0, 0, 0, 0, 0])
const lastMessageTime = ref('N/A')

const messagesByTime = ref({ labels: [], data: [] })
const messagesByRole = ref({ labels: [], data: [] })

const topUsersColumns = [
  { title: 'User', key: 'user_id', dataIndex: 'user_id' },
  { title: 'Threads', dataIndex: 'thread_count', key: 'thread_count', width: 80 },
  { title: 'Messages', key: 'message_count', width: 150 }
]

// Chart data
const lineChartData = computed(() => {
  if (!messagesByTime.value.labels.length) return null
  return {
    labels: messagesByTime.value.labels.map(d => dayjs(d).format('MMM D')),
    datasets: [{
      label: 'Messages',
      data: messagesByTime.value.data,
      borderColor: '#7C6EE4',
      backgroundColor: 'rgba(124, 110, 228, 0.1)',
      fill: true,
      tension: 0.4,
      pointBackgroundColor: '#7C6EE4',
      pointBorderColor: '#fff',
      pointBorderWidth: 2,
      pointRadius: 4
    }]
  }
})

const lineChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1a1a2e',
      titleColor: '#fff',
      bodyColor: '#fff',
      padding: 12,
      cornerRadius: 8
    }
  },
  scales: {
    x: {
      grid: { display: false },
      ticks: { color: '#6b7280' }
    },
    y: {
      grid: { color: '#f0f0f0' },
      ticks: { color: '#6b7280' },
      beginAtZero: true
    }
  }
}

const doughnutChartData = computed(() => {
  if (!messagesByRole.value.labels.length) return null
  return {
    labels: messagesByRole.value.labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
    datasets: [{
      data: messagesByRole.value.data,
      backgroundColor: ['#7C6EE4', '#6B8DD6', '#52c41a', '#faad14'],
      borderWidth: 0,
      hoverOffset: 8
    }]
  }
})

const doughnutChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        padding: 20,
        usePointStyle: true,
        pointStyle: 'circle'
      }
    },
    tooltip: {
      backgroundColor: '#1a1a2e',
      padding: 12,
      cornerRadius: 8
    }
  },
  cutout: '65%'
}

const latencyChartData = computed(() => {
  return {
    labels: ['<100ms', '100-500ms', '500ms-1s', '1-2s', '>2s'],
    datasets: [{
      label: 'Requests',
      data: latencyDistribution.value,
      backgroundColor: [
        'rgba(17, 153, 142, 0.8)',
        'rgba(124, 110, 228, 0.8)',
        'rgba(250, 173, 20, 0.8)',
        'rgba(255, 119, 0, 0.8)',
        'rgba(255, 77, 79, 0.8)'
      ],
      borderRadius: 4
    }]
  }
})

const barChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false }
  },
  scales: {
    y: {
      beginAtZero: true,
      grid: { color: '#f0f0f0' }
    },
    x: {
      grid: { display: false }
    }
  }
}

// Methods
const formatNumber = (num) => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K'
  return num?.toString() || '0'
}

const truncate = (text, length) => {
  if (!text) return ''
  return text.length > length ? text.substring(0, length) + '...' : text
}

const formatTime = (time) => {
  if (!time) return ''
  return dayjs(time).fromNow()
}

const getRoleColor = (role) => {
  const colors = {
    user: '#1890ff',
    assistant: '#7C6EE4',
    system: '#fa8c16',
    tool: '#52c41a'
  }
  return colors[role] || '#7C6EE4'
}

const getUserColor = (userId) => {
  const colors = ['#7C6EE4', '#1890ff', '#52c41a', '#faad14', '#eb2f96']
  const hash = userId ? userId.charCodeAt(0) % colors.length : 0
  return colors[hash]
}

const getProgressPercent = (count) => {
  const max = topUsers.value.length ? Math.max(...topUsers.value.map(u => u.message_count)) : 1
  return (count / max) * 100
}

const getHealthClass = (health) => {
  if (health === 'Excellent' || health === 'Good') return 'health-good'
  if (health === 'Fair') return 'health-fair'
  return 'health-bad'
}

const fetchData = async () => {
  loading.value = true
  try {
    const [statsData, messagesData, configData, timeData, roleData, perfData] = await Promise.all([
      dashboardApi.getStats(),
      dashboardApi.getMessages({ page_size: 5 }),
      dashboardApi.getConfig(),
      dashboardApi.getMessagesByTime(parseInt(timeRange.value)),
      dashboardApi.getMessagesByRole(),
      dashboardApi.getPerformanceStats({ range: '24h' }).catch(() => ({ stats: {}, user_performance: [], latency_distribution: [0,0,0,0,0] }))
    ])

    stats.value = {
      ...statsData,
      message_growth: 12, // Placeholder - would calculate from historical data
      total_sessions: statsData.conversations || 0,
      avg_messages_per_thread: statsData.total_messages && statsData.conversations
        ? Math.round(statsData.total_messages / statsData.conversations)
        : 0
    }
    recentMessages.value = messagesData.messages || []
    config.value = configData
    messagesByTime.value = timeData
    messagesByRole.value = roleData

    // Performance data
    performanceStats.value = perfData.stats || {}
    topUsers.value = (perfData.user_performance || []).slice(0, 5)
    latencyDistribution.value = perfData.latency_distribution || [0,0,0,0,0]

    // Last message time
    if (recentMessages.value.length) {
      lastMessageTime.value = formatTime(recentMessages.value[0].created_at)
    }

    // Calculate system health based on metrics
    const avgLatency = performanceStats.value.avg_response_time_ms || 0
    if (avgLatency < 500) systemHealth.value = 'Excellent'
    else if (avgLatency < 1000) systemHealth.value = 'Good'
    else if (avgLatency < 2000) systemHealth.value = 'Fair'
    else systemHealth.value = 'Poor'

  } catch (err) {
    console.error('Failed to fetch dashboard data:', err)
  } finally {
    loading.value = false
  }
}

watch(timeRange, () => {
  dashboardApi.getMessagesByTime(parseInt(timeRange.value)).then(data => {
    messagesByTime.value = data
  })
})

onMounted(() => {
  fetchData()
})
</script>

<style scoped>
.overview-page {
  padding-bottom: 24px;
}

.stats-row {
  margin-bottom: 24px;
}

.charts-row {
  margin-bottom: 24px;
}

.stat-card {
  background: var(--card-bg);
  padding: 16px 20px;
  border-radius: 10px;
  box-shadow: var(--shadow);
  text-align: center;
}

.stat-title {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.2;
}

.stat-unit {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-left: 2px;
}

.stat-change {
  font-size: 11px;
  font-weight: 500;
  margin-top: 6px;
}

.stat-change.positive {
  color: #52c41a;
}

.stat-change.negative {
  color: #ff4d4f;
}

.stat-change.neutral {
  color: var(--text-secondary);
}

.health-good {
  color: #52c41a !important;
}

.health-fair {
  color: #faad14 !important;
}

.health-bad {
  color: #ff4d4f !important;
}

.chart-container {
  height: 300px;
  position: relative;
}

.doughnut-container {
  height: 280px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.message-preview {
  font-weight: 500;
  color: var(--text-primary);
}

.message-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.message-time {
  color: var(--text-secondary);
  font-size: 12px;
}

.quick-actions h4 {
  color: var(--text-primary);
}
</style>
