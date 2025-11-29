<template>
  <div class="performance-page fade-in">
    <!-- Overview Stats -->
    <a-row :gutter="[16, 16]">
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Avg Response Time</div>
          <div class="stat-value">{{ stats.avg_response_time_ms || 0 }}<span class="stat-unit">ms</span></div>
          <div class="stat-change" :class="stats.latency_change <= 0 ? 'positive' : 'negative'">
            {{ stats.latency_change <= 0 ? '' : '+' }}{{ stats.latency_change || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">P95 Latency</div>
          <div class="stat-value">{{ stats.p95_response_time_ms || 0 }}<span class="stat-unit">ms</span></div>
          <div class="stat-change" :class="stats.p95_change <= 0 ? 'positive' : 'negative'">
            {{ stats.p95_change <= 0 ? '' : '+' }}{{ stats.p95_change || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Avg Enrichment</div>
          <div class="stat-value">{{ stats.avg_enrichment_time_ms || 0 }}<span class="stat-unit">ms</span></div>
          <div class="stat-change" :class="stats.enrichment_change <= 0 ? 'positive' : 'negative'">
            {{ stats.enrichment_change <= 0 ? '' : '+' }}{{ stats.enrichment_change || 0 }}% d/d
          </div>
        </div>
      </a-col>
      <a-col :xs="12" :sm="6">
        <div class="stat-card">
          <div class="stat-title">Total LLM Calls</div>
          <div class="stat-value">{{ stats.total_llm_calls || 0 }}</div>
          <div class="stat-change" :class="stats.calls_growth >= 0 ? 'positive' : 'negative'">
            {{ stats.calls_growth >= 0 ? '+' : '' }}{{ stats.calls_growth || 0 }}% d/d
          </div>
        </div>
      </a-col>
    </a-row>

    <!-- Charts Row -->
    <a-row :gutter="[24, 24]" style="margin-top: 24px;">
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Response Time Trend">
          <template #extra>
            <a-radio-group v-model:value="timeRange" size="small" @change="fetchPerformanceData">
              <a-radio-button value="1h">1h</a-radio-button>
              <a-radio-button value="24h">24h</a-radio-button>
              <a-radio-button value="7d">7d</a-radio-button>
            </a-radio-group>
          </template>
          <div class="chart-container">
            <Line :data="responseTimeChartData" :options="lineChartOptions" />
          </div>
        </a-card>
      </a-col>
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Latency Distribution">
          <div class="chart-container">
            <Bar :data="latencyDistributionData" :options="barChartOptions" />
          </div>
        </a-card>
      </a-col>
    </a-row>

    <!-- Detailed Metrics -->
    <a-row :gutter="[24, 24]" style="margin-top: 24px;">
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Timing Breakdown">
          <a-descriptions :column="1" bordered size="small">
            <a-descriptions-item label="LLM Response Time">
              <a-progress
                :percent="getPercentage(stats.avg_llm_time_ms, stats.avg_total_time_ms)"
                :stroke-color="'#667eea'"
                size="small"
              />
              <span class="metric-value">{{ stats.avg_llm_time_ms || 0 }}ms</span>
            </a-descriptions-item>
            <a-descriptions-item label="Enrichment Time">
              <a-progress
                :percent="getPercentage(stats.avg_enrichment_time_ms, stats.avg_total_time_ms)"
                :stroke-color="'#11998e'"
                size="small"
              />
              <span class="metric-value">{{ stats.avg_enrichment_time_ms || 0 }}ms</span>
            </a-descriptions-item>
            <a-descriptions-item label="Retrieval Time">
              <a-progress
                :percent="getPercentage(stats.avg_retrieval_time_ms, stats.avg_total_time_ms)"
                :stroke-color="'#f093fb'"
                size="small"
              />
              <span class="metric-value">{{ stats.avg_retrieval_time_ms || 0 }}ms</span>
            </a-descriptions-item>
            <a-descriptions-item label="Storage Time">
              <a-progress
                :percent="getPercentage(stats.avg_storage_time_ms, stats.avg_total_time_ms)"
                :stroke-color="'#4facfe'"
                size="small"
              />
              <span class="metric-value">{{ stats.avg_storage_time_ms || 0 }}ms</span>
            </a-descriptions-item>
          </a-descriptions>
        </a-card>
      </a-col>
      <a-col :xs="24" :lg="12">
        <a-card class="dashboard-card" title="Tool Usage">
          <template #extra>
            <a-tag :color="toolStats.success_rate >= 90 ? 'green' : toolStats.success_rate >= 70 ? 'orange' : 'red'">
              {{ toolStats.success_rate || 0 }}% Success
            </a-tag>
          </template>
          <a-table
            :columns="toolColumns"
            :data-source="toolStats.tools || []"
            :pagination="false"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'success_rate'">
                <a-progress
                  :percent="record.success_rate"
                  :status="record.success_rate >= 90 ? 'success' : record.success_rate >= 70 ? 'normal' : 'exception'"
                  size="small"
                  :show-info="false"
                />
                <span>{{ record.success_rate }}%</span>
              </template>
              <template v-else-if="column.key === 'avg_time'">
                {{ record.avg_time_ms }}ms
              </template>
            </template>
          </a-table>
          <div v-if="!toolStats.tools?.length" class="empty-tools">
            <ToolOutlined class="empty-icon" />
            <span>No tool usage recorded yet</span>
          </div>
        </a-card>
      </a-col>
    </a-row>

    <!-- Per-User Performance -->
    <a-card class="dashboard-card" title="Performance by User" style="margin-top: 24px;">
      <a-table
        :columns="userPerfColumns"
        :data-source="userPerformance"
        :pagination="{ pageSize: 10 }"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'avg_response_time'">
            <span :class="getLatencyClass(record.avg_response_time_ms)">
              {{ record.avg_response_time_ms }}ms
            </span>
          </template>
          <template v-else-if="column.key === 'thread_count'">
            <a-badge :count="record.thread_count" :number-style="{ backgroundColor: '#7C6EE4' }" />
          </template>
        </template>
      </a-table>
    </a-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import {
  ToolOutlined
} from '@ant-design/icons-vue'
import { Line, Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { dashboardApi } from '../api'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const timeRange = ref('24h')
const stats = ref({})
const toolStats = ref({})
const userPerformance = ref([])
const responseTimeTrend = ref([])
const latencyDistribution = ref([])

const toolColumns = [
  { title: 'Tool', dataIndex: 'name', key: 'name' },
  { title: 'Calls', dataIndex: 'call_count', key: 'call_count', width: 80 },
  { title: 'Success Rate', key: 'success_rate', width: 150 },
  { title: 'Avg Time', key: 'avg_time', width: 100 }
]

const userPerfColumns = [
  { title: 'User ID', dataIndex: 'user_id', key: 'user_id' },
  { title: 'Threads', key: 'thread_count', width: 100 },
  { title: 'Messages', dataIndex: 'message_count', key: 'message_count', width: 100 },
  { title: 'Avg Response', key: 'avg_response_time', width: 120 },
  { title: 'Total Time', dataIndex: 'total_time_ms', key: 'total_time', width: 120 }
]

const responseTimeChartData = computed(() => ({
  labels: responseTimeTrend.value.map(d => d.time),
  datasets: [
    {
      label: 'Response Time (ms)',
      data: responseTimeTrend.value.map(d => d.value),
      borderColor: '#7C6EE4',
      backgroundColor: 'rgba(124, 110, 228, 0.1)',
      fill: true,
      tension: 0.4
    }
  ]
}))

const latencyDistributionData = computed(() => ({
  labels: ['<100ms', '100-500ms', '500ms-1s', '1-2s', '>2s'],
  datasets: [
    {
      label: 'Requests',
      data: latencyDistribution.value,
      backgroundColor: [
        'rgba(17, 153, 142, 0.8)',
        'rgba(124, 110, 228, 0.8)',
        'rgba(250, 173, 20, 0.8)',
        'rgba(255, 119, 0, 0.8)',
        'rgba(255, 77, 79, 0.8)'
      ]
    }
  ]
}))

const lineChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false }
  },
  scales: {
    y: {
      beginAtZero: true,
      title: { display: true, text: 'ms' }
    }
  }
}

const barChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false }
  }
}

const getPercentage = (value, total) => {
  if (!total || !value) return 0
  return Math.round((value / total) * 100)
}

const getLatencyClass = (ms) => {
  if (ms < 500) return 'latency-good'
  if (ms < 1000) return 'latency-ok'
  return 'latency-bad'
}

const fetchPerformanceData = async () => {
  try {
    // Fetch performance stats
    const data = await dashboardApi.getPerformanceStats({ range: timeRange.value })
    stats.value = data.stats || {}
    toolStats.value = data.tool_stats || {}
    userPerformance.value = data.user_performance || []
    responseTimeTrend.value = data.response_time_trend || []
    latencyDistribution.value = data.latency_distribution || [0, 0, 0, 0, 0]
  } catch (err) {
    console.error('Failed to fetch performance data:', err)
    // Set mock data for demo
    stats.value = {
      avg_response_time_ms: 342,
      p95_response_time_ms: 890,
      avg_enrichment_time_ms: 45,
      total_llm_calls: 156,
      avg_llm_time_ms: 280,
      avg_retrieval_time_ms: 12,
      avg_storage_time_ms: 5,
      avg_total_time_ms: 342
    }
    toolStats.value = {
      success_rate: 94,
      tools: []
    }
    responseTimeTrend.value = [
      { time: '00:00', value: 320 },
      { time: '04:00', value: 380 },
      { time: '08:00', value: 290 },
      { time: '12:00', value: 450 },
      { time: '16:00', value: 320 },
      { time: '20:00', value: 310 }
    ]
    latencyDistribution.value = [45, 32, 15, 6, 2]
  }
}

onMounted(() => {
  fetchPerformanceData()
})
</script>

<style scoped>
.performance-page {
  padding-bottom: 24px;
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

.chart-container {
  height: 300px;
}

.metric-value {
  margin-left: 12px;
  font-weight: 600;
  color: var(--text-primary);
}

.empty-tools {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
  color: var(--text-secondary);
}

.empty-icon {
  font-size: 32px;
  margin-bottom: 8px;
  color: var(--primary-light);
}

.latency-good {
  color: #11998e;
  font-weight: 600;
}

.latency-ok {
  color: #faad14;
  font-weight: 600;
}

.latency-bad {
  color: #ff4d4f;
  font-weight: 600;
}
</style>
