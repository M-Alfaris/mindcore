<template>
  <div class="threads-page fade-in">
    <!-- Header -->
    <a-card class="dashboard-card filter-card">
      <a-row :gutter="16" align="middle">
        <a-col :xs="24" :sm="12">
          <a-input-search
            v-model:value="searchQuery"
            placeholder="Search by user ID..."
            allow-clear
            @search="handleSearch"
          />
        </a-col>
        <a-col :xs="24" :sm="12" style="text-align: right;">
          <span class="total-count">{{ total }} threads total</span>
        </a-col>
      </a-row>
    </a-card>

    <!-- Threads List -->
    <a-row :gutter="[24, 24]">
      <a-col v-for="thread in threads" :key="thread.thread_id" :xs="24" :sm="12" :lg="8">
        <a-card
          class="dashboard-card thread-card"
          hoverable
          @click="viewThread(thread.thread_id)"
        >
          <template #title>
            <div class="thread-header">
              <CommentOutlined class="thread-icon" />
              <span class="thread-title">Thread</span>
            </div>
          </template>
          <template #extra>
            <a-badge
              :count="thread.message_count"
              :number-style="{ backgroundColor: '#7C6EE4' }"
            />
          </template>

          <div class="thread-id-full">
            <a-tooltip :title="thread.thread_id">
              <code>{{ thread.thread_id }}</code>
            </a-tooltip>
            <a-button type="text" size="small" @click.stop="copyThreadId(thread.thread_id)">
              <CopyOutlined />
            </a-button>
          </div>

          <div class="thread-info">
            <div class="thread-stat">
              <UserOutlined />
              <span><strong>User:</strong> {{ thread.user_id || 'anonymous' }}</span>
            </div>
            <div class="thread-stat" v-if="thread.session_id">
              <DesktopOutlined />
              <span><strong>Session:</strong> {{ truncate(thread.session_id, 12) }}</span>
            </div>
            <div class="thread-stat">
              <MessageOutlined />
              <span><strong>Messages:</strong> {{ thread.message_count }}</span>
            </div>
            <div class="thread-stat">
              <ClockCircleOutlined />
              <span><strong>Started:</strong> {{ formatTime(thread.first_message_at) }}</span>
            </div>
            <div class="thread-stat">
              <ClockCircleOutlined />
              <span><strong>Last:</strong> {{ formatTime(thread.last_message_at) }}</span>
            </div>
            <div class="thread-stat" v-if="thread.avg_response_time_ms">
              <ThunderboltOutlined />
              <span><strong>Avg Response:</strong> {{ thread.avg_response_time_ms }}ms</span>
            </div>
          </div>

          <div class="thread-roles" v-if="thread.role_breakdown">
            <a-tag v-for="(count, role) in thread.role_breakdown" :key="role" :class="['role-tag', role]">
              {{ role }}: {{ count }}
            </a-tag>
          </div>

          <template #actions>
            <a-tooltip title="View Messages">
              <EyeOutlined @click.stop="viewThread(thread.thread_id)" />
            </a-tooltip>
            <a-tooltip title="Copy Thread ID">
              <CopyOutlined @click.stop="copyThreadId(thread.thread_id)" />
            </a-tooltip>
          </template>
        </a-card>
      </a-col>
    </a-row>

    <!-- Empty State -->
    <a-card v-if="!loading && threads.length === 0" class="dashboard-card">
      <div class="empty-state">
        <CommentOutlined class="empty-state-icon" />
        <div class="empty-state-title">No threads found</div>
        <div class="empty-state-desc">Conversation threads will appear here</div>
      </div>
    </a-card>

    <!-- Pagination -->
    <div v-if="total > pageSize" class="pagination-container">
      <a-pagination
        v-model:current="currentPage"
        :total="total"
        :page-size="pageSize"
        :show-size-changer="false"
        @change="handlePageChange"
      />
    </div>

    <!-- Loading -->
    <div v-if="loading" class="loading-container">
      <a-spin size="large" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  CommentOutlined,
  UserOutlined,
  MessageOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  CopyOutlined,
  DesktopOutlined,
  ThunderboltOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const router = useRouter()

const threads = ref([])
const loading = ref(false)
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(12)
const searchQuery = ref('')

const truncate = (text, length) => {
  if (!text) return ''
  return text.length > length ? text.substring(0, length) + '...' : text
}

const formatTime = (time) => {
  if (!time) return 'N/A'
  return dayjs(time).fromNow()
}

const fetchThreads = async () => {
  loading.value = true
  try {
    const params = {
      page: currentPage.value,
      page_size: pageSize.value
    }
    if (searchQuery.value) {
      params.user_id = searchQuery.value
    }

    const data = await dashboardApi.getThreads(params)
    threads.value = data.threads || []
    total.value = data.total || 0
  } catch (err) {
    console.error('Failed to fetch threads:', err)
    message.error('Failed to load threads')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  currentPage.value = 1
  fetchThreads()
}

const handlePageChange = (page) => {
  currentPage.value = page
  fetchThreads()
}

const viewThread = (threadId) => {
  router.push(`/threads/${threadId}`)
}

const copyThreadId = async (threadId) => {
  try {
    await navigator.clipboard.writeText(threadId)
    message.success('Thread ID copied')
  } catch {
    message.error('Failed to copy')
  }
}

onMounted(() => {
  fetchThreads()
})
</script>

<style scoped>
.threads-page {
  padding-bottom: 24px;
}

.filter-card {
  margin-bottom: 24px;
}

.total-count {
  color: var(--text-secondary);
  font-size: 14px;
}

.thread-card {
  cursor: pointer;
  transition: all 0.3s ease;
}

.thread-card:hover {
  transform: translateY(-4px);
}

.thread-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.thread-icon {
  color: var(--primary-color);
  font-size: 18px;
}

.thread-title {
  font-weight: 600;
  font-size: 14px;
}

.thread-id-full {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.thread-id-full code {
  font-size: 11px;
  background: var(--bg-color);
  padding: 4px 8px;
  border-radius: 4px;
  color: var(--primary-color);
  word-break: break-all;
  flex: 1;
}

.thread-roles {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
}

.thread-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.thread-stat {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 13px;
}

.thread-stat .anticon {
  color: var(--primary-light);
}

.pagination-container {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

.loading-container {
  display: flex;
  justify-content: center;
  padding: 60px;
}
</style>
