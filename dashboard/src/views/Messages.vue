<template>
  <div class="messages-page fade-in">
    <!-- Filters -->
    <a-card class="dashboard-card filter-card">
      <a-row :gutter="16" align="middle">
        <a-col :xs="24" :sm="8" :md="6">
          <a-input-search
            v-model:value="searchQuery"
            placeholder="Search messages..."
            allow-clear
            @search="handleSearch"
            @change="handleSearchChange"
          />
        </a-col>
        <a-col :xs="12" :sm="8" :md="4">
          <a-select
            v-model:value="roleFilter"
            placeholder="Filter by role"
            allow-clear
            style="width: 100%"
            @change="handleFilter"
          >
            <a-select-option value="user">User</a-select-option>
            <a-select-option value="assistant">Assistant</a-select-option>
            <a-select-option value="system">System</a-select-option>
            <a-select-option value="tool">Tool</a-select-option>
          </a-select>
        </a-col>
        <a-col :xs="12" :sm="8" :md="4">
          <a-button @click="resetFilters">
            <ReloadOutlined /> Reset
          </a-button>
        </a-col>
        <a-col :xs="24" :md="10" style="text-align: right;">
          <span class="total-count">{{ total }} messages total</span>
        </a-col>
      </a-row>
    </a-card>

    <!-- Messages Table -->
    <a-card class="dashboard-card table-card">
      <a-table
        :columns="columns"
        :data-source="messages"
        :loading="loading"
        :pagination="pagination"
        :row-key="record => record.message_id"
        @change="handleTableChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'message_id'">
            <a-tooltip :title="record.message_id">
              <code class="id-cell">{{ truncate(record.message_id, 8) }}</code>
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'session_id'">
            <a-tooltip :title="record.session_id || 'N/A'">
              <code class="id-cell">{{ record.session_id ? truncate(record.session_id, 8) : '-' }}</code>
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'thread_id'">
            <a-tooltip :title="record.thread_id">
              <a @click="goToThread(record.thread_id)" class="id-link">
                {{ truncate(record.thread_id, 8) }}
              </a>
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'role'">
            <a-tag :class="['role-tag', record.role]">
              {{ record.role }}
            </a-tag>
          </template>

          <template v-else-if="column.key === 'raw_text'">
            <a-tooltip :title="record.raw_text" placement="topLeft">
              <span class="message-text">{{ truncate(record.raw_text, 80) }}</span>
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'metadata'">
            <div class="metadata-tags">
              <template v-if="record.metadata?.topics?.length">
                <a-tag v-for="topic in record.metadata.topics.slice(0, 2)" :key="topic" color="purple">
                  {{ topic }}
                </a-tag>
                <a-tag v-if="record.metadata.topics.length > 2" color="default">
                  +{{ record.metadata.topics.length - 2 }}
                </a-tag>
              </template>
              <a-tag v-if="record.metadata?.intent" color="blue">
                {{ record.metadata.intent }}
              </a-tag>
            </div>
          </template>

          <template v-else-if="column.key === 'created_at'">
            <a-tooltip :title="formatDateTime(record.created_at)">
              {{ formatTime(record.created_at) }}
            </a-tooltip>
          </template>

          <template v-else-if="column.key === 'actions'">
            <a-space>
              <a-button type="text" size="small" @click="viewMessage(record)">
                <EyeOutlined />
              </a-button>
              <a-popconfirm
                title="Delete this message?"
                ok-text="Yes"
                cancel-text="No"
                @confirm="deleteMessage(record.message_id)"
              >
                <a-button type="text" size="small" danger>
                  <DeleteOutlined />
                </a-button>
              </a-popconfirm>
            </a-space>
          </template>
        </template>

        <template #emptyText>
          <div class="empty-state">
            <MessageOutlined class="empty-state-icon" />
            <div class="empty-state-title">No messages found</div>
            <div class="empty-state-desc">Messages will appear here once ingested</div>
          </div>
        </template>
      </a-table>
    </a-card>

    <!-- Message Detail Modal -->
    <a-modal
      v-model:open="detailModalVisible"
      title="Message Details"
      width="700px"
      :footer="null"
    >
      <a-descriptions :column="1" bordered v-if="selectedMessage">
        <a-descriptions-item label="Message ID">
          <code>{{ selectedMessage.message_id }}</code>
        </a-descriptions-item>
        <a-descriptions-item label="User ID">{{ selectedMessage.user_id }}</a-descriptions-item>
        <a-descriptions-item label="Thread ID">
          <a @click="goToThread(selectedMessage.thread_id)">{{ selectedMessage.thread_id }}</a>
        </a-descriptions-item>
        <a-descriptions-item label="Role">
          <a-tag :class="['role-tag', selectedMessage.role]">{{ selectedMessage.role }}</a-tag>
        </a-descriptions-item>
        <a-descriptions-item label="Created">{{ formatDateTime(selectedMessage.created_at) }}</a-descriptions-item>
        <a-descriptions-item label="Content">
          <div class="message-content">{{ selectedMessage.raw_text }}</div>
        </a-descriptions-item>
        <a-descriptions-item label="Metadata" v-if="selectedMessage.metadata">
          <div class="metadata-detail">
            <div v-if="selectedMessage.metadata.topics?.length">
              <strong>Topics:</strong>
              <a-tag v-for="topic in selectedMessage.metadata.topics" :key="topic" color="purple">{{ topic }}</a-tag>
            </div>
            <div v-if="selectedMessage.metadata.intent">
              <strong>Intent:</strong> {{ selectedMessage.metadata.intent }}
            </div>
            <div v-if="selectedMessage.metadata.sentiment">
              <strong>Sentiment:</strong> {{ selectedMessage.metadata.sentiment }}
            </div>
            <div v-if="selectedMessage.metadata.importance !== undefined">
              <strong>Importance:</strong>
              <a-progress :percent="selectedMessage.metadata.importance * 100" :show-info="false" size="small" style="width: 100px; display: inline-block; margin-left: 8px;" />
              {{ (selectedMessage.metadata.importance * 100).toFixed(0) }}%
            </div>
          </div>
        </a-descriptions-item>
      </a-descriptions>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ReloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  MessageOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const router = useRouter()

const messages = ref([])
const loading = ref(false)
const total = ref(0)
const searchQuery = ref('')
const roleFilter = ref(null)
const detailModalVisible = ref(false)
const selectedMessage = ref(null)

const pagination = reactive({
  current: 1,
  pageSize: 20,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true,
  pageSizeOptions: ['10', '20', '50', '100'],
  showTotal: (total) => `Total ${total} messages`
})

const columns = [
  {
    title: 'ID',
    key: 'message_id',
    width: 80,
    ellipsis: true
  },
  {
    title: 'Session',
    key: 'session_id',
    width: 100,
    ellipsis: true
  },
  {
    title: 'Thread',
    key: 'thread_id',
    width: 100,
    ellipsis: true
  },
  {
    title: 'Role',
    key: 'role',
    width: 90
  },
  {
    title: 'Message',
    key: 'raw_text',
    ellipsis: true
  },
  {
    title: 'User',
    dataIndex: 'user_id',
    key: 'user_id',
    width: 100,
    ellipsis: true
  },
  {
    title: 'Created',
    key: 'created_at',
    width: 100
  },
  {
    title: 'Actions',
    key: 'actions',
    width: 80,
    align: 'center'
  }
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

const fetchMessages = async () => {
  loading.value = true
  try {
    const params = {
      page: pagination.current,
      page_size: pagination.pageSize
    }
    if (roleFilter.value) params.role = roleFilter.value
    if (searchQuery.value) params.search = searchQuery.value

    const data = await dashboardApi.getMessages(params)
    messages.value = data.messages || []
    total.value = data.total || 0
    pagination.total = data.total || 0
  } catch (err) {
    console.error('Failed to fetch messages:', err)
    message.error('Failed to load messages')
  } finally {
    loading.value = false
  }
}

const handleTableChange = (pag) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  fetchMessages()
}

const handleSearch = () => {
  pagination.current = 1
  fetchMessages()
}

const handleSearchChange = (e) => {
  if (!e.target.value) {
    handleSearch()
  }
}

const handleFilter = () => {
  pagination.current = 1
  fetchMessages()
}

const resetFilters = () => {
  searchQuery.value = ''
  roleFilter.value = null
  pagination.current = 1
  fetchMessages()
}

const viewMessage = (record) => {
  selectedMessage.value = record
  detailModalVisible.value = true
}

const deleteMessage = async (id) => {
  try {
    await dashboardApi.deleteMessage(id)
    message.success('Message deleted')
    fetchMessages()
  } catch (err) {
    console.error('Failed to delete message:', err)
    message.error('Failed to delete message')
  }
}

const goToThread = (threadId) => {
  detailModalVisible.value = false
  router.push(`/threads/${threadId}`)
}

onMounted(() => {
  fetchMessages()
})
</script>

<style scoped>
.messages-page {
  padding-bottom: 24px;
}

.filter-card {
  margin-bottom: 24px;
}

.total-count {
  color: var(--text-secondary);
  font-size: 14px;
}

.message-text {
  color: var(--text-primary);
}

.metadata-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.message-content {
  background: var(--bg-color);
  padding: 12px;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
}

.metadata-detail {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metadata-detail > div {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.id-cell {
  font-size: 11px;
  background: var(--bg-color);
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--text-secondary);
}

.id-link {
  font-size: 11px;
  font-family: 'SF Mono', monospace;
  color: var(--primary-color);
  cursor: pointer;
}

.id-link:hover {
  text-decoration: underline;
}
</style>
