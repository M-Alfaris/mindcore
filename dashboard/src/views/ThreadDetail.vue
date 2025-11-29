<template>
  <div class="thread-detail-page fade-in">
    <!-- Header -->
    <a-card class="dashboard-card header-card">
      <a-row align="middle" justify="space-between">
        <a-col>
          <a-button type="text" @click="goBack">
            <ArrowLeftOutlined /> Back to Threads
          </a-button>
        </a-col>
        <a-col>
          <a-space>
            <span class="thread-label">Thread:</span>
            <code class="thread-id">{{ threadId }}</code>
            <a-button size="small" @click="copyThreadId">
              <CopyOutlined />
            </a-button>
          </a-space>
        </a-col>
      </a-row>
    </a-card>

    <!-- Messages -->
    <a-card class="dashboard-card messages-card">
      <template #title>
        <span>Conversation</span>
        <a-badge :count="total" :number-style="{ backgroundColor: '#7C6EE4', marginLeft: '8px' }" />
      </template>

      <div class="messages-container" ref="messagesContainer">
        <template v-if="messages.length">
          <div
            v-for="msg in messages"
            :key="msg.message_id"
            :class="['message-bubble', msg.role]"
          >
            <div class="message-header">
              <a-avatar
                :style="{ backgroundColor: getRoleColor(msg.role) }"
                size="small"
              >
                {{ msg.role[0].toUpperCase() }}
              </a-avatar>
              <span class="message-role">{{ msg.role }}</span>
              <span class="message-time">{{ formatTime(msg.created_at) }}</span>
            </div>
            <div class="message-content">{{ msg.raw_text }}</div>
            <div class="message-meta" v-if="msg.metadata?.topics?.length">
              <a-tag
                v-for="topic in msg.metadata.topics.slice(0, 3)"
                :key="topic"
                size="small"
                color="purple"
              >
                {{ topic }}
              </a-tag>
            </div>
          </div>
        </template>

        <div v-else-if="!loading" class="empty-state">
          <MessageOutlined class="empty-state-icon" />
          <div class="empty-state-title">No messages in this thread</div>
        </div>
      </div>

      <!-- Load More -->
      <div v-if="hasMore" class="load-more">
        <a-button :loading="loading" @click="loadMore">
          Load More Messages
        </a-button>
      </div>
    </a-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined,
  CopyOutlined,
  MessageOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const router = useRouter()
const route = useRoute()

const threadId = computed(() => route.params.threadId)
const messages = ref([])
const loading = ref(false)
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(50)

const hasMore = computed(() => messages.value.length < total.value)

const formatTime = (time) => {
  if (!time) return ''
  return dayjs(time).format('MMM D, h:mm A')
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

const fetchMessages = async (append = false) => {
  loading.value = true
  try {
    const data = await dashboardApi.getThreadMessages(threadId.value, {
      page: currentPage.value,
      page_size: pageSize.value
    })

    const newMessages = data.messages || []
    if (append) {
      messages.value = [...messages.value, ...newMessages]
    } else {
      // Reverse to show oldest first
      messages.value = newMessages.reverse()
    }
    total.value = data.total || 0
  } catch (err) {
    console.error('Failed to fetch thread messages:', err)
    message.error('Failed to load messages')
  } finally {
    loading.value = false
  }
}

const loadMore = () => {
  currentPage.value++
  fetchMessages(true)
}

const goBack = () => {
  router.push('/threads')
}

const copyThreadId = async () => {
  try {
    await navigator.clipboard.writeText(threadId.value)
    message.success('Thread ID copied')
  } catch {
    message.error('Failed to copy')
  }
}

onMounted(() => {
  fetchMessages()
})
</script>

<style scoped>
.thread-detail-page {
  padding-bottom: 24px;
}

.header-card {
  margin-bottom: 24px;
}

.thread-label {
  color: var(--text-secondary);
}

.thread-id {
  background: var(--primary-bg);
  padding: 4px 12px;
  border-radius: 6px;
  color: var(--primary-color);
  font-size: 13px;
}

.messages-card :deep(.ant-card-head-title) {
  display: flex;
  align-items: center;
}

.messages-container {
  max-height: 600px;
  overflow-y: auto;
  padding: 16px 0;
}

.message-bubble {
  margin-bottom: 20px;
  padding: 16px;
  border-radius: 12px;
  background: var(--bg-color);
  border-left: 3px solid var(--primary-color);
}

.message-bubble.user {
  border-left-color: #1890ff;
  background: #f0f7ff;
}

.message-bubble.assistant {
  border-left-color: #7C6EE4;
  background: var(--primary-bg);
}

.message-bubble.system {
  border-left-color: #fa8c16;
  background: #fff7e6;
}

.message-bubble.tool {
  border-left-color: #52c41a;
  background: #f6ffed;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.message-role {
  font-weight: 600;
  text-transform: capitalize;
  color: var(--text-primary);
}

.message-time {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-secondary);
}

.message-content {
  color: var(--text-primary);
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.message-meta {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.load-more {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}
</style>
