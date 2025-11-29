<template>
  <a-config-provider
    :theme="{
      token: {
        colorPrimary: '#7C6EE4',
        borderRadius: 8,
      },
    }"
  >
    <a-layout class="app-layout">
      <a-layout-sider
        v-model:collapsed="collapsed"
        :trigger="null"
        collapsible
        class="app-sider"
        width="240"
      >
        <div class="logo-container">
          <div class="logo">
            <div class="logo-icon">M</div>
            <span v-if="!collapsed" class="logo-text">Mindcore</span>
          </div>
        </div>
        <a-menu
          v-model:selectedKeys="selectedKeys"
          theme="dark"
          mode="inline"
        >
          <a-menu-item key="overview" @click="navigate('/')">
            <template #icon><DashboardOutlined /></template>
            <span>Overview</span>
          </a-menu-item>
          <a-menu-item key="performance" @click="navigate('/performance')">
            <template #icon><ThunderboltOutlined /></template>
            <span>Performance</span>
          </a-menu-item>
          <a-menu-item key="messages" @click="navigate('/messages')">
            <template #icon><MessageOutlined /></template>
            <span>Messages</span>
          </a-menu-item>
          <a-menu-item key="threads" @click="navigate('/threads')">
            <template #icon><CommentOutlined /></template>
            <span>Threads</span>
          </a-menu-item>
          <a-menu-item key="sessions" @click="navigate('/sessions')">
            <template #icon><DesktopOutlined /></template>
            <span>Sessions</span>
          </a-menu-item>
          <a-menu-item key="logs" @click="navigate('/logs')">
            <template #icon><FileTextOutlined /></template>
            <span>Logs</span>
          </a-menu-item>
          <a-menu-item key="models" @click="navigate('/models')">
            <template #icon><RobotOutlined /></template>
            <span>Models</span>
          </a-menu-item>
          <a-menu-item key="config" @click="navigate('/config')">
            <template #icon><SettingOutlined /></template>
            <span>Configuration</span>
          </a-menu-item>
        </a-menu>
      </a-layout-sider>
      <a-layout>
        <a-layout-header class="app-header">
          <div style="display: flex; align-items: center; gap: 16px;">
            <a-button type="text" @click="collapsed = !collapsed">
              <MenuUnfoldOutlined v-if="collapsed" />
              <MenuFoldOutlined v-else />
            </a-button>
            <span class="header-title">{{ currentPageTitle }}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 16px;">
            <a-badge :count="unreadLogs" :overflow-count="99">
              <a-button type="text" @click="navigate('/logs')">
                <BellOutlined style="font-size: 18px;" />
              </a-button>
            </a-badge>
            <a-dropdown>
              <a-avatar style="background-color: #7C6EE4; cursor: pointer;">
                <template #icon><UserOutlined /></template>
              </a-avatar>
              <template #overlay>
                <a-menu>
                  <a-menu-item key="profile">
                    <UserOutlined /> Profile
                  </a-menu-item>
                  <a-menu-item key="settings" @click="navigate('/config')">
                    <SettingOutlined /> Settings
                  </a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>
          </div>
        </a-layout-header>
        <a-layout-content class="app-content">
          <router-view />
        </a-layout-content>
      </a-layout>
    </a-layout>
  </a-config-provider>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  MessageOutlined,
  CommentOutlined,
  DesktopOutlined,
  FileTextOutlined,
  SettingOutlined,
  RobotOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BellOutlined,
  UserOutlined
} from '@ant-design/icons-vue'

const router = useRouter()
const route = useRoute()

const collapsed = ref(false)
const selectedKeys = ref(['overview'])
const unreadLogs = ref(0)

const pageMap = {
  '/': { key: 'overview', title: 'Overview' },
  '/performance': { key: 'performance', title: 'Performance' },
  '/messages': { key: 'messages', title: 'Messages' },
  '/threads': { key: 'threads', title: 'Threads' },
  '/sessions': { key: 'sessions', title: 'Sessions' },
  '/logs': { key: 'logs', title: 'System Logs' },
  '/models': { key: 'models', title: 'Models' },
  '/config': { key: 'config', title: 'Configuration' }
}

const currentPageTitle = computed(() => {
  const path = route.path
  if (path.startsWith('/threads/')) return 'Thread Detail'
  return pageMap[path]?.title || 'Mindcore'
})

watch(() => route.path, (path) => {
  if (path.startsWith('/threads/')) {
    selectedKeys.value = ['threads']
  } else {
    const page = pageMap[path]
    if (page) {
      selectedKeys.value = [page.key]
    }
  }
}, { immediate: true })

const navigate = (path) => {
  router.push(path)
}
</script>

<style scoped>
.app-sider {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
}

.app-layout > .ant-layout {
  margin-left: 240px;
  transition: margin-left 0.2s;
}

.app-layout > .ant-layout-sider-collapsed + .ant-layout {
  margin-left: 80px;
}
</style>
