<template>
  <div class="models-page fade-in">
    <!-- Active Model Banner -->
    <a-card class="dashboard-card active-model-card">
      <a-row align="middle" :gutter="24">
        <a-col :flex="'auto'">
          <div class="active-model-info">
            <h3>Active Model</h3>
            <p>{{ activeModel?.model || 'No model selected' }}</p>
          </div>
        </a-col>
        <a-col>
          <a-tag :color="getProviderColor(activeModel?.provider)" size="large">
            {{ activeModel?.provider || 'N/A' }}
          </a-tag>
        </a-col>
      </a-row>
    </a-card>

    <!-- Cloud Models -->
    <div class="section-header">
      <CloudOutlined class="section-icon" />
      <h2>Cloud Models (OpenAI)</h2>
    </div>
    <a-row :gutter="[24, 24]">
      <a-col v-for="model in models.cloud" :key="model.id" :xs="24" :sm="12" :lg="6">
        <div
          :class="['model-card', { selected: selectedModel === model.id }]"
          @click="selectModel(model.id)"
        >
          <div class="model-card-icon cloud">
            <ThunderboltOutlined />
          </div>
          <div class="model-card-name">{{ model.name }}</div>
          <div class="model-card-desc">{{ model.description }}</div>
          <div class="model-card-footer">
            <a-tag color="blue">Cloud</a-tag>
            <CheckCircleFilled v-if="selectedModel === model.id" class="check-icon" />
          </div>
        </div>
      </a-col>
    </a-row>

    <!-- Local Models -->
    <div class="section-header">
      <DesktopOutlined class="section-icon" />
      <h2>Local Models (llama.cpp)</h2>
    </div>
    <a-row :gutter="[24, 24]">
      <a-col v-for="model in models.local" :key="model.id" :xs="24" :sm="12" :lg="6">
        <div
          :class="['model-card', { selected: selectedModel === model.id, unavailable: !model.available }]"
          @click="model.available ? selectModel(model.id) : downloadModel(model.id)"
        >
          <div class="model-card-icon local">
            <RobotOutlined />
          </div>
          <div class="model-card-name">{{ model.name }}</div>
          <div class="model-card-desc">{{ model.description }}</div>
          <div class="model-card-meta">
            <HddOutlined /> {{ model.size }}
          </div>
          <div class="model-card-footer">
            <a-tag :color="model.available ? 'green' : 'default'">
              {{ model.available ? 'Available' : 'Not Downloaded' }}
            </a-tag>
            <template v-if="model.available">
              <CheckCircleFilled v-if="selectedModel === model.id" class="check-icon" />
            </template>
            <a-button v-else type="link" size="small" @click.stop="downloadModel(model.id)">
              <DownloadOutlined /> Download
            </a-button>
          </div>
        </div>
      </a-col>
    </a-row>

    <!-- Download Section -->
    <a-card class="dashboard-card download-card">
      <a-row align="middle" :gutter="24">
        <a-col :flex="'auto'">
          <h3>Download Models via CLI</h3>
          <p>Use the Mindcore CLI to download local models:</p>
          <div class="cli-commands">
            <code>mindcore download-model</code>
            <code>mindcore download-model -m qwen2.5-3b</code>
            <code>mindcore list-models -v</code>
          </div>
        </a-col>
        <a-col>
          <a-button type="primary" size="large" @click="openDocs">
            <BookOutlined /> View Documentation
          </a-button>
        </a-col>
      </a-row>
    </a-card>

    <!-- Apply Button -->
    <div class="actions-bar" v-if="selectedModel !== activeModel?.model">
      <a-space>
        <span class="change-notice">
          Model will change from <strong>{{ activeModel?.model }}</strong> to <strong>{{ selectedModel }}</strong>
        </span>
        <a-button type="primary" :loading="applying" @click="applyModel">
          <CheckOutlined /> Apply Changes
        </a-button>
      </a-space>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  CloudOutlined,
  DesktopOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  HddOutlined,
  DownloadOutlined,
  CheckCircleFilled,
  BookOutlined,
  CheckOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'

const models = reactive({
  cloud: [],
  local: []
})
const activeModel = ref(null)
const selectedModel = ref(null)
const applying = ref(false)

const getProviderColor = (provider) => {
  if (!provider) return 'default'
  if (provider.includes('Llama') || provider.includes('llama')) return 'green'
  if (provider.includes('OpenAI') || provider.includes('openai')) return 'blue'
  if (provider.includes('Fallback')) return 'purple'
  return 'default'
}

const fetchModels = async () => {
  try {
    const data = await dashboardApi.getModels()
    models.cloud = data.cloud || []
    models.local = data.local || []

    // Also get active model
    const active = await dashboardApi.getActiveModel()
    activeModel.value = active
    selectedModel.value = active?.model
  } catch (err) {
    console.error('Failed to fetch models:', err)
  }
}

const selectModel = (modelId) => {
  selectedModel.value = modelId
}

const downloadModel = async (modelId) => {
  try {
    await dashboardApi.downloadModel(modelId)
    message.info(`To download ${modelId}, run: mindcore download-model -m ${modelId}`)
  } catch (err) {
    console.error('Failed to trigger download:', err)
  }
}

const applyModel = async () => {
  applying.value = true
  try {
    await dashboardApi.setActiveModel(selectedModel.value)
    message.success('Model changed successfully')
    activeModel.value = { model: selectedModel.value }
  } catch (err) {
    console.error('Failed to apply model:', err)
    message.error('Failed to change model')
  } finally {
    applying.value = false
  }
}

const openDocs = () => {
  window.open('https://github.com/M-Alfaris/mindcore#-local-llm-setup', '_blank')
}

onMounted(() => {
  fetchModels()
})
</script>

<style scoped>
.models-page {
  padding-bottom: 24px;
}

.active-model-card {
  margin-bottom: 32px;
  background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
  color: #fff;
}

.active-model-card h3 {
  color: rgba(255, 255, 255, 0.7);
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.active-model-card p {
  color: #fff;
  margin: 8px 0 0;
  font-size: 24px;
  font-weight: 600;
}

.active-model-card :deep(.ant-tag) {
  font-size: 14px;
  padding: 4px 12px;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 32px 0 16px;
}

.section-icon {
  font-size: 24px;
  color: var(--primary-color);
}

.section-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.model-card {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 24px;
  border: 2px solid var(--border-color);
  cursor: pointer;
  transition: all 0.3s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.model-card:hover {
  border-color: var(--primary-light);
  box-shadow: var(--shadow-hover);
  transform: translateY(-2px);
}

.model-card.selected {
  border-color: var(--primary-color);
  background: var(--primary-bg);
}

.model-card.unavailable {
  opacity: 0.7;
}

.model-card-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  margin-bottom: 16px;
}

.model-card-icon.cloud {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
}

.model-card-icon.local {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  color: #fff;
}

.model-card-name {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.model-card-desc {
  font-size: 13px;
  color: var(--text-secondary);
  flex: 1;
}

.model-card-meta {
  font-size: 12px;
  color: var(--text-secondary);
  margin: 8px 0;
  display: flex;
  align-items: center;
  gap: 4px;
}

.model-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
}

.check-icon {
  color: var(--primary-color);
  font-size: 20px;
}

.download-card {
  margin-top: 32px;
}

.download-card h3 {
  margin: 0 0 8px;
  font-size: 16px;
  font-weight: 600;
}

.download-card p {
  margin: 0 0 16px;
  color: var(--text-secondary);
}

.cli-commands {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cli-commands code {
  background: var(--bg-color);
  padding: 8px 16px;
  border-radius: 6px;
  font-family: 'SF Mono', monospace;
  font-size: 13px;
  color: var(--primary-color);
}

.actions-bar {
  margin-top: 24px;
  padding: 16px 24px;
  background: var(--card-bg);
  border-radius: 12px;
  box-shadow: var(--shadow);
  display: flex;
  justify-content: flex-end;
  position: sticky;
  bottom: 24px;
}

.change-notice {
  color: var(--text-secondary);
  font-size: 14px;
}
</style>
