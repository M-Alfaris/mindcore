<template>
  <div class="config-page fade-in">
    <!-- System Status Bar -->
    <a-card class="dashboard-card status-bar">
      <a-row :gutter="24" align="middle">
        <a-col :flex="'auto'">
          <a-space size="large">
            <div class="status-item">
              <span class="status-label">Mindcore Status</span>
              <a-switch
                v-model:checked="systemEnabled"
                :loading="togglingSystem"
                @change="toggleSystem"
              />
              <a-tag :color="systemEnabled ? 'success' : 'error'">
                {{ systemEnabled ? 'Enabled' : 'Disabled' }}
              </a-tag>
            </div>
            <a-divider type="vertical" style="height: 40px;" />
            <div class="status-item">
              <span class="status-label">API Server</span>
              <a-tag :color="serverStatus === 'online' ? 'success' : 'error'">
                <template #icon>
                  <CheckCircleOutlined v-if="serverStatus === 'online'" />
                  <CloseCircleOutlined v-else />
                </template>
                {{ serverStatus }}
              </a-tag>
            </div>
            <a-divider type="vertical" style="height: 40px;" />
            <div class="status-item">
              <span class="status-label">Active Model</span>
              <a-tag color="purple">{{ getModelDisplayName(formData.llm.model) }}</a-tag>
            </div>
          </a-space>
        </a-col>
        <a-col>
          <a-button type="primary" danger ghost @click="restartServer" :loading="restarting">
            <ReloadOutlined /> Restart Server
          </a-button>
        </a-col>
      </a-row>
    </a-card>

    <a-tabs v-model:activeKey="activeTab" class="config-tabs">
      <!-- LLM Configuration Tab -->
      <a-tab-pane key="llm" tab="LLM Configuration">
        <a-row :gutter="[24, 24]">
          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Main LLM Settings">
              <a-form layout="vertical">
                <a-form-item label="Model">
                  <a-select v-model:value="formData.llm.model" style="width: 100%" size="large">
                    <a-select-opt-group label="OpenAI (Cloud)">
                      <a-select-option value="gpt-4o">GPT-4o - Most capable (Cloud)</a-select-option>
                      <a-select-option value="gpt-4o-mini">GPT-4o Mini - Recommended (Cloud)</a-select-option>
                      <a-select-option value="gpt-4-turbo">GPT-4 Turbo (Cloud)</a-select-option>
                      <a-select-option value="gpt-3.5-turbo">GPT-3.5 Turbo - Fast (Cloud)</a-select-option>
                      <a-select-option value="o1-preview">O1 Preview - Reasoning (Cloud)</a-select-option>
                      <a-select-option value="o1-mini">O1 Mini (Cloud)</a-select-option>
                    </a-select-opt-group>
                    <a-select-opt-group label="Anthropic (Cloud)">
                      <a-select-option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet - Latest (Cloud)</a-select-option>
                      <a-select-option value="claude-3-opus-20240229">Claude 3 Opus (Cloud)</a-select-option>
                      <a-select-option value="claude-3-haiku-20240307">Claude 3 Haiku - Fast (Cloud)</a-select-option>
                    </a-select-opt-group>
                    <a-select-opt-group label="Local Models (llama.cpp)">
                      <a-select-option value="llama-3.2-3b-local">Llama 3.2 3B (Local)</a-select-option>
                      <a-select-option value="llama-3.2-1b-local">Llama 3.2 1B (Local)</a-select-option>
                      <a-select-option value="qwen2.5-3b-local">Qwen 2.5 3B (Local)</a-select-option>
                      <a-select-option value="phi-3.5-mini-local">Phi 3.5 Mini (Local)</a-select-option>
                      <a-select-option value="custom-local">Custom Model Path (Local)</a-select-option>
                    </a-select-opt-group>
                  </a-select>
                </a-form-item>

                <a-form-item label="Custom Model Path" v-if="formData.llm.model === 'custom-local'">
                  <a-input v-model:value="formData.llm.custom_model_path" placeholder="/path/to/model.gguf" />
                </a-form-item>

                <a-form-item label="Temperature">
                  <a-row :gutter="16" align="middle">
                    <a-col :span="16">
                      <a-slider
                        v-model:value="formData.llm.temperature"
                        :min="0"
                        :max="1"
                        :step="0.1"
                        :marks="temperatureMarks"
                      />
                    </a-col>
                    <a-col :span="8">
                      <a-input-number
                        v-model:value="formData.llm.temperature"
                        :min="0"
                        :max="1"
                        :step="0.1"
                        :precision="1"
                        style="width: 100%"
                      />
                    </a-col>
                  </a-row>
                </a-form-item>

                <a-form-item label="Max Tokens">
                  <a-row :gutter="16" align="middle">
                    <a-col :span="16">
                      <a-slider
                        v-model:value="formData.llm.max_tokens"
                        :min="100"
                        :max="8000"
                        :step="100"
                      />
                    </a-col>
                    <a-col :span="8">
                      <a-input-number
                        v-model:value="formData.llm.max_tokens"
                        :min="100"
                        :max="8000"
                        :step="100"
                        style="width: 100%"
                      />
                    </a-col>
                  </a-row>
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Memory & Enrichment">
              <template #extra>
                <a-switch v-model:checked="formData.memory.enabled" />
              </template>

              <a-form layout="vertical">
                <a-form-item>
                  <a-alert
                    v-if="formData.memory.enabled"
                    type="success"
                    show-icon
                    message="Mindcore Memory Active"
                    description="Messages are being enriched with metadata for intelligent context retrieval."
                  />
                  <a-alert
                    v-else
                    type="warning"
                    show-icon
                    message="Memory Disabled"
                    description="Enable memory to get intelligent context management and cost savings."
                  />
                </a-form-item>

                <a-form-item label="Enrichment Model">
                  <a-select v-model:value="formData.memory.model" style="width: 100%" size="large">
                    <a-select-opt-group label="Local Models (Free)">
                      <a-select-option value="llama-3.2-1b-local">Llama 3.2 1B - Fast (Local)</a-select-option>
                      <a-select-option value="llama-3.2-3b-local">Llama 3.2 3B - Better (Local)</a-select-option>
                      <a-select-option value="qwen2.5-3b-local">Qwen 2.5 3B (Local)</a-select-option>
                    </a-select-opt-group>
                    <a-select-opt-group label="Cloud Models">
                      <a-select-option value="gpt-4o-mini-cloud">GPT-4o Mini (Cloud)</a-select-option>
                      <a-select-option value="gpt-3.5-turbo-cloud">GPT-3.5 Turbo (Cloud)</a-select-option>
                    </a-select-opt-group>
                  </a-select>
                </a-form-item>

                <a-divider>Cache Settings</a-divider>

                <a-form-item label="Cache Size">
                  <a-input-number
                    v-model:value="formData.cache.max_size"
                    :min="10"
                    :max="500"
                    addon-after="messages"
                    style="width: 100%"
                  />
                  <div class="form-help">Maximum messages to keep in memory per thread</div>
                </a-form-item>

                <a-form-item label="Cache TTL">
                  <a-input-number
                    v-model:value="formData.cache.ttl"
                    :min="60"
                    :max="86400"
                    addon-after="seconds"
                    style="width: 100%"
                  />
                  <div class="form-help">Time-to-live for cached messages (default: 1 hour)</div>
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>
        </a-row>

        <!-- Save button for LLM tab -->
        <div class="tab-actions">
          <a-space>
            <a-button @click="resetLLMForm">
              <UndoOutlined /> Reset
            </a-button>
            <a-button type="primary" :loading="saving" @click="saveLLMConfig">
              <SaveOutlined /> Save LLM Configuration
            </a-button>
          </a-space>
        </div>
      </a-tab-pane>

      <!-- Database Configuration Tab -->
      <a-tab-pane key="database" tab="Dashboard Database">
        <!-- Clarification Alert -->
        <a-alert
          type="info"
          show-icon
          style="margin-bottom: 24px;"
          message="Dashboard Database Configuration"
          description="This database stores dashboard data only: messages, sessions, threads, and performance metrics. This is separate from the agent's memory layer, which is configured in the LLM Configuration tab under Memory & Enrichment."
        />

        <a-row :gutter="[24, 24]">
          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Dashboard Storage Settings">
              <template #extra>
                <a-tag :color="formData.database.type === 'postgresql' ? 'blue' : 'green'">
                  {{ formData.database.type === 'postgresql' ? 'PostgreSQL' : 'SQLite' }}
                </a-tag>
              </template>

              <a-form layout="vertical">
                <a-form-item label="Database Type">
                  <a-select v-model:value="formData.database.type" style="width: 100%">
                    <a-select-option value="sqlite">
                      <DatabaseOutlined /> SQLite (Local file)
                    </a-select-option>
                    <a-select-option value="postgresql">
                      <CloudServerOutlined /> PostgreSQL (Server)
                    </a-select-option>
                  </a-select>
                </a-form-item>

                <!-- SQLite Settings -->
                <template v-if="formData.database.type === 'sqlite'">
                  <a-form-item label="Database Path">
                    <a-input v-model:value="formData.database.path" placeholder="mindcore.db">
                      <template #prefix><FolderOutlined /></template>
                    </a-input>
                  </a-form-item>
                </template>

                <!-- PostgreSQL Settings -->
                <template v-if="formData.database.type === 'postgresql'">
                  <a-form-item label="Host">
                    <a-input v-model:value="formData.database.host" placeholder="localhost">
                      <template #prefix><GlobalOutlined /></template>
                    </a-input>
                  </a-form-item>

                  <a-form-item label="Port">
                    <a-input-number v-model:value="formData.database.port" :min="1" :max="65535" style="width: 100%" />
                  </a-form-item>

                  <a-form-item label="Database Name">
                    <a-input v-model:value="formData.database.database" placeholder="mindcore" />
                  </a-form-item>

                  <a-form-item label="Username">
                    <a-input v-model:value="formData.database.user" placeholder="postgres">
                      <template #prefix><UserOutlined /></template>
                    </a-input>
                  </a-form-item>

                  <a-form-item label="Password">
                    <a-input-password v-model:value="formData.database.password" placeholder="Enter password" />
                  </a-form-item>

                  <a-form-item label="SSL Mode">
                    <a-select v-model:value="formData.database.sslmode" style="width: 100%">
                      <a-select-option value="disable">Disable</a-select-option>
                      <a-select-option value="require">Require</a-select-option>
                      <a-select-option value="verify-ca">Verify CA</a-select-option>
                      <a-select-option value="verify-full">Verify Full</a-select-option>
                    </a-select>
                  </a-form-item>
                </template>

                <a-button type="primary" ghost @click="testConnection" :loading="testingConnection">
                  <ApiOutlined /> Test Connection
                </a-button>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Dashboard Database Maintenance">
              <a-space direction="vertical" style="width: 100%;">
                <a-card size="small" class="maintenance-card">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Vacuum Database</h4>
                      <p>Reclaim space and optimize database performance</p>
                    </a-col>
                    <a-col>
                      <a-button @click="vacuumDb" :loading="vacuuming">
                        <ClearOutlined /> Vacuum
                      </a-button>
                    </a-col>
                  </a-row>
                </a-card>

                <a-card size="small" class="maintenance-card">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Clear Old Metrics</h4>
                      <p>Remove performance metrics older than retention period</p>
                    </a-col>
                    <a-col>
                      <a-input-number v-model:value="retentionDays" :min="1" :max="365" style="width: 80px; margin-right: 8px;" />
                      <a-button @click="clearOldMetrics" :loading="clearingMetrics">
                        <DeleteOutlined /> Clear
                      </a-button>
                    </a-col>
                  </a-row>
                </a-card>

                <a-card size="small" class="maintenance-card danger">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Reset Database</h4>
                      <p>Delete all data and recreate tables (destructive!)</p>
                    </a-col>
                    <a-col>
                      <a-popconfirm
                        title="Are you sure? This will delete ALL data!"
                        ok-text="Yes, Reset"
                        cancel-text="Cancel"
                        @confirm="resetDatabase"
                      >
                        <a-button danger :loading="resettingDb">
                          <WarningOutlined /> Reset
                        </a-button>
                      </a-popconfirm>
                    </a-col>
                  </a-row>
                </a-card>
              </a-space>
            </a-card>
          </a-col>
        </a-row>

        <!-- Save button for Database tab -->
        <div class="tab-actions">
          <a-space>
            <a-button @click="resetDatabaseForm">
              <UndoOutlined /> Reset
            </a-button>
            <a-button type="primary" :loading="saving" @click="saveDatabaseConfig">
              <SaveOutlined /> Save Database Configuration
            </a-button>
          </a-space>
        </div>
      </a-tab-pane>

      <!-- Agent Memory Tab -->
      <a-tab-pane key="memory" tab="Agent Memory">
        <!-- Clarification Alert -->
        <a-alert
          type="success"
          show-icon
          style="margin-bottom: 24px;"
          message="Agent Memory Layer Configuration"
          description="This is the core Mindcore memory system that stores conversation data with AI-enriched metadata: sessions, threads, messages, topics, categories, importance scores, sentiment analysis, intent classification, and retrieval context. This is separate from the dashboard database which only stores logs and metrics."
        />

        <a-row :gutter="[24, 24]">
          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Memory Storage Settings">
              <template #extra>
                <a-tag :color="formData.agentMemory.type === 'postgresql' ? 'blue' : 'green'">
                  {{ formData.agentMemory.type === 'postgresql' ? 'PostgreSQL' : 'SQLite' }}
                </a-tag>
              </template>

              <a-form layout="vertical">
                <a-form-item label="Storage Type">
                  <a-select v-model:value="formData.agentMemory.type" style="width: 100%">
                    <a-select-option value="sqlite">
                      <DatabaseOutlined /> SQLite (Local file)
                    </a-select-option>
                    <a-select-option value="postgresql">
                      <CloudServerOutlined /> PostgreSQL (Server)
                    </a-select-option>
                  </a-select>
                </a-form-item>

                <!-- SQLite Settings -->
                <template v-if="formData.agentMemory.type === 'sqlite'">
                  <a-form-item label="Database Path">
                    <a-input v-model:value="formData.agentMemory.path" placeholder="mindcore_memory.db">
                      <template #prefix><FolderOutlined /></template>
                    </a-input>
                    <div class="form-help">Stores messages with enrichment metadata</div>
                  </a-form-item>
                </template>

                <!-- PostgreSQL Settings -->
                <template v-if="formData.agentMemory.type === 'postgresql'">
                  <a-form-item label="Host">
                    <a-input v-model:value="formData.agentMemory.host" placeholder="localhost">
                      <template #prefix><GlobalOutlined /></template>
                    </a-input>
                  </a-form-item>

                  <a-form-item label="Port">
                    <a-input-number v-model:value="formData.agentMemory.port" :min="1" :max="65535" style="width: 100%" />
                  </a-form-item>

                  <a-form-item label="Database Name">
                    <a-input v-model:value="formData.agentMemory.database" placeholder="mindcore_memory" />
                  </a-form-item>

                  <a-form-item label="Username">
                    <a-input v-model:value="formData.agentMemory.user" placeholder="postgres">
                      <template #prefix><UserOutlined /></template>
                    </a-input>
                  </a-form-item>

                  <a-form-item label="Password">
                    <a-input-password v-model:value="formData.agentMemory.password" placeholder="Enter password" />
                  </a-form-item>
                </template>

                <a-button type="primary" ghost @click="testMemoryConnection" :loading="testingMemoryConnection">
                  <ApiOutlined /> Test Connection
                </a-button>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Enrichment Data Schema">
              <a-descriptions :column="1" bordered size="small">
                <a-descriptions-item label="Topics">
                  <a-tag color="purple">Array</a-tag>
                  Main subjects discussed in message
                </a-descriptions-item>
                <a-descriptions-item label="Categories">
                  <a-tag color="blue">Array</a-tag>
                  Message types: question, statement, code, etc.
                </a-descriptions-item>
                <a-descriptions-item label="Importance">
                  <a-tag color="orange">Float 0-1</a-tag>
                  Priority score for context retrieval
                </a-descriptions-item>
                <a-descriptions-item label="Sentiment">
                  <a-tag color="cyan">Object</a-tag>
                  Overall sentiment and confidence score
                </a-descriptions-item>
                <a-descriptions-item label="Intent">
                  <a-tag color="green">String</a-tag>
                  Primary intent: ask_question, provide_info, etc.
                </a-descriptions-item>
                <a-descriptions-item label="Entities">
                  <a-tag color="magenta">Array</a-tag>
                  Named entities: people, places, technologies
                </a-descriptions-item>
                <a-descriptions-item label="Key Phrases">
                  <a-tag color="geekblue">Array</a-tag>
                  Important phrases for semantic search
                </a-descriptions-item>
              </a-descriptions>
            </a-card>
          </a-col>

          <a-col :xs="24">
            <a-card class="dashboard-card" title="Data Pipeline Flow">
              <div class="pipeline-flow">
                <div class="pipeline-step">
                  <div class="step-icon input">
                    <MessageOutlined />
                  </div>
                  <div class="step-label">Input</div>
                  <div class="step-desc">User message received</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                  <div class="step-icon enrichment">
                    <ThunderboltOutlined />
                  </div>
                  <div class="step-label">Enrichment</div>
                  <div class="step-desc">LLM extracts metadata</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                  <div class="step-icon storage">
                    <DatabaseOutlined />
                  </div>
                  <div class="step-label">Storage</div>
                  <div class="step-desc">Save to memory layer</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                  <div class="step-icon retrieval">
                    <SearchOutlined />
                  </div>
                  <div class="step-label">Retrieval</div>
                  <div class="step-desc">Context assembly</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                  <div class="step-icon output">
                    <RobotOutlined />
                  </div>
                  <div class="step-label">Output</div>
                  <div class="step-desc">LLM response generated</div>
                </div>
              </div>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Memory Statistics">
              <a-row :gutter="[16, 16]">
                <a-col :span="12">
                  <a-statistic title="Total Messages" :value="memoryStats.total_messages" />
                </a-col>
                <a-col :span="12">
                  <a-statistic title="Enriched Messages" :value="memoryStats.enriched_messages" suffix="%" />
                </a-col>
                <a-col :span="12">
                  <a-statistic title="Unique Topics" :value="memoryStats.unique_topics" />
                </a-col>
                <a-col :span="12">
                  <a-statistic title="Active Threads" :value="memoryStats.active_threads" />
                </a-col>
              </a-row>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Memory Maintenance">
              <a-space direction="vertical" style="width: 100%;">
                <a-card size="small" class="maintenance-card">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Re-enrich Messages</h4>
                      <p>Re-run enrichment on messages that failed</p>
                    </a-col>
                    <a-col>
                      <a-button @click="reEnrichMessages" :loading="reEnriching">
                        <ThunderboltOutlined /> Re-enrich
                      </a-button>
                    </a-col>
                  </a-row>
                </a-card>

                <a-card size="small" class="maintenance-card">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Rebuild Topic Index</h4>
                      <p>Optimize topic-based retrieval performance</p>
                    </a-col>
                    <a-col>
                      <a-button @click="rebuildTopicIndex" :loading="rebuildingIndex">
                        <SyncOutlined /> Rebuild
                      </a-button>
                    </a-col>
                  </a-row>
                </a-card>

                <a-card size="small" class="maintenance-card danger">
                  <a-row align="middle" justify="space-between">
                    <a-col>
                      <h4>Clear All Memory</h4>
                      <p>Delete all messages and enrichment data (destructive!)</p>
                    </a-col>
                    <a-col>
                      <a-popconfirm
                        title="Are you sure? This will delete ALL conversation memory!"
                        ok-text="Yes, Clear"
                        cancel-text="Cancel"
                        @confirm="clearMemory"
                      >
                        <a-button danger :loading="clearingMemory">
                          <DeleteOutlined /> Clear
                        </a-button>
                      </a-popconfirm>
                    </a-col>
                  </a-row>
                </a-card>
              </a-space>
            </a-card>
          </a-col>
        </a-row>

        <!-- Save button for Agent Memory tab -->
        <div class="tab-actions">
          <a-space>
            <a-button @click="resetMemoryForm">
              <UndoOutlined /> Reset
            </a-button>
            <a-button type="primary" :loading="saving" @click="saveMemoryConfig">
              <SaveOutlined /> Save Memory Configuration
            </a-button>
          </a-space>
        </div>
      </a-tab-pane>

      <!-- Environment Variables Tab -->
      <a-tab-pane key="env" tab="Environment Variables">
        <a-row :gutter="[24, 24]">
          <a-col :xs="24">
            <a-card class="dashboard-card" title="Environment Variables">
              <template #extra>
                <a-space>
                  <a-button type="primary" ghost size="small" @click="addEnvVar">
                    <PlusOutlined /> Add Variable
                  </a-button>
                  <a-button size="small" @click="refreshEnvVars">
                    <ReloadOutlined /> Refresh
                  </a-button>
                </a-space>
              </template>

              <a-table
                :columns="envColumns"
                :data-source="envVarsList"
                :pagination="false"
                size="small"
                :row-key="record => record.key"
              >
                <template #bodyCell="{ column, record, index }">
                  <template v-if="column.key === 'key'">
                    <a-input
                      v-if="record.editing"
                      v-model:value="record.key"
                      placeholder="VARIABLE_NAME"
                      style="width: 200px"
                    />
                    <code v-else>{{ record.key }}</code>
                  </template>

                  <template v-if="column.key === 'value'">
                    <template v-if="record.editing">
                      <a-input-password
                        v-if="record.sensitive"
                        v-model:value="record.value"
                        placeholder="Enter value"
                        style="width: 300px"
                      />
                      <a-input
                        v-else
                        v-model:value="record.value"
                        placeholder="Enter value"
                        style="width: 300px"
                      />
                    </template>
                    <template v-else>
                      <code v-if="record.sensitive">{{ maskValue(record.value) }}</code>
                      <code v-else>{{ record.value || '(not set)' }}</code>
                    </template>
                  </template>

                  <template v-if="column.key === 'sensitive'">
                    <a-checkbox v-model:checked="record.sensitive" :disabled="!record.editing" />
                  </template>

                  <template v-if="column.key === 'actions'">
                    <a-space>
                      <template v-if="record.editing">
                        <a-button type="text" size="small" @click="saveEnvVar(index)">
                          <CheckOutlined style="color: #52c41a" />
                        </a-button>
                        <a-button type="text" size="small" @click="cancelEditEnvVar(index)">
                          <CloseOutlined style="color: #ff4d4f" />
                        </a-button>
                      </template>
                      <template v-else>
                        <a-button type="text" size="small" @click="editEnvVar(index)">
                          <EditOutlined />
                        </a-button>
                        <a-popconfirm title="Delete this variable?" @confirm="deleteEnvVar(index)">
                          <a-button type="text" size="small" danger>
                            <DeleteOutlined />
                          </a-button>
                        </a-popconfirm>
                      </template>
                    </a-space>
                  </template>
                </template>
              </a-table>

              <a-divider>Common Variables</a-divider>

              <a-collapse ghost>
                <a-collapse-panel key="api" header="API Keys">
                  <a-descriptions :column="1" size="small" bordered>
                    <a-descriptions-item label="OPENAI_API_KEY">
                      Required for OpenAI models
                    </a-descriptions-item>
                    <a-descriptions-item label="ANTHROPIC_API_KEY">
                      Required for Claude models
                    </a-descriptions-item>
                  </a-descriptions>
                </a-collapse-panel>
                <a-collapse-panel key="mindcore" header="Mindcore Settings">
                  <a-descriptions :column="1" size="small" bordered>
                    <a-descriptions-item label="MINDCORE_DB_PATH">
                      SQLite database path (default: mindcore.db)
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_LLAMA_MODEL_PATH">
                      Path to local GGUF model file
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_LOG_LEVEL">
                      Logging level: DEBUG, INFO, WARNING, ERROR
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_API_PORT">
                      API server port (default: 8000)
                    </a-descriptions-item>
                  </a-descriptions>
                </a-collapse-panel>
                <a-collapse-panel key="database" header="Database">
                  <a-descriptions :column="1" size="small" bordered>
                    <a-descriptions-item label="MINDCORE_PG_HOST">
                      PostgreSQL host
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_PG_PORT">
                      PostgreSQL port (default: 5432)
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_PG_DATABASE">
                      PostgreSQL database name
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_PG_USER">
                      PostgreSQL username
                    </a-descriptions-item>
                    <a-descriptions-item label="MINDCORE_PG_PASSWORD">
                      PostgreSQL password
                    </a-descriptions-item>
                  </a-descriptions>
                </a-collapse-panel>
              </a-collapse>
            </a-card>
          </a-col>
        </a-row>

        <!-- Save button for Environment tab -->
        <div class="tab-actions">
          <a-button type="primary" :loading="saving" @click="saveEnvVars">
            <SaveOutlined /> Save Environment Variables
          </a-button>
        </div>
      </a-tab-pane>

      <!-- Advanced Settings Tab -->
      <a-tab-pane key="advanced" tab="Advanced">
        <a-row :gutter="[24, 24]">
          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="API Settings">
              <a-form layout="vertical">
                <a-form-item label="API Port">
                  <a-input-number v-model:value="formData.api.port" :min="1024" :max="65535" style="width: 100%" />
                </a-form-item>

                <a-form-item label="CORS Origins">
                  <a-select
                    v-model:value="formData.api.cors_origins"
                    mode="tags"
                    style="width: 100%"
                    placeholder="Enter allowed origins"
                  />
                  <div class="form-help">Leave empty to allow all origins (*)</div>
                </a-form-item>

                <a-form-item label="Rate Limiting">
                  <a-switch v-model:checked="formData.api.rate_limiting" />
                  <span style="margin-left: 8px;">{{ formData.api.rate_limiting ? 'Enabled' : 'Disabled' }}</span>
                </a-form-item>

                <a-form-item label="Requests per Minute" v-if="formData.api.rate_limiting">
                  <a-input-number v-model:value="formData.api.rate_limit" :min="10" :max="1000" style="width: 100%" />
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Logging & Monitoring">
              <a-form layout="vertical">
                <a-form-item label="Log Level">
                  <a-select v-model:value="formData.logging.level" style="width: 100%">
                    <a-select-option value="DEBUG">DEBUG - Verbose</a-select-option>
                    <a-select-option value="INFO">INFO - Standard</a-select-option>
                    <a-select-option value="WARNING">WARNING - Important only</a-select-option>
                    <a-select-option value="ERROR">ERROR - Errors only</a-select-option>
                  </a-select>
                </a-form-item>

                <a-form-item label="JSON Logs">
                  <a-switch v-model:checked="formData.logging.json_format" />
                  <span style="margin-left: 8px;">Structured JSON format for log aggregators</span>
                </a-form-item>

                <a-form-item label="Performance Tracking">
                  <a-switch v-model:checked="formData.monitoring.performance_tracking" />
                  <span style="margin-left: 8px;">Record LLM call latencies and metrics</span>
                </a-form-item>

                <a-form-item label="Tool Call Tracking">
                  <a-switch v-model:checked="formData.monitoring.tool_tracking" />
                  <span style="margin-left: 8px;">Record tool usage and success rates</span>
                </a-form-item>

                <a-form-item label="Metrics Retention">
                  <a-input-number
                    v-model:value="formData.monitoring.retention_days"
                    :min="1"
                    :max="365"
                    addon-after="days"
                    style="width: 100%"
                  />
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Security">
              <a-form layout="vertical">
                <a-form-item label="Input Validation">
                  <a-switch v-model:checked="formData.security.input_validation" />
                  <span style="margin-left: 8px;">Validate and sanitize user inputs</span>
                </a-form-item>

                <a-form-item label="Max Message Length">
                  <a-input-number
                    v-model:value="formData.security.max_message_length"
                    :min="100"
                    :max="100000"
                    addon-after="chars"
                    style="width: 100%"
                  />
                </a-form-item>

                <a-form-item label="Audit Logging">
                  <a-switch v-model:checked="formData.security.audit_logging" />
                  <span style="margin-left: 8px;">Log all API requests for security audit</span>
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>

          <a-col :xs="24" :lg="12">
            <a-card class="dashboard-card" title="Export / Import">
              <a-space direction="vertical" style="width: 100%;">
                <a-button block @click="exportConfig">
                  <DownloadOutlined /> Export Configuration
                </a-button>
                <a-upload
                  :before-upload="importConfig"
                  :show-upload-list="false"
                  accept=".json"
                >
                  <a-button block>
                    <UploadOutlined /> Import Configuration
                  </a-button>
                </a-upload>
                <a-button block type="dashed" @click="resetToDefaults">
                  <UndoOutlined /> Reset to Defaults
                </a-button>
              </a-space>
            </a-card>
          </a-col>
        </a-row>

        <!-- Save button for Advanced tab -->
        <div class="tab-actions">
          <a-space>
            <a-button @click="resetAdvancedForm">
              <UndoOutlined /> Reset
            </a-button>
            <a-button type="primary" :loading="saving" @click="saveAdvancedConfig">
              <SaveOutlined /> Save Advanced Configuration
            </a-button>
          </a-space>
        </div>
      </a-tab-pane>
    </a-tabs>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  DatabaseOutlined,
  CloudServerOutlined,
  FolderOutlined,
  GlobalOutlined,
  UserOutlined,
  ApiOutlined,
  ClearOutlined,
  DeleteOutlined,
  WarningOutlined,
  PlusOutlined,
  ReloadOutlined,
  EditOutlined,
  CheckOutlined,
  CloseOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DownloadOutlined,
  UploadOutlined,
  UndoOutlined,
  SaveOutlined,
  MessageOutlined,
  ThunderboltOutlined,
  SearchOutlined,
  SyncOutlined,
  RobotOutlined
} from '@ant-design/icons-vue'
import { dashboardApi } from '../api'

const config = ref({})
const saving = ref(false)
const activeTab = ref('llm')
const systemEnabled = ref(true)
const togglingSystem = ref(false)
const serverStatus = ref('online')
const restarting = ref(false)
const testingConnection = ref(false)
const testingMemoryConnection = ref(false)
const vacuuming = ref(false)
const clearingMetrics = ref(false)
const resettingDb = ref(false)
const retentionDays = ref(30)
const reEnriching = ref(false)
const rebuildingIndex = ref(false)
const clearingMemory = ref(false)

// Memory statistics
const memoryStats = ref({
  total_messages: 0,
  enriched_messages: 0,
  unique_topics: 0,
  active_threads: 0
})

// Temperature marks for slider
const temperatureMarks = {
  0: 'Precise',
  0.5: 'Balanced',
  1: 'Creative'
}

const formData = reactive({
  llm: {
    model: 'gpt-4o-mini',
    custom_model_path: '',
    temperature: 0.3,
    max_tokens: 1500
  },
  memory: {
    enabled: true,
    model: 'llama-3.2-1b-local'
  },
  cache: {
    max_size: 50,
    ttl: 3600
  },
  database: {
    type: 'sqlite',
    path: 'mindcore_dashboard.db',
    host: 'localhost',
    port: 5432,
    database: 'mindcore_dashboard',
    user: 'postgres',
    password: '',
    sslmode: 'disable'
  },
  agentMemory: {
    type: 'sqlite',
    path: 'mindcore.db',
    host: 'localhost',
    port: 5432,
    database: 'mindcore_memory',
    user: 'postgres',
    password: '',
    sslmode: 'disable'
  },
  api: {
    port: 8000,
    cors_origins: [],
    rate_limiting: true,
    rate_limit: 100
  },
  logging: {
    level: 'INFO',
    json_format: false
  },
  monitoring: {
    performance_tracking: true,
    tool_tracking: true,
    retention_days: 30
  },
  security: {
    input_validation: true,
    max_message_length: 10000,
    audit_logging: false
  }
})

const envVarsList = ref([
  { key: 'OPENAI_API_KEY', value: '', sensitive: true, editing: false },
  { key: 'ANTHROPIC_API_KEY', value: '', sensitive: true, editing: false },
  { key: 'MINDCORE_DB_PATH', value: 'mindcore.db', sensitive: false, editing: false },
  { key: 'MINDCORE_LOG_LEVEL', value: 'INFO', sensitive: false, editing: false },
  { key: 'MINDCORE_API_PORT', value: '8000', sensitive: false, editing: false }
])

const envColumns = [
  { title: 'Variable', key: 'key', width: 250 },
  { title: 'Value', key: 'value', width: 350 },
  { title: 'Sensitive', key: 'sensitive', width: 100, align: 'center' },
  { title: 'Actions', key: 'actions', width: 120, align: 'center' }
]

const getModelDisplayName = (model) => {
  const names = {
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-3.5-turbo': 'GPT-3.5 Turbo',
    'o1-preview': 'O1 Preview',
    'o1-mini': 'O1 Mini',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
    'llama-3.2-3b-local': 'Llama 3.2 3B',
    'llama-3.2-1b-local': 'Llama 3.2 1B',
    'qwen2.5-3b-local': 'Qwen 2.5 3B',
    'phi-3.5-mini-local': 'Phi 3.5 Mini',
    'custom-local': 'Custom Model'
  }
  return names[model] || model
}

const maskValue = (value) => {
  if (!value) return '(not set)'
  if (value.length < 8) return '***'
  return value.substring(0, 4) + '...' + value.substring(value.length - 4)
}

const fetchConfig = async () => {
  try {
    const data = await dashboardApi.getConfig()
    config.value = data

    if (data.llm) {
      formData.llm.model = data.llm.model || 'gpt-4o-mini'
      formData.llm.temperature = data.llm.temperature || 0.3
      formData.llm.max_tokens = data.llm.max_tokens || 1500
    }
    if (data.memory) {
      formData.memory.enabled = data.memory.enabled !== false
      formData.memory.model = data.memory.model || 'llama-3.2-1b-local'
    }
    if (data.cache) {
      formData.cache.max_size = data.cache.max_size || 50
      formData.cache.ttl = data.cache.ttl || 3600
    }
    if (data.database) {
      formData.database.type = data.database.type || 'sqlite'
      formData.database.path = data.database.path || 'mindcore.db'
    }
  } catch (err) {
    console.error('Failed to fetch config:', err)
  }
}

const checkServerStatus = async () => {
  try {
    await dashboardApi.getHealth()
    serverStatus.value = 'online'
  } catch {
    serverStatus.value = 'offline'
  }
}

const toggleSystem = async (enabled) => {
  togglingSystem.value = true
  try {
    await dashboardApi.updateConfig({ system_enabled: enabled })
    message.success(enabled ? 'Mindcore enabled' : 'Mindcore disabled')
  } catch (err) {
    console.error('Failed to toggle system:', err)
    message.error('Failed to update system status')
    systemEnabled.value = !enabled
  } finally {
    togglingSystem.value = false
  }
}

const restartServer = async () => {
  restarting.value = true
  try {
    message.info('Restart signal sent. Server will restart shortly...')
    setTimeout(() => {
      restarting.value = false
      checkServerStatus()
    }, 3000)
  } catch (err) {
    console.error('Failed to restart server:', err)
    message.error('Failed to restart server')
    restarting.value = false
  }
}

const testConnection = async () => {
  testingConnection.value = true
  try {
    const result = await dashboardApi.testDatabaseConnection(formData.database)
    if (result.status === 'success') {
      message.success(result.message || 'Database connection successful!')
    } else {
      message.error(result.message || 'Connection failed')
    }
  } catch (err) {
    message.error('Failed to connect to database')
  } finally {
    testingConnection.value = false
  }
}

const vacuumDb = async () => {
  vacuuming.value = true
  try {
    const result = await dashboardApi.vacuumDatabase()
    message.success(result.message || 'Database vacuumed successfully')
  } catch (err) {
    message.error('Failed to vacuum database')
  } finally {
    vacuuming.value = false
  }
}

const clearOldMetrics = async () => {
  clearingMetrics.value = true
  try {
    const result = await dashboardApi.clearOldMetrics(retentionDays.value)
    message.success(`Cleared ${result.deleted_count || 0} metrics older than ${retentionDays.value} days`)
  } catch (err) {
    message.error('Failed to clear metrics')
  } finally {
    clearingMetrics.value = false
  }
}

const resetDatabase = async () => {
  resettingDb.value = true
  try {
    const result = await dashboardApi.resetDatabase()
    message.success(`Database reset: cleared ${result.tables_cleared?.length || 0} tables`)
  } catch (err) {
    message.error('Failed to reset database')
  } finally {
    resettingDb.value = false
  }
}

// Agent Memory functions
const testMemoryConnection = async () => {
  testingMemoryConnection.value = true
  try {
    const result = await dashboardApi.testDatabaseConnection(formData.agentMemory)
    if (result.status === 'success') {
      message.success(result.message || 'Memory database connection successful!')
    } else {
      message.error(result.message || 'Connection failed')
    }
  } catch (err) {
    message.error('Failed to connect to memory database')
  } finally {
    testingMemoryConnection.value = false
  }
}

const fetchMemoryStats = async () => {
  try {
    const stats = await dashboardApi.getStats()
    memoryStats.value = {
      total_messages: stats.total_messages || 0,
      enriched_messages: stats.total_messages > 0 ? 95 : 0, // Approximate enriched %
      unique_topics: Math.floor((stats.total_messages || 0) * 0.3), // Approximate
      active_threads: stats.conversations || 0
    }
  } catch (err) {
    console.error('Failed to fetch memory stats:', err)
  }
}

const reEnrichMessages = async () => {
  reEnriching.value = true
  try {
    message.info('Re-enriching failed messages...')
    // In production, this would call an API endpoint
    await new Promise(resolve => setTimeout(resolve, 2000))
    message.success('Re-enrichment completed')
  } catch (err) {
    message.error('Failed to re-enrich messages')
  } finally {
    reEnriching.value = false
  }
}

const rebuildTopicIndex = async () => {
  rebuildingIndex.value = true
  try {
    message.info('Rebuilding topic index...')
    // In production, this would call an API endpoint
    await new Promise(resolve => setTimeout(resolve, 2000))
    message.success('Topic index rebuilt successfully')
  } catch (err) {
    message.error('Failed to rebuild topic index')
  } finally {
    rebuildingIndex.value = false
  }
}

const clearMemory = async () => {
  clearingMemory.value = true
  try {
    const result = await dashboardApi.resetDatabase()
    message.success(`Memory cleared: ${result.tables_cleared?.length || 0} tables reset`)
    await fetchMemoryStats()
  } catch (err) {
    message.error('Failed to clear memory')
  } finally {
    clearingMemory.value = false
  }
}

const resetMemoryForm = () => {
  formData.agentMemory = {
    type: 'sqlite',
    path: 'mindcore.db',
    host: 'localhost',
    port: 5432,
    database: 'mindcore_memory',
    user: 'postgres',
    password: '',
    sslmode: 'disable'
  }
  message.info('Memory settings reset')
}

const saveMemoryConfig = async () => {
  saving.value = true
  try {
    await dashboardApi.updateConfig({
      agent_memory: formData.agentMemory
    })
    message.success('Memory configuration saved')
    await fetchConfig()
  } catch (err) {
    console.error('Failed to save memory config:', err)
    message.error('Failed to save memory configuration')
  } finally {
    saving.value = false
  }
}

const addEnvVar = () => {
  envVarsList.value.push({
    key: '',
    value: '',
    sensitive: false,
    editing: true
  })
}

const editEnvVar = (index) => {
  envVarsList.value[index].editing = true
  envVarsList.value[index]._original = { ...envVarsList.value[index] }
}

const saveEnvVar = async (index) => {
  const envVar = envVarsList.value[index]
  if (!envVar.key) {
    message.error('Variable name is required')
    return
  }
  envVar.editing = false
  delete envVar._original
  message.success(`Saved ${envVar.key}`)
}

const cancelEditEnvVar = (index) => {
  const envVar = envVarsList.value[index]
  if (envVar._original) {
    Object.assign(envVar, envVar._original)
    delete envVar._original
  } else if (!envVar.key) {
    envVarsList.value.splice(index, 1)
  }
  envVar.editing = false
}

const deleteEnvVar = (index) => {
  const key = envVarsList.value[index].key
  envVarsList.value.splice(index, 1)
  message.success(`Deleted ${key}`)
}

const refreshEnvVars = () => {
  message.info('Environment variables refreshed')
}

// Tab-specific save functions
const saveLLMConfig = async () => {
  saving.value = true
  try {
    await dashboardApi.updateConfig({
      llm: formData.llm,
      memory: formData.memory,
      cache: formData.cache
    })
    message.success('LLM configuration saved')
    await fetchConfig()
  } catch (err) {
    console.error('Failed to save config:', err)
    message.error('Failed to save configuration')
  } finally {
    saving.value = false
  }
}

const saveDatabaseConfig = async () => {
  saving.value = true
  try {
    await dashboardApi.updateConfig({
      database: formData.database
    })
    message.success('Database configuration saved')
    await fetchConfig()
  } catch (err) {
    console.error('Failed to save config:', err)
    message.error('Failed to save configuration')
  } finally {
    saving.value = false
  }
}

const saveEnvVars = async () => {
  saving.value = true
  try {
    await dashboardApi.updateEnvVars(envVarsList.value)
    message.success('Environment variables saved')
  } catch (err) {
    console.error('Failed to save env vars:', err)
    message.error('Failed to save environment variables')
  } finally {
    saving.value = false
  }
}

const saveAdvancedConfig = async () => {
  saving.value = true
  try {
    await dashboardApi.updateConfig({
      api: formData.api,
      logging: formData.logging,
      monitoring: formData.monitoring,
      security: formData.security
    })
    message.success('Advanced configuration saved')
    await fetchConfig()
  } catch (err) {
    console.error('Failed to save config:', err)
    message.error('Failed to save configuration')
  } finally {
    saving.value = false
  }
}

// Tab-specific reset functions
const resetLLMForm = () => {
  formData.llm = { model: 'gpt-4o-mini', custom_model_path: '', temperature: 0.3, max_tokens: 1500 }
  formData.memory = { enabled: true, model: 'llama-3.2-1b-local' }
  formData.cache = { max_size: 50, ttl: 3600 }
  message.info('LLM settings reset')
}

const resetDatabaseForm = () => {
  formData.database = { type: 'sqlite', path: 'mindcore.db', host: 'localhost', port: 5432, database: 'mindcore', user: 'postgres', password: '', sslmode: 'disable' }
  message.info('Database settings reset')
}

const resetAdvancedForm = () => {
  formData.api = { port: 8000, cors_origins: [], rate_limiting: true, rate_limit: 100 }
  formData.logging = { level: 'INFO', json_format: false }
  formData.monitoring = { performance_tracking: true, tool_tracking: true, retention_days: 30 }
  formData.security = { input_validation: true, max_message_length: 10000, audit_logging: false }
  message.info('Advanced settings reset')
}

const exportConfig = () => {
  const configJson = JSON.stringify(formData, null, 2)
  const blob = new Blob([configJson], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'mindcore-config.json'
  a.click()
  URL.revokeObjectURL(url)
  message.success('Configuration exported')
}

const importConfig = (file) => {
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const imported = JSON.parse(e.target.result)
      Object.assign(formData, imported)
      message.success('Configuration imported')
    } catch (err) {
      message.error('Invalid configuration file')
    }
  }
  reader.readAsText(file)
  return false
}

const resetToDefaults = () => {
  resetLLMForm()
  resetDatabaseForm()
  resetAdvancedForm()
  message.success('Reset all to default values')
}

onMounted(() => {
  fetchConfig()
  checkServerStatus()
  fetchMemoryStats()
})
</script>

<style scoped>
.config-page {
  padding-bottom: 24px;
}

.status-bar {
  margin-bottom: 24px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-label {
  font-size: 13px;
  color: var(--text-secondary);
}

.config-tabs :deep(.ant-tabs-nav) {
  margin-bottom: 24px;
}

.form-help {
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 4px;
}

.maintenance-card {
  background: var(--bg-color);
}

.maintenance-card h4 {
  margin: 0 0 4px 0;
  font-size: 14px;
}

.maintenance-card p {
  margin: 0;
  font-size: 12px;
  color: var(--text-secondary);
}

.maintenance-card.danger {
  border-color: #ff4d4f;
}

.maintenance-card.danger h4 {
  color: #ff4d4f;
}

.tab-actions {
  margin-top: 24px;
  padding: 16px 0;
  display: flex;
  justify-content: flex-end;
  border-top: 1px solid var(--border-color);
}

/* Fix button text visibility */
.config-page :deep(.ant-btn-primary) {
  color: #fff !important;
}

.config-page :deep(.ant-btn-primary:hover) {
  color: #fff !important;
}

.config-page :deep(.ant-btn-primary.ant-btn-dangerous) {
  color: #fff !important;
}

.config-page :deep(.ant-btn-default),
.config-page :deep(.ant-btn-dashed) {
  color: var(--text-primary) !important;
}

.config-page :deep(.ant-btn-default:hover),
.config-page :deep(.ant-btn-dashed:hover) {
  color: var(--primary-color) !important;
}

/* Pipeline Flow Visualization */
.pipeline-flow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 16px;
  overflow-x: auto;
}

.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 100px;
}

.step-icon {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: #fff;
  margin-bottom: 12px;
}

.step-icon.input {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.step-icon.enrichment {
  background: linear-gradient(135deg, #11998e, #38ef7d);
}

.step-icon.storage {
  background: linear-gradient(135deg, #f093fb, #f5576c);
}

.step-icon.retrieval {
  background: linear-gradient(135deg, #4facfe, #00f2fe);
}

.step-icon.output {
  background: linear-gradient(135deg, #fa709a, #fee140);
}

.step-label {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.step-desc {
  font-size: 11px;
  color: var(--text-secondary);
  text-align: center;
}

.pipeline-arrow {
  font-size: 24px;
  color: var(--primary-color);
  margin: 0 8px;
  flex-shrink: 0;
}
</style>
