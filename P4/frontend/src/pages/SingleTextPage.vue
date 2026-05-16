<script setup>
import { computed, onMounted, ref } from "vue";
import {
  tokenizeText,
  recognizeEntities,
  classifyText,
  getBackendMeta,
  getApiBaseUrl,
} from "../api/nlp";
import TokenResultCard from "../components/TokenResultCard.vue";
import NerResultCard from "../components/NerResultCard.vue";
import ClassifyResultCard from "../components/ClassifyResultCard.vue";
import TextStatsCharts from "../components/TextStatsCharts.vue";

const ENTITY_LABEL_NAMES = {
  PER: "人名",
  LOC: "地点",
  GPE: "行政地域",
  FAC: "设施/景点",
  ORG: "机构",
  COMPANY: "公司",
  INSTITUTION: "单位",
  MISC: "其他实体",
};

const TEXT_EXAMPLES = [
  {
    key: "tech",
    label: "科技",
    text: "北京人工智能研究院发布了多模态模型评测报告，研究团队表示新系统在中文问答和图文理解任务上表现稳定。",
  },
  {
    key: "finance",
    label: "财经",
    text: "上海证券市场今日关注新能源板块表现，多家机构分析师认为企业季度财报将影响后续资金配置节奏。",
  },
  {
    key: "sports",
    label: "体育",
    text: "中国女排在杭州举行的邀请赛中战胜强敌，主教练表示球队在发球和拦网环节取得了明显提升。",
  },
  {
    key: "education",
    label: "教育",
    text: "北京师范大学近期启动教育数字化实验项目，相关负责人介绍该项目将支持课堂反馈与个性化学习分析。",
  },
];

const text = ref("");
const loading = ref(false);
const error = ref("");
const tokens = ref([]);
const entities = ref([]);
const label = ref("");
const confidence = ref(null);
const hasSubmitted = ref(false);
const activeResultTab = ref("overview");
const analysisCost = ref(null);
const copyMessage = ref("");
const backendStatus = ref({
  online: false,
  service: "未知服务",
  lastUsedPath: "unknown",
  error: "",
});
const backendStatusLoading = ref(false);
const backendBaseUrl = getApiBaseUrl();

const trimmedText = computed(() => text.value.trim());
const charCount = computed(() => trimmedText.value.length);
const tokenCount = computed(() => tokens.value.length);
const entityCount = computed(() => entities.value.length);
const hasResults = computed(
  () => tokens.value.length > 0 || entities.value.length > 0 || Boolean(label.value),
);
const analysisStateText = computed(() => {
  if (loading.value) {
    return "分析中";
  }
  if (hasResults.value) {
    return "已完成";
  }
  if (hasSubmitted.value) {
    return "已提交";
  }
  return "待分析";
});

function resetResults() {
  tokens.value = [];
  entities.value = [];
  label.value = "";
  confidence.value = null;
  analysisCost.value = null;
}

function fillExample(exampleText) {
  text.value = exampleText;
  error.value = "";
  copyMessage.value = "";
}

function clearAll() {
  text.value = "";
  error.value = "";
  hasSubmitted.value = false;
  copyMessage.value = "";
  resetResults();
}

function formatConfidence(value) {
  if (value === null || value === "") {
    return "未提供";
  }

  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return String(value);
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

function formatCost(value) {
  if (value === null || value === undefined) {
    return "未记录";
  }
  return `${Math.round(Number(value))} ms`;
}

function buildAnalysisSummary() {
  const tokenText = tokens.value.length > 0 ? tokens.value.join(" / ") : "无";
  const entityText =
    entities.value.length > 0
      ? entities.value
          .map(
            (item) =>
              `${item.text} [${item.label}] (${item.start}-${item.end})`,
          )
          .join("\n")
      : "无";

  return [
    "P4 单文本分析结果",
    "",
    `原文：${trimmedText.value || "无"}`,
    "",
    `分词结果：${tokenText}`,
    "",
    "实体识别：",
    entityText,
    "",
    `分类标签：${label.value || "无"}`,
    `置信度：${formatConfidence(confidence.value)}`,
  ].join("\n");
}

function buildAnalysisReportDescription() {
  const confidenceText = formatConfidence(confidence.value);
  const labelText = label.value || "未返回";
  const entityList = entities.value
    .map((item) => {
      const entityLabel = String(item?.label || "MISC").toUpperCase();
      const entityName = ENTITY_LABEL_NAMES[entityLabel] || entityLabel;
      return `${entityName}“${item.text}”`;
    })
    .filter(Boolean);

  const entityDescription =
    entityList.length > 0
      ? `从实体识别结果看，文本中包含${entityList.slice(0, 4).join("、")}${
          entityList.length > 4 ? "等实体" : ""
        }，说明该文本具有较明确的命名实体线索。`
      : "从实体识别结果看，文本中未识别出明确的命名实体，说明该文本可能更偏向普通叙述或实体信息较少。";

  return `本次输入文本共 ${charCount.value} 个字符，系统分词得到 ${tokenCount.value} 个词元，识别出 ${entityCount.value} 个命名实体，文本分类结果为“${labelText}”，置信度为 ${confidenceText}。本次分析耗时约 ${formatCost(
    analysisCost.value,
  )}。${entityDescription}`;
}

async function writeTextToClipboard(textValue) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(textValue);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = textValue;
  textarea.setAttribute("readonly", "readonly");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

async function copyAnalysisResult() {
  if (!hasResults.value) {
    copyMessage.value = "当前没有可复制的分析结果。";
    return;
  }

  const summary = buildAnalysisSummary();

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(summary);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = summary;
      textarea.setAttribute("readonly", "readonly");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }

    copyMessage.value = "分析结果已复制到剪贴板。";
  } catch {
    copyMessage.value = "复制失败，请稍后重试。";
  }
}

async function copyAnalysisReportDescription() {
  if (!hasResults.value) {
    copyMessage.value = "当前没有可复制的报告描述。";
    return;
  }

  try {
    await writeTextToClipboard(buildAnalysisReportDescription());
    copyMessage.value = "报告描述已复制到剪贴板。";
  } catch {
    copyMessage.value = "复制失败，请稍后重试。";
  }
}

async function handleAnalyze() {
  const input = trimmedText.value;

  if (!input) {
    error.value = "请输入需要分析的中文文本。";
    hasSubmitted.value = false;
    copyMessage.value = "";
    resetResults();
    return;
  }

  loading.value = true;
  error.value = "";
  hasSubmitted.value = true;
  copyMessage.value = "";
  resetResults();
  const startTime = performance.now();

  try {
    const [tokenizeRes, nerRes, classifyRes] = await Promise.all([
      tokenizeText(input),
      recognizeEntities(input),
      classifyText(input),
    ]);

    tokens.value = tokenizeRes?.tokens || [];
    entities.value = nerRes?.entities || [];
    label.value = classifyRes?.label || "";
    confidence.value = classifyRes?.confidence ?? null;
    activeResultTab.value = "overview";
  } catch (e) {
    error.value = e instanceof Error ? e.message : "分析失败，请稍后重试。";
  } finally {
    analysisCost.value = performance.now() - startTime;
    loading.value = false;
  }
}

async function refreshBackendStatus() {
  backendStatusLoading.value = true;

  try {
    const meta = await getBackendMeta();
    backendStatus.value = {
      online: true,
      service: meta?.service || "未知服务",
      lastUsedPath: meta?.ner_status?.last_used_path || "unknown",
      error: "",
    };
  } catch (e) {
    backendStatus.value = {
      online: false,
      service: "未知服务",
      lastUsedPath: "unknown",
      error: e instanceof Error ? e.message : "无法连接后端服务",
    };
  } finally {
    backendStatusLoading.value = false;
  }
}

onMounted(() => {
  refreshBackendStatus();
});
</script>

<template>
  <section class="page-stack">
    <header class="page-hero">
      <div class="page-hero-copy">
        <p class="section-kicker">Single Text</p>
        <h2>单文本分析</h2>
        <p>
          输入一段中文文本，系统将同时返回分词结果、命名实体识别结果以及文本分类结果，
          方便课堂演示与实验报告截图。
        </p>
      </div>

      <div class="hero-metrics">
        <article class="metric-card">
          <span>字符数</span>
          <strong>{{ charCount }}</strong>
        </article>
        <article class="metric-card">
          <span>当前状态</span>
          <strong>{{ analysisStateText }}</strong>
        </article>
      </div>
    </header>

    <section class="backend-status-bar">
      <div class="backend-status-main">
        <div class="backend-status-chip">
          <span>服务状态</span>
          <strong :class="backendStatus.online ? 'status-ok' : 'status-down'">
            {{ backendStatus.online ? "在线" : "离线" }}
          </strong>
        </div>
        <div class="backend-status-chip">
          <span>NER 路径</span>
          <strong>{{ backendStatus.lastUsedPath }}</strong>
        </div>
        <div class="backend-status-chip">
          <span>后端地址</span>
          <strong>{{ backendBaseUrl }}</strong>
        </div>
      </div>

      <button
        class="ghost-btn backend-refresh-btn"
        type="button"
        :disabled="backendStatusLoading"
        @click="refreshBackendStatus"
      >
        {{ backendStatusLoading ? "刷新中..." : "刷新状态" }}
      </button>

      <p v-if="backendStatus.error" class="status-text status-text-error">
        后端连接失败：{{ backendStatus.error }}
      </p>
    </section>


    <section class="workspace-card input-card">
      <div class="section-head">
        <div>
          <h3>输入文本</h3>
          <p>建议输入 1 至 3 句中文，便于分词、实体与分类结果同时展示。</p>
        </div>
        <div class="meta-pill">{{ charCount }} 字</div>
      </div>

      <textarea
        v-model="text"
        class="editor-textarea"
        rows="5"
        placeholder="例如：小明在北京大学学习自然语言处理，并计划下周去北京参加人工智能学术活动。"
      />

      <div class="example-toolbar">
        <span class="example-label">快速示例</span>
        <div class="example-row">
        <button
          v-for="example in TEXT_EXAMPLES"
          :key="example.key"
          class="example-chip"
          type="button"
          :disabled="loading"
          @click="fillExample(example.text)"
        >
          {{ example.label }}
        </button>
        </div>
      </div>

      <div class="action-row">
        <button class="primary-btn" :disabled="loading" @click="handleAnalyze">
          {{ loading ? "分析中..." : "开始分析" }}
        </button>
        <button
          class="ghost-btn soft-action-btn"
          :disabled="loading || !hasResults"
          @click="copyAnalysisResult"
        >
          复制分析结果
        </button>
        <button
          class="ghost-btn soft-action-btn"
          :disabled="loading || !hasResults"
          @click="copyAnalysisReportDescription"
        >
          复制报告描述
        </button>
        <button class="ghost-btn soft-action-btn" :disabled="loading" @click="clearAll">清空内容</button>
      </div>

      <p v-if="copyMessage" class="copy-feedback">{{ copyMessage }}</p>
    </section>

    <section class="workspace-card result-tabs-card">
      <div class="result-tabs-head">
        <div>
          <h3>分析结果</h3>
          <p>将概览、可视化和详细结果集中在同一卡片中，便于演示和截图。</p>
        </div>

        <div class="result-head-actions">
          <div class="result-quick-stats" aria-label="分析结果概要">
            <span>词数 <strong>{{ tokenCount }}</strong></span>
            <span>实体 <strong>{{ entityCount }}</strong></span>
            <span>分类 <strong>{{ label || "待返回" }}</strong></span>
            <span>状态 <strong>{{ analysisStateText }}</strong></span>
            <span>耗时 <strong>{{ formatCost(analysisCost) }}</strong></span>
          </div>

          <div class="result-tab-list" role="tablist" aria-label="分析结果">
          <button
            class="result-tab-btn"
            :class="{ active: activeResultTab === 'overview' }"
            type="button"
            role="tab"
            :aria-selected="activeResultTab === 'overview'"
            @click="activeResultTab = 'overview'"
          >
            结果概览
          </button>
          <button
            class="result-tab-btn"
            :class="{ active: activeResultTab === 'visualization' }"
            type="button"
            role="tab"
            :aria-selected="activeResultTab === 'visualization'"
            @click="activeResultTab = 'visualization'"
          >
            结果可视化
          </button>
          <button
            class="result-tab-btn"
            :class="{ active: activeResultTab === 'detail' }"
            type="button"
            role="tab"
            :aria-selected="activeResultTab === 'detail'"
            @click="activeResultTab = 'detail'"
          >
            结果详情
          </button>
        </div>
      </div>

      </div>

      <div v-if="activeResultTab === 'overview'" class="result-tab-panel" role="tabpanel">
        <div class="overview-grid">
          <article class="overview-item">
            <span>分词数量</span>
            <strong>{{ tokenCount }}</strong>
          </article>
          <article class="overview-item">
            <span>实体数量</span>
            <strong>{{ entityCount }}</strong>
          </article>
          <article class="overview-item">
            <span>分类标签</span>
            <strong>{{ label || "待返回" }}</strong>
          </article>
        </div>

        <div
          v-if="error"
          class="status-banner status-error"
          role="alert"
        >
          {{ error }}
        </div>
        <div v-else-if="loading" class="status-banner status-loading">
          正在调用后端接口并汇总三类结果，请稍候。
        </div>
        <div v-else-if="hasSubmitted && !hasResults" class="status-banner status-empty">
          本次请求已完成，但暂未返回可展示结果。
        </div>
        <div v-else class="status-banner status-neutral">
          输入文本后可以在本卡片中切换查看分词、实体识别、文本分类和可视化统计。
        </div>

        <article v-if="hasResults" class="report-preview-card">
          <div class="report-preview-head">
            <h4>报告描述预览</h4>
            <span>{{ formatCost(analysisCost) }}</span>
          </div>
          <p>{{ buildAnalysisReportDescription() }}</p>
        </article>
      </div>

      <div
        v-else-if="activeResultTab === 'visualization'"
        class="result-tab-panel"
        role="tabpanel"
      >
        <TextStatsCharts
          :tokens="tokens"
          :entities="entities"
          :loading="loading"
          :ready="hasSubmitted"
        />
      </div>

      <div v-else class="result-tab-panel" role="tabpanel">
        <section class="result-grid">
          <TokenResultCard :tokens="tokens" :loading="loading" :ready="hasSubmitted" />
          <NerResultCard
            :entities="entities"
            :source-text="trimmedText"
            :loading="loading"
            :ready="hasSubmitted"
          />
          <ClassifyResultCard
            :label="label"
            :confidence="confidence"
            :loading="loading"
            :ready="hasSubmitted"
          />
        </section>
      </div>
    </section>
  </section>
</template>

<style scoped>
.page-stack {
  display: grid;
  gap: var(--space-5);
}

.page-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(220px, 0.8fr);
  gap: var(--space-4);
  align-items: stretch;
}

.page-hero-copy h2 {
  margin: 0;
  font-size: 1.9rem;
  color: var(--color-text-strong);
}

.page-hero-copy p:last-child {
  margin: var(--space-3) 0 0;
  color: var(--color-text-muted);
  line-height: 1.75;
}

.section-kicker {
  margin: 0 0 var(--space-2);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-accent);
}

.hero-metrics {
  display: grid;
  gap: var(--space-3);
}

.metric-card {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
  box-shadow: var(--shadow-soft);
}

.metric-card span {
  color: var(--color-text-muted);
  font-size: 0.88rem;
}

.metric-card strong {
  color: var(--color-text-strong);
  font-size: 1.35rem;
}

.workspace-card {
  display: grid;
  gap: var(--space-4);
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  background: var(--color-surface);
  box-shadow: var(--shadow-soft);
}

.input-card {
  gap: var(--space-3);
}

.backend-status-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--space-3);
  align-items: center;
  padding: 0.85rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.backend-status-main {
  display: grid;
  grid-template-columns: minmax(120px, 0.5fr) minmax(0, 1fr) minmax(0, 0.9fr);
  gap: var(--space-3);
  min-width: 0;
}

.backend-status-chip {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.65rem;
  border-radius: 999px;
  background: #f8fbff;
  border: 1px solid var(--color-border-soft);
}

.backend-status-chip span {
  color: var(--color-text-muted);
  font-size: 0.82rem;
  font-weight: 700;
  white-space: nowrap;
}

.backend-status-chip strong {
  min-width: 0;
  color: var(--color-text-strong);
  font-size: 0.88rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.backend-refresh-btn {
  min-height: 36px;
  padding: 0.52rem 0.85rem;
}

.result-tabs-card {
  gap: var(--space-4);
}

.result-tabs-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  align-items: flex-start;
}

.result-head-actions {
  display: grid;
  gap: var(--space-2);
  justify-items: end;
}

.result-quick-stats {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.45rem;
}

.result-quick-stats span {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-height: 30px;
  padding: 0.32rem 0.65rem;
  border: 1px solid var(--color-border-soft);
  border-radius: 999px;
  background: #f8fbff;
  color: var(--color-text-muted);
  font-size: 0.82rem;
  font-weight: 700;
  white-space: nowrap;
}

.result-quick-stats strong {
  max-width: 108px;
  overflow: hidden;
  color: var(--color-text-strong);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-tabs-head h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--color-text-strong);
}

.result-tabs-head p {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
}

.result-tab-list {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  padding: 0.3rem;
  border: 1px solid var(--color-border-soft);
  border-radius: 999px;
  background: #f6f8fb;
}

.result-tab-btn {
  min-height: 36px;
  padding: 0.5rem 0.9rem;
  border-radius: 999px;
  background: transparent;
  color: var(--color-text-muted);
  font-size: 0.9rem;
  font-weight: 700;
  cursor: pointer;
  transition:
    background 0.18s ease,
    color 0.18s ease,
    box-shadow 0.18s ease;
}

.result-tab-btn.active {
  background: #ffffff;
  color: var(--color-accent-strong);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}

.result-tab-panel {
  display: grid;
  gap: var(--space-4);
}

.report-preview-card {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-lg);
  background: #fbfdff;
}

.report-preview-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.report-preview-head h4 {
  margin: 0;
  color: var(--color-text-strong);
  font-size: 0.98rem;
}

.report-preview-head span {
  color: var(--color-text-muted);
  font-size: 0.86rem;
  font-weight: 700;
}

.report-preview-card p {
  margin: 0;
  color: var(--color-text-muted);
  line-height: 1.7;
}

.section-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: flex-start;
}

.section-head h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--color-text-strong);
}

.section-head p {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
}

.meta-pill {
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  background: var(--color-accent-soft);
  color: var(--color-accent-strong);
  font-size: 0.86rem;
  font-weight: 700;
  white-space: nowrap;
}

.editor-textarea {
  width: 100%;
  min-height: 150px;
  padding: 1rem 1.05rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: #fbfdff;
  color: var(--color-text-strong);
  font: inherit;
  line-height: 1.7;
  resize: vertical;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.input-card .editor-textarea {
  min-height: 140px;
}

.editor-textarea:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: 0 0 0 4px rgba(40, 94, 255, 0.12);
  background: #ffffff;
}

.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}

.input-card .action-row {
  align-items: center;
  gap: 0.65rem;
}

.soft-action-btn {
  min-height: 38px;
  padding: 0.55rem 0.85rem;
  font-size: 0.88rem;
  box-shadow: none;
}

.example-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  align-items: center;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-lg);
  background: #fbfdff;
}

.example-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  align-items: center;
}

.example-label {
  color: var(--color-text-muted);
  font-size: 0.9rem;
  font-weight: 700;
}

.example-chip {
  min-height: 32px;
  padding: 0.38rem 0.7rem;
  border-radius: 999px;
  border: 1px solid #d6e0f5;
  background: #f7faff;
  color: var(--color-text-strong);
  font-weight: 700;
  font-size: 0.86rem;
  cursor: pointer;
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background 0.18s ease;
}

.example-chip:hover {
  transform: translateY(-1px);
  border-color: #c0d2f3;
  background: #eef4ff;
}

.example-chip:disabled {
  cursor: not-allowed;
  opacity: 0.7;
  transform: none;
}

.copy-feedback {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 0.92rem;
}

.status-ok {
  color: #067647;
}

.status-down {
  color: #b42318;
}

.status-text {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 0.9rem;
}

.status-text-error {
  color: #b42318;
}

.primary-btn,
.secondary-btn,
.ghost-btn {
  min-width: 124px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-3);
}

.overview-item {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border-soft);
}

.overview-item span {
  font-size: 0.86rem;
  color: var(--color-text-muted);
}

.overview-item strong {
  font-size: 1.15rem;
  color: var(--color-text-strong);
}

.status-banner {
  padding: 0.95rem 1rem;
  border-radius: var(--radius-lg);
  font-size: 0.95rem;
  line-height: 1.6;
}

.status-neutral {
  background: #f6f8fb;
  color: var(--color-text-muted);
  border: 1px solid var(--color-border-soft);
}

.status-loading {
  background: #eef4ff;
  color: #1d4ed8;
  border: 1px solid #c7d7ff;
}

.status-empty {
  background: #fff8eb;
  color: #9a6700;
  border: 1px solid #f1d79f;
}

.status-error {
  background: #fff1f1;
  color: #b42318;
  border: 1px solid #f7c3c3;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-4);
}

@media (max-width: 960px) {
  .page-hero,
  .result-grid,
  .overview-grid,
  .backend-status-bar,
  .backend-status-main {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .workspace-card {
    padding: var(--space-4);
  }

  .section-head,
  .result-tabs-head {
    flex-direction: column;
  }

  .result-tab-list {
    width: 100%;
    border-radius: var(--radius-lg);
  }

  .result-head-actions,
  .result-quick-stats {
    width: 100%;
    justify-items: stretch;
    justify-content: flex-start;
  }

  .result-tab-btn {
    flex: 1 1 auto;
  }

  .backend-status-bar {
    padding: var(--space-3);
  }

  .backend-status-chip {
    border-radius: var(--radius-lg);
  }

  .page-hero-copy h2 {
    font-size: 1.6rem;
  }

  .example-row {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
