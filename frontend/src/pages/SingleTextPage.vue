<script setup>
import { computed, ref } from "vue";
import { tokenizeText, recognizeEntities, classifyText } from "../api/nlp";
import TokenResultCard from "../components/TokenResultCard.vue";
import NerResultCard from "../components/NerResultCard.vue";
import ClassifyResultCard from "../components/ClassifyResultCard.vue";

const SAMPLE_TEXT =
  "小明在北京大学学习自然语言处理，并计划下周去北京参加人工智能学术活动。";

const text = ref("");
const loading = ref(false);
const error = ref("");
const tokens = ref([]);
const entities = ref([]);
const label = ref("");
const confidence = ref(null);
const hasSubmitted = ref(false);

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
}

function fillExample() {
  text.value = SAMPLE_TEXT;
  error.value = "";
}

function clearAll() {
  text.value = "";
  error.value = "";
  hasSubmitted.value = false;
  resetResults();
}

async function handleAnalyze() {
  const input = trimmedText.value;

  if (!input) {
    error.value = "请输入需要分析的中文文本。";
    hasSubmitted.value = false;
    resetResults();
    return;
  }

  loading.value = true;
  error.value = "";
  hasSubmitted.value = true;
  resetResults();

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
  } catch (e) {
    error.value = e instanceof Error ? e.message : "分析失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="page-stack">
    <header class="page-hero">
      <div class="page-hero-copy">
        <p class="section-kicker">Single Text Workflow</p>
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

    <section class="workspace-card">
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
        rows="7"
        placeholder="例如：小明在北京大学学习自然语言处理，并计划下周去北京参加人工智能学术活动。"
      />

      <div class="action-row">
        <button class="primary-btn" :disabled="loading" @click="handleAnalyze">
          {{ loading ? "分析中..." : "开始分析" }}
        </button>
        <button class="secondary-btn" :disabled="loading" @click="fillExample">
          填入示例
        </button>
        <button class="ghost-btn" :disabled="loading" @click="clearAll">清空内容</button>
      </div>
    </section>

    <section class="workspace-card">
      <div class="section-head">
        <div>
          <h3>结果概览</h3>
          <p>先看总体状态，再查看下方三张结果卡片。</p>
        </div>
      </div>

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
        输入文本后可以在下方同时查看分词、实体识别和文本分类结果。
      </div>
    </section>

    <section class="result-grid">
      <TokenResultCard :tokens="tokens" :loading="loading" :ready="hasSubmitted" />
      <NerResultCard :entities="entities" :loading="loading" :ready="hasSubmitted" />
      <ClassifyResultCard
        :label="label"
        :confidence="confidence"
        :loading="loading"
        :ready="hasSubmitted"
      />
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
  min-height: 188px;
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
  .overview-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .workspace-card {
    padding: var(--space-4);
  }

  .section-head {
    flex-direction: column;
  }

  .page-hero-copy h2 {
    font-size: 1.6rem;
  }
}
</style>
