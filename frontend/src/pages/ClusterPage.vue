<script setup>
import { computed, ref } from "vue";
import { clusterDocuments } from "../api/nlp";
import ClusterChart from "../components/ClusterChart.vue";
import { getClusterColor } from "../constants/clusterPalette";

const SAMPLE_INPUT = [
  "科技观察::北京人工智能实验室发布多模态模型评测结果，研究团队表示中文理解能力继续提升。",
  "科技产业::上海芯片企业公布新一代算力平台，计划服务自动驾驶与智能制造场景。",
  "财经快讯::多家基金公司调整新能源板块持仓结构，市场关注季度财报与现金流表现。",
  "资本市场::券商分析师认为消费与高端制造板块有望在下阶段获得更多资金关注。",
  "体育赛报::中国女排在国际邀请赛中连胜两场，主教练强调拦网和一传质量明显提高。",
  "体育评论::杭州马拉松吸引大量选手参赛，赛事组织与城市服务获得跑者积极评价。",
].join("\n");

const rawInput = ref("");
const clusterCount = ref("3");
const loading = ref(false);
const error = ref("");
const points = ref([]);
const hasSubmitted = ref(false);
const copyMessage = ref("");

function parseLinesToDocuments(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const sepIndex = line.indexOf("::");
      if (sepIndex === -1) {
        throw new Error("每一行都必须使用“标题::内容”的输入格式。");
      }

      const title = line.slice(0, sepIndex).trim();
      const content = line.slice(sepIndex + 2).trim();

      if (!title || !content) {
        throw new Error("每一行都需要同时包含非空标题和正文内容。");
      }

      return {
        title,
        text: content,
      };
    });
}

const parsedPreview = computed(() => {
  try {
    return parseLinesToDocuments(rawInput.value);
  } catch {
    return [];
  }
});

const documentCount = computed(() => parsedPreview.value.length);
const previewRows = computed(() =>
  parsedPreview.value.map((item, index) => ({
    index: index + 1,
    title: item.title,
    summary: item.text.length > 48 ? `${item.text.slice(0, 48)}...` : item.text,
  })),
);
const clusterGroups = computed(() => {
  const groups = new Map();

  for (const point of points.value) {
    const key = Number(point.cluster);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(point.title);
  }

  return [...groups.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([cluster, titles]) => ({
      cluster,
      count: titles.length,
      titles,
      color: getClusterColor(cluster),
    }));
});
const hasResults = computed(() => points.value.length > 0);
const effectiveClusterCount = computed(() => {
  if (clusterCount.value.trim()) {
    return clusterCount.value.trim();
  }
  return hasResults.value ? String(clusterGroups.value.length) : "自动";
});

function fillExample() {
  rawInput.value = SAMPLE_INPUT;
  clusterCount.value = "3";
  error.value = "";
  copyMessage.value = "";
}

function clearAll() {
  rawInput.value = "";
  clusterCount.value = "3";
  error.value = "";
  points.value = [];
  hasSubmitted.value = false;
  copyMessage.value = "";
}

function buildClusterSummary() {
  return [
    "P4 聚类摘要",
    "",
    ...clusterGroups.value.map((group) => [
      `簇 ${group.cluster}（${group.count} 篇）`,
      ...group.titles.map((title) => `- ${title}`),
      "",
    ]).flat(),
  ]
    .join("\n")
    .trim();
}

async function copyClusterSummary() {
  if (!hasResults.value) {
    copyMessage.value = "当前没有可复制的聚类摘要。";
    return;
  }

  const summaryText = buildClusterSummary();

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(summaryText);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = summaryText;
      textarea.setAttribute("readonly", "readonly");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }

    copyMessage.value = "聚类摘要已复制到剪贴板。";
  } catch {
    copyMessage.value = "复制失败，请稍后重试。";
  }
}

async function handleCluster() {
  error.value = "";
  points.value = [];
  copyMessage.value = "";

  let documents;
  try {
    documents = parseLinesToDocuments(rawInput.value);
  } catch (e) {
    error.value = e instanceof Error ? e.message : "输入格式有误，请检查后重试。";
    hasSubmitted.value = false;
    return;
  }

  if (documents.length < 2) {
    error.value = "请至少输入两条文档，再进行聚类分析。";
    hasSubmitted.value = false;
    return;
  }

  let count;
  if (clusterCount.value.trim()) {
    count = Number(clusterCount.value);
    if (!Number.isInteger(count) || count < 2) {
      error.value = "cluster_count 必须是大于等于 2 的整数。";
      hasSubmitted.value = false;
      return;
    }
  }

  hasSubmitted.value = true;
  loading.value = true;

  try {
    const res = await clusterDocuments(documents, count);
    points.value = res?.points || [];
  } catch (e) {
    error.value = e instanceof Error ? e.message : "聚类失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="page-stack">
    <header class="page-hero">
      <div class="page-hero-copy">
        <p class="section-kicker">Document Clustering</p>
        <h2>聚类分析</h2>
        <p>
          输入多条文档后，系统将返回二维聚类散点图，帮助观察不同文本在主题上的分布情况，
          适合在实验报告中展示聚类结果与文档归属。
        </p>
      </div>

      <div class="hero-metrics">
        <article class="metric-card">
          <span>文档条数</span>
          <strong>{{ documentCount }}</strong>
        </article>
        <article class="metric-card">
          <span>聚类数量</span>
          <strong>{{ effectiveClusterCount }}</strong>
        </article>
      </div>
    </header>

    <section class="workspace-card">
      <div class="section-head">
        <div>
          <h3>输入文档</h3>
          <p>每行一条文档，格式为“标题::内容”。至少输入两条文档以生成聚类结果。</p>
        </div>
      </div>

      <textarea
        v-model="rawInput"
        class="editor-textarea"
        rows="9"
        placeholder="科技新闻::人工智能与芯片技术持续推动科技产业升级。"
      />

      <div class="form-grid">
        <label class="field-block">
          <span>cluster_count（可选）</span>
          <input
            v-model="clusterCount"
            type="number"
            min="2"
            placeholder="为空时由后端默认值决定"
          />
        </label>

        <div class="field-note">
          <strong>输入提示</strong>
          <p>建议准备 3 至 6 条主题差异明显的文本，更容易得到适合截图的聚类分布。</p>
        </div>
      </div>

      <div class="preview-block">
        <div class="preview-head">
          <h4>文档预览表</h4>
          <span class="meta-pill">{{ previewRows.length }} 条</span>
        </div>

        <div v-if="previewRows.length === 0" class="preview-empty">
          按“标题::内容”格式输入后，这里会自动展示解析出的标题和正文摘要。
        </div>

        <div v-else class="preview-table-wrap">
          <table class="preview-table">
            <thead>
              <tr>
                <th>序号</th>
                <th>标题</th>
                <th>正文摘要</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in previewRows" :key="`${row.title}-${row.summary}`">
                <td>{{ row.index }}</td>
                <td>{{ row.title }}</td>
                <td>{{ row.summary }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="action-row">
        <button class="primary-btn" :disabled="loading" @click="handleCluster">
          {{ loading ? "聚类中..." : "开始聚类" }}
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
          <h3>结果说明</h3>
          <p>图表用于展示文档相对位置，右侧摘要用于说明每个簇中包含哪些文档。</p>
        </div>
      </div>

      <div
        v-if="error"
        class="status-banner status-error"
        role="alert"
      >
        {{ error }}
      </div>
      <div v-else-if="loading" class="status-banner status-loading">
        正在执行聚类与二维降维，请稍候查看散点图。
      </div>
      <div v-else-if="hasSubmitted && !hasResults" class="status-banner status-empty">
        本次请求已完成，但暂未返回可展示的聚类点。
      </div>
      <div v-else class="status-banner status-neutral">
        完成聚类后，系统会同时展示散点图、文档条数和各簇文档摘要。
      </div>

      <div class="result-layout">
        <ClusterChart :points="points" :loading="loading" />

        <aside class="summary-card">
          <div class="summary-head">
            <h3>聚类摘要</h3>
            <div class="summary-actions">
              <span class="meta-pill">{{ hasResults ? `${points.length} 个点` : "等待结果" }}</span>
              <button
                class="summary-copy-btn"
                type="button"
                :disabled="loading || !hasResults"
                @click="copyClusterSummary"
              >
                复制聚类摘要
              </button>
            </div>
          </div>

          <div class="summary-stats">
            <article class="overview-item">
              <span>返回文档</span>
              <strong>{{ points.length }}</strong>
            </article>
            <article class="overview-item">
              <span>识别簇数</span>
              <strong>{{ clusterGroups.length }}</strong>
            </article>
          </div>

          <p v-if="!hasResults" class="summary-empty">
            聚类完成后，这里会列出每个簇包含的文档标题，便于报告中解释结果。
          </p>
          <p v-if="hasResults && copyMessage" class="summary-copy-feedback">{{ copyMessage }}</p>

          <div v-if="hasResults" class="cluster-group-list">
            <article
              v-for="group in clusterGroups"
              :key="group.cluster"
              class="cluster-group-card"
            >
              <div class="cluster-group-head">
                <div class="cluster-badge">
                  <span
                    class="cluster-dot"
                    :style="{ backgroundColor: group.color }"
                  />
                  <strong>簇 {{ group.cluster }}</strong>
                </div>
                <span>{{ group.count }} 篇文档</span>
              </div>

              <ul class="cluster-title-list">
                <li v-for="title in group.titles" :key="title">{{ title }}</li>
              </ul>
            </article>
          </div>
        </aside>
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
  background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
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

.section-head h3,
.summary-head h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--color-text-strong);
}

.section-head p {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
}

.editor-textarea,
.field-block input {
  width: 100%;
  padding: 1rem 1.05rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: #fbfdff;
  color: var(--color-text-strong);
  font: inherit;
  line-height: 1.7;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.editor-textarea {
  min-height: 220px;
  resize: vertical;
}

.editor-textarea:focus,
.field-block input:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: 0 0 0 4px rgba(40, 94, 255, 0.12);
  background: #ffffff;
}

.form-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
  gap: var(--space-4);
}

.field-block {
  display: grid;
  gap: var(--space-2);
}

.field-block span {
  color: var(--color-text-muted);
  font-size: 0.9rem;
}

.field-note {
  padding: var(--space-4);
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-lg);
  background: var(--color-surface-muted);
}

.field-note strong {
  color: var(--color-text-strong);
}

.field-note p,
.summary-empty {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
}

.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}

.preview-block {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #fbfdff;
  border: 1px solid var(--color-border-soft);
}

.preview-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.preview-head h4 {
  margin: 0;
  color: var(--color-text-strong);
  font-size: 1rem;
}

.preview-empty {
  padding: 0.9rem 1rem;
  border-radius: var(--radius-lg);
  background: var(--color-surface-muted);
  color: var(--color-text-muted);
  line-height: 1.6;
}

.preview-table-wrap {
  overflow-x: auto;
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
}

.preview-table th,
.preview-table td {
  padding: 0.9rem 0.85rem;
  border-bottom: 1px solid var(--color-border-soft);
  text-align: left;
  vertical-align: top;
}

.preview-table th {
  color: var(--color-text-strong);
  font-size: 0.88rem;
}

.preview-table th:first-child,
.preview-table td:first-child {
  width: 76px;
}

.preview-table td {
  color: var(--color-text-muted);
  line-height: 1.6;
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

.result-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
  gap: var(--space-4);
  align-items: start;
}

.summary-card {
  display: grid;
  gap: var(--space-4);
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
}

.summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
}

.summary-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.75rem;
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

.summary-copy-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0.52rem 0.9rem;
  border-radius: 999px;
  border: 1px solid #d3dfff;
  background: #eef4ff;
  color: var(--color-accent-strong);
  font-size: 0.88rem;
  font-weight: 700;
  cursor: pointer;
}

.summary-copy-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
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

.cluster-group-list {
  display: grid;
  gap: var(--space-3);
}

.summary-copy-feedback {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 0.92rem;
}

.cluster-group-card {
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #fbfdff;
  border: 1px solid var(--color-border-soft);
}

.cluster-group-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
  color: var(--color-text-strong);
}

.cluster-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

.cluster-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  box-shadow: 0 0 0 3px rgba(43, 99, 240, 0.12);
}

.cluster-group-head span {
  color: var(--color-text-muted);
  font-size: 0.9rem;
}

.cluster-title-list {
  margin: 0;
  padding-left: 1.1rem;
  color: var(--color-text-muted);
  display: grid;
  gap: var(--space-2);
}

@media (max-width: 960px) {
  .page-hero,
  .form-grid,
  .result-layout,
  .summary-stats {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .workspace-card,
  .summary-card {
    padding: var(--space-4);
  }

  .section-head,
  .summary-head {
    flex-direction: column;
  }

  .summary-actions,
  .preview-head,
  .cluster-group-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .page-hero-copy h2 {
    font-size: 1.6rem;
  }
}
</style>
