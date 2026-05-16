<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";

const props = defineProps({
  tokens: {
    type: Array,
    default: () => [],
  },
  entities: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  ready: {
    type: Boolean,
    default: false,
  },
});

const ENTITY_LABEL_NAMES = {
  PER: "人名",
  LOC: "地点",
  ORG: "机构",
  GPE: "行政地域",
  FAC: "设施/景点",
  COMPANY: "公司",
  INSTITUTION: "单位",
  MISC: "其他实体",
};

const ENTITY_LABEL_ORDER = [
  "PER",
  "LOC",
  "GPE",
  "FAC",
  "ORG",
  "COMPANY",
  "INSTITUTION",
  "MISC",
];

const tokenChartRef = ref(null);
const entityChartRef = ref(null);
let tokenChart = null;
let entityChart = null;

const hasTokens = computed(() => props.tokens.length > 0);
const hasEntities = computed(() => props.entities.length > 0);
const hasAnyResult = computed(() => hasTokens.value || hasEntities.value);

const tokenLengthRows = computed(() => {
  const buckets = new Map([
    ["1", 0],
    ["2", 0],
    ["3", 0],
    ["4", 0],
    ["5+", 0],
  ]);

  for (const token of props.tokens) {
    const length = Array.from(String(token || "")).length;
    if (length <= 0) {
      continue;
    }
    const key = length >= 5 ? "5+" : String(length);
    buckets.set(key, (buckets.get(key) || 0) + 1);
  }

  return [...buckets.entries()].map(([name, value]) => ({ name, value }));
});

const entityLabelRows = computed(() => {
  const counts = new Map();

  for (const entity of props.entities) {
    const label = String(entity?.label || "MISC").toUpperCase();
    counts.set(label, (counts.get(label) || 0) + 1);
  }

  return [...counts.entries()]
    .sort(([a], [b]) => {
      const indexA = ENTITY_LABEL_ORDER.indexOf(a);
      const indexB = ENTITY_LABEL_ORDER.indexOf(b);
      const orderA = indexA === -1 ? ENTITY_LABEL_ORDER.length : indexA;
      const orderB = indexB === -1 ? ENTITY_LABEL_ORDER.length : indexB;
      return orderA - orderB || a.localeCompare(b);
    })
    .map(([label, value]) => ({
      name: label,
      displayName: ENTITY_LABEL_NAMES[label] || label,
      value,
    }));
});

function buildBarOption({ title, xName, yName, rows, color, formatter }) {
  return {
    color: [color],
    title: {
      text: title,
      left: 12,
      top: 8,
      textStyle: {
        color: "#1f2937",
        fontSize: 15,
        fontWeight: 700,
      },
    },
    grid: {
      left: 46,
      right: 20,
      top: 64,
      bottom: 48,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
      borderWidth: 1,
      borderColor: "#d6deea",
      backgroundColor: "rgba(255,255,255,0.96)",
      textStyle: {
        color: "#1f2937",
      },
      formatter,
    },
    xAxis: {
      type: "category",
      name: xName,
      nameLocation: "middle",
      nameGap: 30,
      data: rows.map((item) => item.name),
      axisLabel: {
        color: "#4b5563",
      },
      axisLine: {
        lineStyle: {
          color: "#94a3b8",
        },
      },
    },
    yAxis: {
      type: "value",
      name: yName,
      minInterval: 1,
      axisLabel: {
        color: "#4b5563",
      },
      splitLine: {
        lineStyle: {
          color: "#e6edf8",
        },
      },
    },
    series: [
      {
        type: "bar",
        data: rows.map((item) => item.value),
        barMaxWidth: 42,
        itemStyle: {
          borderRadius: [6, 6, 0, 0],
        },
        label: {
          show: true,
          position: "top",
          color: "#1f2937",
          fontWeight: 700,
        },
      },
    ],
  };
}

function renderTokenChart() {
  if (!tokenChartRef.value || props.loading || !hasTokens.value) {
    disposeTokenChart();
    return;
  }

  if (!tokenChart) {
    tokenChart = echarts.init(tokenChartRef.value);
  }

  tokenChart.setOption(
    buildBarOption({
      title: "分词长度分布",
      xName: "词语长度",
      yName: "数量",
      rows: tokenLengthRows.value,
      color: "#2f6bff",
      formatter(params) {
        const item = params?.[0];
        return `${item?.axisValue || ""} 字词语<br/>数量：${item?.value ?? 0}`;
      },
    }),
    true,
  );
}

function renderEntityChart() {
  if (!entityChartRef.value || props.loading || !hasEntities.value) {
    disposeEntityChart();
    return;
  }

  if (!entityChart) {
    entityChart = echarts.init(entityChartRef.value);
  }

  entityChart.setOption(
    buildBarOption({
      title: "实体类别统计",
      xName: "实体类别",
      yName: "数量",
      rows: entityLabelRows.value,
      color: "#16a37a",
      formatter(params) {
        const item = params?.[0];
        const row = entityLabelRows.value.find((entry) => entry.name === item?.axisValue);
        return `${row?.displayName || item?.axisValue || ""} (${item?.axisValue || ""})<br/>数量：${
          item?.value ?? 0
        }`;
      },
    }),
    true,
  );
}

async function renderCharts() {
  await nextTick();
  renderTokenChart();
  renderEntityChart();
}

function handleResize() {
  tokenChart?.resize();
  entityChart?.resize();
}

function disposeTokenChart() {
  if (tokenChart) {
    tokenChart.dispose();
    tokenChart = null;
  }
}

function disposeEntityChart() {
  if (entityChart) {
    entityChart.dispose();
    entityChart = null;
  }
}

function disposeCharts() {
  disposeTokenChart();
  disposeEntityChart();
}

onMounted(() => {
  renderCharts();
  window.addEventListener("resize", handleResize);
});

watch(
  () => [props.tokens, props.entities, props.loading],
  () => {
    renderCharts();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
  disposeCharts();
});
</script>

<template>
  <section class="stats-card">
    <div class="card-head">
      <div>
        <p class="section-kicker">Result Visualization</p>
        <h3>结果可视化</h3>
        <p>基于当前分词与实体识别结果生成统计图，便于课堂展示和实验报告截图。</p>
      </div>
      <span class="meta-pill">{{ tokens.length }} 词 / {{ entities.length }} 实体</span>
    </div>

    <div v-if="loading" class="state-box state-loading">正在生成统计图表，请稍候。</div>
    <div v-else-if="!ready || !hasAnyResult" class="state-box state-empty">
      完成一次文本分析后，这里会展示分词长度分布和实体类别统计。
    </div>
    <div v-else class="chart-grid">
      <article class="chart-panel">
        <div v-if="hasTokens" ref="tokenChartRef" class="chart-canvas" />
        <div v-else class="mini-empty">暂无分词结果，无法生成长度分布。</div>
      </article>

      <article class="chart-panel">
        <div v-if="hasEntities" ref="entityChartRef" class="chart-canvas" />
        <div v-else class="mini-empty">暂无实体结果，无法生成类别统计。</div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.stats-card {
  display: grid;
  gap: var(--space-4);
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
  box-shadow: var(--shadow-soft);
}

.card-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: flex-start;
}

.section-kicker {
  margin: 0 0 var(--space-2);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-accent);
}

.card-head h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--color-text-strong);
}

.card-head p:last-child {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
}

.meta-pill {
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  background: var(--color-accent-soft);
  color: var(--color-accent-strong);
  font-size: 0.84rem;
  font-weight: 700;
  white-space: nowrap;
}

.state-box,
.mini-empty {
  padding: 0.95rem 1rem;
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border-soft);
  line-height: 1.6;
}

.state-loading {
  background: #eef4ff;
  color: #1d4ed8;
}

.state-empty,
.mini-empty {
  background: var(--color-surface-muted);
  color: var(--color-text-muted);
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-4);
}

.chart-panel {
  min-width: 0;
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-lg);
  background: #f9fbff;
  overflow: hidden;
}

.chart-canvas {
  width: 100%;
  height: 340px;
}

.mini-empty {
  display: grid;
  min-height: 340px;
  place-items: center;
  text-align: center;
}

@media (max-width: 900px) {
  .chart-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .stats-card {
    padding: var(--space-4);
  }

  .card-head {
    flex-direction: column;
  }

  .chart-canvas,
  .mini-empty {
    height: 300px;
    min-height: 300px;
  }
}
</style>
