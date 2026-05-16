<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";

const props = defineProps({
  groups: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
});

const chartRef = ref(null);
let chartInstance = null;

const chartRows = computed(() =>
  props.groups.map((group) => ({
    name: `Cluster ${group.cluster}`,
    label: `簇 ${group.cluster}`,
    value: group.count,
    color: group.color,
  })),
);

const hasRows = computed(() => chartRows.value.length > 0);

function buildOption() {
  return {
    grid: {
      left: 38,
      right: 14,
      top: 18,
      bottom: 34,
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
      formatter(params) {
        const item = params?.[0];
        const row = chartRows.value.find((entry) => entry.label === item?.axisValue);
        return `${row?.label || item?.axisValue || ""}<br/>文档数量：${item?.value ?? 0}`;
      },
    },
    xAxis: {
      type: "category",
      data: chartRows.value.map((item) => item.label),
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
        name: "文档数量",
        type: "bar",
        barMaxWidth: 34,
        data: chartRows.value.map((item) => ({
          value: item.value,
          itemStyle: {
            color: item.color,
            borderRadius: [6, 6, 0, 0],
          },
        })),
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

function renderChart() {
  if (!chartRef.value || props.loading || !hasRows.value) {
    disposeChart();
    return;
  }

  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value);
  }

  chartInstance.setOption(buildOption(), true);
}

async function refreshChart() {
  await nextTick();
  renderChart();
}

function handleResize() {
  chartInstance?.resize();
}

function disposeChart() {
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
}

onMounted(() => {
  refreshChart();
  window.addEventListener("resize", handleResize);
});

watch(
  () => [props.groups, props.loading],
  () => {
    refreshChart();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
  disposeChart();
});
</script>

<template>
  <section class="cluster-stats-card">
    <div class="chart-head">
      <h4>各簇文档数量</h4>
      <span>{{ groups.length }} 个簇</span>
    </div>

    <div v-if="loading" class="chart-state">正在统计簇内文档数量...</div>
    <div v-else-if="!hasRows" class="chart-state">聚类完成后展示各簇文档数量。</div>
    <div v-else ref="chartRef" class="chart-canvas" />
  </section>
</template>

<style scoped>
.cluster-stats-card {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-3);
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-lg);
  background: #f9fbff;
}

.chart-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.chart-head h4 {
  margin: 0;
  color: var(--color-text-strong);
  font-size: 0.98rem;
}

.chart-head span {
  color: var(--color-text-muted);
  font-size: 0.84rem;
  font-weight: 700;
  white-space: nowrap;
}

.chart-canvas {
  width: 100%;
  height: 220px;
}

.chart-state {
  display: grid;
  min-height: 160px;
  place-items: center;
  padding: 0.9rem;
  border-radius: var(--radius-lg);
  background: var(--color-surface-muted);
  color: var(--color-text-muted);
  line-height: 1.6;
  text-align: center;
}

@media (max-width: 640px) {
  .chart-canvas {
    height: 200px;
  }
}
</style>
