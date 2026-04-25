<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { CLUSTER_PALETTE, getClusterColor } from "../constants/clusterPalette";

const props = defineProps({
  points: {
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

const groupedSeries = computed(() => {
  const groups = new Map();

  for (const point of props.points) {
    const cluster = Number(point.cluster || 0);
    if (!groups.has(cluster)) {
      groups.set(cluster, []);
    }
    groups.get(cluster).push({
      value: [point.x, point.y],
      title: point.title,
      cluster,
    });
  }

  return [...groups.entries()].sort((a, b) => a[0] - b[0]);
});

function buildOption() {
  return {
    color: CLUSTER_PALETTE,
    legend: {
      top: 8,
      icon: "circle",
      itemWidth: 10,
      textStyle: {
        color: "#4b5563",
      },
    },
    grid: {
      left: 48,
      right: 20,
      top: 56,
      bottom: 48,
    },
    xAxis: {
      type: "value",
      name: "降维坐标 X",
      nameLocation: "middle",
      nameGap: 28,
      axisLine: {
        lineStyle: {
          color: "#94a3b8",
        },
      },
      splitLine: {
        lineStyle: {
          color: "#e6edf8",
        },
      },
    },
    yAxis: {
      type: "value",
      name: "降维坐标 Y",
      nameLocation: "middle",
      nameGap: 38,
      axisLine: {
        lineStyle: {
          color: "#94a3b8",
        },
      },
      splitLine: {
        lineStyle: {
          color: "#e6edf8",
        },
      },
    },
    tooltip: {
      trigger: "item",
      borderWidth: 1,
      borderColor: "#d6deea",
      backgroundColor: "rgba(255,255,255,0.96)",
      textStyle: {
        color: "#1f2937",
      },
      formatter(params) {
        const point = params.data || {};
        const [x, y] = params.value || [];
        return [
          `<strong>${point.title || "未命名文档"}</strong>`,
          `所属簇：${point.cluster}`,
          `X：${typeof x === "number" ? x.toFixed(3) : x}`,
          `Y：${typeof y === "number" ? y.toFixed(3) : y}`,
        ].join("<br/>");
      },
    },
    series: groupedSeries.value.map(([cluster, data], index) => ({
      name: `簇 ${cluster}`,
      type: "scatter",
      symbolSize: 15,
      data,
      emphasis: {
        scale: true,
      },
      itemStyle: {
        color: getClusterColor(cluster, index),
        shadowBlur: 8,
        shadowColor: "rgba(15, 23, 42, 0.12)",
      },
    })),
  };
}

function renderChart() {
  if (!chartRef.value || props.loading || props.points.length === 0) {
    return;
  }

  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value);
  }

  chartInstance.setOption(buildOption(), true);
}

function handleResize() {
  if (chartInstance) {
    chartInstance.resize();
  }
}

function disposeChart() {
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
}

onMounted(async () => {
  await nextTick();
  renderChart();
  window.addEventListener("resize", handleResize);
});

watch(
  () => [props.points, props.loading],
  async ([newPoints, isLoading]) => {
    await nextTick();
    if (isLoading || !newPoints || newPoints.length === 0) {
      disposeChart();
      return;
    }
    renderChart();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
  disposeChart();
});
</script>

<template>
  <section class="cluster-card">
    <div class="card-head">
      <div>
        <h3 class="chart-title">聚类散点图</h3>
        <p class="chart-desc">不同颜色代表不同簇，点与点之间的相对距离可帮助观察文档分布差异。</p>
      </div>
      <span class="meta-pill">{{ points.length }} 个点</span>
    </div>

    <div v-if="loading" class="state-box state-loading">正在生成聚类图...</div>
    <div v-else-if="!points || points.length === 0" class="state-box state-empty">
      聚类完成后，这里会展示二维散点图。
    </div>
    <div v-else ref="chartRef" class="chart-canvas" />
  </section>
</template>

<style scoped>
.cluster-card {
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

.chart-title {
  margin: 0;
  font-size: 1.08rem;
  color: var(--color-text-strong);
}

.chart-desc {
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

.state-box {
  padding: 0.95rem 1rem;
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border-soft);
  line-height: 1.6;
}

.state-loading {
  background: #eef4ff;
  color: #1d4ed8;
}

.state-empty {
  background: var(--color-surface-muted);
  color: var(--color-text-muted);
}

.chart-canvas {
  width: 100%;
  height: 420px;
  border-radius: var(--radius-lg);
  background: #f9fbff;
}

@media (max-width: 640px) {
  .cluster-card {
    padding: var(--space-4);
  }

  .card-head {
    flex-direction: column;
  }

  .chart-canvas {
    height: 320px;
  }
}
</style>
