<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";

const props = defineProps({
  points: {
    type: Array,
    default: () => [],
  },
});

const chartRef = ref(null);
let chartInstance = null;

function buildOption() {
  const data = props.points.map((item) => ({
    value: [item.x, item.y],
    title: item.title,
    cluster: item.cluster,
  }));

  return {
    grid: {
      left: 40,
      right: 20,
      top: 30,
      bottom: 40,
    },
    xAxis: {
      type: "value",
      name: "X",
      nameLocation: "middle",
      nameGap: 24,
      splitLine: {
        lineStyle: {
          color: "#eef1f5",
        },
      },
    },
    yAxis: {
      type: "value",
      name: "Y",
      nameLocation: "middle",
      nameGap: 30,
      splitLine: {
        lineStyle: {
          color: "#eef1f5",
        },
      },
    },
    tooltip: {
      trigger: "item",
      formatter(params) {
        const point = params.data || {};
        return `title: ${point.title}<br/>cluster: ${point.cluster}<br/>x: ${params.value?.[0]}<br/>y: ${params.value?.[1]}`;
      },
    },
    series: [
      {
        type: "scatter",
        symbolSize: 12,
        data,
        itemStyle: {
          color(params) {
            const palette = [
              "#2563eb",
              "#16a34a",
              "#f59e0b",
              "#ef4444",
              "#8b5cf6",
              "#06b6d4",
            ];
            const cluster = Number(params.data?.cluster || 0);
            const idx = Math.abs(cluster) % palette.length;
            return palette[idx];
          },
        },
      },
    ],
  };
}

function renderChart() {
  if (!chartRef.value || props.points.length === 0) {
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
  () => props.points,
  async (newPoints) => {
    await nextTick();
    if (!newPoints || newPoints.length === 0) {
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
    <h3 class="chart-title">聚类散点图</h3>

    <p v-if="!points || points.length === 0" class="empty-text">暂无聚类结果</p>
    <div v-else ref="chartRef" class="chart-canvas" />
  </section>
</template>

<style scoped>
.cluster-card {
  border: 1px solid #dcdfe6;
  border-radius: 10px;
  background: #ffffff;
  padding: 14px;
}

.chart-title {
  margin: 0 0 10px;
  font-size: 16px;
  color: #111827;
}

.empty-text {
  margin: 0;
  color: #6b7280;
  line-height: 1.6;
}

.chart-canvas {
  width: 100%;
  height: 360px;
  border-radius: 8px;
  background: #fafbfd;
}
</style>
