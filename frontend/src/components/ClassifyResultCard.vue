<script setup>
import { computed } from "vue";

const props = defineProps({
  label: {
    type: String,
    default: "",
  },
  confidence: {
    type: [Number, String, null],
    default: null,
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

const confidenceValue = computed(() => {
  if (props.confidence === null || props.confidence === "") {
    return null;
  }
  const value = Number(props.confidence);
  if (Number.isNaN(value)) {
    return null;
  }
  return Math.max(0, Math.min(1, value));
});

const confidencePercent = computed(() => {
  if (confidenceValue.value === null) {
    return null;
  }
  return `${(confidenceValue.value * 100).toFixed(1)}%`;
});
</script>

<template>
  <section class="result-card">
    <div class="card-head">
      <div>
        <h3 class="result-title">文本分类</h3>
        <p class="result-desc">突出显示预测标签，并用置信度刻画当前分类结果的稳定程度。</p>
      </div>
    </div>

    <div v-if="loading" class="state-box state-loading">正在预测文本类别...</div>
    <div v-else-if="!ready" class="state-box state-empty">提交文本后，这里将显示分类结果。</div>
    <div v-else-if="!label" class="state-box state-empty">当前文本未返回可展示的分类标签。</div>

    <div v-else class="classification-panel">
      <div class="label-block">
        <span class="label-caption">预测标签</span>
        <strong class="label-value">{{ label }}</strong>
      </div>

      <div class="confidence-block">
        <div class="confidence-row">
          <span>置信度</span>
          <strong>{{ confidencePercent || "未提供" }}</strong>
        </div>

        <div class="confidence-bar">
          <div
            class="confidence-fill"
            :style="{ width: confidencePercent || '0%' }"
          />
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.result-card {
  display: grid;
  gap: var(--space-4);
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
  box-shadow: var(--shadow-soft);
}

.result-title {
  margin: 0;
  font-size: 1.08rem;
  color: var(--color-text-strong);
}

.result-desc {
  margin: var(--space-2) 0 0;
  color: var(--color-text-muted);
  line-height: 1.65;
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

.classification-panel {
  display: grid;
  gap: var(--space-4);
}

.label-block {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #f3f7ff;
  border: 1px solid #d7e3ff;
}

.label-caption,
.confidence-row span {
  color: var(--color-text-muted);
  font-size: 0.9rem;
}

.label-value {
  font-size: 1.65rem;
  color: var(--color-text-strong);
}

.confidence-block {
  display: grid;
  gap: var(--space-3);
}

.confidence-row {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.confidence-row strong {
  color: var(--color-text-strong);
}

.confidence-bar {
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: #e7eefb;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #3f7cff 0%, #1d4ed8 100%);
}
</style>
