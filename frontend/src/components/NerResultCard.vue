<script setup>
const LABEL_TEXT_MAP = {
  PER: "人名",
  LOC: "地点",
  ORG: "机构",
};

defineProps({
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

function getLabelText(label) {
  return LABEL_TEXT_MAP[label] || label;
}
</script>

<template>
  <section class="result-card">
    <div class="card-head">
      <div>
        <h3 class="result-title">命名实体识别</h3>
        <p class="result-desc">展示文本中的实体内容、实体类型，以及它在原文中的字符位置。</p>
      </div>
    </div>

    <div v-if="loading" class="state-box state-loading">正在识别实体...</div>
    <div v-else-if="!ready" class="state-box state-empty">提交文本后，这里将显示实体识别结果。</div>
    <div v-else-if="entities.length === 0" class="state-box state-empty">当前文本未识别到可展示的命名实体。</div>

    <div v-else class="entity-list">
      <article v-for="(item, index) in entities" :key="`${item.text}-${index}`" class="entity-card">
        <div class="entity-head">
          <strong>{{ item.text }}</strong>
          <span class="entity-label">{{ item.label }}</span>
        </div>

        <div class="entity-meta">
          <span>{{ getLabelText(item.label) }}</span>
          <span>起始 {{ item.start }}</span>
          <span>结束 {{ item.end }}</span>
        </div>
      </article>
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

.card-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: flex-start;
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

.entity-list {
  display: grid;
  gap: var(--space-3);
}

.entity-card {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #fcfdff;
  border: 1px solid var(--color-border-soft);
}

.entity-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.entity-head strong {
  font-size: 1rem;
  color: var(--color-text-strong);
}

.entity-label {
  padding: 0.35rem 0.72rem;
  border-radius: 999px;
  background: #eef4ff;
  color: #2354cc;
  font-size: 0.82rem;
  font-weight: 700;
}

.entity-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  color: var(--color-text-muted);
  font-size: 0.9rem;
}
</style>
