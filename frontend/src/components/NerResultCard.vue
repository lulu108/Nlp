<script setup>
import { computed } from "vue";

const LABEL_TEXT_MAP = {
  PER: "人名",
  LOC: "地点",
  ORG: "机构",
};

const LABEL_CLASS_MAP = {
  PER: "entity-highlight-per",
  LOC: "entity-highlight-loc",
  ORG: "entity-highlight-org",
};

const ENTITY_LEGEND = [
  { label: "PER", text: "人名" },
  { label: "LOC", text: "地点" },
  { label: "ORG", text: "机构" },
];

const props = defineProps({
  entities: {
    type: Array,
    default: () => [],
  },
  sourceText: {
    type: String,
    default: "",
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

function getLabelClass(label) {
  return LABEL_CLASS_MAP[label] || "entity-highlight-default";
}

const normalizedEntities = computed(() =>
  [...props.entities]
    .filter(
      (item) =>
        typeof item?.start === "number" &&
        typeof item?.end === "number" &&
        item.start >= 0 &&
        item.end > item.start,
    )
    .sort((a, b) => a.start - b.start || b.end - a.end),
);

const highlightedSegments = computed(() => {
  if (!props.sourceText) {
    return [];
  }

  const segments = [];
  let cursor = 0;

  for (const entity of normalizedEntities.value) {
    const start = Math.max(cursor, entity.start);
    const end = Math.min(props.sourceText.length, entity.end);

    if (start > cursor) {
      segments.push({
        text: props.sourceText.slice(cursor, start),
        highlight: false,
      });
    }

    if (end > start) {
      segments.push({
        text: props.sourceText.slice(start, end),
        highlight: true,
        label: entity.label,
        className: getLabelClass(entity.label),
      });
      cursor = end;
    }
  }

  if (cursor < props.sourceText.length) {
    segments.push({
      text: props.sourceText.slice(cursor),
      highlight: false,
    });
  }

  return segments.length > 0
    ? segments
    : [
        {
          text: props.sourceText,
          highlight: false,
        },
      ];
});
</script>

<template>
  <section class="result-card">
    <div class="card-head">
      <div>
        <h3 class="result-title">命名实体识别</h3>
        <p class="result-desc">
          展示文本中的实体内容、实体类型，以及它在原文中的字符位置。
        </p>
      </div>
    </div>

    <div v-if="loading" class="state-box state-loading">正在识别实体...</div>
    <div v-else-if="!ready" class="state-box state-empty">
      提交文本后，这里将显示实体识别结果。
    </div>
    <div v-else-if="entities.length === 0" class="state-box state-empty">
      当前文本未识别到可展示的命名实体。
    </div>

    <div v-else class="entity-list">
      <div class="legend-row" aria-label="实体颜色图例">
        <span class="legend-label">颜色图例</span>
        <div class="legend-items">
          <span
            v-for="item in ENTITY_LEGEND"
            :key="item.label"
            class="legend-chip"
            :class="getLabelClass(item.label)"
          >
            <span class="legend-dot" />
            {{ item.text }}
          </span>
        </div>
      </div>

      <div class="highlight-panel">
        <div class="highlight-head">
          <strong>原文高亮</strong>
          <span>按 start / end 位置标注实体</span>
        </div>

        <p class="highlight-text">
          <template
            v-for="(segment, index) in highlightedSegments"
            :key="`${segment.text}-${index}`"
          >
            <mark
              v-if="segment.highlight"
              class="entity-highlight"
              :class="segment.className"
            >
              {{ segment.text }}
              <span class="highlight-inline-label">{{ getLabelText(segment.label) }}</span>
            </mark>
            <span v-else>{{ segment.text }}</span>
          </template>
        </p>
      </div>

      <article
        v-for="(item, index) in entities"
        :key="`${item.text}-${index}`"
        class="entity-card"
      >
        <div class="entity-head">
          <strong>{{ item.text }}</strong>
          <span class="entity-label" :class="getLabelClass(item.label)">{{
            item.label
          }}</span>
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

.legend-row {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #fbfdff;
  border: 1px solid var(--color-border-soft);
}

.legend-label {
  color: var(--color-text-muted);
  font-size: 0.88rem;
  font-weight: 700;
}

.legend-items {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
}

.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.48rem 0.8rem;
  border-radius: 999px;
  font-size: 0.86rem;
  font-weight: 700;
  color: var(--color-text-strong);
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: currentColor;
  opacity: 0.72;
}

.highlight-panel {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: #f8fbff;
  border: 1px solid #dfe8f7;
}

.highlight-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-3);
  align-items: center;
}

.highlight-head strong {
  color: var(--color-text-strong);
}

.highlight-head span {
  color: var(--color-text-muted);
  font-size: 0.88rem;
}

.highlight-text {
  margin: 0;
  color: var(--color-text-strong);
  line-height: 1.9;
}

.entity-highlight {
  display: inline;
  padding: 0.16rem 0.22rem;
  border-radius: 0.45rem;
  color: var(--color-text-strong);
}

.entity-highlight small {
  margin-left: 0.25rem;
  font-size: 0.72rem;
  font-weight: 700;
  opacity: 0.78;
}

.highlight-inline-label {
  margin-left: 0.35rem;
  font-size: 0.74rem;
  font-weight: 700;
  opacity: 0.82;
}

.entity-highlight-per {
  background: #dfe9ff;
  color: #2450be;
}

.entity-highlight-loc {
  background: #dff6eb;
  color: #0f7a4c;
}

.entity-highlight-org {
  background: #fff1d6;
  color: #a16207;
}

.entity-highlight-default {
  background: #eceff4;
  color: #475569;
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

@media (max-width: 640px) {
  .highlight-head,
  .entity-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .legend-items {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
