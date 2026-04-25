<script setup>
import { computed } from "vue";

const props = defineProps({
  tokens: {
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

const tokenCount = computed(() => props.tokens.length);
</script>

<template>
  <section class="result-card">
    <div class="card-head">
      <div>
        <h3 class="result-title">分词结果</h3>
        <p class="result-desc">以词元卡片形式展示文本切分结果，适合课堂讲解与截图展示。</p>
      </div>
      <span class="count-pill">{{ tokenCount }} 个词元</span>
    </div>

    <div v-if="loading" class="state-box state-loading">正在生成分词结果...</div>
    <div v-else-if="!ready" class="state-box state-empty">提交文本后，这里将显示分词结果。</div>
    <div v-else-if="tokens.length === 0" class="state-box state-empty">本次文本未返回可展示的分词结果。</div>

    <div v-else class="token-cloud">
      <span v-for="(token, index) in tokens" :key="`${token}-${index}`" class="token-chip">
        {{ token }}
      </span>
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

.count-pill {
  padding: 0.4rem 0.82rem;
  border-radius: 999px;
  background: #eef4ff;
  color: #2354cc;
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

.token-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
}

.token-chip {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 0.55rem 0.85rem;
  border-radius: 999px;
  background: #f3f7ff;
  border: 1px solid #d7e3ff;
  color: var(--color-text-strong);
  font-size: 0.94rem;
  line-height: 1.2;
}
</style>
