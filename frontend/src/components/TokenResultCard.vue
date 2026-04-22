<script setup>
defineProps({
  tokens: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <section class="result-card">
    <h3 class="result-title">分词结果</h3>
    <p v-if="tokens.length === 0" class="empty-text">暂无结果</p>
    <p v-else class="result-text">{{ tokens.join(" / ") }}</p>
  </section>
</template>

<style scoped>
.result-card {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  padding: 12px;
  background: #fff;
}

.result-title {
  margin: 0 0 8px;
  font-size: 16px;
}

.empty-text,
.result-text {
  margin: 0;
  color: #4b5563;
  line-height: 1.6;
}
</style>
