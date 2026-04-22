<script setup>
defineProps({
  label: {
    type: String,
    default: "",
  },
  confidence: {
    type: [Number, String, null],
    default: null,
  },
});
</script>

<template>
  <section class="result-card">
    <h3 class="result-title">文本分类结果</h3>

    <p v-if="!label" class="empty-text">暂无结果</p>

    <p v-else class="result-text">
      标签：{{ label }}
      <span v-if="confidence !== null && confidence !== ''">
        ，置信度：{{ confidence }}
      </span>
    </p>
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
