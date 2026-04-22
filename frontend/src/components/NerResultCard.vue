<script setup>
defineProps({
  entities: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <section class="result-card">
    <h3 class="result-title">命名实体识别结果</h3>

    <p v-if="entities.length === 0" class="empty-text">暂无结果</p>

    <ul v-else class="entity-list">
      <li v-for="(item, index) in entities" :key="index" class="entity-item">
        <span>text: {{ item.text }}</span>
        <span>label: {{ item.label }}</span>
        <span>start: {{ item.start }}</span>
        <span>end: {{ item.end }}</span>
      </li>
    </ul>
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

.empty-text {
  margin: 0;
  color: #4b5563;
}

.entity-list {
  margin: 0;
  padding-left: 18px;
}

.entity-item {
  display: grid;
  gap: 2px;
  margin-bottom: 8px;
  color: #374151;
}

.entity-item:last-child {
  margin-bottom: 0;
}
</style>
