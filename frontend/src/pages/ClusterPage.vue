<script setup>
import { ref } from "vue";
import { clusterDocuments } from "../api/nlp";
import ClusterChart from "../components/ClusterChart.vue";

const rawInput = ref("");
const clusterCount = ref("");
const loading = ref(false);
const error = ref("");
const points = ref([]);

function parseLinesToDocuments(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const sepIndex = line.indexOf("::");
      if (sepIndex === -1) {
        throw new Error("每行必须是“标题::内容”格式");
      }

      const title = line.slice(0, sepIndex).trim();
      const content = line.slice(sepIndex + 2).trim();

      if (!title || !content) {
        throw new Error("每行的标题和内容都不能为空");
      }

      return {
        title,
        text: content,
      };
    });
}

async function handleCluster() {
  error.value = "";
  points.value = [];

  let documents;
  try {
    documents = parseLinesToDocuments(rawInput.value);
  } catch (e) {
    error.value = e instanceof Error ? e.message : "输入格式有误";
    return;
  }

  if (documents.length < 2) {
    error.value = "请至少输入两条文本进行聚类";
    return;
  }

  let count;
  if (clusterCount.value.trim()) {
    count = Number(clusterCount.value);
    if (!Number.isInteger(count) || count < 2) {
      error.value = "cluster_count 必须是大于等于 2 的整数";
      return;
    }
  }

  loading.value = true;
  try {
    const res = await clusterDocuments(documents, count);
    points.value = res?.points || [];
  } catch (e) {
    error.value = e instanceof Error ? e.message : "聚类失败，请稍后重试";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="cluster-page">
    <h2 class="page-title">聚类分析</h2>

    <div class="input-block">
      <textarea
        v-model="rawInput"
        rows="10"
        placeholder="每行一条，格式：标题::内容"
      />
    </div>

    <div class="input-block">
      <label>
        cluster_count：
        <input
          v-model="clusterCount"
          type="number"
          min="2"
          placeholder="可为空"
        />
      </label>
    </div>

    <div class="input-block">
      <button :disabled="loading" @click="handleCluster">
        {{ loading ? "聚类中..." : "开始聚类" }}
      </button>
    </div>

    <p v-if="error" class="error-text">错误：{{ error }}</p>

    <div class="result-block">
      <ClusterChart :points="points" />
    </div>
  </section>
</template>

<style scoped>
.cluster-page {
  display: grid;
  gap: 12px;
}

.page-title {
  margin: 0;
}

.input-block {
  display: grid;
  gap: 8px;
}

textarea,
input {
  width: 100%;
  padding: 8px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font: inherit;
}

button {
  width: 120px;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #ffffff;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.error-text {
  margin: 0;
  color: #b91c1c;
}

.result-block {
  margin-top: 4px;
}
</style>
