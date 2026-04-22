<script setup>
import { ref } from "vue";
import { tokenizeText, recognizeEntities, classifyText } from "../api/nlp";
import TokenResultCard from "../components/TokenResultCard.vue";
import NerResultCard from "../components/NerResultCard.vue";
import ClassifyResultCard from "../components/ClassifyResultCard.vue";

const text = ref("");
const loading = ref(false);
const error = ref("");
const tokens = ref([]);
const entities = ref([]);
const label = ref("");
const confidence = ref(null);

async function handleAnalyze() {
  const input = text.value.trim();

  if (!input) {
    error.value = "请输入要分析的文本";
    return;
  }

  loading.value = true;
  error.value = "";
  tokens.value = [];
  entities.value = [];
  label.value = "";
  confidence.value = null;

  try {
    const [tokenizeRes, nerRes, classifyRes] = await Promise.all([
      tokenizeText(input),
      recognizeEntities(input),
      classifyText(input),
    ]);

    tokens.value = tokenizeRes?.tokens || [];
    entities.value = nerRes?.entities || [];
    label.value = classifyRes?.label || "";
    confidence.value = classifyRes?.confidence ?? null;
  } catch (e) {
    error.value = e instanceof Error ? e.message : "分析失败，请稍后重试";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section>
    <h2>单文本分析</h2>

    <div>
      <textarea v-model="text" rows="8" placeholder="请输入一段中文文本" />
    </div>

    <div>
      <button :disabled="loading" @click="handleAnalyze">
        {{ loading ? "分析中..." : "开始分析" }}
      </button>
    </div>

    <p v-if="loading">正在调用后端接口，请稍候...</p>
    <p v-if="error">错误：{{ error }}</p>

    <div class="result-area">
      <TokenResultCard :tokens="tokens" />
      <NerResultCard :entities="entities" />
      <ClassifyResultCard :label="label" :confidence="confidence" />
    </div>
  </section>
</template>
