<script setup>
import { ref } from "vue";
import SingleTextPage from "./pages/SingleTextPage.vue";
import ClusterPage from "./pages/ClusterPage.vue";

const activeTab = ref("single");
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <p class="app-kicker">Experiment 4 · NLP Demo</p>
        <h1>P4 自然语言处理应用系统</h1>
        <p class="app-subtitle">
          面向课程演示与实验报告的中文文本分析前端，提供单文本分析与聚类可视化两类页面。
        </p>
      </div>
    </header>

    <section class="flow-strip" aria-label="分析流程">
      <div class="flow-step">
        <span class="flow-index">01</span>
        <strong>文本输入</strong>
      </div>
      <span class="flow-arrow">→</span>
      <div class="flow-step">
        <span class="flow-index">02</span>
        <strong>接口调用</strong>
      </div>
      <span class="flow-arrow">→</span>
      <div class="flow-step">
        <span class="flow-index">03</span>
        <strong>NLP 分析</strong>
      </div>
      <span class="flow-arrow">→</span>
      <div class="flow-step">
        <span class="flow-index">04</span>
        <strong>结果展示</strong>
      </div>
    </section>

    <main class="app-main">
      <nav class="tab-actions" aria-label="页面导航">
        <button
          class="tab-btn"
          :class="{ active: activeTab === 'single' }"
          @click="activeTab = 'single'"
        >
          <span class="tab-label">单文本分析</span>
          <span class="tab-hint">分词 · NER · 分类</span>
        </button>
        <button
          class="tab-btn"
          :class="{ active: activeTab === 'cluster' }"
          @click="activeTab = 'cluster'"
        >
          <span class="tab-label">聚类分析</span>
          <span class="tab-hint">二维散点图展示</span>
        </button>
      </nav>

      <section v-if="activeTab === 'single'" class="panel">
        <SingleTextPage />
      </section>

      <section v-else class="panel">
        <ClusterPage />
      </section>
    </main>
  </div>
</template>
