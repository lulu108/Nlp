<script setup>
import { ref } from "vue";

const activeTab = ref("single");
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <h1>P4 自然语言处理应用系统</h1>
      <p>实验四前端第一版基础壳子</p>
    </header>

    <main class="app-main">
      <div class="tab-actions">
        <button
          class="tab-btn"
          :class="{ active: activeTab === 'single' }"
          @click="activeTab = 'single'"
        >
          单文本分析
        </button>
        <button
          class="tab-btn"
          :class="{ active: activeTab === 'cluster' }"
          @click="activeTab = 'cluster'"
        >
          聚类分析
        </button>
      </div>

      <section v-if="activeTab === 'single'" class="panel">
        <h2>单文本分析页面</h2>
        <p>这里是占位区域，后续接入分词、NER、分类模块。</p>
      </section>

      <section v-else class="panel">
        <h2>聚类分析页面</h2>
        <p>这里是占位区域，后续接入文本聚类与可视化模块。</p>
      </section>
    </main>
  </div>
</template>
