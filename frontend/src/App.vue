<script setup>
import { ref } from "vue";
import SingleTextPage from "./pages/SingleTextPage.vue";
import ClusterPage from "./pages/ClusterPage.vue";

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
        <SingleTextPage />
      </section>

      <section v-else class="panel">
        <ClusterPage />
      </section>
    </main>
  </div>
</template>
