<template>
  <div class="layout">
    <sidebar class="sidebar"/>
    <tabbar class="tabbar"/>
    <router-view class="main"></router-view>
  </div>
  

</template>

<script setup lang="ts">
import sidebar from './sidebar/index.vue'
import tabbar from './tabbar/index.vue'
</script>

<style scoped lang="scss">
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    overflow: hidden;
}
.layout {
    height: 100vh;
    width: 100vw;
    margin: 0;
    padding: 0;
}
.sidebar {
    position: absolute;
    left: 0px;
    top: 0px;
    height: 100%;
    width: 200px;
}
.tabbar {
  margin-left: 200px;
  width: calc(100% - 200px);
  height: 80px;
  border: 1px solid;
  border-color: transparent transparent rgb(212, 212, 212) transparent;
}
.main {
  margin-left: 200px;
  height: calc(100vh - 80px);
  overflow: auto;
}
</style>
