<template>
  <div class="sidebar">
    <el-menu
      :router="true"
      class="menu"
      default-active="/grid"
      background-color="#f9f9f9"
      text-color="#333"
      active-text-color="#42b983"
    >
      <el-menu-item index="/grid">
        <i-ep-HomeFilled />
        <span class="menu-text">栅格数据</span>
      </el-menu-item>
      <el-menu-item index="/analyze">
        <i-ep-Histogram /> <!-- 更换为柱状图icon -->
        <span class="menu-text">关键因素</span>
      </el-menu-item>
      <el-menu-item index="/compare">
        <i-ep-Star />
        <span class="menu-text">模型对比</span>
      </el-menu-item>
      <el-menu-item index="/predict">
        <i-ep-DataLine />
        <span class="menu-text">光热预测</span>
      </el-menu-item>
    </el-menu>
  </div>
</template>

<script setup lang="ts">

</script>

<style scoped lang="scss">
.sidebar {
  width: 220px;
  height: 100vh;
  background-color: #f9f9f9;
  border-right: 1px solid #e0e0e0;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
  transition: all 0.3s;
  .menu {
    position: sticky;
    top: 0px;
    border-right: none;
    height: 100%;
    .el-menu-item {
      height: 50px;
      line-height: 50px;
      transition: background-color 0.3s ease;
      border-radius: 6px;
      margin: 4px 10px;
      gap: 10px;
      :hover {
        background-color: #eafaf1;
        color: #42b983;
      }

      .el-icon {
        margin-right: 8px;
      }

      .menu-text {
        font-size: 15px;
        font-weight: 500;
      }
    }

    .el-menu-item.is-active {
      background-color: #dff5eb;
      color: #42b983 ;
    }
  }
}
</style>

