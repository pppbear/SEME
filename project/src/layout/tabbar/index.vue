<template>
  <div class="tabbar">
    <div class="tab-route">
    </div>
    <div class="tab-btn-contain">
      <el-button class="tab-btns" circle @click="refresh">
          <i-ep-RefreshRight />
      </el-button>
      <el-button class="tab-btns" circle @click="fullscreen">
          <i-ep-FullScreen />
      </el-button>
      <el-dropdown placement="bottom-end" class="tab-dropdown">
        <el-button class="tab-btns" circle>
          <i-ep-User />
        </el-button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item @click="logout" type="button">退出</el-dropdown-item>
          </el-dropdown-menu>
        </template> 
      </el-dropdown>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const refresh = () => {
  location.reload()
}
const fullscreen = () => {
  document.documentElement.requestFullscreen()
}
const logout = () => {
  userStore.logout()
  window.location.reload() // 强制刷新，确保新界面渲染
}
</script>

<style lang="scss">
.tabbar {
  height: 50px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  .tab-btn-contain {
    padding-right: 20px;
    display: flex;
    align-items: center;
    gap: 15px;
    .tab-dropdown {
      margin-left: 10px;
    }
  }
}
</style>