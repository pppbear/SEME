<template>
  <!-- 注册组件 -->
  <div class="register">
    <RegDraw :visible="regValue" @close="regValue = false"/>
  </div>
  <div class="reset">
    <ResetDraw :visible="resetValue" @close="resetValue = false"/>
  </div>
  <!-- 登录组件 -->
  <div class="container">
    <!-- 将标题放到screen上面 -->
    <div class="login-title">欢迎登陆上海市宜居性系统</div>
    <div class="screen">
      <div class="screen-content">
        <form class="login">
          <div class="login-field">
            <input type="text" class="login-input" placeholder="Email/UserName" v-model="user.username">
          </div>
          <div class="login-field">
            <input type="password" class="login-input" placeholder="Password" v-model="user.password">
          </div>
          <button
            type="button"
            class="button login-submit"
            @click="login"
            :disabled="loginLoading"
          >
            <span class="button-text">
              <el-icon v-if="loginLoading" class="loading-icon"><i-ep-loading /></el-icon>
              登录
            </span>
          </button>			
          <button type="button" class="button login-submit" @click="register">
            <span class="button-text">注册</span>
          </button>	
          <div class="reset-box">
            <span class="reset-text" @click="resetPassword">忘记密码？</span>
          </div>	
        </form>
      </div>
      <!-- 表单背景图 -->
      <div class="screen-background">
        <span class="screen-background-shape screen-background-shape4"></span>
        <span class="screen-background-shape screen-background-shape3"></span>		
        <span class="screen-background-shape screen-background-shape2"></span>
        <span class="screen-background-shape screen-background-shape1"></span>
      </div>		
    </div>
  </div>
</template>

<script setup lang="ts">
import {type UserLogin} from '@/api/user/type'
import { ref } from 'vue';
import router from '@/router/index'
import { useUserStore } from '@/stores/user';
import pinia from '@/stores/store';
import RegDraw from '@/views/register/index.vue'
import ResetDraw from '@/views/reset/index.vue'
import { ElMessage } from 'element-plus';

const user = ref<UserLogin>({
  username: "",
  password: "",
  remember: false
})
const regValue = ref<boolean>(false)
const resetValue = ref<boolean>(false)
const userStore = useUserStore(pinia)  
const loginLoading = ref(false) // 新增loading状态

// 登录函数
function login(): void {
  if(user.value.username && user.value.password) {
    loginLoading.value = true
    userStore.login(user.value) 
    .then((username: string) => {
      ElMessage({
        showClose: true,
        message: `${username}欢迎回来`,
        type: 'success',
      })
      router.push({ name: 'Layout' })
    })
    .catch(() => {
      user.value = {
        username: "",
        password: "",
        remember: false
      }
      ElMessage({
        showClose: true,
        message: '登录失败，请检查用户名和密码',
        type: 'error',
      })
    })
    .finally(() => {
      loginLoading.value = false
    })
  }
  else {
    ElMessage({
      showClose: true,
      message: '请完整填写表单',
      type: 'error',
    })
  }
}
// 注册组件展示
function register(): void {
  // 注册组件展示
  regValue.value = true
}
// 密码重设
function resetPassword(): void {
  resetValue.value = true
}
</script>

<style scoped lang="scss">
@import url('https://fonts.googleapis.com/css?family=Raleway:400,700');
html {
  margin: 0;
  padding: 0;	
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    width: 100vw;
    background: linear-gradient(90deg, #C7C5F4, #776BCC);		
    .reset-box {
      margin-top: 10px;
      position: relative;
    }
    .reset-text:hover {
      text-decoration: underline;
    }
  }
  .screen {		
    background: linear-gradient(90deg, #5D54A4, #7C78B8);		
    position: relative;	
    height: 600px;
    width: 400px;	
    box-shadow: 0px 0px 24px #5C5696;
    border-radius: 20px;
  }
  .login-title {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #4C489D;
    margin-top: 40px;
    margin-bottom: 10px;
    letter-spacing: 2px;
  }
  .screen-content {
    z-index: 1;
    position: relative;	
    height: 100%;
  }
  .screen-background {		
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 0;
    -webkit-clip-path: inset(0 0 0 0);
    clip-path: inset(0 0 0 0);	
  }
  .screen-background-shape {
    transform: rotate(45deg);
    position: absolute;
  }

  .screen-background-shape1 {
    height: 520px;
    width: 520px;
    background: #FFF;	
    top: -50px;
    right: 120px;	
    border-radius: 0 72px 0 0;
  }

  .screen-background-shape2 {
    height: 220px;
    width: 220px;
    background: #6C63AC;	
    top: -172px;
    right: 0;	
    border-radius: 32px;
  }

  .screen-background-shape3 {
    height: 540px;
    width: 190px;
    background: linear-gradient(270deg, #5D54A4, #6A679E);
    top: -24px;
    right: 0;	
    border-radius: 32px;
  }

  .screen-background-shape4 {
    height: 300px;
    width: 100px;
    background: #7E7BB9;	
    top: 420px;
    right: 50px;	
    border-radius: 60px;
  }
  
  .login {
    width: 320px;
    padding: 30px;
    padding-top: 80px;
  }

  .login-field {
    padding: 20px 0px;	
    position: relative;	
  }

  .login-input {
    border: none;
    border-bottom: 2px solid #D1D1D4;
    background: none;
    padding: 10px;
    padding-left: 24px;
    font-weight: 700;
    width: 75%;
    transition: .2s;
  }

  .login-input:active,
  .login-input:focus,
  .login-input:hover {
    outline: none;
    border-bottom-color: #6A679E;
  }

  .login-submit {
    background: #fff;
    font-size: 14px;
    margin-top: 20px;
    padding: 16px 20px;
    border-radius: 26px;
    border: 1px solid #D4D3E8;
    text-transform: uppercase;
    font-weight: 700;
    display: flex;
    align-items: center;
    width: 100%;
    color: #4C489D;
    box-shadow: 0px 2px 2px #5C5696;
    cursor: pointer;
    transition: .2s;
  }

  .login-submit:active,
  .login-submit:focus,
  .login-submit:hover {
    border-color: #6A679E;
    outline: none;
  }
}
</style>