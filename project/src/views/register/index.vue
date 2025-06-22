<template>
  <div v-if="visible" class="modal-overlay">
    <div class="modal-content">
      <span class="close-button" @click="close">×</span>
      <form class="login">
        <div class="login-field">
          <input type="text" class="login-input" placeholder="Email" v-model="form.email" />
        </div>
        <div class="error-text" v-if="emailError">{{ emailError }}</div>
        <div class="login-field">
          <input type="text" class="login-input" placeholder="Username" v-model="form.username" />
        </div>
        <div class="login-field">
          <input type="password" class="login-input" placeholder="Password" v-model="form.password" />
        </div>
        <div class="login-field code-row">
          <input type="text" class="login-input" placeholder="Email Code" v-model="form.verification_code" />
          <button type="button" class="code-button" @click="sendCode" :disabled="codeSending">
            {{ codeSending ? countdown + 's' : '获取验证码' }}
          </button>
        </div>
        <button type="button" class="button login-submit" @click="submitRegister">
          <span class="button-text">注册</span>
        </button>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, defineProps, defineEmits, watch } from 'vue'
import { type UserSignup } from '@/api/user/type';
import { sendCodeRequest, reqRegister } from '@/api/user';
import { ElMessage } from 'element-plus';
// prop传递注册组件是否显示
defineProps<{ visible: boolean }>()
// 发送关闭事件
const emit = defineEmits(['close'])

const close = () => emit('close')

const form = ref<UserSignup>({
  email: '',
  username: '',
  password: '',
  verification_code: '',
})

const codeSending = ref<boolean>(false)
const countdown = ref<number>(60)
let timer: ReturnType<typeof setInterval> | undefined = undefined

const emailError = ref<string>('')  // 邮箱错误信息

const validateEmail = (email: string) => {
  const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return pattern.test(email)
}

watch(() => form.value.email, (newVal) => {
  if (!newVal) {
    emailError.value = '请输入邮箱'
  } else if (!validateEmail(newVal)) {
    emailError.value = '邮箱格式不正确'
  } else {
    emailError.value = ''
  }
})

const sendCode = () => {
  if (!form.value.email) {
    ElMessage.warning('请输入邮箱')
    return
  }
  codeSending.value = true
  countdown.value = 60
  // 开始计时器
  timer = setInterval(() => {
    countdown.value--
    if (countdown.value <= 0) {
      clearInterval(timer)
      codeSending.value = false
    }
  }, 1000)
  console.log('发送验证码到邮箱:', form.value.email)
  sendCodeRequest({email: form.value.email}).then((data) => {
    console.log(data)
    if (data.code === 200){
      ElMessage({
        showClose: true,
        message: `验证码已发送至邮箱地址：${form.value.email}，有效期为10分钟`,
        type: 'success',
      })
    }
    else {
      console.warn(`请求地址：/api/v1/reset-password/send-code，状态码：${data.code}，错误信息：${data.message}`)
      ElMessage({
        showClose: true,
        message: data.message,
        type: 'error',
      })
    }
  })
}

const submitRegister = () => {
  console.log('提交注册信息', form.value)
  if(form.value.email && form.value.password && form.value.verification_code && form.value.username && validateEmail(form.value.email)) {
    reqRegister(form.value).then((data) => {
      if(data.code === 200) {
        ElMessage({
          showClose: true,
          message: "注册成功，请尝试登录",
          type: 'success',
        })
        close()
      }
      else {
        console.warn(`请求地址：/api/v1/auth/register，状态码：${data.code}，错误信息：${data.message}`)
        ElMessage({
          showClose: true,
          message: data.message,
          type: 'error',
        })
      }
    })
    .catch(err => {
      console.warn(err) 
      ElMessage({
        showClose: true,
        message: "发生了一些错误，请稍后再试！",
        type: 'error',
      })
    })
  }
  else {
    ElMessage({
      showClose: true,
      message: "请完善表单内容",
      type: 'error',
    })
  }
}
</script>

<style scoped lang="scss">
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
}

.modal-content {
  position: relative;
  width: 400px;
  padding: 40px 30px;
  background: linear-gradient(90deg, #5D54A4, #7C78B8);
  border-radius: 20px;
  box-shadow: 0 0 24px #5C5696;
}

.close-button {
  position: absolute;
  top: 10px;
  right: 16px;
  font-size: 24px;
  color: white;
  cursor: pointer;
}

.login-field {
  padding: 20px 0;
}

.login-input {
  border: none;
  border-bottom: 2px solid #D1D1D4;
  background: none;
  padding: 10px;
  padding-left: 10px;
  font-weight: 700;
  width: 100%;
  color: white;
  transition: .2s;
}

.login-input:focus {
  outline: none;
  border-bottom-color: #fff;
}

.code-row {
  display: flex;
  justify-content: space-between;

  .login-input {
    width: 60%;
  }

  .code-button {
    margin-left: 10px;
    padding: 10px 14px;
    background-color: #fff;
    color: #4C489D;
    font-weight: bold;
    border-radius: 20px;
    border: 1px solid #D4D3E8;
    cursor: pointer;
  }

  .code-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
}

.login-submit {
  background: #fff;
  font-size: 14px;
  margin-top: 30px;
  padding: 14px 20px;
  border-radius: 26px;
  border: 1px solid #D4D3E8;
  font-weight: 700;
  width: 100%;
  color: #4C489D;
  cursor: pointer;
}

.error-text {
  color: #ffcccc;
  font-size: 12px;
  margin-top: 4px;
}
</style>
