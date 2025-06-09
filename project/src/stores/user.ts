import { defineStore } from 'pinia'
import type{ UserLogin, LoginResponse} from '@/api/user/type'
import { reqLogin } from '@/api/user';
export const useUserStore = defineStore('user', () => {
  async function login(info: UserLogin): Promise<string> {
    try {
      const data: LoginResponse = await reqLogin(info)
      if(data.code === 200) {
        const token = data.data.token
        const expires = data.data.expire
        localStorage.setItem("token", token)
        localStorage.setItem("expires", String(expires))
        return Promise.resolve(data.data.username)
      }
      else {
       throw new Error(data.message)
      }
    }
    catch(error: unknown) {
      if (error instanceof Error) {
        throw new Error(error.message || '登录失败');
      } else {
        throw new Error(`未知类型错误: ${error}`);
      }
    }
  }
  function logout(): void {
    localStorage.clear()
  }
  function getToken(): string {
    const token = localStorage.getItem("token")
    // 没有token
    if(!token) {
      return ""
    }
    // 检查是否过期
    const expires = Number(localStorage.getItem("expires"))
    // 将 expire 转换为日期对象进行比较(秒转亳秒)
    const expireDate=new Date(expires * 1000);
    const now = new Date();
    if(expireDate <= now) {
      console.log("Token expired");
      logout()
      return ""
    }
    return token
  }
  return { login, logout, getToken }
})
