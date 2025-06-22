import axios, { type AxiosResponse } from "axios";
import { ElMessage } from 'element-plus'

const request = axios.create({
    baseURL: "http://localhost:8000",
    timeout: 600000,
    headers: {
        'Content-Type': 'application/json',
    }
})
// 请求拦截器
request.interceptors.request.use((config) => {
    console.log(config);
    const token = localStorage.getItem('token')
    config.headers.Authorization = `Bearer ${token}`
    return config
})

// 响应拦截器
request.interceptors.response.use(
  (res: AxiosResponse) => {
    return res.data
  },
  (error) => {
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      ElMessage.error('请求超时，请稍后再试')
    } else {
      ElMessage.error('请求失败，请稍后再试')
    }
    return Promise.reject(error)
  }
)

export default request