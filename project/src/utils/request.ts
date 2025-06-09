import axios, { type AxiosResponse } from "axios";
const request = axios.create({
    baseURL: "http://42.192.209.62:8000",
    timeout: 60000,
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
request.interceptors.response.use((res: AxiosResponse) => {
    console.log(res);
    return res.data
})

export default request