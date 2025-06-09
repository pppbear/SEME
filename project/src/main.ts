import './assets/main.css'

import { createApp } from 'vue'
import pinia from './stores/store'
import router from './router'
import App from './App.vue'
import 'element-plus/dist/index.css' // 样式引入

const app = createApp(App)

app.use(pinia)
app.use(router)

app.mount('#app')
