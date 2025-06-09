<template>
  <div class="kan-page">
    <el-card class="card">
      <div class="button-row">
        <el-button @click="showFormat = true">查看文件格式</el-button>
        <el-upload :before-upload="handleFile" accept=".xlsx,.xls" class="upload-button">
          <el-button type="primary">上传样本文件</el-button>
        </el-upload>
        <el-select v-model="selectedData" placeholder="请选择因变量" class="select">
          <el-option v-for="col in options" :key="col.value" :label="col.label" :value="col.value" />
        </el-select>
        <el-button
          type="primary"
          :disabled="!selectedData || !excelFile || loading"
          :loading="loading"
          @click="handlePredict"
          class="predict-btn"
        >预测</el-button>
      </div>
      <div v-if="excelFileName" class="filename">已上传：{{ excelFileName }}</div>
    </el-card>

    <el-card class="card">
      <div v-if="!hasResult" class="no-result">
        暂无数据，请上传文件并点击“预测”
      </div>
      <div v-else ref="lineChartRef" class="chart"></div>
    </el-card>
    <el-dialog v-model="showFormat" title="输入数据格式要求" width="600">
      <p class="dialog-tip">
        请确保上传的 Excel 文件包含以下字段，字段顺序可不固定，但列名需正确，请勿包含预测值。
      </p>
      <el-table :data="fieldList" style="width: 100%" border>
        <el-table-column prop="name" label="字段名" />
        <el-table-column prop="desc" label="说明" />
      </el-table>
    </el-dialog>

    <el-card class="card key-vars-card">
      <div class="key-vars-title">
        当前因变量的关键自变量
      </div>
      <el-table :data="mainVars[selectedData] || []" style="width: 100%" border>
        <el-table-column prop="name" label="自变量" />
        <el-table-column prop="desc" label="说明" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from 'vue'
import * as echarts from 'echarts'
import { predictResult } from '@/api/predict'

const options = [
  { label: '白天地表平均温度', value: 'lst_day_c' },
  { label: '夜晚地表平均温度', value: 'lst_night_c' },
  { label: '夜晚灯光辐射值', value: 'nighttime_' }
]
const selectedData = ref<string>('lst_day_c')
const showFormat = ref<boolean>(false)
const excelFile = ref<File | null>(null)
const excelFileName = ref<string>('')
const loading = ref(false)
const hasResult = ref(false)
const fieldList = [
  { name: 'Land01 ~ Land05', desc: '土地利用类型相关特征' },
  { name: 'NDVI_MEAN', desc: '植被指数均值' },
  { name: 'sidewalk_M', desc: '人行道距离' },
  { name: 'building_M', desc: '建筑物距离' },
  { name: 'vegetation', desc: '绿地覆盖率' },
  { name: 'sky_MEAN', desc: '天空可视率均值' },
  { name: 'POI餐饮 ~ POI政府', desc: '各类兴趣点数量' },
  { name: '不透水', desc: '不透水面比例' },
  { name: '建筑密', desc: '建筑密度' },
  { name: '容积率', desc: '建筑容积率' },
  { name: 'railway_m, Subway_m, car_road_m', desc: '各类交通设施距离' },
  { name: 'high_grade', desc: '高等级道路指标' },
  { name: 'nighttime_', desc: '夜晚灯光辐射值（因变量）' },
  { name: 'lst_day_c', desc: '白天地表温度（因变量）' },
  { name: 'lst_night_c', desc: '夜晚地表温度（因变量）' }
]

// 关键自变量静态数据
const mainVars: Record<string, Array<{ name: string, desc: string }>> = {
  lst_day_c: [
    { name: 'Land01', desc: '土地利用类型相关特征' },
    { name: '容积率', desc: '建筑容积率' },
    { name: 'PNT_COUNT', desc: '兴趣点数量' },
    { name: 'MEAN', desc: '植被指数均值' },
    { name: 'POI购物', desc: '购物类兴趣点数量' },
    { name: '不透水', desc: '不透水面比例' },
    { name: 'Land03', desc: '土地利用类型相关特征' },
    { name: 'POI生活', desc: '生活类兴趣点数量' },
    { name: '建筑密', desc: '建筑密度' },
    { name: 'NDVI_MEAN', desc: '植被指数均值' },
    { name: 'POI餐饮', desc: '餐饮类兴趣点数量' },
    { name: 'POI商务', desc: '商务类兴趣点数量' },
    { name: 'POI公司', desc: '公司类兴趣点数量' }
  ],
  lst_night_c: [
    { name: '容积率', desc: '建筑容积率' },
    { name: 'PNT_COUNT', desc: '兴趣点数量' },
    { name: 'NDVI_MEAN', desc: '植被指数均值' },
    { name: 'POI商务', desc: '商务类兴趣点数量' },
    { name: 'POI科教', desc: '科教类兴趣点数量' },
    { name: 'POI政府', desc: '政府类兴趣点数量' },
    { name: 'POI餐饮', desc: '餐饮类兴趣点数量' },
    { name: 'POI公司', desc: '公司类兴趣点数量' },
    { name: 'Land505', desc: '土地利用类型相关特征' },
    { name: 'car_road_m', desc: '各类交通设施距离' },
    { name: 'POI总数', desc: '兴趣点总数' },
    { name: 'POI医疗', desc: '医疗类兴趣点数量' },
    { name: 'Land03', desc: '土地利用类型相关特征' }
  ],
  nighttime_: [
    { name: '不透水', desc: '不透水面比例' },
    { name: 'POI商务', desc: '商务类兴趣点数量' },
    { name: 'Land01', desc: '土地利用类型相关特征' },
    { name: 'MEAN', desc: '植被指数均值' },
    { name: '建筑密', desc: '建筑密度' },
    { name: 'POI科教', desc: '科教类兴趣点数量' },
    { name: 'POI体育', desc: '体育类兴趣点数量' },
    { name: 'Land03', desc: '土地利用类型相关特征' },
    { name: 'POI生活', desc: '生活类兴趣点数量' },
    { name: 'POI公司', desc: '公司类兴趣点数量' },
    { name: 'car_road_m', desc: '各类交通设施距离' },
    { name: 'POI风景', desc: '风景类兴趣点数量' },
    { name: '容积率', desc: '建筑容积率' }
  ]
}

const chartData = ref<number[]>([])
const chartLabels = ref<string[]>([])
const lineChartRef = ref<HTMLElement>()
let lineChart: echarts.ECharts

function handleFile(file: File) {
  excelFile.value = file
  excelFileName.value = file.name
  hasResult.value = false
  return false // 阻止自动上传
}

const handlePredict = async () => {
  if (!excelFile.value || !selectedData.value) return
  loading.value = true
  hasResult.value = true
  await nextTick()
  lineChart = echarts.init(lineChartRef.value)
  lineChart.showLoading()
  try {
    // 兼容后端接收 formData
    const formData = new FormData()
    formData.append('file', excelFile.value)
    formData.append('dependent_name', selectedData.value)
    const res = await predictResult(formData as any)
    if (res.code === 200 && res.data && Array.isArray(res.data)) {
      chartData.value = res.data
      chartLabels.value = res.data.map((_, idx) => (idx + 1).toString())
      hasResult.value = true
      updateChart()
    }
  } finally {
    loading.value = false
    if (lineChart) lineChart.hideLoading()
  }
}

function updateChart() {
  const option = {
    title: { text: 'KAN预测值', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['KAN预测值'], top: 'bottom' },
    xAxis: { type: 'category', data: chartLabels.value },
    yAxis: { type: 'value', name: 'KAN预测值' },
    series: [
      { name: 'KAN预测值', type: 'line', data: chartData.value }
    ]
  }
  lineChart.setOption(option)
}
</script>

<style scoped>
.card {
  padding: 20px;
}
.button-row {
  flex-wrap: wrap;
  align-items: center;
  justify-items: center;
}
.button-row > * {
  margin-right: 12px;
}
.predict-btn {
  margin-left: 12px;
}
.upload-button {
  display: inline-block;
}
.select {
  width: 240px;
  margin: 20px 0;
  gap: 10px;
}
.filename {
  margin-top: 10px;
  font-style: italic;
  color: #409EFF;
}
.chart {
  width: 100%;
  height: 400px;
  margin-bottom: 40px;
}
.no-result {
  height: 400px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #999;
}
.dialog-tip {
  margin-bottom: 10px;
}
.key-vars-card {
  margin-bottom: 20px;
}
.key-vars-title {
  font-weight: bold;
  margin-bottom: 10px;
}
</style>
