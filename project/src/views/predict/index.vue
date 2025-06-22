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
        <el-button
          type="success"
          class="download-btn"
          :disabled="!download"
          @click="downloadResult"
        >
          下载结果
        </el-button>
      </div>
      <div v-if="excelFileName" class="filename">已上传：{{ excelFileName }}</div>
    </el-card>

    <el-card class="card">
      <div v-if="!hasResult" class="no-result">
        暂无数据，请上传文件并点击“预测”
      </div>
      <div v-else ref="lineChartRef" class="chart"></div>
    </el-card>
    <el-dialog v-model="showFormat" title="输入数据格式要求" width="800">
      <p class="dialog-tip">
        请确保上传的 Excel 文件包含以下字段，每个预测值必须包含的值见页面下方的关键自变量，字段顺序可不固定，但列名需正确，请勿包含预测值。
      </p>
      <el-table :data="fieldList" style="width: 100%" border>
        <el-table-column prop="name" label="字段名" />
        <el-table-column prop="unit" label="单位" />
        <el-table-column prop="description" label="说明" />
        <el-table-column prop="role" label="角色" />
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
import { predictResult, type PredictResponse } from '@/api/predict'
import * as XLSX from 'xlsx'

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
const download = ref(false)
const fieldList = [
  { name: "Land01", unit: "百分比", description: "住宅用地面积占栅格面积的百分比", role: "自变量" },
  { name: "Land02", unit: "百分比", description: "商业服务用地面积占栅格面积的百分比", role: "自变量" },
  { name: "Land03", unit: "百分比", description: "工业用地面积占栅格面积的百分比", role: "自变量" },
  { name: "Land50234", unit: "百分比", description: "教育科研、医疗、体育文化用地面积占栅格面积的百分比", role: "自变量" },
  { name: "Land505", unit: "百分比", description: "公园与绿地用地面积占栅格面积的百分比", role: "自变量" },
  { name: "NDVI_MEAN", unit: "比值", description: "归一化差异植被指数（值在 -1 到 1 之间）", role: "自变量" },
  { name: "sidewalk_M", unit: "百分比", description: "栅格平均街景人行横道占比", role: "自变量" },
  { name: "building_M", unit: "百分比", description: "栅格平均街景建筑立面占比", role: "自变量" },
  { name: "vegetation_M", unit: "百分比", description: "栅格平均街景绿视率", role: "自变量" },
  { name: "sky_MEAN", unit: "百分比", description: "栅格平均街景天空占比", role: "自变量" },
  { name: "POI餐饮", unit: "个", description: "餐饮服务设施数量", role: "自变量" },
  { name: "POI风景", unit: "个", description: "风景名胜设施数量", role: "自变量" },
  { name: "POI公司", unit: "个", description: "公司企业设施数量", role: "自变量" },
  { name: "POI购物", unit: "个", description: "购物服务设施数量", role: "自变量" },
  { name: "POI科教", unit: "个", description: "科教文化设施数量", role: "自变量" },
  { name: "POI商务", unit: "个", description: "商务住宅设施数量", role: "自变量" },
  { name: "POI生活", unit: "个", description: "生活服务设施数量", role: "自变量" },
  { name: "POI体育", unit: "个", description: "体育休闲设施数量", role: "自变量" },
  { name: "POI医疗", unit: "个", description: "医疗保健设施数量", role: "自变量" },
  { name: "POI政府", unit: "个", description: "政府机构设施数量", role: "自变量" },
  { name: "不透水", unit: "百分比", description: "不透水面面积占栅格面积比例", role: "自变量" },
  { name: "建筑密", unit: "百分比", description: "建筑基底面积占栅格面积比例", role: "自变量" },
  { name: "容积率", unit: "米", description: "建筑总体积与栅格总面积之比", role: "自变量" },
  { name: "railway_m", unit: "米", description: "铁路总长度", role: "自变量" },
  { name: "Subway_m", unit: "米", description: "地铁总长度", role: "自变量" },
  { name: "car_road_m", unit: "米", description: "车行道总长度", role: "自变量" },
  { name: "high_grade_road_m", unit: "米", description: "高等级道路总长度", role: "自变量" },
];

// 关键自变量静态数据
const mainVars: Record<string, Array<{ name: string, desc: string }>> = {
  lst_day_c: [
    { name: 'Land01', desc: '住宅用地面积占栅格面积的百分比' },
    { name: '容积率', desc: '建筑总体积/栅格总面积(1km*1km)' },
    { name: 'PNT_COUNT', desc: '兴趣点数量' },
    { name: 'MEAN', desc: '植被指数均值' },
    { name: 'POI购物', desc: '购物服务设施个数' },
    { name: '不透水', desc: '不透水面面积占栅格面积的比例' },
    { name: 'Land03', desc: '工业用地面积占栅格面积的百分比' },
    { name: 'POI生活', desc: '生活服务设施个数' },
    { name: '建筑密度', desc: '建筑基底面积占栅格面积的比例' },
    { name: 'NDVI_MEAN', desc: '归一化差异植被指数' },
    { name: 'POI餐饮', desc: '餐饮服务设施个数' },
    { name: 'POI商务', desc: '商务住宅设施个数' },
    { name: 'POI公司', desc: '公司企业设施个数' }
  ],
  lst_night_c: [
    { name: '容积率', desc: '建筑总体积/栅格总面积(1km*1km)' },
    { name: 'PNT_COUNT', desc: '兴趣点数量' },
    { name: 'NDVI_MEAN', desc: '归一化差异植被指数' },
    { name: 'POI商务', desc: '商务住宅设施个数' },
    { name: 'POI科教', desc: '科教文化设施个数' },
    { name: 'POI政府', desc: '政府机构设施个数' },
    { name: 'POI餐饮', desc: '餐饮服务设施个数' },
    { name: 'POI公司', desc: '公司企业设施个数' },
    { name: 'Land505', desc: '公园与绿地用地面积占栅格面积的百分比' },
    { name: 'car_road_m', desc: '每个栅格中的车行道总长度' },
    { name: 'POI总数', desc: '兴趣点总数' },
    { name: 'POI医疗', desc: '每个栅格中医疗保健设施个数' },
    { name: 'Land03', desc: '工业用地面积占栅格面积的百分比' }
  ],
  nighttime_: [
    { name: '不透水', desc: '不透水面面积占栅格面积的比例' },
    { name: 'POI商务', desc: '商务住宅设施个数' },
    { name: 'Land01', desc: '住宅用地面积占栅格面积的百分比' },
    { name: 'MEAN', desc: '植被指数均值' },
    { name: '建筑密度', desc: '建筑基底面积占栅格面积的比例' },
    { name: 'POI科教', desc: '科教文化设施个数' },
    { name: 'POI体育', desc: '体育休闲设施个数' },
    { name: 'Land03', desc: '工业用地面积占栅格面积的百分比' },
    { name: 'POI生活', desc: '生活服务设施个数' },
    { name: 'POI公司', desc: '公司企业设施个数' },
    { name: 'car_road_m', desc: '每个栅格中的车行道总长度' },
    { name: 'POI风景', desc: '风景名胜设施个数' },
    { name: '容积率', desc: '建筑总体积/栅格总面积(1km*1km)' }
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
    const res = await predictResult({file: excelFile.value, dependent_name: selectedData.value}) as PredictResponse
    if (res.code === 200 && res.data && Array.isArray(res.data)) {
      chartData.value = res.data
      chartLabels.value = res.data.map((_, idx) => (idx + 1).toString())
      download.value = true
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

function downloadResult() {
  if (!hasResult.value || !chartData.value.length || !excelFile.value) return

  // 读取原始文件
  const reader = new FileReader()
  reader.onload = (e) => {
    const data = new Uint8Array(e.target?.result as ArrayBuffer)
    const workbook = XLSX.read(data, { type: 'array' })
    const sheetName = workbook.SheetNames[0]
    const worksheet = workbook.Sheets[sheetName]
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const json = XLSX.utils.sheet_to_json<any>(worksheet)

    // 将预测结果加到每一行
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    json.forEach((row: any, idx: number) => {
      row['预测值'] = chartData.value[idx] ?? ''
    })

    // 生成新sheet和workbook
    const newSheet = XLSX.utils.json_to_sheet(json)
    const newWb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(newWb, newSheet, 'Sheet1')

    // 导出
    XLSX.writeFile(newWb, 'predict_result.xlsx')
  }
  reader.readAsArrayBuffer(excelFile.value)
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
.download-btn {
  margin-left: 12px;
}
</style>
