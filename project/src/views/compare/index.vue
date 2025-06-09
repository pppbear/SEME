<template>
  <div class="container">
    <el-card class="card">
      <div class="button-row">
        <el-button @click="showFormat = true">查看文件格式</el-button>
        <el-upload :before-upload="handleFile" accept=".xlsx,.xls" class="upload-button">
          <el-button type="primary">上传样本文件</el-button>
        </el-upload>
        <el-select v-model="selectedVariable" placeholder="请选择因变量" class="select">
          <el-option v-for="col in options" :key="col.value" :label="col.label" :value="col.value" />
        </el-select>
        <el-button
          type="primary"
          :disabled="!selectedVariable || !excelFile || loading"
          :loading="loading"
          @click="compare"
        >预测对比</el-button>
      </div>
      <div v-if="excelFileName" class="filename">已上传：{{ excelFileName }}</div>
    </el-card>

    <el-card>
      <div v-if="!hasResult" class="no-result">
        暂无数据，请上传文件并点击“预测对比”
      </div>
      <div v-else class="chart" ref="lineChartRef"></div>
    </el-card>

    <el-card>
      <div v-if="!hasResult" class="no-result">
        暂无数据，请上传文件并点击“预测对比”
      </div>
      <div v-else class="chart" ref="barChartRef"></div>
    </el-card>

    <el-dialog v-model="showFormat" title="输入数据格式要求" width="600">
      <p class="dialog-tip">
        请确保上传的 Excel 文件包含以下字段，字段顺序可不固定，但列名需正确，且必须包含选定的因变量列（如 <b>lst_day_c</b>）。
      </p>
      <el-table :data="fieldList" style="width: 100%" border>
        <el-table-column prop="name" label="字段名" />
        <el-table-column prop="desc" label="说明" />
      </el-table>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from 'vue'
import { getCompareResult, type CompareResult } from '@/api/compare/index'
import * as echarts from 'echarts'
const excelFile = ref<File | null>(null) // 上传的原文件
const excelFileName = ref<string>('') // 展示上传的文件名
const selectedVariable = ref<string>("lst_day_c") // 选定的参数
const showFormat = ref<boolean>(false)
const lineChartRef = ref<HTMLElement>()
const barChartRef = ref<HTMLElement>()
let myLineChart: echarts.ECharts
let myBarChart: echarts.ECharts
const hasResult = ref(false)
const loading = ref(false) // 新增loading状态

// 选项列表
const options = [
  { label: '白天地表平均温度', value: 'lst_day_c' },
  { label: '夜晚地表平均温度', value: 'lst_night_c' },
  { label: '夜晚灯光辐射值', value: 'nighttime_' }
]
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

// 处理上传的文件
function handleFile(file: File) {
  const reader = new FileReader()
  reader.onload = () => {
    excelFile.value = file
    excelFileName.value = file.name
  }
  reader.readAsArrayBuffer(file)
  return false // 阻止自动上传
}

async function compare() {
  loading.value = true // 开始loading
  hasResult.value = true
  try {
    await nextTick()
    myLineChart = echarts.init(lineChartRef.value)
    myBarChart = echarts.init(barChartRef.value)
    myLineChart.showLoading()
    myBarChart.showLoading()
    const result: CompareResult = (await getCompareResult({file: excelFile.value as unknown as File, dependent_name: selectedVariable.value})).data[0]
    updateChart(result)
    updateBarChart(result)
  } finally {
    loading.value = false // 结束loading
    myLineChart.hideLoading()
    myBarChart.hideLoading()
  }
}

function updateChart(result: CompareResult) {
  // 折线图
  myLineChart.hideLoading()
  const option = {
    title: {
      text: '模型预测值对比折线图'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['真实值', 'MLP预测', 'RF预测', 'KAN预测']
    },
    xAxis: {
      type: 'category',
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '真实值',
        type: 'line',
        data: result.true_values
      },
      {
        name: 'MLP预测',
        type: 'line',
        data: result.mlp_predictions
      },
      {
        name: 'RF预测',
        type: 'line',
        data: result.rf_predictions
      },
      {
        name: 'KAN预测',
        type: 'line',
        data: result.kan_predictions
      }
    ]
  };

  myLineChart.setOption(option);
}
function updateBarChart(result: CompareResult) {
  myBarChart.hideLoading()
  const option = {
    title: { text: '模型性能指标对比（MSE 和 R²）' },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    legend: {
      data: ['MSE', 'R²']
    },
    xAxis: {
      type: 'category',
      data: ['MLP', 'RF', 'KAN']
    },
    yAxis: [
      {
        type: 'value',
        name: 'MSE',
        position: 'left',
        axisLine: { lineStyle: { color: '#5470c6' } },
        axisLabel: { formatter: '{value}' }
      },
      {
        type: 'value',
        name: 'R²',
        position: 'right',
        min: 0,
        max: 1,
        axisLine: { lineStyle: { color: '#91cc75' } },
        axisLabel: { formatter: '{value}' }
      }
    ],
    series: [
      {
        name: 'MSE',
        type: 'bar',
        data: [result.mse_mlp, result.mse_rf, result.mse_kan],
        itemStyle: { color: '#5470c6' },
        yAxisIndex: 0
      },
      {
        name: 'R²',
        type: 'bar',
        data: [result.r2_mlp, result.r2_rf, result.r2_kan],
        itemStyle: { color: '#91cc75' },
        yAxisIndex: 1
      }
    ]
  }
  myBarChart.setOption(option)
}

</script>

<style scoped lang="scss">
.card {
  padding: 20px;
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
.button-row {
  flex-wrap: wrap;
  align-items: center;
  justify-items: center;
}
.button-row > * {
  margin-right: 12px;
}
.upload-button {
  display: inline-block;
}
.no-result {
  height: 400px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #999;
}
.chart {
  width: 100%;
  height: 400px;
}
.dialog-tip {
  margin-bottom: 10px;
}
</style>
