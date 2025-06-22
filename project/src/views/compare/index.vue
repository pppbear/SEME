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

    <el-dialog v-model="showFormat" title="输入数据格式要求" width="800">
      <p class="dialog-tip">
        请确保上传的 Excel 文件包含以下字段，字段顺序可不固定，但列名需正确，且必须包含选定的因变量列。
      </p>
      <el-table :data="fieldList" style="width: 100%" border>
        <el-table-column prop="name" label="字段名" />
        <el-table-column prop="unit" label="单位" />
        <el-table-column prop="description" label="说明" />
        <el-table-column prop="role" label="角色" />
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
  { name: "nighttime_", unit: "nW/sr/cm²", description: "夜间灯光辐射值", role: "因变量" },
  { name: "lst_day_c", unit: "℃", description: "白天地表平均温度", role: "因变量" },
  { name: "lst_night_c", unit: "℃", description: "夜晚地表平均温度", role: "因变量" },
];

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
