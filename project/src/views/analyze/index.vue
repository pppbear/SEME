<template>
  <div class="kan-page">
    <el-card class="card">
      <div class="button-row">
        <el-button @click="showFormat = true">查看数据格式</el-button>
        <el-upload :before-upload="handleFile" accept=".xlsx,.xls" class="upload-button">
          <el-button type="primary">上传数据文件</el-button>
        </el-upload>
        <el-select v-model="selectedData" placeholder="请选择因变量" class="select">
          <el-option v-for="col in options" :key="col.value" :label="col.label" :value="col.value" />
        </el-select>
        <el-button
          type="primary"
          :disabled="!selectedData || !excelFile || loading"
          :loading="loading"
          @click="handleAnalyze"
          class="analyze-btn"
        >分析</el-button>
      </div>
      <div v-if="excelFileName" class="filename">已上传：{{ excelFileName }}</div>
    </el-card>

    <el-card class="card result-card">
      <div class="result-title">
        关键自变量分析结果
      </div>
      <div v-if="!hasResult" class="no-result">
        暂无数据，请上传文件并点击“分析”
      </div>
      <el-table
        v-else
        :data="resultList"
        style="width: 100%"
        border
        :loading="loading"
      >
        <el-table-column prop="feature_name" label="自变量" />
        <el-table-column label="重要性">
          <template #default="{ row }">
            <el-progress
              :percentage="Math.round(row.feature_value * 100)"
              :text-inside="true"
              :stroke-width="18"
              status="success"
              color="#409EFF"
              style="width: 90%;"
            >
            </el-progress>
          </template>
        </el-table-column>
      </el-table>
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
import { ref } from 'vue'
import { predictResult } from '@/api/analyze'

const options = [
  { label: '白天地表平均温度', value: 'lst_day_c' },
  { label: '夜晚地表平均温度', value: 'lst_night_c' },
  { label: '夜晚灯光辐射值', value: 'nighttime_' }
]
const selectedData = ref<string>('')
const excelFile = ref<File | null>(null)
const excelFileName = ref<string>('')
const loading = ref(false)
const hasResult = ref(false)
const showFormat = ref(false)
const resultList = ref<{ feature_name: string, feature_value: number }[]>([])

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

function handleFile(file: File) {
  excelFile.value = file
  excelFileName.value = file.name
  hasResult.value = false
  resultList.value = []
  return false // 阻止自动上传
}

async function handleAnalyze() {
  if (!excelFile.value || !selectedData.value) return
  loading.value = true
  hasResult.value = false
  resultList.value = []
  try {
    const formData = new FormData()
    formData.append('file', excelFile.value)
    formData.append('dependent_name', selectedData.value)
    const res = await predictResult(formData as any)
    if (res.code === 200 && Array.isArray(res.data)) {
      resultList.value = res.data
      hasResult.value = true
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.card {
  padding: 20px;
}
.result-card {
  margin-bottom: 20px;
}
.button-row {
  flex-wrap: wrap;
  align-items: center;
  justify-items: center;
}
.button-row > * {
  margin-right: 12px;
}
.analyze-btn {
  margin-left: 16px;
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
.upload-button {
  display: inline-block;
}
.result-title {
  font-weight: bold;
  margin-bottom: 10px;
}
.no-result {
  height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #999;
}
.dialog-tip {
  margin-bottom: 10px;
}
</style>