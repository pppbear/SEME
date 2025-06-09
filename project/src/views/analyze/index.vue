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

    <el-dialog v-model="showFormat" title="输入数据格式要求" width="600">
      <p class="dialog-tip">
        请确保上传的 Excel 文件包含以下字段，字段顺序可不固定，但列名需正确，且必须包含选定的因变量列。
      </p>
      <el-table :data="fieldList" style="width: 100%" border>
        <el-table-column prop="name" label="字段名" />
        <el-table-column prop="desc" label="说明" />
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