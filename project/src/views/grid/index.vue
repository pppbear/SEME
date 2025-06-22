<template>
  <div class="container">
    <!-- 选择框 -->
    <el-select v-model="selectedVar" placeholder="选择展示的栅格数据" style="width: 200px; margin-bottom: 10px">
      <el-option v-for="item in variableOptions" :key="item.value" :label="item.label" :value="item.value" />
    </el-select>

    <!-- 图表 -->
    <div class="heapChart" ref="chartRef"></div>
  </div>
</template>

<script setup lang="ts">
import * as echarts from "echarts";
import { ref, onMounted, watch } from "vue";
import { getGrid, type GridResponse, type GridRow } from "@/api/grid/index"

const chartRef = ref<HTMLElement | null>(null)

// 选项列表
const variableOptions = [
  { label: '白天地表平均温度', value: 'lst_day_c' },
  { label: '夜晚地表平均温度', value: 'lst_night_c' },
  { label: '夜晚灯光辐射值', value: 'nighttime_' }
]

// 当前选择的变量
const selectedVar = ref('lst_day_c')
const selectedLabel = ref('白天的地表平均温度')
// 监听选择变化更新图表
watch(selectedVar, (newVal) => {
  updateChart(newVal)
})

let myChart: echarts.ECharts 

// 初始化图表和地图
onMounted(async () => {
  const res = await fetch('/data/ShangHai.json')
  const geoJson = await res.json()
  echarts.registerMap('SH', geoJson)

  myChart = echarts.init(chartRef.value)
    window.addEventListener('resize', () => {
    myChart.resize()
  })
  myChart.showLoading()
  // 初始绘制
  updateChart(selectedVar.value)
})

// 监听变量变化并更新图表
watch(selectedVar, (newVal) => {
  switch(newVal) {
    case 'lst_day_c':
      selectedLabel.value = '白天地表平均温度'
      break
    case 'lst_night_c':
      selectedLabel.value = '夜晚地表平均温度'
      break
    case 'nighttime_':
      selectedLabel.value = '夜晚灯光辐射值'
      break
  }
  myChart.showLoading()
  updateChart(newVal)
})

// 图表更新函数
async function updateChart(variable: string) {
  if (!chartRef.value) return

  try {
    const response: GridResponse = await getGrid({ data: variable })
    const heatmapData: GridRow[] = response.data
    const formattedData = heatmapData.map((item) => {
      return {
        value: [item.longitude, item.latitude, item.value]
      }
    })
    
    let max: number = 0
    let min: number = Infinity
    heatmapData.map((val) => {
      if(val.value === 0) {
        return
      }
      max = Math.max(max, val.value)
      min = Math.min(min, val.value)
    })
    console.log(max, min)
    const option: echarts.EChartsOption = {
      title: {
        text: `上海${selectedLabel.value}热力图`,
        left: 'center'
      },
      visualMap: {
        min: min,
        max: max,
        left: 'left',
        top: 'bottom',
        text: ['高', '低'],
        inRange: {
          color: ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']
        },
        calculable: true
      },
      geo: {
        map: 'SH',
        roam: true,
        label: {
          show: true
        },
        itemStyle: {
          borderColor: '#999'
        }
      },
      series: [
        {
          name: '热力值',
          type: 'heatmap',
          coordinateSystem: 'geo',
          data: formattedData,
          pointSize: 3,
          blurSize: 6
        }
      ]
    }
    myChart.hideLoading()
    myChart.setOption(option)
  } catch (err) {
    console.error('更新图表失败:', err)
  }
}
</script>

<style scoped lang="scss">
.container {
  height: 100%;
  .heapChart {
    height: 80%;
    width: 80%;
  }
}
</style>