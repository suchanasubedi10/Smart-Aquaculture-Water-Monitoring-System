/**
 * Water Intelligence Dashboard - Charts Module
 * Chart.js based visualizations for water quality data
 */

// ============================================================================
// CHART INSTANCES
// ============================================================================
let forecastChart = null
let anomalyTimelineChart = null
let riskHeatmapChart = null
let driftTrendChart = null

// ============================================================================
// CHART COLORS (Theme-aware)
// ============================================================================
function getChartColors() {
  const isDark = document.documentElement.classList.contains('dark')
  return {
    text: isDark ? '#e5e7eb' : '#374151',
    grid: isDark ? '#374151' : '#e5e7eb',
    background: isDark ? '#1f2937' : '#ffffff',
    ph: {
      border: 'rgb(59, 130, 246)',
      background: 'rgba(59, 130, 246, 0.1)',
    },
    tds: {
      border: 'rgb(147, 51, 234)',
      background: 'rgba(147, 51, 234, 0.1)',
    },
    phForecast: {
      border: 'rgb(34, 197, 94)',
      background: 'rgba(34, 197, 94, 0.2)',
    },
    tdsForecast: {
      border: 'rgb(249, 115, 22)',
      background: 'rgba(249, 115, 22, 0.2)',
    },
    anomaly: {
      normal: 'rgba(34, 197, 94, 0.7)',
      anomaly: 'rgba(239, 68, 68, 0.9)',
    },
    risk: {
      safe: 'rgba(34, 197, 94, 0.7)',
      stress: 'rgba(234, 179, 8, 0.7)',
      danger: 'rgba(239, 68, 68, 0.7)',
      empty: isDark ? 'rgba(55, 65, 81, 0.3)' : 'rgba(229, 231, 235, 0.5)',
    },
  }
}

// ============================================================================
// FORECAST CHART (Task 3)
// ============================================================================
function initForecastChart() {
  const ctx = document.getElementById('forecastChart')
  if (!ctx) return null

  const colors = getChartColors()

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'pH (Historical)',
          data: [],
          borderColor: colors.ph.border,
          backgroundColor: colors.ph.background,
          tension: 0.3,
          fill: false,
          pointRadius: 3,
        },
        {
          label: 'TDS (Historical)',
          data: [],
          borderColor: colors.tds.border,
          backgroundColor: colors.tds.background,
          tension: 0.3,
          fill: false,
          pointRadius: 3,
          yAxisID: 'y1',
        },
        {
          label: 'pH (Forecast)',
          data: [],
          borderColor: colors.phForecast.border,
          backgroundColor: colors.phForecast.background,
          borderDash: [5, 5],
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointStyle: 'triangle',
        },
        {
          label: 'TDS (Forecast)',
          data: [],
          borderColor: colors.tdsForecast.border,
          backgroundColor: colors.tdsForecast.background,
          borderDash: [5, 5],
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointStyle: 'triangle',
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          labels: { color: colors.text },
        },
        tooltip: {
          backgroundColor: colors.background,
          titleColor: colors.text,
          bodyColor: colors.text,
          borderColor: colors.grid,
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Time', color: colors.text },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: { display: true, text: 'pH', color: colors.text },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
          min: 5,
          max: 10,
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: { display: true, text: 'TDS (mg/L)', color: colors.text },
          ticks: { color: colors.text },
          grid: { drawOnChartArea: false },
          min: 0,
          max: 1500,
        },
      },
    },
  })

  return forecastChart
}

function updateForecastChart(historicalData, forecastData) {
  if (!forecastChart) {
    initForecastChart()
  }
  if (!forecastChart) return

  const colors = getChartColors()

  // Historical labels (timestamps)
  const histLabels = historicalData.map((d, i) => {
    if (d.timestamp) {
      return new Date(d.timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      })
    }
    return `T-${historicalData.length - i}`
  })

  // Forecast labels
  const forecastLabels = forecastData.ph
    ? forecastData.ph.map((_, i) => `+${i + 1}`)
    : []

  // Combine labels
  const allLabels = [...histLabels, ...forecastLabels]

  // Historical data
  const histPh = historicalData.map((d) => d.ph)
  const histTds = historicalData.map((d) => d.tds)

  // Forecast data with null padding for historical period
  const nullPadding = new Array(historicalData.length).fill(null)
  const forecastPh = forecastData.ph ? [...nullPadding, ...forecastData.ph] : []
  const forecastTds = forecastData.tds
    ? [...nullPadding, ...forecastData.tds]
    : []

  // Extend historical with nulls for forecast period
  const histPhExtended = [
    ...histPh,
    ...new Array(forecastLabels.length).fill(null),
  ]
  const histTdsExtended = [
    ...histTds,
    ...new Array(forecastLabels.length).fill(null),
  ]

  forecastChart.data.labels = allLabels
  forecastChart.data.datasets[0].data = histPhExtended
  forecastChart.data.datasets[1].data = histTdsExtended
  forecastChart.data.datasets[2].data = forecastPh
  forecastChart.data.datasets[3].data = forecastTds
  forecastChart.update()
}

function clearForecastChart() {
  if (forecastChart) {
    forecastChart.data.labels = []
    forecastChart.data.datasets.forEach((ds) => (ds.data = []))
    forecastChart.update()
  }
}

// ============================================================================
// ANOMALY TIMELINE CHART (Task 4)
// ============================================================================
function initAnomalyTimelineChart() {
  // Prevent re-initialization
  if (anomalyTimelineChart) return anomalyTimelineChart

  const ctx = document.getElementById('anomalyTimelineChart')
  if (!ctx) return null

  const colors = getChartColors()

  anomalyTimelineChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Normal',
          data: [],
          backgroundColor: colors.anomaly.normal,
          pointRadius: 6,
          pointHoverRadius: 8,
        },
        {
          label: 'Anomaly',
          data: [],
          backgroundColor: colors.anomaly.anomaly,
          pointRadius: 8,
          pointHoverRadius: 10,
          pointStyle: 'triangle',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: colors.text },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return `Score: ${context.parsed.y.toFixed(3)} at ${
                context.raw.time
              }`
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Time (minutes ago)',
            color: colors.text,
          },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
          reverse: true,
        },
        y: {
          title: { display: true, text: 'Anomaly Score', color: colors.text },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
          min: -0.5,
          max: 0.5,
        },
      },
    },
  })

  // Add placeholder data
  updateAnomalyTimelineWithPlaceholder()

  return anomalyTimelineChart
}

function updateAnomalyTimelineWithPlaceholder() {
  const normalData = []
  const anomalyData = []

  // Generate 30 data points
  for (let i = 0; i < 30; i++) {
    const score = (Math.random() - 0.5) * 0.4
    const isAnomaly = score < -0.15 || score > 0.2
    const point = {
      x: i * 5, // minutes ago
      y: score,
      time: `${i * 5}m ago`,
    }

    if (isAnomaly) {
      anomalyData.push(point)
    } else {
      normalData.push(point)
    }
  }

  if (anomalyTimelineChart) {
    anomalyTimelineChart.data.datasets[0].data = normalData
    anomalyTimelineChart.data.datasets[1].data = anomalyData
    anomalyTimelineChart.update()
  }
}

function addAnomalyPoint(score, isAnomaly, timestamp) {
  if (!anomalyTimelineChart) return

  const point = {
    x: 0,
    y: score,
    time: new Date(timestamp).toLocaleTimeString(),
  }

  // Shift existing points
  anomalyTimelineChart.data.datasets.forEach((ds) => {
    ds.data.forEach((p) => (p.x += 5))
    // Remove points older than 150 minutes
    ds.data = ds.data.filter((p) => p.x <= 150)
  })

  // Add new point
  if (isAnomaly) {
    anomalyTimelineChart.data.datasets[1].data.push(point)
  } else {
    anomalyTimelineChart.data.datasets[0].data.push(point)
  }

  anomalyTimelineChart.update()
}

// ============================================================================
// RISK HEATMAP (Task 4) - 24x7 Grid
// ============================================================================
function initRiskHeatmap() {
  // Prevent re-initialization
  if (riskHeatmapChart) return riskHeatmapChart

  const ctx = document.getElementById('riskHeatmapChart')
  if (!ctx) return null

  const colors = getChartColors()
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`)

  // Generate placeholder data (24 hours x 7 days)
  const data = []
  for (let day = 0; day < 7; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const risk = Math.random()
      data.push({
        x: hour,
        y: day,
        v: risk < 0.6 ? 0 : risk < 0.85 ? 1 : 2, // 0=SAFE, 1=STRESS, 2=DANGER
      })
    }
  }

  riskHeatmapChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Risk Level',
          data: data.map((d) => ({ x: d.x, y: d.y })),
          backgroundColor: data.map((d) => {
            if (d.v === 0) return colors.risk.safe
            if (d.v === 1) return colors.risk.stress
            return colors.risk.danger
          }),
          pointRadius: 12,
          pointStyle: 'rect',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              const riskLevels = ['SAFE', 'STRESS', 'DANGER']
              const idx = context.dataIndex
              return `${days[data[idx].y]} ${hours[data[idx].x]}: ${
                riskLevels[data[idx].v]
              }`
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          min: -0.5,
          max: 23.5,
          title: { display: true, text: 'Hour of Day', color: colors.text },
          ticks: {
            color: colors.text,
            stepSize: 4,
            callback: (val) => `${val}:00`,
          },
          grid: { display: false },
        },
        y: {
          type: 'linear',
          min: -0.5,
          max: 6.5,
          title: { display: true, text: 'Day', color: colors.text },
          ticks: {
            color: colors.text,
            stepSize: 1,
            callback: (val) => days[val] || '',
          },
          grid: { display: false },
        },
      },
    },
  })

  return riskHeatmapChart
}

function updateRiskHeatmap(riskData) {
  if (!riskHeatmapChart) return

  const colors = getChartColors()

  riskHeatmapChart.data.datasets[0].data = riskData.map((d) => ({
    x: d.hour,
    y: d.day,
  }))
  riskHeatmapChart.data.datasets[0].backgroundColor = riskData.map((d) => {
    if (d.risk === 0) return colors.risk.safe
    if (d.risk === 1) return colors.risk.stress
    return colors.risk.danger
  })

  riskHeatmapChart.update()
}

// ============================================================================
// SENSOR DRIFT CHART (Task 6)
// ============================================================================
function initDriftTrendChart() {
  const ctx = document.getElementById('driftTrendChart')
  if (!ctx) return null

  const colors = getChartColors()

  driftTrendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'pH Sensor Drift (%)',
          data: [],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.3,
          fill: true,
        },
        {
          label: 'TDS Sensor Drift (%)',
          data: [],
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: colors.text },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Time', color: colors.text },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
        },
        y: {
          title: { display: true, text: 'Drift %', color: colors.text },
          ticks: { color: colors.text },
          grid: { color: colors.grid },
          min: -5,
          max: 10,
        },
      },
    },
  })

  // Add placeholder data
  updateDriftTrendWithPlaceholder()

  return driftTrendChart
}

function updateDriftTrendWithPlaceholder() {
  const labels = []
  const phDrift = []
  const tdsDrift = []

  for (let i = 0; i < 24; i++) {
    labels.push(`${i}:00`)
    phDrift.push((Math.random() - 0.3) * 3)
    tdsDrift.push((Math.random() - 0.2) * 4)
  }

  if (driftTrendChart) {
    driftTrendChart.data.labels = labels
    driftTrendChart.data.datasets[0].data = phDrift
    driftTrendChart.data.datasets[1].data = tdsDrift
    driftTrendChart.update()
  }
}

function updateDriftTrendChart(data) {
  if (!driftTrendChart) return

  driftTrendChart.data.labels = data.labels || []
  driftTrendChart.data.datasets[0].data = data.phDrift || []
  driftTrendChart.data.datasets[1].data = data.tdsDrift || []
  driftTrendChart.update()
}

// ============================================================================
// THEME UPDATE
// ============================================================================
function updateChartsTheme() {
  const colors = getChartColors()
  const charts = [
    forecastChart,
    anomalyTimelineChart,
    riskHeatmapChart,
    driftTrendChart,
  ]

  charts.forEach((chart) => {
    if (!chart) return

    // Update legend
    if (chart.options.plugins?.legend?.labels) {
      chart.options.plugins.legend.labels.color = colors.text
    }

    // Update scales
    Object.values(chart.options.scales || {}).forEach((scale) => {
      if (scale.title) scale.title.color = colors.text
      if (scale.ticks) scale.ticks.color = colors.text
      if (scale.grid) scale.grid.color = colors.grid
    })

    chart.update()
  })
}

// ============================================================================
// INITIALIZATION
// ============================================================================
function initAllCharts() {
  initForecastChart()
  initAnomalyTimelineChart()
  initRiskHeatmap()
  initDriftTrendChart()
}

// Getter for forecast chart instance
function getForecastChart() {
  return forecastChart
}

// Export functions
window.initAllCharts = initAllCharts
window.initForecastChart = initForecastChart
window.updateForecastChart = updateForecastChart
window.clearForecastChart = clearForecastChart
window.getForecastChart = getForecastChart
window.initAnomalyTimelineChart = initAnomalyTimelineChart
window.addAnomalyPoint = addAnomalyPoint
window.initRiskHeatmap = initRiskHeatmap
window.updateRiskHeatmap = updateRiskHeatmap
window.initDriftTrendChart = initDriftTrendChart
window.updateDriftTrendChart = updateDriftTrendChart
window.updateChartsTheme = updateChartsTheme
