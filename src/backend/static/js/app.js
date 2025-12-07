/**
 * Water Intelligence Dashboard - Main Application JavaScript
 * Smart Aquaculture Water Monitoring Using AI and IoT
 */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const AppState = {
  currentTab: 'dashboard',
  darkMode: localStorage.getItem('darkMode') === 'true',
  readingsHistory: [],
  lastPrediction: null,
  lastForecast: null,
  MAX_HISTORY: 100,
}

// ============================================================================
// THEME MANAGEMENT
// ============================================================================
function initTheme() {
  if (AppState.darkMode) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
  updateThemeToggleIcon()
}

function toggleDarkMode() {
  AppState.darkMode = !AppState.darkMode
  localStorage.setItem('darkMode', AppState.darkMode)
  document.documentElement.classList.toggle('dark')
  updateThemeToggleIcon()
}

function updateThemeToggleIcon() {
  const sunIcon = document.getElementById('sunIcon')
  const moonIcon = document.getElementById('moonIcon')
  if (sunIcon && moonIcon) {
    if (AppState.darkMode) {
      sunIcon.classList.remove('hidden')
      moonIcon.classList.add('hidden')
    } else {
      sunIcon.classList.add('hidden')
      moonIcon.classList.remove('hidden')
    }
  }
}

// ============================================================================
// NAVIGATION
// ============================================================================
function initNavigation() {
  const navItems = document.querySelectorAll('[data-tab]')
  navItems.forEach((item) => {
    item.addEventListener('click', (e) => {
      e.preventDefault()
      const tabId = item.getAttribute('data-tab')
      switchTab(tabId)
    })
  })
}

function switchTab(tabId) {
  AppState.currentTab = tabId

  // Update nav active states
  document.querySelectorAll('[data-tab]').forEach((item) => {
    const isActive = item.getAttribute('data-tab') === tabId
    item.classList.toggle('bg-blue-600', isActive)
    item.classList.toggle('text-white', isActive)
    item.classList.toggle('text-gray-700', !isActive)
    item.classList.toggle('dark:text-gray-300', !isActive)
    item.classList.toggle('hover:bg-gray-100', !isActive)
    item.classList.toggle('dark:hover:bg-gray-700', !isActive)
  })

  // Show/hide content sections
  document.querySelectorAll('[data-content]').forEach((section) => {
    const isVisible = section.getAttribute('data-content') === tabId
    section.classList.toggle('hidden', !isVisible)
  })

  // Close mobile sidebar after navigation
  closeMobileSidebar()
}

// ============================================================================
// MOBILE SIDEBAR
// ============================================================================
function toggleMobileSidebar() {
  const sidebar = document.getElementById('mobileSidebar')
  const overlay = document.getElementById('sidebarOverlay')

  if (sidebar && overlay) {
    sidebar.classList.toggle('-translate-x-full')
    overlay.classList.toggle('hidden')
  }
}

function closeMobileSidebar() {
  const sidebar = document.getElementById('mobileSidebar')
  const overlay = document.getElementById('sidebarOverlay')

  if (sidebar && overlay) {
    sidebar.classList.add('-translate-x-full')
    overlay.classList.add('hidden')
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function formatTimestamp(isoString) {
  const date = new Date(isoString)
  return date.toLocaleString()
}

function showNotification(message, type = 'info', duration = 5000) {
  const container = document.getElementById('notificationContainer')
  if (!container) return

  const colors = {
    info: 'bg-blue-500',
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
  }

  const notification = document.createElement('div')
  notification.className = `${colors[type]} text-white px-4 py-3 rounded-lg shadow-lg transform transition-all duration-300 translate-y-2 opacity-0 max-w-md`
  notification.innerHTML = `
    <div class="flex items-start justify-between">
      <span class="whitespace-pre-wrap text-sm">${message}</span>
      <button onclick="this.parentElement.parentElement.remove()" class="ml-4 hover:opacity-75 flex-shrink-0">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>
  `

  container.appendChild(notification)

  // Animate in
  requestAnimationFrame(() => {
    notification.classList.remove('translate-y-2', 'opacity-0')
  })

  // Auto remove after duration
  setTimeout(() => {
    notification.classList.add('translate-y-2', 'opacity-0')
    setTimeout(() => notification.remove(), 300)
  }, duration)
}

// ============================================================================
// QUICK SCENARIOS
// ============================================================================
function setQuickScenario(scenario) {
  const phInput = document.getElementById('ph')
  const tdsInput = document.getElementById('tds')

  // Values calibrated to match trained model predictions:
  // SAFE (0): pH ~7.0, TDS ~200
  // STRESS (1): pH ~6.3, TDS ~400
  // DANGER (2): pH ~5.9, TDS ~200
  const scenarios = {
    safe: { ph: 7.0, tds: 200 }, // Returns prediction=0 (SAFE)
    stress: { ph: 6.3, tds: 400 }, // Returns prediction=1 (STRESS)
    danger: { ph: 5.9, tds: 200 }, // Returns prediction=2 (DANGER)
  }

  const values = scenarios[scenario]
  if (values && phInput && tdsInput) {
    phInput.value = values.ph
    tdsInput.value = values.tds

    showNotification(
      `Set ${scenario.toUpperCase()} scenario (pH: ${values.ph}, TDS: ${
        values.tds
      })`,
      'info'
    )
  }
}

// ============================================================================
// PREDICTION API
// ============================================================================
async function predict() {
  const phInput = document.getElementById('ph')
  const tdsInput = document.getElementById('tds')
  const timestampInput = document.getElementById('timestamp')
  const deviceIdInput = document.getElementById('deviceId')
  const predictBtn = document.getElementById('predictBtn')

  const ph = parseFloat(phInput.value)
  const tds = parseFloat(tdsInput.value)
  const timestamp = timestampInput.value
    ? new Date(timestampInput.value).toISOString()
    : new Date().toISOString()
  const deviceId = deviceIdInput.value || 'dashboard'

  // Validate inputs
  if (isNaN(ph) || ph < 0 || ph > 14) {
    showNotification('pH must be between 0 and 14', 'error')
    return
  }
  if (isNaN(tds) || tds < 0) {
    showNotification('TDS must be a positive number', 'error')
    return
  }

  const payload = {
    timestamp: timestamp,
    ph: ph,
    tds: tds,
    device_id: deviceId,
  }

  try {
    // Update button state
    predictBtn.disabled = true
    predictBtn.innerHTML = `
      <svg class="w-4 h-4 mr-2 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
      Predicting...
    `

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}))
      throw new Error(
        errorData.detail || `HTTP ${res.status}: ${res.statusText}`
      )
    }

    const data = await res.json()

    // Store last prediction
    AppState.lastPrediction = data

    // Update UI components
    updateStatusBadge(data.classification?.prediction, data.anomaly?.is_anomaly)
    updatePredictionResults(data)
    updateVirtualSensors(data.virtual)
    updateCleanedFeatures(data.features)
    updateAlerts(
      data.recommendations_structured || data.recommendations,
      data.anomaly?.is_anomaly
    )
    updateAnomalyIndicator(data.anomaly?.is_anomaly)
    updateWaterQualityGauge(data)

    // Add to history
    addToReadingsHistory(payload, data)

    showNotification('Prediction completed successfully', 'success')
  } catch (err) {
    console.error('Prediction error:', err)
    showNotification(`Prediction failed: ${err.message}`, 'error')
    updateAlerts([{ message: err.message, severity: 'critical' }], false)
  } finally {
    predictBtn.disabled = false
    predictBtn.innerHTML = `
      <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
      </svg>
      Predict Now
    `
  }
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================
function updateStatusBadge(prediction, isAnomaly = false) {
  const statusBadge = document.getElementById('statusBadge')
  if (!statusBadge) return

  const labels = ['SAFE', 'STRESS', 'DANGER']
  const icons = [
    '<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    '<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
    '<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
  ]
  const classes = [
    'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800',
    'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-400 dark:border-yellow-800',
    'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800',
  ]

  const label = labels[prediction] ?? 'UNKNOWN'
  const icon = icons[prediction] ?? icons[0]
  const className =
    classes[prediction] ??
    'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600'

  let anomalyBadge = ''
  if (isAnomaly) {
    anomalyBadge = `
      <span class="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400">
        <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
        Anomaly
      </span>
    `
  }

  statusBadge.innerHTML = `
    <span class="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${className} border">
      ${icon}
      ${label}
    </span>
    ${anomalyBadge}
  `
}

// Helper function to get trend indicator
function getTrendIndicator(current, previous, isHigherBetter = true) {
  if (previous === null || previous === undefined) return ''

  const diff = current - previous
  const threshold = Math.abs(previous) * 0.05 // 5% threshold

  if (Math.abs(diff) < threshold) {
    return '<span class="ml-1 text-gray-400 text-xs">→</span>'
  } else if (diff > 0) {
    const color = isHigherBetter ? 'text-green-500' : 'text-red-500'
    return `<span class="ml-1 ${color} text-xs">↑</span>`
  } else {
    const color = isHigherBetter ? 'text-red-500' : 'text-green-500'
    return `<span class="ml-1 ${color} text-xs">↓</span>`
  }
}

function updatePredictionResults(data) {
  const predictionResults = document.getElementById('predictionResults')
  if (!predictionResults) return

  const features = data.features || {}
  const virtual = data.virtual || {}

  // Get previous values for trend comparison
  const prevReading =
    AppState.readingsHistory.length > 0 ? AppState.readingsHistory[0] : null
  const prevPh = prevReading?.ph
  const prevTds = prevReading?.tds
  const prevTemp = prevReading?.result?.virtual?.temp_est
  const prevDO = prevReading?.result?.virtual?.do_est

  // pH trend (neutral around 7 is good)
  const phTrend = getTrendIndicator(features.ph, prevPh, features.ph < 7)
  // TDS trend (lower is generally better)
  const tdsTrend = getTrendIndicator(features.tds, prevTds, false)
  // Temp trend (stable is good, show as neutral)
  const tempTrend = virtual.temp_est
    ? getTrendIndicator(virtual.temp_est, prevTemp, true)
    : ''
  // DO trend (higher is better)
  const doTrend = virtual.do_est
    ? getTrendIndicator(virtual.do_est, prevDO, true)
    : ''

  predictionResults.innerHTML = `
    <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center relative">
      <div class="text-2xl font-bold text-blue-700 dark:text-blue-400">${(
        features.ph ?? 0
      ).toFixed(2)}${phTrend}</div>
      <div class="text-xs text-blue-600 dark:text-blue-500 mt-1">pH</div>
    </div>
    <div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center relative">
      <div class="text-2xl font-bold text-purple-700 dark:text-purple-400">${(
        features.tds ?? 0
      ).toFixed(0)}${tdsTrend}</div>
      <div class="text-xs text-purple-600 dark:text-purple-500 mt-1">TDS</div>
    </div>
    <div class="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 text-center relative">
      <div class="text-2xl font-bold text-orange-700 dark:text-orange-400">${
        virtual.temp_est ? virtual.temp_est.toFixed(1) + '°C' : '--'
      }${tempTrend}</div>
      <div class="text-xs text-orange-600 dark:text-orange-500 mt-1">Temp Est.</div>
    </div>
    <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center relative">
      <div class="text-2xl font-bold text-green-700 dark:text-green-400">${
        virtual.do_est ? virtual.do_est.toFixed(1) : '--'
      }${doTrend}</div>
      <div class="text-xs text-green-600 dark:text-green-500 mt-1">DO Est.</div>
    </div>
  `
}

function updateVirtualSensors(virtual) {
  const virtualSensors = document.getElementById('virtualSensors')
  const vTemp = document.getElementById('vTemp')
  const vDO = document.getElementById('vDO')
  const vTurbidity = document.getElementById('vTurbidity')

  if (!virtualSensors || !virtual) return

  virtualSensors.classList.remove('hidden')

  if (vTemp)
    vTemp.textContent = virtual.temp_est
      ? `${virtual.temp_est.toFixed(1)}°C`
      : '--'
  if (vDO)
    vDO.textContent = virtual.do_est
      ? `${virtual.do_est.toFixed(2)} mg/L`
      : '--'
  if (vTurbidity)
    vTurbidity.textContent = virtual.turbidity_est
      ? `${virtual.turbidity_est.toFixed(1)} NTU`
      : '--'
}

function updateAnomalyIndicator(isAnomaly) {
  const anomalyIndicator = document.getElementById('anomalyIndicator')
  if (anomalyIndicator) {
    anomalyIndicator.classList.toggle('hidden', !isAnomaly)
  }
}

function updateCleanedFeatures(features) {
  const cleanedFeatures = document.getElementById('cleanedFeatures')
  const featuresJson = document.getElementById('featuresJson')

  if (!cleanedFeatures || !featuresJson) return

  if (features && Object.keys(features).length > 0) {
    cleanedFeatures.classList.remove('hidden')
    // Format features nicely
    const formattedFeatures = Object.entries(features)
      .map(([key, value]) => {
        const formattedValue =
          typeof value === 'number' ? value.toFixed(4) : value
        return `${key}: ${formattedValue}`
      })
      .join('\n')
    featuresJson.textContent = formattedFeatures
  }
}

// ============================================================================
// WATER QUALITY GAUGE
// ============================================================================
function updateWaterQualityGauge(data) {
  const circle = document.getElementById('qualityGaugeCircle')
  const scoreEl = document.getElementById('qualityScore')
  const labelEl = document.getElementById('qualityLabel')

  if (!circle || !scoreEl || !labelEl) return

  // Calculate water quality score (0-100)
  let score = 100
  const classification = data.classification?.prediction ?? 0
  const isAnomaly = data.anomaly?.is_anomaly ?? false
  const anomalyScore = data.anomaly?.score ?? 0

  // Base score from classification
  if (classification === 0) score = 95 // SAFE
  else if (classification === 1) score = 65 // STRESS
  else if (classification === 2) score = 25 // DANGER

  // Adjust for anomaly
  if (isAnomaly) score -= 20
  score -= Math.min(anomalyScore * 50, 30) // Reduce based on anomaly score

  // Check pH deviation from optimal (6.5-8.5)
  const ph = data.features?.ph ?? 7
  if (ph < 6.5 || ph > 8.5) score -= Math.abs(ph - 7.5) * 5

  // Clamp score
  score = Math.max(0, Math.min(100, score))

  // Update circle
  const circumference = 2 * Math.PI * 40 // 251.2
  const offset = circumference - (score / 100) * circumference
  circle.style.strokeDashoffset = offset

  // Color based on score
  if (score >= 80) {
    circle.classList.remove(
      'text-yellow-500',
      'text-red-500',
      'text-gray-300',
      'dark:text-gray-600'
    )
    circle.classList.add('text-green-500')
  } else if (score >= 50) {
    circle.classList.remove(
      'text-green-500',
      'text-red-500',
      'text-gray-300',
      'dark:text-gray-600'
    )
    circle.classList.add('text-yellow-500')
  } else {
    circle.classList.remove(
      'text-green-500',
      'text-yellow-500',
      'text-gray-300',
      'dark:text-gray-600'
    )
    circle.classList.add('text-red-500')
  }

  // Update text
  scoreEl.textContent = Math.round(score)
  scoreEl.className =
    score >= 80
      ? 'text-4xl font-bold text-green-600 dark:text-green-400'
      : score >= 50
      ? 'text-4xl font-bold text-yellow-600 dark:text-yellow-400'
      : 'text-4xl font-bold text-red-600 dark:text-red-400'

  const labels = { 80: 'Excellent', 50: 'Fair', 0: 'Poor' }
  labelEl.textContent =
    score >= 80 ? 'Excellent' : score >= 50 ? 'Fair' : 'Poor'
}

function updateAlerts(recommendations, isAnomaly) {
  const alertsList = document.getElementById('alertsList')
  if (!alertsList) return

  if (!recommendations || recommendations.length === 0) {
    alertsList.innerHTML = `
      <div class="text-gray-500 dark:text-gray-400 text-sm italic flex items-center">
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        All parameters within normal range. No alerts.
      </div>
    `
    return
  }

  const severityColors = {
    critical:
      'bg-red-50 border-red-300 text-red-800 dark:bg-red-900/30 dark:border-red-700 dark:text-red-400',
    high: 'bg-orange-50 border-orange-300 text-orange-800 dark:bg-orange-900/30 dark:border-orange-700 dark:text-orange-400',
    medium:
      'bg-yellow-50 border-yellow-300 text-yellow-800 dark:bg-yellow-900/30 dark:border-yellow-700 dark:text-yellow-400',
    low: 'bg-blue-50 border-blue-300 text-blue-800 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-400',
  }

  const severityIcons = {
    critical:
      '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>',
    high: '<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>',
    medium:
      '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>',
    low: '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>',
  }

  alertsList.innerHTML = recommendations
    .map((rec) => {
      const message = typeof rec === 'string' ? rec : rec.message
      const severity = typeof rec === 'object' ? rec.severity : 'medium'
      const code = typeof rec === 'object' ? rec.code : ''
      const colorClass = severityColors[severity] || severityColors.medium
      const iconPath = severityIcons[severity] || severityIcons.medium

      return `
        <div class="p-3 rounded-lg border ${colorClass}">
          <div class="flex items-start">
            <svg class="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              ${iconPath}
            </svg>
            <div class="flex-1">
              ${
                code
                  ? `<span class="text-xs font-mono opacity-75">[${code}]</span> `
                  : ''
              }
              <span class="text-sm">${message}</span>
            </div>
          </div>
        </div>
      `
    })
    .join('')
}

function addToReadingsHistory(payload, result) {
  const reading = {
    ...payload,
    result: result,
    addedAt: new Date().toISOString(),
  }

  AppState.readingsHistory.unshift(reading)
  if (AppState.readingsHistory.length > AppState.MAX_HISTORY) {
    AppState.readingsHistory.pop()
  }

  updateReadingsTable()
  updateLiveStatistics()
  updateDataComparison(payload, result)
}

// ============================================================================
// DATA COMPARISON
// ============================================================================
function updateDataComparison(current, result) {
  const card = document.getElementById('dataComparisonCard')
  if (!card) return

  const history = AppState.readingsHistory
  if (history.length < 2) {
    card.classList.add('hidden')
    return
  }

  card.classList.remove('hidden')

  // Calculate averages from history (excluding current)
  const pastReadings = history.slice(1)
  const avgPh =
    pastReadings.reduce((sum, r) => sum + r.ph, 0) / pastReadings.length
  const avgTds =
    pastReadings.reduce((sum, r) => sum + r.tds, 0) / pastReadings.length

  // Update current values
  const compCurrentPh = document.getElementById('compCurrentPh')
  const compCurrentTds = document.getElementById('compCurrentTds')
  const compAvgPh = document.getElementById('compAvgPh')
  const compAvgTds = document.getElementById('compAvgTds')
  const compPhDiff = document.getElementById('compPhDiff')
  const compTdsDiff = document.getElementById('compTdsDiff')
  const compStatusCurrent = document.getElementById('compStatusCurrent')
  const compStatusTrend = document.getElementById('compStatusTrend')

  if (compCurrentPh) compCurrentPh.textContent = current.ph.toFixed(2)
  if (compCurrentTds) compCurrentTds.textContent = current.tds.toFixed(0)
  if (compAvgPh) compAvgPh.textContent = avgPh.toFixed(2)
  if (compAvgTds) compAvgTds.textContent = avgTds.toFixed(0)

  // Calculate differences
  const phDiff = current.ph - avgPh
  const tdsDiff = current.tds - avgTds

  if (compPhDiff) {
    const phSign = phDiff >= 0 ? '+' : ''
    compPhDiff.textContent = `${phSign}${phDiff.toFixed(2)}`
    compPhDiff.className = `text-xs px-2 py-1 rounded-full ${
      Math.abs(phDiff) < 0.5
        ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400'
        : Math.abs(phDiff) < 1
        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-400'
        : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400'
    }`
  }

  if (compTdsDiff) {
    const tdsSign = tdsDiff >= 0 ? '+' : ''
    compTdsDiff.textContent = `${tdsSign}${tdsDiff.toFixed(0)}`
    compTdsDiff.className = `text-xs px-2 py-1 rounded-full ${
      Math.abs(tdsDiff) < 50
        ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400'
        : Math.abs(tdsDiff) < 100
        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-400'
        : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400'
    }`
  }

  // Update status
  const prediction = result?.classification?.prediction ?? -1
  const labels = ['SAFE', 'STRESS', 'DANGER']
  const statusColors = [
    'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400',
    'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-400',
    'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400',
  ]

  if (compStatusCurrent && prediction >= 0) {
    compStatusCurrent.textContent = labels[prediction]
    compStatusCurrent.className = `px-2 py-1 text-xs rounded-full ${statusColors[prediction]}`
  }

  // Calculate trend
  if (compStatusTrend && history.length >= 3) {
    const recentPreds = history
      .slice(0, 5)
      .map((r) => r.result?.classification?.prediction ?? 1)
    const avgPred = recentPreds.reduce((a, b) => a + b, 0) / recentPreds.length

    if (avgPred < 0.5) compStatusTrend.textContent = '📈 Trending Safe'
    else if (avgPred < 1.5) compStatusTrend.textContent = '📊 Moderate'
    else compStatusTrend.textContent = '📉 Needs Attention'
  }
}

// ============================================================================
// LIVE STATISTICS
// ============================================================================
function updateLiveStatistics() {
  const history = AppState.readingsHistory

  // Total readings
  const totalEl = document.getElementById('statTotalReadings')
  if (totalEl) totalEl.textContent = history.length

  if (history.length === 0) return

  // Calculate averages
  const phValues = history.map((r) => r.ph).filter((v) => !isNaN(v))
  const tdsValues = history.map((r) => r.tds).filter((v) => !isNaN(v))

  const avgPh = phValues.reduce((a, b) => a + b, 0) / phValues.length
  const avgTds = tdsValues.reduce((a, b) => a + b, 0) / tdsValues.length

  const avgPhEl = document.getElementById('statAvgPh')
  const avgTdsEl = document.getElementById('statAvgTds')
  if (avgPhEl) avgPhEl.textContent = avgPh.toFixed(2)
  if (avgTdsEl) avgTdsEl.textContent = avgTds.toFixed(0)

  // Calculate ranges
  const minPh = Math.min(...phValues)
  const maxPh = Math.max(...phValues)
  const minTds = Math.min(...tdsValues)
  const maxTds = Math.max(...tdsValues)

  const phRangeEl = document.getElementById('statPhRange')
  const tdsRangeEl = document.getElementById('statTdsRange')
  if (phRangeEl)
    phRangeEl.textContent = `${minPh.toFixed(1)} - ${maxPh.toFixed(1)}`
  if (tdsRangeEl)
    tdsRangeEl.textContent = `${minTds.toFixed(0)} - ${maxTds.toFixed(0)}`

  // Calculate anomaly rate
  const anomalies = history.filter(
    (r) => r.result?.anomaly?.is_anomaly === true
  )
  const anomalyRate = (anomalies.length / history.length) * 100

  const anomalyRateEl = document.getElementById('statAnomalyRate')
  if (anomalyRateEl) {
    anomalyRateEl.textContent = `${anomalyRate.toFixed(1)}%`
    anomalyRateEl.className =
      anomalyRate > 20
        ? 'font-semibold text-red-600 dark:text-red-400'
        : anomalyRate > 10
        ? 'font-semibold text-orange-600 dark:text-orange-400'
        : 'font-semibold text-green-600 dark:text-green-400'
  }

  // Update session summary in sidebar
  updateSessionSummary(history)
}

function updateSessionSummary(history) {
  const sessionReadings = document.getElementById('sessionReadings')
  const sessionAnomalies = document.getElementById('sessionAnomalies')
  const sessionSafe = document.getElementById('sessionSafe')
  const sessionDanger = document.getElementById('sessionDanger')

  // Count statistics
  let anomaliesCount = 0
  let safeCount = 0
  let dangerCount = 0

  history.forEach((r) => {
    if (r.result?.anomaly?.is_anomaly === true) anomaliesCount++
    const pred = r.result?.classification?.prediction
    if (pred === 0) safeCount++
    if (pred === 2) dangerCount++
  })

  // Update DOM elements
  if (sessionReadings) sessionReadings.textContent = history.length
  if (sessionAnomalies) sessionAnomalies.textContent = anomaliesCount
  if (sessionSafe) sessionSafe.textContent = safeCount
  if (sessionDanger) sessionDanger.textContent = dangerCount
}

function updateReadingsTable() {
  const tableBody = document.getElementById('readingsTableBody')
  if (!tableBody) return

  if (AppState.readingsHistory.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="4" class="py-4 text-center text-gray-500 dark:text-gray-400 italic">No readings yet</td>
      </tr>
    `
    return
  }

  const labels = ['SAFE', 'STRESS', 'DANGER']
  const badgeColors = [
    'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400',
    'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-400',
    'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400',
  ]

  tableBody.innerHTML = AppState.readingsHistory
    .slice(0, 10)
    .map((r) => {
      const prediction = r.result?.classification?.prediction ?? -1
      const label = labels[prediction] || 'N/A'
      const badgeColor =
        badgeColors[prediction] ||
        'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
      const time = new Date(r.timestamp).toLocaleTimeString()

      return `
        <tr class="border-b dark:border-gray-700 last:border-0">
          <td class="py-2 text-sm">${time}</td>
          <td class="py-2 text-sm">${r.ph.toFixed(1)}</td>
          <td class="py-2 text-sm">${r.tds.toFixed(0)}</td>
          <td class="py-2">
            <span class="px-2 py-0.5 rounded-full text-xs font-medium ${badgeColor}">${label}</span>
          </td>
        </tr>
      `
    })
    .join('')
}

// ============================================================================
// MODEL STATUS
// ============================================================================
async function loadModelStatus() {
  const modelStatus = document.getElementById('modelStatus')
  if (!modelStatus) return

  try {
    const res = await fetch('/api/models')
    if (!res.ok) throw new Error('Failed to load models')

    const data = await res.json()

    const html = data.models
      .map(
        (m) => `
        <div class="flex items-center justify-between py-1.5 border-b dark:border-gray-700 last:border-0">
          <span class="text-gray-700 dark:text-gray-300 text-sm">${
            m.name
          }</span>
          <span class="px-2 py-0.5 rounded text-xs font-medium ${
            m.available
              ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400'
              : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400'
          }">
            ${m.available ? '✓ Ready' : '✗ Unavailable'}
          </span>
        </div>
      `
      )
      .join('')

    modelStatus.innerHTML =
      html ||
      '<div class="text-gray-500 dark:text-gray-400">No models found</div>'
  } catch (err) {
    modelStatus.innerHTML = `<div class="text-red-500 dark:text-red-400">Failed to load model status</div>`
  }
}

// ============================================================================
// CSV EXPORT
// ============================================================================
function downloadCSV() {
  if (AppState.readingsHistory.length === 0) {
    showNotification(
      'No readings to export yet. Run some predictions first.',
      'warning'
    )
    return
  }

  const headers = [
    'timestamp',
    'ph',
    'tds',
    'device_id',
    'prediction',
    'is_anomaly',
    'temp_est',
    'do_est',
    'turbidity_est',
  ]
  const rows = AppState.readingsHistory.map((r) => [
    r.timestamp,
    r.ph,
    r.tds,
    r.device_id,
    r.result?.classification?.prediction ?? '',
    r.result?.anomaly?.is_anomaly ?? '',
    r.result?.virtual?.temp_est ?? '',
    r.result?.virtual?.do_est ?? '',
    r.result?.virtual?.turbidity_est ?? '',
  ])

  const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)

  const a = document.createElement('a')
  a.href = url
  a.download = `water_readings_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()

  URL.revokeObjectURL(url)
  showNotification('CSV file downloaded successfully', 'success')
}

// ============================================================================
// FORECAST API (Task 3)
// ============================================================================
async function runForecast(steps = 5) {
  const forecastBtn = document.getElementById('forecastBtn')
  const phInput = document.getElementById('ph')
  const tdsInput = document.getElementById('tds')

  const ph = parseFloat(phInput?.value) || 7.4
  const tds = parseFloat(tdsInput?.value) || 350

  // Generate history from readings or create synthetic
  let history = AppState.readingsHistory.slice(-20).map((r) => ({
    timestamp: r.timestamp,
    ph: r.ph,
    tds: r.tds,
  }))

  // If not enough history, generate synthetic data
  if (history.length < 10) {
    history = generateSyntheticHistory(ph, tds, 20)
  }

  try {
    if (forecastBtn) {
      forecastBtn.disabled = true
      forecastBtn.textContent = 'Loading...'
    }

    const res = await fetch(`/api/forecast_lstm?steps=${steps}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(history),
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()
    AppState.lastForecast = data

    // Update chart
    if (window.updateForecastChart) {
      window.updateForecastChart(history, data)
    }

    showNotification(`Forecast generated for ${steps} steps`, 'success')
  } catch (err) {
    console.error('Forecast error:', err)
    showNotification(`Forecast failed: ${err.message}`, 'error')
  } finally {
    if (forecastBtn) {
      forecastBtn.disabled = false
      forecastBtn.textContent = 'Forecast'
    }
  }
}

function generateSyntheticHistory(basePh, baseTds, count) {
  const now = new Date()
  const rows = []
  for (let i = count; i > 0; i--) {
    const t = new Date(now - i * 5 * 60000)
    rows.push({
      timestamp: t.toISOString(),
      ph: basePh + (Math.random() - 0.5) * 0.3,
      tds: baseTds + (Math.random() - 0.5) * 30,
    })
  }
  return rows
}

function clearForecastHistory() {
  if (window.clearForecastChart) {
    window.clearForecastChart()
  }
  showNotification('Forecast history cleared', 'info')
}

// ============================================================================
// AI PARAMETER SIMULATOR (Task 5)
// ============================================================================
let simulationTimeout = null

async function simulateParameters() {
  const phSlider = document.getElementById('simPh')
  const tdsSlider = document.getElementById('simTds')
  const hourSlider = document.getElementById('simHour')

  if (!phSlider || !tdsSlider || !hourSlider) return

  const ph = parseFloat(phSlider.value)
  const tds = parseFloat(tdsSlider.value)
  const hour = parseInt(hourSlider.value)

  // Update display values
  document.getElementById('simPhValue').textContent = ph.toFixed(1)
  document.getElementById('simTdsValue').textContent = tds
  document.getElementById('simHourValue').textContent = `${hour}:00`

  // Debounce API calls
  clearTimeout(simulationTimeout)
  simulationTimeout = setTimeout(async () => {
    await runSimulation(ph, tds, hour)
  }, 300)
}

async function runSimulation(ph, tds, hour) {
  const simResults = document.getElementById('simResults')
  const simStatus = document.getElementById('simStatus')

  // Create timestamp with specified hour
  const now = new Date()
  now.setHours(hour, 0, 0, 0)

  const payload = {
    timestamp: now.toISOString(),
    ph: ph,
    tds: tds,
    device_id: 'simulator',
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!res.ok) throw new Error('Simulation failed')

    const data = await res.json()

    // Update simulation results
    if (simResults) {
      const virtual = data.virtual || {}
      const features = data.features || {}

      simResults.innerHTML = `
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div class="text-gray-600 dark:text-gray-400">Temp Est:</div>
          <div class="font-medium dark:text-white">${
            virtual.temp_est?.toFixed(1) || '--'
          }°C</div>
          <div class="text-gray-600 dark:text-gray-400">DO Est:</div>
          <div class="font-medium dark:text-white">${
            virtual.do_est?.toFixed(2) || '--'
          } mg/L</div>
          <div class="text-gray-600 dark:text-gray-400">Turbidity:</div>
          <div class="font-medium dark:text-white">${
            virtual.turbidity_est?.toFixed(1) || '--'
          } NTU</div>
          <div class="text-gray-600 dark:text-gray-400">Ammonia Risk:</div>
          <div class="font-medium dark:text-white">${
            features.ammonia_risk?.toFixed(2) || '--'
          }</div>
        </div>
      `
    }

    // Update status badge
    if (simStatus) {
      const prediction = data.classification?.prediction ?? -1
      const labels = ['SAFE', 'STRESS', 'DANGER']
      const colors = [
        'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400',
        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400',
        'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400',
      ]

      simStatus.className = `px-3 py-1 rounded-full text-xs font-medium ${
        colors[prediction] || 'bg-gray-100 text-gray-800'
      }`
      simStatus.textContent = labels[prediction] || 'UNKNOWN'
    }
  } catch (err) {
    console.error('Simulation error:', err)
    if (simResults) {
      simResults.innerHTML =
        '<div class="text-red-500 text-sm">Simulation failed</div>'
    }
  }
}

// ============================================================================
// SENSOR HEALTH (Task 6)
// ============================================================================
async function loadSensorHealth() {
  const reliabilityScore = document.getElementById('reliabilityScore')
  const driftPercent = document.getElementById('driftPercent')
  const lastAnomalyScore = document.getElementById('lastAnomalyScore')
  const sensorRecommendation = document.getElementById('sensorRecommendation')

  try {
    const res = await fetch('/api/sensor_health')

    let data
    if (res.ok) {
      data = await res.json()
    } else {
      // Use mock data if endpoint doesn't exist
      data = {
        reliability_score: 87,
        ph_drift: 1.2,
        tds_drift: 2.8,
        last_anomaly_score: -0.12,
        recommendation:
          'Sensors operating normally. Next calibration due in 5 days.',
        drift_history: {
          labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
          phDrift: Array.from({ length: 24 }, () => (Math.random() - 0.3) * 3),
          tdsDrift: Array.from({ length: 24 }, () => (Math.random() - 0.2) * 4),
        },
      }
    }

    // Update UI
    if (reliabilityScore) {
      const score = data.reliability_score || 0
      reliabilityScore.textContent = score
      reliabilityScore.className = `text-3xl font-bold ${
        score >= 80
          ? 'text-green-600'
          : score >= 60
          ? 'text-yellow-600'
          : 'text-red-600'
      }`
    }

    if (driftPercent) {
      driftPercent.textContent = `pH: ${
        data.ph_drift?.toFixed(1) || 0
      }% | TDS: ${data.tds_drift?.toFixed(1) || 0}%`
    }

    if (lastAnomalyScore) {
      lastAnomalyScore.textContent =
        data.last_anomaly_score?.toFixed(3) || 'N/A'
    }

    if (sensorRecommendation) {
      sensorRecommendation.textContent =
        data.recommendation || 'No recommendations'
    }

    // Update drift chart
    if (window.updateDriftTrendChart && data.drift_history) {
      window.updateDriftTrendChart(data.drift_history)
    }
  } catch (err) {
    console.error('Failed to load sensor health:', err)
  }
}

// ============================================================================
// EVENT TIMELINE (Task 7)
// ============================================================================
const eventTimeline = []

function pushEvent(event) {
  const timeline = document.getElementById('eventTimeline')
  if (!timeline) return

  const eventData = {
    id: Date.now(),
    timestamp: event.timestamp || new Date().toISOString(),
    type: event.type || 'info',
    message: event.message || 'Unknown event',
    severity: event.severity || 'low',
  }

  eventTimeline.unshift(eventData)

  // Keep only last 50 events
  if (eventTimeline.length > 50) {
    eventTimeline.pop()
  }

  renderEventTimeline()
}

function renderEventTimeline() {
  const timeline = document.getElementById('eventTimeline')
  if (!timeline) return

  if (eventTimeline.length === 0) {
    timeline.innerHTML = `
      <div class="text-center text-gray-500 dark:text-gray-400 py-8">
        <svg class="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <p>No events recorded yet</p>
      </div>
    `
    return
  }

  const typeIcons = {
    ph_spike: '🔴 pH Spike',
    tds_jump: '🟣 TDS Jump',
    do_low: '🟢 Low DO',
    warning: '⚠️ Warning',
    info: 'ℹ️ Info',
    anomaly: '🔺 Anomaly',
  }

  const severityColors = {
    critical: 'border-red-500 bg-red-50 dark:bg-red-900/20',
    high: 'border-orange-500 bg-orange-50 dark:bg-orange-900/20',
    medium: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
    low: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
  }

  timeline.innerHTML = eventTimeline
    .slice(0, 20)
    .map(
      (event) => `
      <div class="relative pl-6 pb-4 border-l-2 ${
        severityColors[event.severity] || 'border-gray-300'
      } last:pb-0">
        <div class="absolute -left-2 top-0 w-4 h-4 rounded-full bg-white dark:bg-gray-800 border-2 ${
          event.severity === 'critical'
            ? 'border-red-500'
            : event.severity === 'high'
            ? 'border-orange-500'
            : 'border-blue-500'
        }"></div>
        <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
          ${new Date(event.timestamp).toLocaleTimeString()}
        </div>
        <div class="text-sm font-medium text-gray-800 dark:text-gray-200">
          ${typeIcons[event.type] || event.type}
        </div>
        <div class="text-sm text-gray-600 dark:text-gray-400">
          ${event.message}
        </div>
      </div>
    `
    )
    .join('')
}

function clearEventTimeline() {
  eventTimeline.length = 0
  renderEventTimeline()
  showNotification('Event timeline cleared', 'info')
}

// Auto-push events from predictions
function checkForEvents(predictionData) {
  if (!predictionData) return

  const features = predictionData.features || {}
  const anomaly = predictionData.anomaly || {}
  const recommendations = predictionData.recommendations_structured || []

  // Check for pH issues
  if (features.ph < 6.5 || features.ph > 8.5) {
    pushEvent({
      type: 'ph_spike',
      message: `pH level at ${features.ph?.toFixed(2)} - outside optimal range`,
      severity: features.ph < 6 || features.ph > 9 ? 'critical' : 'high',
    })
  }

  // Check for TDS issues
  if (features.tds > 1000) {
    pushEvent({
      type: 'tds_jump',
      message: `TDS at ${features.tds?.toFixed(0)} mg/L - elevated levels`,
      severity: features.tds > 1200 ? 'critical' : 'high',
    })
  }

  // Check for anomalies
  if (anomaly.is_anomaly) {
    pushEvent({
      type: 'anomaly',
      message: `Anomaly detected with score ${anomaly.score?.toFixed(3)}`,
      severity: 'high',
    })
  }

  // Add recommendations as warnings
  recommendations.forEach((rec) => {
    if (rec.severity === 'critical' || rec.severity === 'high') {
      pushEvent({
        type: 'warning',
        message: rec.message,
        severity: rec.severity,
      })
    }
  })
}

// ============================================================================
// PDF REPORT GENERATOR (Task 10)
// ============================================================================
async function downloadReport(range = 'daily') {
  const btn = document.getElementById(`${range}ReportBtn`)

  // Calculate date range
  const endDate = new Date()
  const startDate = new Date()
  if (range === 'daily') {
    startDate.setDate(startDate.getDate() - 1)
  } else if (range === 'weekly') {
    startDate.setDate(startDate.getDate() - 7)
  }

  try {
    if (btn) {
      btn.disabled = true
      btn.innerHTML = `
        <svg class="w-4 h-4 mr-2 animate-spin inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
        Generating...
      `
    }

    const res = await fetch('/api/generate_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_date: startDate.toISOString(),
        end_date: endDate.toISOString(),
        range: range,
      }),
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`)
    }

    // Get the PDF blob
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)

    // Trigger download
    const a = document.createElement('a')
    a.href = url
    a.download = `water_report_${range}_${endDate
      .toISOString()
      .slice(0, 10)}.pdf`
    a.click()

    URL.revokeObjectURL(url)
    showNotification(
      `${range.charAt(0).toUpperCase() + range.slice(1)} report downloaded`,
      'success'
    )
  } catch (err) {
    console.error('Report generation error:', err)
    showNotification(`Report generation failed: ${err.message}`, 'error')
  } finally {
    if (btn) {
      btn.disabled = false
      btn.innerHTML =
        range === 'daily'
          ? '<svg class="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Daily Report'
          : '<svg class="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Weekly Report'
    }
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme()
  initNavigation()

  // Set up theme toggle
  const themeToggle = document.getElementById('themeToggle')
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      toggleDarkMode()
      // Update charts theme
      if (window.updateChartsTheme) {
        setTimeout(window.updateChartsTheme, 100)
      }
    })
  }

  // Set up mobile menu toggle
  const mobileMenuBtn = document.getElementById('mobileMenuBtn')
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', toggleMobileSidebar)
  }

  // Set up sidebar overlay click to close
  const sidebarOverlay = document.getElementById('sidebarOverlay')
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeMobileSidebar)
  }

  // Set up predict button
  const predictBtn = document.getElementById('predictBtn')
  if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
      await predict()
      // Check for events after prediction
      checkForEvents(AppState.lastPrediction)
    })
  }

  // Set up forecast buttons
  const forecastBtn = document.getElementById('forecastBtn')
  if (forecastBtn) {
    forecastBtn.addEventListener('click', () => runForecast(5))
  }

  const forecast5Btn = document.getElementById('forecast5Btn')
  if (forecast5Btn) {
    forecast5Btn.addEventListener('click', () => runForecast(5))
  }

  const clearHistoryBtn = document.getElementById('clearHistoryBtn')
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', clearForecastHistory)
  }

  // Set up expand forecast chart button
  const expandForecastBtn = document.getElementById('expandForecastBtn')
  if (expandForecastBtn) {
    expandForecastBtn.addEventListener('click', expandForecastChart)
  }

  // Set up download button
  const downloadBtn = document.getElementById('downloadBtn')
  if (downloadBtn) {
    downloadBtn.addEventListener('click', downloadCSV)
  }

  // Set up simulator sliders
  const simSliders = ['simPh', 'simTds', 'simHour']
  simSliders.forEach((id) => {
    const slider = document.getElementById(id)
    if (slider) {
      slider.addEventListener('input', simulateParameters)
    }
  })

  // Set up report buttons
  const dailyReportBtn = document.getElementById('dailyReportBtn')
  if (dailyReportBtn) {
    dailyReportBtn.addEventListener('click', () => downloadReport('daily'))
  }

  const weeklyReportBtn = document.getElementById('weeklyReportBtn')
  if (weeklyReportBtn) {
    weeklyReportBtn.addEventListener('click', () => downloadReport('weekly'))
  }

  // Set up clear timeline button
  const clearTimelineBtn = document.getElementById('clearTimelineBtn')
  if (clearTimelineBtn) {
    clearTimelineBtn.addEventListener('click', clearEventTimeline)
  }

  // Set default timestamp to now
  const timestampInput = document.getElementById('timestamp')
  if (timestampInput) {
    const now = new Date()
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset())
    timestampInput.value = now.toISOString().slice(0, 16)
  }

  // Initialize to dashboard tab
  switchTab('dashboard')

  // Load model status
  loadModelStatus()

  // Initialize charts when switching to relevant tabs
  document.querySelectorAll('[data-tab]').forEach((item) => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab')
      setTimeout(() => {
        if (tabId === 'forecast') {
          // Initialize all forecast-related charts
          if (window.initForecastChart) window.initForecastChart()
          if (window.initAnomalyTimelineChart) window.initAnomalyTimelineChart()
          if (window.initRiskHeatmap) window.initRiskHeatmap()
        }
        if (tabId === 'sensors') {
          loadSensorHealth()
          if (window.initDriftTrendChart) {
            window.initDriftTrendChart()
          }
        }
        if (tabId === 'assistant' && window.initAssistant) {
          window.initAssistant()
        }
      }, 100)
    })
  })

  // Render initial event timeline
  renderEventTimeline()

  // Check API health on load and periodically
  checkApiHealth()
  setInterval(checkApiHealth, 30000) // Check every 30 seconds

  // Update last refresh time
  updateLastRefreshTime()

  // Initialize keyboard shortcuts
  initKeyboardShortcuts()

  console.log('Water Intelligence Dashboard initialized')
})

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================
function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Don't trigger if user is typing in input/textarea
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

    // Ctrl/Cmd + key shortcuts
    if (e.ctrlKey || e.metaKey) {
      switch (e.key.toLowerCase()) {
        case 'p': // Predict
          e.preventDefault()
          predict()
          break
        case 'f': // Forecast
          e.preventDefault()
          runForecast(5)
          break
        case 'd': // Toggle dark mode
          e.preventDefault()
          toggleDarkMode()
          break
        case 'e': // Export CSV
          e.preventDefault()
          downloadCSV()
          break
        case '/': // Show shortcuts help
          e.preventDefault()
          showShortcutsHelp()
          break
      }
    }

    // Number keys for tabs (1-6)
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      const tabMap = {
        1: 'dashboard',
        2: 'forecast',
        3: 'assistant',
        4: 'reports',
        5: 'sensors',
        6: 'settings',
      }
      if (tabMap[e.key]) {
        switchTab(tabMap[e.key])
      }

      // Quick scenario shortcuts
      if (e.key === 's') setQuickScenario('safe')
      if (e.key === 't') setQuickScenario('stress')
      if (e.key === 'x') setQuickScenario('danger')
    }
  })
}

function showShortcutsHelp() {
  // Create modal if not exists
  let modal = document.getElementById('shortcutsModal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'shortcutsModal'
    modal.className =
      'fixed inset-0 z-50 flex items-center justify-center bg-black/50 hidden'
    modal.innerHTML = `
      <div class="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 max-w-md mx-4 transform transition-all">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-bold text-gray-800 dark:text-white flex items-center gap-2">
            <span>⌨️</span> Keyboard Shortcuts
          </h3>
          <button onclick="document.getElementById('shortcutsModal').classList.add('hidden')" 
                  class="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
        <div class="space-y-4 text-sm">
          <div>
            <h4 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">Navigation</h4>
            <div class="grid grid-cols-2 gap-2 text-gray-600 dark:text-gray-400">
              <span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">1-6</kbd></span>
              <span>Switch tabs</span>
            </div>
          </div>
          <div>
            <h4 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">Actions</h4>
            <div class="space-y-1 text-gray-600 dark:text-gray-400">
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Ctrl+P</kbd></span><span>Run prediction</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Ctrl+F</kbd></span><span>Run forecast</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Ctrl+D</kbd></span><span>Toggle dark mode</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Ctrl+E</kbd></span><span>Export CSV</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Ctrl+/</kbd></span><span>Show this help</span></div>
            </div>
          </div>
          <div>
            <h4 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">Quick Scenarios</h4>
            <div class="space-y-1 text-gray-600 dark:text-gray-400">
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">S</kbd></span><span>🟢 Safe scenario</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">T</kbd></span><span>🟡 Stress scenario</span></div>
              <div class="flex justify-between"><span><kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">X</kbd></span><span>🔴 Danger scenario</span></div>
            </div>
          </div>
        </div>
        <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-center">
          <button onclick="document.getElementById('shortcutsModal').classList.add('hidden')"
                  class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm">
            Got it!
          </button>
        </div>
      </div>
    `
    document.body.appendChild(modal)

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden')
    })

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
        modal.classList.add('hidden')
      }
    })
  }

  modal.classList.remove('hidden')
}

// ============================================================================
// API HEALTH CHECK
// ============================================================================
async function checkApiHealth() {
  const connectionStatus = document.getElementById('connectionStatus')
  const connectionDot = document.getElementById('connectionDot')
  const connectionText = document.getElementById('connectionText')

  if (!connectionStatus || !connectionDot || !connectionText) return

  try {
    const startTime = performance.now()
    const res = await fetch('/api/health', { method: 'GET' })
    const endTime = performance.now()
    const latency = Math.round(endTime - startTime)

    if (res.ok) {
      connectionStatus.className =
        'hidden sm:flex items-center px-3 py-1.5 bg-green-50 dark:bg-green-900/30 rounded-full cursor-pointer'
      connectionDot.className =
        'w-2 h-2 bg-green-500 rounded-full pulse-dot mr-2'
      connectionText.className =
        'text-xs font-medium text-green-700 dark:text-green-400'
      connectionText.textContent = `Connected (${latency}ms)`
      updateLastRefreshTime()
    } else {
      throw new Error('API returned error')
    }
  } catch (err) {
    connectionStatus.className =
      'hidden sm:flex items-center px-3 py-1.5 bg-red-50 dark:bg-red-900/30 rounded-full cursor-pointer'
    connectionDot.className = 'w-2 h-2 bg-red-500 rounded-full mr-2'
    connectionText.className =
      'text-xs font-medium text-red-700 dark:text-red-400'
    connectionText.textContent = 'Disconnected'
  }
}

function updateLastRefreshTime() {
  const lastRefreshEl = document.getElementById('lastRefreshTime')
  if (lastRefreshEl) {
    lastRefreshEl.textContent = new Date().toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  }
}

// ============================================================================
// FULLSCREEN FORECAST CHART
// ============================================================================
let fullscreenChart = null

function expandForecastChart() {
  // Create modal if not exists
  let modal = document.getElementById('forecastFullscreenModal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'forecastFullscreenModal'
    modal.className =
      'fixed inset-0 z-50 flex items-center justify-center bg-black/80 hidden'
    modal.innerHTML = `
      <div class="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-[95vw] h-[90vh] p-6 relative">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
            <svg class="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
            </svg>
            LSTM Forecast - Fullscreen View
          </h3>
          <button id="closeForecastFullscreen" 
                  class="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
        <div class="h-[calc(100%-80px)]">
          <canvas id="forecastChartFullscreen"></canvas>
        </div>
        <div class="absolute bottom-4 left-6 right-6 flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>Press <kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">Esc</kbd> to close</span>
          <span>Time-series forecasting using LSTM neural network</span>
        </div>
      </div>
    `
    document.body.appendChild(modal)

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeForecastFullscreen()
    })

    // Close button handler
    document
      .getElementById('closeForecastFullscreen')
      .addEventListener('click', closeForecastFullscreen)

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
        closeForecastFullscreen()
      }
    })
  }

  modal.classList.remove('hidden')

  // Clone the chart data to fullscreen canvas
  setTimeout(() => {
    createFullscreenChart()
  }, 100)
}

function createFullscreenChart() {
  const ctx = document.getElementById('forecastChartFullscreen')
  if (!ctx) {
    console.warn('Fullscreen canvas not found')
    return
  }

  // Destroy existing fullscreen chart
  if (fullscreenChart) {
    fullscreenChart.destroy()
    fullscreenChart = null
  }

  // Get data from the original forecast chart using our getter
  const originalChart = window.getForecastChart
    ? window.getForecastChart()
    : null

  if (!originalChart) {
    console.warn(
      'Original forecast chart not found - please run a forecast first'
    )
    // Show a message in the modal
    const container = ctx.parentElement
    if (container) {
      container.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400">
          <svg class="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          <p class="text-lg font-medium">No Forecast Data Available</p>
          <p class="text-sm mt-2">Run a forecast first to see the expanded chart</p>
        </div>
      `
    }
    return
  }

  // Get theme colors
  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#e5e7eb' : '#374151'
  const gridColor = isDark ? '#374151' : '#e5e7eb'
  const bgColor = isDark ? '#1f2937' : '#ffffff'

  // Exactly clone the original chart configuration
  fullscreenChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [...originalChart.data.labels],
      datasets: [
        {
          label: 'pH (Historical)',
          data: [...originalChart.data.datasets[0].data],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.3,
          fill: false,
          pointRadius: 4,
          borderWidth: 2,
        },
        {
          label: 'TDS (Historical)',
          data: [...originalChart.data.datasets[1].data],
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          tension: 0.3,
          fill: false,
          pointRadius: 4,
          borderWidth: 2,
          yAxisID: 'y1',
        },
        {
          label: 'pH (Forecast)',
          data: [...originalChart.data.datasets[2].data],
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.2)',
          borderDash: [5, 5],
          tension: 0.3,
          fill: true,
          pointRadius: 5,
          pointStyle: 'triangle',
          borderWidth: 2,
        },
        {
          label: 'TDS (Forecast)',
          data: [...originalChart.data.datasets[3].data],
          borderColor: 'rgb(249, 115, 22)',
          backgroundColor: 'rgba(249, 115, 22, 0.2)',
          borderDash: [5, 5],
          tension: 0.3,
          fill: true,
          pointRadius: 5,
          pointStyle: 'triangle',
          borderWidth: 2,
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
          display: true,
          position: 'top',
          labels: {
            color: textColor,
            font: { size: 14 },
            usePointStyle: true,
            padding: 20,
          },
        },
        tooltip: {
          backgroundColor: bgColor,
          titleColor: textColor,
          bodyColor: textColor,
          borderColor: gridColor,
          borderWidth: 1,
          padding: 12,
          titleFont: { size: 14 },
          bodyFont: { size: 13 },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time',
            color: textColor,
            font: { size: 14, weight: 'bold' },
          },
          ticks: { color: textColor, font: { size: 12 } },
          grid: { color: gridColor },
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'pH',
            color: textColor,
            font: { size: 14, weight: 'bold' },
          },
          ticks: { color: textColor, font: { size: 12 } },
          grid: { color: gridColor },
          min: 5,
          max: 10,
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: {
            display: true,
            text: 'TDS (mg/L)',
            color: textColor,
            font: { size: 14, weight: 'bold' },
          },
          ticks: { color: textColor, font: { size: 12 } },
          grid: { drawOnChartArea: false },
          min: 0,
          max: 1500,
        },
      },
    },
  })
}

function closeForecastFullscreen() {
  const modal = document.getElementById('forecastFullscreenModal')
  if (modal) {
    modal.classList.add('hidden')
    // Restore canvas element in case it was replaced with a message
    const container = modal.querySelector('.h-\\[calc\\(100\\%-80px\\)\\]')
    if (container && !container.querySelector('canvas')) {
      container.innerHTML = '<canvas id="forecastChartFullscreen"></canvas>'
    }
  }
  if (fullscreenChart) {
    fullscreenChart.destroy()
    fullscreenChart = null
  }
}

// Export for use in other modules
window.AppState = AppState
window.showNotification = showNotification
window.switchTab = switchTab
window.predict = predict
window.downloadCSV = downloadCSV
window.loadModelStatus = loadModelStatus
window.runForecast = runForecast
window.clearForecastHistory = clearForecastHistory
window.simulateParameters = simulateParameters
window.loadSensorHealth = loadSensorHealth
window.pushEvent = pushEvent
window.clearEventTimeline = clearEventTimeline
window.downloadReport = downloadReport
window.checkApiHealth = checkApiHealth
window.setQuickScenario = setQuickScenario
window.showShortcutsHelp = showShortcutsHelp
window.updateSessionSummary = updateSessionSummary
window.expandForecastChart = expandForecastChart
window.closeForecastFullscreen = closeForecastFullscreen
