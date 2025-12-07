/**
 * Water Intelligence Dashboard - AI Assistant Module
 * GenAI chat interface with voice support and LangGraph conversation history
 */

// ============================================================================
// ASSISTANT STATE
// ============================================================================
const AssistantState = {
  isListening: false,
  isSpeaking: false,
  recognition: null,
  synthesis: window.speechSynthesis,
  chatHistory: [],
  sessionId: null, // Session ID for LangGraph persistence
}

// Generate or retrieve session ID
function getSessionId() {
  if (!AssistantState.sessionId) {
    // Try to get from localStorage or generate new one
    AssistantState.sessionId = localStorage.getItem('aquabot_session_id')
    if (!AssistantState.sessionId) {
      AssistantState.sessionId = `session-${Date.now()}-${Math.random()
        .toString(36)
        .substr(2, 9)}`
      localStorage.setItem('aquabot_session_id', AssistantState.sessionId)
    }
  }
  return AssistantState.sessionId
}

// Start a new conversation session
function startNewSession() {
  AssistantState.sessionId = `session-${Date.now()}-${Math.random()
    .toString(36)
    .substr(2, 9)}`
  localStorage.setItem('aquabot_session_id', AssistantState.sessionId)
  AssistantState.chatHistory = []
  clearChat()
  addMessage(
    'assistant',
    'Started a new conversation. How can I help you with your water quality monitoring today?'
  )
}

// ============================================================================
// CHAT FUNCTIONS (Task 8)
// ============================================================================
async function sendMessage(message = null) {
  const input = document.getElementById('chatInput')
  const sendBtn = document.getElementById('sendMessageBtn')
  const query = message || input?.value?.trim()

  if (!query) return

  // Clear input
  if (input) input.value = ''

  // Add user message to chat
  addMessage('user', query)

  // Build context from AppState
  const context = {
    last_prediction: window.AppState?.lastPrediction || null,
    last_forecast: window.AppState?.lastForecast || null,
    readings_count: window.AppState?.readingsHistory?.length || 0,
  }

  // Show loading
  const loadingId = addMessage('assistant', '', true)

  try {
    if (sendBtn) {
      sendBtn.disabled = true
      sendBtn.innerHTML = `
        <svg class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
      `
    }

    const res = await fetch('/api/generative_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: getSessionId(),
        context,
      }),
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()
    const reply =
      data.reply ||
      data.response ||
      data.message ||
      'I apologize, but I could not generate a response.'

    // Update session ID if server provided a new one
    if (data.session_id) {
      AssistantState.sessionId = data.session_id
      localStorage.setItem('aquabot_session_id', data.session_id)
    }

    // Remove loading and add response
    removeMessage(loadingId)
    addMessage('assistant', reply, false, data.source || 'unknown')

    // Speak the response if TTS is enabled
    if (document.getElementById('ttsToggle')?.checked) {
      speak(reply)
    }

    // Save to history
    AssistantState.chatHistory.push({ role: 'user', content: query })
    AssistantState.chatHistory.push({ role: 'assistant', content: reply })
  } catch (err) {
    console.error('Chat error:', err)
    removeMessage(loadingId)
    addMessage(
      'assistant',
      `Sorry, I encountered an error: ${err.message}. Please try again.`
    )
  } finally {
    if (sendBtn) {
      sendBtn.disabled = false
      sendBtn.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
        </svg>
      `
    }
  }
}

function addMessage(role, content, isLoading = false, source = null) {
  const container = document.getElementById('chatMessages')
  if (!container) return null

  const messageId = `msg-${Date.now()}`
  const isUser = role === 'user'

  const messageEl = document.createElement('div')
  messageEl.id = messageId
  messageEl.className = `flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`

  if (isLoading) {
    messageEl.innerHTML = `
      <div class="max-w-[80%] px-4 py-3 rounded-2xl bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <div class="flex items-center space-x-2">
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
        </div>
      </div>
    `
  } else {
    const bgClass = isUser
      ? 'bg-blue-600 text-white'
      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
    const roundedClass = isUser ? 'rounded-br-md' : 'rounded-bl-md'

    // Show source indicator for assistant messages
    const sourceIndicator =
      !isUser && source
        ? `<span class="text-xs opacity-50 mt-1 block">${
            source === 'langgraph' ? '🤖 AI' : '📋 Rule-based'
          }</span>`
        : ''

    // Use markdown rendering for assistant messages, escape HTML for user messages
    const renderedContent = isUser
      ? escapeHtml(content)
      : renderMarkdown(content)

    messageEl.innerHTML = `
      <div class="max-w-[80%] px-4 py-3 rounded-2xl ${bgClass} ${roundedClass}">
        <div class="text-sm prose prose-sm dark:prose-invert max-w-none">${renderedContent}</div>
        ${sourceIndicator}
      </div>
    `
  }

  container.appendChild(messageEl)
  container.scrollTop = container.scrollHeight

  return messageId
}

function removeMessage(messageId) {
  const el = document.getElementById(messageId)
  if (el) el.remove()
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

/**
 * Convert basic markdown to HTML for chat messages
 * Supports: **bold**, *italic*, `code`, ```code blocks```, - lists, numbered lists, newlines
 */
function renderMarkdown(text) {
  if (!text) return ''

  // First escape HTML to prevent XSS
  let html = escapeHtml(text)

  // Code blocks (``` ... ```) - must be before inline code
  html = html.replace(
    /```([\s\S]*?)```/g,
    '<pre class="bg-gray-200 dark:bg-gray-600 rounded px-2 py-1 my-1 text-xs overflow-x-auto"><code>$1</code></pre>'
  )

  // Inline code (`code`)
  html = html.replace(
    /`([^`]+)`/g,
    '<code class="bg-gray-200 dark:bg-gray-600 rounded px-1 text-xs">$1</code>'
  )

  // Bold (**text** or __text__)
  html = html.replace(
    /\*\*([^*]+)\*\*/g,
    '<strong class="font-semibold">$1</strong>'
  )
  html = html.replace(
    /__([^_]+)__/g,
    '<strong class="font-semibold">$1</strong>'
  )

  // Italic (*text* or _text_) - be careful not to match inside words
  html = html.replace(/(?<![*\w])\*([^*]+)\*(?![*\w])/g, '<em>$1</em>')
  html = html.replace(/(?<![_\w])_([^_]+)_(?![_\w])/g, '<em>$1</em>')

  // Process lines to handle lists properly
  const lines = html.split('\n')
  let result = []
  let inOrderedList = false
  let inUnorderedList = false
  let listCounter = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const numberedMatch = line.match(/^\d+\.\s+(.+)$/)
    const bulletMatch = line.match(/^[\-•\*]\s+(.+)$/)

    if (numberedMatch) {
      if (!inOrderedList) {
        if (inUnorderedList) {
          result.push('</ul>')
          inUnorderedList = false
        }
        result.push('<ol class="list-decimal ml-5 my-1">')
        inOrderedList = true
        listCounter = 0
      }
      listCounter++
      result.push(`<li class="py-0.5">${numberedMatch[1]}</li>`)
    } else if (bulletMatch) {
      if (!inUnorderedList) {
        if (inOrderedList) {
          result.push('</ol>')
          inOrderedList = false
        }
        result.push('<ul class="list-disc ml-5 my-1">')
        inUnorderedList = true
      }
      result.push(`<li class="py-0.5">${bulletMatch[1]}</li>`)
    } else {
      // Close any open lists
      if (inOrderedList) {
        result.push('</ol>')
        inOrderedList = false
      }
      if (inUnorderedList) {
        result.push('</ul>')
        inUnorderedList = false
      }
      result.push(line)
    }
  }

  // Close any remaining open lists
  if (inOrderedList) result.push('</ol>')
  if (inUnorderedList) result.push('</ul>')

  html = result.join('\n')

  // Headers (## Header)
  html = html.replace(
    /^###\s+(.+)$/gm,
    '<h4 class="font-semibold text-sm mt-1">$1</h4>'
  )
  html = html.replace(/^##\s+(.+)$/gm, '<h3 class="font-semibold mt-1">$1</h3>')
  html = html.replace(/^#\s+(.+)$/gm, '<h2 class="font-bold mt-1">$1</h2>')

  // Line breaks - convert double newlines to single break
  html = html.replace(/\n\n+/g, '<br>')

  // Single newlines to <br> but skip if inside list tags
  html = html.replace(/\n(?![<])/g, '<br>')

  // Remove excessive <br> tags
  html = html.replace(/(<br\s*\/?>){2,}/gi, '<br>')

  // Remove <br> right after opening list tags or before closing
  html = html.replace(/<(ul|ol)[^>]*><br>/gi, '<$1>')
  html = html.replace(/<br><\/(ul|ol)>/gi, '</$1>')
  html = html.replace(/<\/li><br><li/gi, '</li><li')

  return html
}

function clearChat() {
  const container = document.getElementById('chatMessages')
  if (container) {
    container.innerHTML = `
      <div class="flex justify-start mb-3">
        <div class="max-w-[80%] px-4 py-3 rounded-2xl rounded-bl-md bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <p class="text-sm">Hello! I'm your AI water quality assistant. Ask me anything about your sensor data, predictions, or water management best practices.</p>
        </div>
      </div>
    `
  }
  AssistantState.chatHistory = []
}

async function explainPrediction() {
  const lastPred = window.AppState?.lastPrediction

  if (!lastPred) {
    window.showNotification?.(
      'No prediction to explain. Run a prediction first.',
      'warning'
    )
    return
  }

  const classification =
    ['SAFE', 'STRESS', 'DANGER'][lastPred.classification?.prediction] ||
    'Unknown'
  const query = `Explain this water quality prediction in simple terms:
- Classification: ${classification}
- pH: ${lastPred.features?.ph?.toFixed(2) || 'N/A'}
- TDS: ${lastPred.features?.tds?.toFixed(0) || 'N/A'} mg/L
- Estimated Temperature: ${lastPred.virtual?.temp_est?.toFixed(1) || 'N/A'}°C
- Estimated DO: ${lastPred.virtual?.do_est?.toFixed(2) || 'N/A'} mg/L
- Anomaly Detected: ${lastPred.anomaly?.is_anomaly ? 'Yes' : 'No'}
- Recommendations: ${lastPred.recommendations?.join(', ') || 'None'}

What does this mean for my fish, and what should I do?`

  // Switch to assistant tab
  window.switchTab?.('assistant')

  await sendMessage(query)
}

// ============================================================================
// VOICE FUNCTIONS (Task 9)
// ============================================================================
function initVoiceAssistant() {
  // Check for browser support
  if (
    !('webkitSpeechRecognition' in window) &&
    !('SpeechRecognition' in window)
  ) {
    console.warn('Speech recognition not supported')
    const micBtn = document.getElementById('micButton')
    if (micBtn) {
      micBtn.disabled = true
      micBtn.title = 'Speech recognition not supported in this browser'
    }
    return
  }

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition
  AssistantState.recognition = new SpeechRecognition()
  AssistantState.recognition.continuous = false
  AssistantState.recognition.interimResults = true
  AssistantState.recognition.lang = 'en-US'

  AssistantState.recognition.onstart = () => {
    AssistantState.isListening = true
    updateMicButton(true)
  }

  AssistantState.recognition.onend = () => {
    AssistantState.isListening = false
    updateMicButton(false)
  }

  AssistantState.recognition.onresult = (event) => {
    const input = document.getElementById('chatInput')
    if (!input) return

    let finalTranscript = ''
    let interimTranscript = ''

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript
      if (event.results[i].isFinal) {
        finalTranscript += transcript
      } else {
        interimTranscript += transcript
      }
    }

    input.value = finalTranscript || interimTranscript

    // Auto-send on final result
    if (finalTranscript) {
      setTimeout(() => sendMessage(), 500)
    }
  }

  AssistantState.recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error)
    AssistantState.isListening = false
    updateMicButton(false)

    if (event.error === 'not-allowed') {
      window.showNotification?.(
        'Microphone access denied. Please allow microphone access.',
        'error'
      )
    }
  }
}

function toggleListening() {
  if (!AssistantState.recognition) {
    window.showNotification?.('Speech recognition not available', 'error')
    return
  }

  if (AssistantState.isListening) {
    AssistantState.recognition.stop()
  } else {
    try {
      AssistantState.recognition.start()
    } catch (e) {
      console.error('Failed to start recognition:', e)
    }
  }
}

function updateMicButton(isActive) {
  const micBtn = document.getElementById('micButton')
  const micIcon = document.getElementById('micIcon')
  const waveIndicator = document.getElementById('waveIndicator')

  if (micBtn) {
    micBtn.classList.toggle('bg-red-500', isActive)
    micBtn.classList.toggle('bg-gray-200', !isActive)
    micBtn.classList.toggle('dark:bg-gray-600', !isActive)
  }

  if (micIcon) {
    micIcon.classList.toggle('text-white', isActive)
    micIcon.classList.toggle('text-gray-600', !isActive)
    micIcon.classList.toggle('dark:text-gray-300', !isActive)
  }

  if (waveIndicator) {
    waveIndicator.classList.toggle('hidden', !isActive)
  }
}

function speak(text) {
  if (!AssistantState.synthesis) {
    console.warn('Speech synthesis not supported')
    return
  }

  // Cancel any ongoing speech
  AssistantState.synthesis.cancel()

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 1.0
  utterance.pitch = 1.0
  utterance.volume = 1.0

  // Try to use a natural-sounding voice
  const voices = AssistantState.synthesis.getVoices()
  const preferredVoice = voices.find(
    (v) =>
      v.name.includes('Google') ||
      v.name.includes('Natural') ||
      v.name.includes('Samantha')
  )
  if (preferredVoice) {
    utterance.voice = preferredVoice
  }

  utterance.onstart = () => {
    AssistantState.isSpeaking = true
  }

  utterance.onend = () => {
    AssistantState.isSpeaking = false
  }

  AssistantState.synthesis.speak(utterance)
}

function stopSpeaking() {
  if (AssistantState.synthesis) {
    AssistantState.synthesis.cancel()
    AssistantState.isSpeaking = false
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================
function initAssistant() {
  initVoiceAssistant()

  // Set up event listeners
  const chatInput = document.getElementById('chatInput')
  const sendBtn = document.getElementById('sendMessageBtn')
  const micBtn = document.getElementById('micButton')
  const clearBtn = document.getElementById('clearChatBtn')

  if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        sendMessage()
      }
    })
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', () => sendMessage())
  }

  if (micBtn) {
    micBtn.addEventListener('click', toggleListening)
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', clearChat)
  }

  // Load voices (needed for some browsers)
  if (AssistantState.synthesis) {
    AssistantState.synthesis.getVoices()
    speechSynthesis.onvoiceschanged = () => {
      AssistantState.synthesis.getVoices()
    }
  }
}

// Export functions
window.sendMessage = sendMessage
window.addMessage = addMessage
window.clearChat = clearChat
window.explainPrediction = explainPrediction
window.toggleListening = toggleListening
window.speak = speak
window.stopSpeaking = stopSpeaking
window.initAssistant = initAssistant
