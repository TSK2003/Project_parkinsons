document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analysisForm")
  if (!form) {
    return
  }

  const confirmForms = Array.from(document.querySelectorAll(".report-confirm-form"))
  const fileInput = document.getElementById("audioFile")
  const startButton = document.getElementById("recordStart")
  const stopButton = document.getElementById("recordStop")
  const analyzeButton = document.getElementById("analyzeBtn")
  const status = document.getElementById("analysisStatus")
  const timer = document.getElementById("recordTimer")
  const preview = document.getElementById("recordedPreview")
  const waveformCanvas = document.getElementById("waveformCanvas")
  const waveformContext = waveformCanvas ? waveformCanvas.getContext("2d") : null

  const TARGET_SAMPLE_RATE = 44100
  const MAX_RECORDING_MS = 10000
  const HARD_MIN_DURATION_SECONDS = 2
  const RECOMMENDED_DURATION_SECONDS = 8
  const MIN_CLIENT_RMS = 0.003
  const MAX_CLIENT_CLIPPED_FRACTION = 0.01
  const SUPPORTED_FILE_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".webm"]

  let stream = null
  let recorder = null
  let chunks = []
  let recordedBlob = null
  let previewUrl = null
  let timerInterval = null
  let autoStopTimer = null
  let startedAt = null
  let waveformAudioContext = null
  let waveformSource = null
  let waveformAnalyser = null
  let waveformData = null
  let waveformFrameId = null

  function setStatus(message, tone = "") {
    status.textContent = message
    status.className = "inline-status"
    if (tone) {
      status.classList.add(`is-${tone}`)
    }
  }

  function clearPreview() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      previewUrl = null
    }
    preview.hidden = true
    preview.removeAttribute("src")
  }

  function renderWaveformSamples(channelData, label = "Recorded waveform") {
    if (!waveformContext || !waveformCanvas || !channelData?.length) {
      return
    }

    const { width, height } = waveformCanvas
    waveformContext.clearRect(0, 0, width, height)

    const gradient = waveformContext.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, "rgba(255, 255, 255, 0.96)")
    gradient.addColorStop(1, "rgba(255, 247, 237, 0.94)")
    waveformContext.fillStyle = gradient
    waveformContext.fillRect(0, 0, width, height)

    waveformContext.strokeStyle = "rgba(18, 32, 41, 0.08)"
    waveformContext.lineWidth = 1
    waveformContext.beginPath()
    waveformContext.moveTo(0, height / 2)
    waveformContext.lineTo(width, height / 2)
    waveformContext.stroke()

    const step = Math.max(1, Math.floor(channelData.length / width))
    waveformContext.strokeStyle = "#14746f"
    waveformContext.lineWidth = 2

    for (let x = 0; x < width; x += 1) {
      const start = x * step
      const end = Math.min(start + step, channelData.length)
      let minValue = 1
      let maxValue = -1

      for (let index = start; index < end; index += 1) {
        const sample = channelData[index]
        if (sample < minValue) {
          minValue = sample
        }
        if (sample > maxValue) {
          maxValue = sample
        }
      }

      const y1 = ((1 + minValue) * height) / 2
      const y2 = ((1 + maxValue) * height) / 2
      waveformContext.beginPath()
      waveformContext.moveTo(x, y1)
      waveformContext.lineTo(x, y2)
      waveformContext.stroke()
    }

    waveformContext.fillStyle = "rgba(91, 107, 114, 0.88)"
    waveformContext.font = "13px 'Plus Jakarta Sans', sans-serif"
    waveformContext.textAlign = "right"
    waveformContext.fillText(label, width - 14, 24)
  }

  async function renderWaveformFromBlob(blob, label) {
    try {
      const decoded = await decodeAudioBlob(blob)
      renderWaveformSamples(decoded.getChannelData(0), label)
    } catch (_error) {
      drawWaveformPlaceholder("Wave signal preview is not available for this audio.")
    }
  }

  function stopTracks() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      stream = null
    }
  }

  function drawWaveformPlaceholder(label = "Wave signal will appear here while recording.") {
    if (!waveformContext || !waveformCanvas) {
      return
    }

    const { width, height } = waveformCanvas
    waveformContext.clearRect(0, 0, width, height)

    const gradient = waveformContext.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, "rgba(20, 116, 111, 0.16)")
    gradient.addColorStop(1, "rgba(245, 158, 11, 0.12)")
    waveformContext.fillStyle = gradient
    waveformContext.fillRect(0, 0, width, height)

    waveformContext.strokeStyle = "rgba(18, 32, 41, 0.18)"
    waveformContext.lineWidth = 2
    waveformContext.beginPath()
    waveformContext.moveTo(0, height / 2)
    waveformContext.lineTo(width, height / 2)
    waveformContext.stroke()

    waveformContext.fillStyle = "rgba(91, 107, 114, 0.9)"
    waveformContext.font = "14px 'Plus Jakarta Sans', sans-serif"
    waveformContext.textAlign = "center"
    waveformContext.fillText(label, width / 2, height / 2 - 16)
  }

  function stopWaveform() {
    if (waveformFrameId) {
      window.cancelAnimationFrame(waveformFrameId)
      waveformFrameId = null
    }

    if (waveformSource) {
      waveformSource.disconnect()
      waveformSource = null
    }

    if (waveformAnalyser) {
      waveformAnalyser.disconnect()
      waveformAnalyser = null
    }

    waveformData = null

    if (waveformAudioContext) {
      waveformAudioContext.close().catch(() => {})
      waveformAudioContext = null
    }

    drawWaveformPlaceholder()
  }

  function drawWaveformFrame() {
    if (!waveformContext || !waveformCanvas || !waveformAnalyser || !waveformData) {
      return
    }

    const { width, height } = waveformCanvas
    waveformAnalyser.getByteTimeDomainData(waveformData)

    waveformContext.clearRect(0, 0, width, height)
    waveformContext.fillStyle = "rgba(255, 247, 237, 0.9)"
    waveformContext.fillRect(0, 0, width, height)

    waveformContext.strokeStyle = "rgba(18, 32, 41, 0.08)"
    waveformContext.lineWidth = 1
    waveformContext.beginPath()
    waveformContext.moveTo(0, height / 2)
    waveformContext.lineTo(width, height / 2)
    waveformContext.stroke()

    waveformContext.lineWidth = 3
    waveformContext.strokeStyle = "#14746f"
    waveformContext.beginPath()

    const sliceWidth = width / waveformData.length
    let x = 0

    for (let index = 0; index < waveformData.length; index += 1) {
      const value = waveformData[index] / 128
      const y = (value * height) / 2

      if (index === 0) {
        waveformContext.moveTo(x, y)
      } else {
        waveformContext.lineTo(x, y)
      }
      x += sliceWidth
    }

    waveformContext.lineTo(width, height / 2)
    waveformContext.stroke()

    waveformContext.fillStyle = "rgba(91, 107, 114, 0.85)"
    waveformContext.font = "13px 'Plus Jakarta Sans', sans-serif"
    waveformContext.textAlign = "right"
    waveformContext.fillText("Live wave signal", width - 14, 24)

    waveformFrameId = window.requestAnimationFrame(drawWaveformFrame)
  }

  async function startWaveform(activeStream) {
    if (!waveformCanvas) {
      return
    }

    stopWaveform()

    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) {
      drawWaveformPlaceholder("Wave signal preview is not supported in this browser.")
      return
    }

    waveformAudioContext = new AudioContextClass()
    waveformSource = waveformAudioContext.createMediaStreamSource(activeStream)
    waveformAnalyser = waveformAudioContext.createAnalyser()
    waveformAnalyser.fftSize = 2048
    waveformAnalyser.smoothingTimeConstant = 0.85
    waveformData = new Uint8Array(waveformAnalyser.fftSize)
    waveformSource.connect(waveformAnalyser)

    if (waveformAudioContext.state === "suspended") {
      await waveformAudioContext.resume()
    }

    drawWaveformFrame()
  }

  function resetRecordingTimers() {
    window.clearInterval(timerInterval)
    window.clearTimeout(autoStopTimer)
    timerInterval = null
    autoStopTimer = null
    timer.textContent = "00:00"
  }

  function updateTimer() {
    const elapsedMs = Date.now() - startedAt
    const elapsedSeconds = Math.min(Math.floor(elapsedMs / 1000), 99)
    const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0")
    const seconds = String(elapsedSeconds % 60).padStart(2, "0")
    timer.textContent = `${minutes}:${seconds}`
  }

  function getAudioContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    return AudioContextClass ? new AudioContextClass() : null
  }

  function getPreferredMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
    ]
    for (const candidate of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidate)) {
        return candidate
      }
    }
    return ""
  }

  function isSupportedAudioFile(file) {
    const lowerName = (file.name || "").toLowerCase()
    return SUPPORTED_FILE_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
  }

  async function decodeAudioBlob(blob) {
    const context = getAudioContext()
    if (!context) {
      throw new Error("Audio decoding is not supported in this browser.")
    }

    try {
      const arrayBuffer = await blob.arrayBuffer()
      return await context.decodeAudioData(arrayBuffer.slice(0))
    } finally {
      await context.close()
    }
  }

  async function toWav(blob) {
    const decoded = await decodeAudioBlob(blob)
    const channelData = decoded.getChannelData(0)
    const sampleRate = decoded.sampleRate

    const wavBuffer = new ArrayBuffer(44 + channelData.length * 2)
    const view = new DataView(wavBuffer)

    const writeString = (offset, value) => {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index))
      }
    }

    writeString(0, "RIFF")
    view.setUint32(4, 36 + channelData.length * 2, true)
    writeString(8, "WAVE")
    writeString(12, "fmt ")
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(36, "data")
    view.setUint32(40, channelData.length * 2, true)

    let offset = 44
    for (let index = 0; index < channelData.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[index]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
      offset += 2
    }

    return new Blob([wavBuffer], { type: "audio/wav" })
  }

  async function inspectAudioBlob(blob) {
    const decoded = await decodeAudioBlob(blob)
    const channelData = decoded.getChannelData(0)

    let sumSquares = 0
    let peak = 0
    let clippedSamples = 0
    for (let index = 0; index < channelData.length; index += 1) {
      const sample = channelData[index]
      const absSample = Math.abs(sample)
      sumSquares += sample * sample
      peak = Math.max(peak, absSample)
      if (absSample >= 0.98) {
        clippedSamples += 1
      }
    }

    return {
      duration: decoded.duration,
      sampleRate: decoded.sampleRate,
      rms: channelData.length ? Math.sqrt(sumSquares / channelData.length) : 0,
      peak,
      clippedFraction: channelData.length ? clippedSamples / channelData.length : 0,
    }
  }

  function evaluateAudioMetrics(metrics) {
    if (metrics.duration < HARD_MIN_DURATION_SECONDS) {
      return {
        blocking: true,
        tone: "error",
        message: "Please capture at least 2 seconds of a steady 'Ahh' so the server has enough signal to analyze.",
      }
    }

    if (metrics.rms < MIN_CLIENT_RMS) {
      return {
        blocking: true,
        tone: "error",
        message: "The recording is too quiet. Move closer to the microphone and speak a little louder.",
      }
    }

    if (metrics.clippedFraction > MAX_CLIENT_CLIPPED_FRACTION) {
      return {
        blocking: true,
        tone: "error",
        message: "The recording sounds clipped or distorted. Lower the microphone gain and record again.",
      }
    }

    if (metrics.duration < RECOMMENDED_DURATION_SECONDS) {
      return {
        blocking: false,
        tone: "warning",
        message: "This recording is shorter than the recommended 8 to 10 seconds. The server can still analyze it, but a longer sample usually improves reliability.",
      }
    }

    if (Math.round(metrics.sampleRate) !== TARGET_SAMPLE_RATE) {
      return {
        blocking: false,
        tone: "warning",
        message: "The sample rate is not exactly 44.1 kHz, but the server will resample it. You can still analyze this recording.",
      }
    }

    return {
      blocking: false,
      tone: "success",
      message: "Recording quality looks acceptable. You can now analyze and save this report.",
    }
  }

  async function inspectCurrentAudio(sourceBlob) {
    try {
      const metrics = await inspectAudioBlob(sourceBlob)
      return evaluateAudioMetrics(metrics)
    } catch (_error) {
      return {
        blocking: false,
        tone: "working",
        message: "Audio captured. The server will run the final quality checks during analysis.",
      }
    }
  }

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
    } catch (_error) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      } catch (_fallbackError) {
        setStatus("Microphone access was blocked. Please allow microphone access and try again.", "error")
        return
      }
    }

    await startWaveform(stream)
    chunks = []
    recordedBlob = null
    clearPreview()
    fileInput.value = ""

    const mimeType = getPreferredMimeType()
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data)
      }
    }

    recorder.onstop = async () => {
      stopWaveform()
      stopTracks()
      resetRecordingTimers()
      startButton.disabled = false
      stopButton.disabled = true

      if (!chunks.length) {
        setStatus("No audio was captured. Please try recording again.", "error")
        return
      }

      try {
        const rawBlob = new Blob(chunks, { type: mimeType || "audio/webm" })
        recordedBlob = await toWav(rawBlob)
        const qualityCheck = await inspectCurrentAudio(recordedBlob)
        await renderWaveformFromBlob(recordedBlob, "Recorded waveform")
        previewUrl = URL.createObjectURL(recordedBlob)
        preview.src = previewUrl
        preview.hidden = false
        setStatus(qualityCheck.message, qualityCheck.tone)
      } catch (_error) {
        recordedBlob = new Blob(chunks, { type: mimeType || "audio/webm" })
        await renderWaveformFromBlob(recordedBlob, "Recorded waveform")
        previewUrl = URL.createObjectURL(recordedBlob)
        preview.src = previewUrl
        preview.hidden = false
        setStatus("Recording captured. The server will run the final quality checks during analysis.", "working")
      }
    }

    recorder.start()
    startedAt = Date.now()
    updateTimer()
    timerInterval = window.setInterval(updateTimer, 200)
    autoStopTimer = window.setTimeout(stopRecording, MAX_RECORDING_MS)
    startButton.disabled = true
    stopButton.disabled = false
    setStatus("Recording in progress. Browser audio enhancements are disabled for cleaner capture, so hold a steady 'Ahh' until the timer completes.", "working")
  }

  function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    } else {
      stopWaveform()
      stopTracks()
      resetRecordingTimers()
      startButton.disabled = false
      stopButton.disabled = true
    }
  }

  startButton.addEventListener("click", startRecording)
  stopButton.addEventListener("click", stopRecording)

  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      if (!isSupportedAudioFile(fileInput.files[0])) {
        fileInput.value = ""
        setStatus("Please choose a supported audio file: WAV, MP3, M4A, AAC, OGG, FLAC, MP4, or WebM.", "error")
        return
      }
      recordedBlob = null
      clearPreview()
      setStatus("Checking the selected audio file before upload.", "working")
      await renderWaveformFromBlob(fileInput.files[0], "Selected file waveform")
      const qualityCheck = await inspectCurrentAudio(fileInput.files[0])
      setStatus(qualityCheck.message, qualityCheck.tone)
    }
  })

  form.addEventListener("submit", async (event) => {
    event.preventDefault()

    const selectedFile = fileInput.files[0]
    if (!selectedFile && !recordedBlob) {
      setStatus("Please upload an audio file or record a voice sample first.", "error")
      return
    }
    if (selectedFile && !isSupportedAudioFile(selectedFile)) {
      setStatus("Please upload a supported audio file: WAV, MP3, M4A, AAC, OGG, FLAC, MP4, or WebM.", "error")
      return
    }

    const sourceBlob = selectedFile || recordedBlob
    const qualityCheck = await inspectCurrentAudio(sourceBlob)
    if (qualityCheck.blocking) {
      setStatus(qualityCheck.message, qualityCheck.tone)
      return
    }

    const payload = new FormData()
    if (selectedFile) {
      payload.append("audio", selectedFile)
    } else {
      payload.append("audio", recordedBlob, "patient_recording.wav")
    }
    const consentCheckbox = form.querySelector('input[name="consent_training"]')
    if (consentCheckbox?.checked) {
      payload.append("consent_training", "1")
    }

    analyzeButton.disabled = true
    startButton.disabled = true
    stopButton.disabled = true
    setStatus("Analyzing the audio and saving the patient report. This will take a moment.", "working")

    try {
      const response = await fetch(form.dataset.endpoint, {
        method: "POST",
        body: payload,
      })
      const data = await response.json()

      if (!response.ok || data.error) {
        throw new Error(data.error || "Could not save the report.")
      }

      const report = data.report || {}
      if (report.needs_retake) {
        setStatus("Report saved, but a repeat recording is recommended because the sample quality or confidence was weak.", "warning")
      } else {
        setStatus("Report saved successfully. Refreshing the page with the new result.", "success")
      }
      window.setTimeout(() => {
        window.location.reload()
      }, 1200)
    } catch (error) {
      setStatus(error.message || "Could not analyze the audio sample.", "error")
      analyzeButton.disabled = false
      startButton.disabled = false
      stopButton.disabled = false
    }
  })

  confirmForms.forEach((confirmForm) => {
    const statusNode = confirmForm.querySelector(".report-confirm-status")
    confirmForm.addEventListener("submit", async (event) => {
      event.preventDefault()

      const select = confirmForm.querySelector('select[name="clinician_confirmed_label"]')
      if (!select) {
        return
      }

      if (statusNode) {
        statusNode.textContent = "Saving clinician confirmation."
        statusNode.className = "inline-status report-confirm-status is-working"
      }

      try {
        const response = await fetch(confirmForm.dataset.endpoint, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            clinician_confirmed_label: select.value,
          }),
        })
        const data = await response.json()
        if (!response.ok || data.error) {
          throw new Error(data.error || "Could not save clinician confirmation.")
        }

        if (statusNode) {
          statusNode.textContent = "Clinician confirmation saved."
          statusNode.className = "inline-status report-confirm-status is-success"
        }

        window.setTimeout(() => {
          window.location.reload()
        }, 800)
      } catch (error) {
        if (statusNode) {
          statusNode.textContent = error.message || "Could not save clinician confirmation."
          statusNode.className = "inline-status report-confirm-status is-error"
        }
      }
    })
  })

  window.addEventListener("beforeunload", () => {
    stopWaveform()
    stopTracks()
    resetRecordingTimers()
    clearPreview()
  })

  drawWaveformPlaceholder()
})
