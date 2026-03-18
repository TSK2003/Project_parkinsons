document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analysisForm")
  if (!form) {
    return
  }

  const fileInput = document.getElementById("audioFile")
  const startButton = document.getElementById("recordStart")
  const stopButton = document.getElementById("recordStop")
  const analyzeButton = document.getElementById("analyzeBtn")
  const status = document.getElementById("analysisStatus")
  const timer = document.getElementById("recordTimer")
  const preview = document.getElementById("recordedPreview")

  const MAX_RECORDING_MS = 10000

  let stream = null
  let recorder = null
  let chunks = []
  let recordedBlob = null
  let previewUrl = null
  let timerInterval = null
  let autoStopTimer = null
  let startedAt = null

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

  function stopTracks() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      stream = null
    }
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

  async function toWav(blob) {
    const arrayBuffer = await blob.arrayBuffer()
    const context = new AudioContext()
    const decoded = await context.decodeAudioData(arrayBuffer)
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

    context.close()
    return new Blob([wavBuffer], { type: "audio/wav" })
  }

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (_error) {
      setStatus("Microphone access was blocked. Please allow microphone access and try again.", "error")
      return
    }

    chunks = []
    recordedBlob = null
    clearPreview()
    fileInput.value = ""

    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" })
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data)
      }
    }

    recorder.onstop = async () => {
      stopTracks()
      resetRecordingTimers()
      startButton.disabled = false
      stopButton.disabled = true

      if (!chunks.length) {
        setStatus("No audio was captured. Please try recording again.", "error")
        return
      }

      try {
        recordedBlob = await toWav(new Blob(chunks, { type: "audio/webm" }))
      } catch (_error) {
        recordedBlob = new Blob(chunks, { type: "audio/webm" })
      }

      previewUrl = URL.createObjectURL(recordedBlob)
      preview.src = previewUrl
      preview.hidden = false
      setStatus("Recording captured. You can now analyze and save this report.", "success")
    }

    recorder.start()
    startedAt = Date.now()
    updateTimer()
    timerInterval = window.setInterval(updateTimer, 200)
    autoStopTimer = window.setTimeout(stopRecording, MAX_RECORDING_MS)
    startButton.disabled = true
    stopButton.disabled = false
    setStatus("Recording in progress. Hold a steady 'Ahh' until the timer completes.", "working")
  }

  function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    } else {
      stopTracks()
      resetRecordingTimers()
      startButton.disabled = false
      stopButton.disabled = true
    }
  }

  startButton.addEventListener("click", startRecording)
  stopButton.addEventListener("click", stopRecording)

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      recordedBlob = null
      clearPreview()
      setStatus("Audio file selected. Click analyze to create a saved report.", "working")
    }
  })

  form.addEventListener("submit", async (event) => {
    event.preventDefault()

    const selectedFile = fileInput.files[0]
    if (!selectedFile && !recordedBlob) {
      setStatus("Please upload an audio file or record a voice sample first.", "error")
      return
    }

    const payload = new FormData()
    if (selectedFile) {
      payload.append("audio", selectedFile)
    } else {
      payload.append("audio", recordedBlob, "patient_recording.wav")
    }

    analyzeButton.disabled = true
    startButton.disabled = true
    stopButton.disabled = true
    setStatus("Analyzing your audio and saving the report. This will take a moment.", "working")

    try {
      const response = await fetch(form.dataset.endpoint, {
        method: "POST",
        body: payload,
      })
      const data = await response.json()

      if (!response.ok || data.error) {
        throw new Error(data.error || "Could not save the report.")
      }

      setStatus("Report saved successfully. Refreshing the dashboard with your new result.", "success")
      window.setTimeout(() => {
        window.location.reload()
      }, 900)
    } catch (error) {
      setStatus(error.message || "Could not analyze the audio sample.", "error")
      analyzeButton.disabled = false
      startButton.disabled = false
      stopButton.disabled = false
    }
  })

  window.addEventListener("beforeunload", () => {
    stopTracks()
    resetRecordingTimers()
    clearPreview()
  })
})
