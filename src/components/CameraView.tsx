import React, { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowDown, Camera, Film, Hand, Pause, Play, Sparkles, Upload, VideoOff, Wand2, X } from 'lucide-react'
import { GESTURE_PRESETS } from '../data'
import { BrowserKeypointExtractor } from '../keypointExtractor'
import { GesturePreset } from '../types'

interface CameraViewProps {
  isActive: boolean
  onToggleActive: () => void
  onTriggerTranslation: (text: string, category: string) => void
  onStreamStart: () => Promise<boolean>
  onKeypointFrame: (keypoints: number[]) => Promise<void>
  onStreamEnd: () => Promise<void>
  isPredicting?: boolean
}

type InputMode = 'camera' | 'video'
type CameraCaptureStatus = 'idle' | 'countdown' | 'recording'

const CAMERA_EXTRACTION_FPS = 30
const CAMERA_FRAME_INTERVAL_MS = 1000 / CAMERA_EXTRACTION_FPS
const UPLOAD_EXTRACTION_FPS = 30

export const CameraView: React.FC<CameraViewProps> = ({
  isActive,
  onToggleActive,
  onTriggerTranslation,
  onStreamStart,
  onKeypointFrame,
  onStreamEnd,
  isPredicting = false,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const extractorRef = useRef<BrowserKeypointExtractor | null>(null)
  const loopRef = useRef<number | null>(null)
  const lastFrameAtRef = useRef(0)
  const countdownTimerRef = useRef<number | null>(null)
  const pendingSpaceStartRef = useRef(false)
  const isSendingFrameRef = useRef(false)

  const [inputMode, setInputMode] = useState<InputMode>('camera')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraCaptureStatus, setCameraCaptureStatus] = useState<CameraCaptureStatus>('idle')
  const [countdown, setCountdown] = useState(0)
  const [activeGesture, setActiveGesture] = useState<GesturePreset | null>(null)
  const [trackingScore, setTrackingScore] = useState(0)
  const [sentFrames, setSentFrames] = useState(0)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)
  const [isVideoPaused, setIsVideoPaused] = useState(false)
  const [videoDuration, setVideoDuration] = useState(0)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)

  const stopFrameLoop = () => {
    if (loopRef.current !== null) cancelAnimationFrame(loopRef.current)
    loopRef.current = null
    isSendingFrameRef.current = false
  }

  const stopCountdown = () => {
    if (countdownTimerRef.current !== null) window.clearInterval(countdownTimerRef.current)
    countdownTimerRef.current = null
    setCountdown(0)
    setCameraCaptureStatus((status) => (status === 'countdown' ? 'idle' : status))
  }

  const stopCameraStream = () => {
    stopFrameLoop()
    stopCountdown()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraReady(false)
    setCameraCaptureStatus('idle')
  }

  const ensureExtractor = async () => {
    if (!extractorRef.current) {
      extractorRef.current = new BrowserKeypointExtractor()
      await extractorRef.current.init()
    }
    return extractorRef.current
  }

  const sendCurrentVideoFrame = async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return false

    const extractor = await ensureExtractor()
    const keypoints = await extractor.extract(video)
    if (!keypoints) return false

    await onKeypointFrame(keypoints)
    setSentFrames((count) => count + 1)
    return true
  }

  const startCameraLoop = () => {
    stopFrameLoop()
    lastFrameAtRef.current = 0
    const tick = async (time: number) => {
      if (time - lastFrameAtRef.current >= CAMERA_FRAME_INTERVAL_MS && !isSendingFrameRef.current) {
        lastFrameAtRef.current = time
        isSendingFrameRef.current = true
        sendCurrentVideoFrame().catch((err) => {
          setCameraError(err instanceof Error ? err.message : '키포인트 추출에 실패했습니다.')
        }).finally(() => {
          isSendingFrameRef.current = false
        })
      }
      loopRef.current = requestAnimationFrame(tick)
    }
    loopRef.current = requestAnimationFrame(tick)
  }

  const startCameraCaptureAfterCountdown = () => {
    if (cameraCaptureStatus !== 'idle') return
    setCameraError(null)
    setCountdown(3)
    setCameraCaptureStatus('countdown')

    countdownTimerRef.current = window.setInterval(() => {
      setCountdown((value) => {
        if (value > 1) return value - 1

        if (countdownTimerRef.current !== null) window.clearInterval(countdownTimerRef.current)
        countdownTimerRef.current = null

        onStreamStart()
          .then((started) => {
            if (!started) {
              setCameraCaptureStatus('idle')
              return
            }
            setSentFrames(0)
            setCameraCaptureStatus('recording')
            startCameraLoop()
          })
          .catch((err) => {
            setCameraError(err instanceof Error ? err.message : '실시간 번역 세션을 시작하지 못했습니다.')
            setCameraCaptureStatus('idle')
          })

        return 0
      })
    }, 1000)
  }

  const finishCameraCapture = async () => {
    stopCountdown()
    stopFrameLoop()
    if (cameraCaptureStatus === 'recording') {
      setCameraCaptureStatus('idle')
      await onStreamEnd()
      stopCameraStream()
      onToggleActive()
    }
  }

  const handleSpaceCaptureControl = () => {
    if (inputMode !== 'camera' || isPredicting) return

    if (!isActive) {
      pendingSpaceStartRef.current = true
      onToggleActive()
      return
    }

    if (cameraCaptureStatus === 'recording') {
      finishCameraCapture().catch(console.error)
      return
    }

    if (cameraCaptureStatus === 'countdown') {
      stopCountdown()
      return
    }

    if (cameraReady) startCameraCaptureAfterCountdown()
    else pendingSpaceStartRef.current = true
  }

  const seekVideo = (video: HTMLVideoElement, time: number) => new Promise<void>((resolve, reject) => {
    const duration = video.duration || 0
    const targetTime = Math.min(Math.max(time, 0), Math.max(duration - 0.001, 0))

    if (Math.abs(video.currentTime - targetTime) < 0.001 && video.readyState >= 2) {
      resolve()
      return
    }

    const handleSeeked = () => {
      cleanup()
      resolve()
    }
    const handleError = () => {
      cleanup()
      reject(new Error('동영상 프레임 이동에 실패했습니다.'))
    }
    const cleanup = () => {
      window.clearTimeout(timeout)
      video.removeEventListener('seeked', handleSeeked)
      video.removeEventListener('error', handleError)
    }
    const timeout = window.setTimeout(() => {
      cleanup()
      reject(new Error('동영상 프레임 이동 시간이 초과되었습니다.'))
    }, 5000)

    video.addEventListener('seeked', handleSeeked, { once: true })
    video.addEventListener('error', handleError, { once: true })
    video.currentTime = targetTime
  })

  useEffect(() => {
    if (inputMode !== 'camera') return

    if (!isActive) {
      stopCameraStream()
      return
    }

    let cancelled = false
    async function startCamera() {
      try {
        setSentFrames(0)
        setCameraError(null)
        setCameraReady(false)
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        await ensureExtractor()
        setCameraReady(true)
        if (pendingSpaceStartRef.current && !cancelled) {
          pendingSpaceStartRef.current = false
          startCameraCaptureAfterCountdown()
        }
      } catch (err) {
        setCameraError(err instanceof Error ? err.message : '카메라를 연결할 수 없습니다.')
      }
    }

    startCamera()
    return () => {
      cancelled = true
      stopCameraStream()
    }
  }, [isActive, inputMode])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target as HTMLElement | null
      const tagName = target?.tagName?.toLowerCase()
      if (tagName === 'input' || tagName === 'textarea' || target?.isContentEditable) return

      event.preventDefault()
      handleSpaceCaptureControl()
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [inputMode, isActive, cameraReady, cameraCaptureStatus, isPredicting])

  useEffect(() => {
    if (inputMode !== 'video' || !videoRef.current || !videoObjectUrl) return

    stopCameraStream()
    videoRef.current.srcObject = null
    videoRef.current.src = videoObjectUrl
    videoRef.current.load()
    videoRef.current.play().catch(() => setIsVideoPaused(true))
  }, [videoObjectUrl, inputMode])

  useEffect(() => {
    const video = videoRef.current
    if (!video || inputMode !== 'video') return

    const onTimeUpdate = () => setVideoCurrentTime(video.currentTime)
    const onDurationChange = () => setVideoDuration(video.duration || 0)
    const onPause = () => setIsVideoPaused(true)
    const onPlay = () => setIsVideoPaused(false)
    video.addEventListener('timeupdate', onTimeUpdate)
    video.addEventListener('durationchange', onDurationChange)
    video.addEventListener('pause', onPause)
    video.addEventListener('play', onPlay)
    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate)
      video.removeEventListener('durationchange', onDurationChange)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('play', onPlay)
    }
  }, [inputMode])

  useEffect(() => {
    if (!isActive || inputMode !== 'camera') return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    let animationId = 0
    let frame = 0
    const render = () => {
      frame += 1
      const width = (canvas.width = canvas.parentElement?.clientWidth || 640)
      const height = (canvas.height = canvas.parentElement?.clientHeight || 400)
      ctx.clearRect(0, 0, width, height)
      ctx.strokeStyle = 'rgba(15, 194, 158, 0.15)'
      ctx.beginPath()
      ctx.moveTo(0, (frame * 3) % height)
      ctx.lineTo(width, (frame * 3) % height)
      ctx.stroke()
      ctx.fillStyle = 'rgba(15, 194, 158, 0.7)'
      ctx.font = '10px monospace'
      ctx.fillText(
        `${cameraCaptureStatus === 'recording' ? 'REC 30FPS' : 'READY'} | FRAMES: ${sentFrames} | LATENCY: local keypoints`,
        16,
        height - 20,
      )
      setTrackingScore(Math.min(99, Math.round(88 + sentFrames / 3)))
      animationId = requestAnimationFrame(render)
    }
    render()
    return () => cancelAnimationFrame(animationId)
  }, [cameraCaptureStatus, isActive, inputMode, sentFrames])

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl)
    setVideoFile(file)
    setVideoObjectUrl(URL.createObjectURL(file))
    setSentFrames(0)
    setIsVideoPaused(false)
    setInputMode('video')
    if (!isActive) onToggleActive()
  }

  const handleClearVideo = () => {
    videoRef.current?.pause()
    if (videoRef.current) videoRef.current.src = ''
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl)
    setVideoFile(null)
    setVideoObjectUrl(null)
    setIsVideoPaused(false)
    setInputMode('camera')
    if (isActive) onToggleActive()
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const toggleVideoPlayback = () => {
    const video = videoRef.current
    if (!video) return
    if (isVideoPaused) video.play().catch(console.error)
    else video.pause()
  }

  const handleCameraStopAndPredict = async () => {
    if (cameraCaptureStatus === 'recording') {
      await finishCameraCapture()
      return
    }
    stopCameraStream()
    onToggleActive()
  }

  const handlePredictUploadedVideo = async () => {
    const video = videoRef.current
    if (!video || !Number.isFinite(video.duration) || video.duration <= 0) return

    setCameraError(null)
    setSentFrames(0)
    const started = await onStreamStart()
    if (!started) return

    video.pause()
    setIsVideoPaused(true)

    try {
      const frameCount = Math.max(1, Math.floor(video.duration * UPLOAD_EXTRACTION_FPS))
      for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
        await seekVideo(video, frameIndex / UPLOAD_EXTRACTION_FPS)
        setVideoCurrentTime(video.currentTime)
        // eslint-disable-next-line no-await-in-loop
        await sendCurrentVideoFrame()
      }
      await onStreamEnd()
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : '동영상 예측에 실패했습니다.')
    }
  }

  const handlePresetSelect = (preset: GesturePreset) => {
    setActiveGesture(preset)
    onTriggerTranslation(preset.translationText, preset.category)
    setTimeout(() => setActiveGesture(null), 1500)
  }

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const rest = Math.floor(seconds % 60)
    return `${minutes}:${rest.toString().padStart(2, '0')}`
  }

  const isVideoMode = inputMode === 'video' && !!videoFile

  return (
    <div className="w-full">
      <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={handleFileChange} />

      {!isActive && (
        <div className="flex items-center gap-2 mb-3">
          <button
            onClick={() => setInputMode('camera')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-bold border transition-all ${
              inputMode === 'camera'
                ? 'bg-brand-green/10 border-brand-green text-brand-green'
                : 'bg-white border-neutral-200 text-neutral-500 hover:border-neutral-300'
            }`}
          >
            <Camera className="w-4 h-4" />
            실시간 카메라
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-bold border transition-all ${
              inputMode === 'video'
                ? 'bg-brand-green/10 border-brand-green text-brand-green'
                : 'bg-white border-neutral-200 text-neutral-500 hover:border-neutral-300'
            }`}
          >
            <Upload className="w-4 h-4" />
            동영상 업로드
          </button>
        </div>
      )}

      <div
        className="relative w-full aspect-video min-h-[280px] sm:min-h-[420px] bg-brand-emerald rounded-2xl flex flex-col items-center justify-center overflow-hidden border border-emerald-950/20 shadow-inner group cursor-pointer"
        onClick={() => {
          if (!isActive && !isVideoMode) onToggleActive()
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
            isActive || isVideoMode ? 'opacity-100' : 'hidden'
          }`}
        />
        {isActive && inputMode === 'camera' && (
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />
        )}
        {(!isActive || isVideoMode) && (
          <>
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/10 pointer-events-none z-10" />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(15,194,158,0.1)_0%,rgba(0,0,0,0.4)_100%)] pointer-events-none z-10" />
          </>
        )}

        <AnimatePresence mode="wait">
          {!isActive && !isVideoMode && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center justify-center p-6 text-center select-none z-20"
            >
              <motion.div
                whileHover={{ scale: 1.08 }}
                className="w-20 h-20 rounded-full bg-white/10 hover:bg-white/15 text-white flex items-center justify-center border border-white/20 shadow-lg cursor-pointer transition-all duration-300"
              >
                <Hand className="w-10 h-10 text-white -rotate-12 animate-bounce" />
              </motion.div>
              <div className="mt-6 flex flex-col items-center gap-1.5">
                <div className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-white/90">
                  <Camera className="w-5 h-5 animate-pulse" />
                </div>
                <ArrowDown className="w-5 h-5 text-brand-green mt-1" />
              </div>
              <h3 className="mt-5 text-xl font-bold text-white tracking-wide">화면을 눌러 수어 번역 시작하기</h3>
              <p className="text-sm text-neutral-300 max-w-sm mt-2 font-medium">
                카메라를 통해 환자의 수어 동작을 인식하고 실시간으로 번역합니다.
              </p>  
              {cameraError && <p className="mt-3 text-xs text-red-200">{cameraError}</p>}
            </motion.div>
          )}

          {isActive && inputMode === 'camera' && (
            <motion.div
              key="camera-hud"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none"
            >
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
                <span className="text-xs font-bold text-white tracking-wider ml-0.5">
                  {cameraCaptureStatus === 'recording'
                    ? 'LIVE 30fps 전송 중'
                    : cameraCaptureStatus === 'countdown'
                      ? `${countdown}초 후 촬영 시작`
                      : '스페이스바로 촬영 시작'}
                </span>
              </div>
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <Sparkles className="w-4 h-4 text-brand-green animate-pulse" />
                <span className="text-xs font-bold text-white tracking-wider">프레임: {sentFrames}</span>
              </div>
            </motion.div>
          )}

          {isActive && inputMode === 'camera' && cameraCaptureStatus === 'countdown' && (
            <motion.div
              key="countdown"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none"
            >
              <div className="w-28 h-28 rounded-full bg-black/50 border border-white/20 backdrop-blur-md flex items-center justify-center text-5xl font-black text-white">
                {countdown}
              </div>
              <p className="mt-4 text-sm font-bold text-white/90">촬영 준비 중입니다</p>
            </motion.div>
          )}

          {isVideoMode && (
            <motion.div
              key="video-hud"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none"
            >
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10 max-w-[60%]">
                <Film className="w-3.5 h-3.5 text-brand-green shrink-0" />
                <span className="text-xs font-bold text-white truncate">{videoFile.name}</span>
              </div>
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <span className="text-xs font-bold text-white tabular-nums">
                  {formatTime(videoCurrentTime)} / {formatTime(videoDuration)}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {isActive && inputMode === 'camera' && (
          <div className="absolute bottom-4 right-4 z-20">
            <button
              onClick={(event) => {
                event.stopPropagation()
                handleCameraStopAndPredict().catch(console.error)
              }}
              disabled={isPredicting}
              className="px-3.5 py-2 bg-black/60 hover:bg-black/80 disabled:opacity-60 backdrop-blur-md text-white rounded-lg border border-white/10 text-sm font-semibold flex items-center gap-1.5 transition-colors"
            >
              <VideoOff className="w-4 h-4 text-red-400" />
              {isPredicting
                ? '예측 중...'
                : cameraCaptureStatus === 'recording'
                  ? '촬영 종료 / 예측'
                  : '카메라 끄기'}
            </button>
          </div>
        )}

        {isVideoMode && (
          <div className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/80 to-transparent px-4 pb-4 pt-8">
            <input
              type="range"
              min={0}
              max={videoDuration || 0}
              step={0.1}
              value={videoCurrentTime}
              onChange={(event) => {
                if (videoRef.current) videoRef.current.currentTime = Number(event.target.value)
              }}
              onClick={(event) => event.stopPropagation()}
              className="w-full h-1 accent-brand-green cursor-pointer mb-3"
            />
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    toggleVideoPlayback()
                  }}
                  className="w-8 h-8 rounded-full bg-white/15 hover:bg-white/25 flex items-center justify-center transition-colors"
                >
                  {isVideoPaused ? <Play className="w-4 h-4 text-white ml-0.5" /> : <Pause className="w-4 h-4 text-white" />}
                </button>
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    fileInputRef.current?.click()
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-bold transition-colors"
                >
                  <Upload className="w-3.5 h-3.5" />
                  다른 영상
                </button>
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    handlePredictUploadedVideo().catch(console.error)
                  }}
                  disabled={isPredicting}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-brand-green hover:bg-brand-green/90 disabled:opacity-60 text-white rounded-lg text-xs font-bold transition-colors"
                >
                  <Wand2 className="w-3.5 h-3.5" />
                  {isPredicting ? '예측 중...' : 'AI 예측'}
                </button>
              </div>
              <button
                onClick={(event) => {
                  event.stopPropagation()
                  handleClearVideo()
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg text-xs font-bold transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                제거
              </button>
            </div>
          </div>
        )}
      </div>

      {(isActive || isVideoMode) && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-4 bg-white border border-neutral-100 rounded-xl p-4 shadow-xs">
          <div className="flex items-center justify-between border-b pb-2 mb-3">
            <span className="text-sm font-bold text-on-surface flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-secondary" />
              임상 수어 시뮬레이터
            </span>
            <span className="text-xs text-on-surface-variant font-medium bg-neutral-100 px-2 py-0.5 rounded-sm">테스트용 즉시 번역</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {GESTURE_PRESETS.map((preset) => {
              const active = activeGesture?.id === preset.id
              return (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset)}
                  className={`p-2 rounded-lg border text-left flex flex-col transition-all duration-200 ${
                    active
                      ? 'border-brand-green bg-brand-green/10 ring-1 ring-brand-green'
                      : 'border-neutral-200 bg-neutral-50 hover:bg-neutral-100 hover:border-neutral-300'
                  }`}
                >
                  <span className="text-xs font-bold text-on-surface truncate">{preset.category} 수어</span>
                  <span className="text-xs text-neutral-500 line-clamp-1 mt-0.5 font-normal">{preset.gestureName}</span>
                </button>
              )
            })}
          </div>
        </motion.div>
      )}
    </div>
  )
}
