import React, { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  Camera,
  Film,
  Hand,
  Pause,
  Play,
  Sparkles,
  Upload,
  VideoOff,
  Wand2,
  X,
  ArrowDown,
} from 'lucide-react'
import { GESTURE_PRESETS } from '../data'
import { BrowserKeypointExtractor } from '../keypointExtractor'
import { GesturePreset } from '../types'

interface CameraViewProps {
  isActive: boolean
  onToggleActive: () => void
  onTriggerTranslation: (text: string, category: string) => void
  onStreamStart: () => Promise<boolean>
  onKeypointFrame: (keypoints: number[], frameId: number) => Promise<void>
  onWordBoundary: () => Promise<void>
  onStreamEnd: () => Promise<void>
  isPredicting?: boolean
}

type InputMode = 'camera' | 'video'
type CameraCaptureStatus = 'idle' | 'recording'

const CAMERA_EXTRACTION_FPS = 30
const CAMERA_FRAME_INTERVAL_MS = 1000 / CAMERA_EXTRACTION_FPS
const UPLOAD_EXTRACTION_FPS = 30

const CALIBRATION_FRAMES = 20
const START_THRESHOLD_MULTIPLIER = 6
const END_THRESHOLD_MULTIPLIER = 2.5
const MIN_START_THRESHOLD = 0.0015
const MIN_END_THRESHOLD = 0.0006

const MOTION_START_FRAMES = 3
const WORD_END_FRAMES = 8
const SENTENCE_END_FRAMES = 30
const POST_END_COOLDOWN_FRAMES = 8

function computeMotionScore(
  prev: number[] | null,
  curr: number[],
): number {
  if (!prev || prev.length !== curr.length || curr.length === 0) {
    return 0
  }

  let sum = 0

  for (let i = 0; i < curr.length; i += 1) {
    sum += Math.abs(curr[i] - prev[i])
  }

  return sum / curr.length
}

export const CameraView: React.FC<CameraViewProps> = ({
  isActive,
  onToggleActive,
  onTriggerTranslation,
  onStreamStart,
  onKeypointFrame,
  onWordBoundary,
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
  const isSendingFrameRef = useRef(false)

  // ---- 리렌더링 방지용 Ref ----
  const sentFramesRef = useRef(0)

  // ---- 모션 게이트 상태 ----
  const motionActiveRef = useRef(false)
  const startCountRef = useRef(0)
  const endCountRef = useRef(0)
  const cooldownRef = useRef(0)
  const wordBoundarySentRef = useRef(false)
  const debugFrameCounterRef = useRef(0)

  // 실제 카메라에서 추출한 프레임 번호.
  // 서버로 전송하지 않는 정지 프레임도 계속 증가한다.
  const cameraFrameIdRef = useRef(0)

  // ---- 자동 캘리브레이션 상태 ----
  const calibrationSamplesRef = useRef<number[]>([])
  const motionStartThresholdRef = useRef(MIN_START_THRESHOLD)
  const motionEndThresholdRef = useRef(MIN_END_THRESHOLD)
  const isCalibratedRef = useRef(false)
  const prevKeypointsRef = useRef<number[] | null>(null)
  const streamStartingRef = useRef(false)

  // ---- 스트림 종료 중복 방지 ----
  const streamEndingRef = useRef(false)
  const awaitingResultRef = useRef(false)

  const [inputMode, setInputMode] = useState<InputMode>('camera')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [cameraCaptureStatus, setCameraCaptureStatus] =
    useState<CameraCaptureStatus>('idle')

  const [activeGesture, setActiveGesture] =
    useState<GesturePreset | null>(null)

  const [sentFramesDisplay, setSentFramesDisplay] = useState(0)

  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoObjectUrl, setVideoObjectUrl] =
    useState<string | null>(null)

  const [isVideoPaused, setIsVideoPaused] = useState(false)
  const [videoDuration, setVideoDuration] = useState(0)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)

  // ------------------------------------------------------------
  // 모션 게이트 초기화
  // ------------------------------------------------------------
  const resetMotionGate = () => {
    motionActiveRef.current = false
    startCountRef.current = 0
    endCountRef.current = 0
    cooldownRef.current = 0
    wordBoundarySentRef.current = false

    calibrationSamplesRef.current = []

    motionStartThresholdRef.current = MIN_START_THRESHOLD
    motionEndThresholdRef.current = MIN_END_THRESHOLD

    isCalibratedRef.current = false
    prevKeypointsRef.current = null
    streamStartingRef.current = false

    debugFrameCounterRef.current = 0
    cameraFrameIdRef.current = 0
    sentFramesRef.current = 0
  }

  // ------------------------------------------------------------
  // 프레임 루프 종료
  // ------------------------------------------------------------
  const stopFrameLoop = () => {
    if (loopRef.current !== null) {
      cancelAnimationFrame(loopRef.current)
    }

    loopRef.current = null
    isSendingFrameRef.current = false
  }

  // ------------------------------------------------------------
  // 스트리밍 상태 전체 초기화
  // ------------------------------------------------------------
  const resetStreamingState = () => {
    resetMotionGate()
    streamEndingRef.current = false
    awaitingResultRef.current = false
    setSentFramesDisplay(0)
    setCameraCaptureStatus('idle')
    setCameraError(null)
  }

  // ------------------------------------------------------------
  // AI 스트림 종료
  //
  // 이미 종료된 상태에서 다시 onStreamEnd()를 호출하지 않도록
  // 방어 로직을 추가한다.
  // ------------------------------------------------------------
  const endSignStream = async () => {
    if (streamEndingRef.current) {
      return
    }

    if (
      !motionActiveRef.current &&
      !streamStartingRef.current
    ) {
      return
    }

    streamEndingRef.current = true
    awaitingResultRef.current = true

    try {
      await onStreamEnd()
    } catch (error) {
      awaitingResultRef.current = false
      console.error('[CAMERA] stream end error:', error)
    } finally {
      streamEndingRef.current = false
      motionActiveRef.current = false
      streamStartingRef.current = false
      wordBoundarySentRef.current = false
      endCountRef.current = 0
      setCameraCaptureStatus('idle')
    }
  }

  // ------------------------------------------------------------
  // 카메라 종료
  // ------------------------------------------------------------
  const handleCameraOff = async () => {
    stopFrameLoop()

    if (
      motionActiveRef.current ||
      streamStartingRef.current
    ) {
      await endSignStream()
    }

    stopCameraStream()
    resetStreamingState()
    onToggleActive()
  }

  // ------------------------------------------------------------
  // 카메라 MediaStream 종료
  // ------------------------------------------------------------
  const stopCameraStream = () => {
    stopFrameLoop()

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop())

      streamRef.current = null
    }

    if (
      videoRef.current &&
      inputMode === 'camera'
    ) {
      videoRef.current.srcObject = null
    }

    resetMotionGate()
    setCameraCaptureStatus('idle')
  }

  // ------------------------------------------------------------
  // MediaPipe / Keypoint Extractor 초기화
  // ------------------------------------------------------------
  const ensureExtractor = async () => {
    if (!extractorRef.current) {
      extractorRef.current =
        new BrowserKeypointExtractor()

      await extractorRef.current.init()
    }

    return extractorRef.current
  }

  // ------------------------------------------------------------
  // 현재 video에서 keypoint 추출
  // ------------------------------------------------------------
  const extractCurrentVideoKeypoints =
    async (): Promise<number[] | null> => {
      const video = videoRef.current

      if (!video || video.readyState < 2) {
        return null
      }

      const extractor = await ensureExtractor()

      return await extractor.extract(video)
    }

  // ------------------------------------------------------------
  // 업로드 영상 처리
  // ------------------------------------------------------------
  const handleVideoFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      setCameraError('동영상 파일만 선택할 수 있습니다.')
      return
    }

    if (videoObjectUrl) {
      URL.revokeObjectURL(videoObjectUrl)
    }

    const objectUrl = URL.createObjectURL(file)

    setVideoFile(file)
    setVideoObjectUrl(objectUrl)
    setInputMode('video')
    setIsVideoPaused(false)
    setVideoDuration(0)
    setVideoCurrentTime(0)
    setCameraError(null)
  }

  // ------------------------------------------------------------
  // 업로드 영상 종료
  // ------------------------------------------------------------
  const clearVideoFile = () => {
    if (videoObjectUrl) {
      URL.revokeObjectURL(videoObjectUrl)
    }

    setVideoFile(null)
    setVideoObjectUrl(null)
    setIsVideoPaused(false)
    setVideoDuration(0)
    setVideoCurrentTime(0)
    setInputMode('camera')
  }

  // ------------------------------------------------------------
  // 영상 재생 / 일시정지
  // ------------------------------------------------------------
  const toggleVideoPlayback = async () => {
    const video = videoRef.current

    if (!video || inputMode !== 'video') {
      return
    }

    try {
      if (video.paused) {
        await video.play()
        setIsVideoPaused(false)
      } else {
        video.pause()
        setIsVideoPaused(true)
      }
    } catch (error) {
      console.error(
        '[VIDEO] playback error:',
        error,
      )
    }
  }

  // ------------------------------------------------------------
  // 카메라 모션 감지 루프
  // ------------------------------------------------------------
  const startWatchLoop = () => {
    stopFrameLoop()

    lastFrameAtRef.current = 0

    resetMotionGate()

    setCameraCaptureStatus('idle')

    const tick = async (time: number) => {
      if (
        time - lastFrameAtRef.current >=
        CAMERA_FRAME_INTERVAL_MS
      ) {
        if (!isSendingFrameRef.current) {
          lastFrameAtRef.current = time
          isSendingFrameRef.current = true

          try {
            const keypoints =
              await extractCurrentVideoKeypoints()

            if (keypoints) {
              // 실제 카메라 프레임 번호.
              // 전송 여부와 관계없이 증가한다.
              const cameraFrameId =
                cameraFrameIdRef.current

              cameraFrameIdRef.current += 1

              const score = computeMotionScore(
                prevKeypointsRef.current,
                keypoints,
              )

              prevKeypointsRef.current = keypoints

              // ------------------------------------------------
              // 자동 캘리브레이션
              // ------------------------------------------------
              if (!isCalibratedRef.current) {
                if (
                  calibrationSamplesRef.current
                    .length < CALIBRATION_FRAMES
                ) {
                  calibrationSamplesRef.current.push(
                    score,
                  )
                }

                if (
                  calibrationSamplesRef.current
                    .length >= CALIBRATION_FRAMES
                ) {
                  const samples = [
                    ...calibrationSamplesRef.current,
                  ].sort((a, b) => a - b)

                  const percentileIndex =
                    Math.floor(samples.length * 0.2)

                  const baseline =
                    samples[percentileIndex]

                  motionStartThresholdRef.current =
                    Math.max(
                      MIN_START_THRESHOLD,
                      baseline *
                        START_THRESHOLD_MULTIPLIER,
                    )

                  motionEndThresholdRef.current =
                    Math.max(
                      MIN_END_THRESHOLD,
                      baseline *
                        END_THRESHOLD_MULTIPLIER,
                    )

                  isCalibratedRef.current = true

                  console.log(
                    '[CAMERA] motion calibration complete:',
                    {
                      baseline,
                      startThreshold:
                        motionStartThresholdRef.current,
                      endThreshold:
                        motionEndThresholdRef.current,
                    },
                  )
                }
              } else {
                debugFrameCounterRef.current += 1

                // ----------------------------------------------
                // 아직 수어 세션이 시작되지 않은 상태
                // ----------------------------------------------
                if (!motionActiveRef.current) {
                  if (awaitingResultRef.current) {
                    startCountRef.current = 0
                  } else if (cooldownRef.current > 0) {
                    cooldownRef.current -= 1
                  } else if (
                    score >=
                    motionStartThresholdRef.current
                  ) {
                    startCountRef.current += 1
                  } else {
                    startCountRef.current = 0
                  }

                  // --------------------------------------------
                  // 모션이 일정 프레임 이상 감지되면
                  // AI 스트림 시작
                  // --------------------------------------------
                  if (
                    cooldownRef.current === 0 &&
                    startCountRef.current >=
                      MOTION_START_FRAMES &&
                    !streamStartingRef.current
                  ) {
                    streamStartingRef.current = true
                    motionActiveRef.current = true

                    startCountRef.current = 0
                    endCountRef.current = 0
                    wordBoundarySentRef.current = false

                    sentFramesRef.current = 0
                    setSentFramesDisplay(0)

                    setCameraCaptureStatus(
                      'recording',
                    )

                    const started =
                      await onStreamStart()

                    streamStartingRef.current = false

                    if (!started) {
                      motionActiveRef.current = false

                      setCameraCaptureStatus(
                        'idle',
                      )
                    } else {
                      console.log(
                        '[CAMERA] sending frame',
                        cameraFrameId,
                        'motion:',
                        score,
                      )

                      await onKeypointFrame(
                        keypoints,
                        cameraFrameId,
                      )

                      sentFramesRef.current += 1
                    }
                  }
                } else {
                  // --------------------------------------------
                  // 수어 세션 중
                  //
                  // - 움직이면 즉시 frame 전송
                  // - 멈추기 시작한 초반 WORD_END_FRAMES 동안
                  //   정지 프레임 전송
                  // - WORD_END_FRAMES 이후에는 정지 프레임
                  //   전송하지 않음
                  // - 다시 움직이면 즉시 전송 재개
                  // --------------------------------------------

                  const isStill =
                    score <=
                    motionEndThresholdRef.current

                  if (isStill) {
                    endCountRef.current += 1
                  } else {
                    endCountRef.current = 0
                    wordBoundarySentRef.current =
                      false
                  }

                  const shouldSendFrame =
                    !isStill ||
                    endCountRef.current <=
                      WORD_END_FRAMES

                  // --------------------------------------------
                  // frame 전송
                  // --------------------------------------------
                  if (shouldSendFrame) {
                    await onKeypointFrame(
                      keypoints,
                      cameraFrameId,
                    )

                    sentFramesRef.current += 1

                    console.log(
                      '[CAMERA] frame sent',
                      {
                        cameraFrameId,
                        motion: score,
                        still: isStill,
                        stillCount:
                          endCountRef.current,
                      },
                    )
                  }

                  // --------------------------------------------
                  // 단어 경계
                  // --------------------------------------------
                  else if (
                    wordBoundarySentRef.current ===
                    false
                  ) {
                    // WORD_END_FRAMES만큼 마지막 정지 프레임을
                    // 보낸 뒤 더 이상 정지 프레임을 보내지 않는다.

                    wordBoundarySentRef.current =
                      true

                    console.log(
                      '[CAMERA] motion paused - stop sending frames',
                      {
                        cameraFrameId,
                        motion: score,
                        stillCount:
                          endCountRef.current,
                      },
                    )

                    try {
                      await onWordBoundary()
                    } catch (error) {
                      console.error(
                        '[CAMERA] word boundary error:',
                        error,
                      )
                    }
                  }

                  // --------------------------------------------
                  // 15프레임마다 UI 업데이트
                  // --------------------------------------------
                  if (
                    sentFramesRef.current > 0 &&
                    sentFramesRef.current % 15 === 0
                  ) {
                    setSentFramesDisplay(
                      sentFramesRef.current,
                    )
                  }

                  // --------------------------------------------
                  // 문장 종료
                  //
                  // 일정 시간 계속 정지하면 스트림 종료.
                  // 정지 구간에서는 frame을 보내지 않고
                  // WebSocket 연결만 유지한다.
                  // --------------------------------------------
                  if (
                    endCountRef.current >=
                    SENTENCE_END_FRAMES
                  ) {
                    motionActiveRef.current = false

                    endCountRef.current = 0

                    wordBoundarySentRef.current =
                      false

                    cooldownRef.current =
                      POST_END_COOLDOWN_FRAMES

                    setCameraCaptureStatus('idle')

                    setSentFramesDisplay(
                      sentFramesRef.current,
                    )

                    console.log(
                      '[CAMERA] sentence end - close stream',
                      {
                        lastCameraFrameId:
                          cameraFrameId,
                        sentFrames:
                          sentFramesRef.current,
                      },
                    )

                    if (
                      !streamEndingRef.current
                    ) {
                      streamEndingRef.current =
                        true
                      awaitingResultRef.current =
                        true

                      try {
                        await onStreamEnd()
                      } catch (error) {
                        awaitingResultRef.current =
                          false
                        console.error(
                          '[CAMERA] stream end error:',
                          error,
                        )
                      } finally {
                        streamEndingRef.current =
                          false
                      }
                    }
                  }
                }
              }
            }
          } catch (err) {
            console.error(
              '[CAMERA] keypoint extraction/send error:',
              err,
            )

            setCameraError(
              err instanceof Error
                ? err.message
                : '키포인트 추출/전송 실패',
            )

            // 프레임 전송 중 오류가 발생해도
            // 다음 frame loop는 계속 동작하도록 한다.
          } finally {
            isSendingFrameRef.current = false
          }
        }
      }

      loopRef.current =
        requestAnimationFrame(tick)
    }

    loopRef.current =
    requestAnimationFrame(tick)
  }

  // 최종 translation/error 처리로 부모의 예측 상태가 해제되면
  // 다음 문장을 감지할 수 있도록 로컬 잠금도 해제한다.
  useEffect(() => {
    if (!isPredicting && awaitingResultRef.current) {
      awaitingResultRef.current = false
    }
  }, [isPredicting])

  // ------------------------------------------------------------
  // 카메라 활성화 / 비활성화
  // ------------------------------------------------------------
  useEffect(() => {
    if (inputMode !== 'camera') {
      return
    }

    if (!isActive) {
      stopCameraStream()
      return
    }

    let cancelled = false
    let activeStream: MediaStream | null = null

    async function startCamera() {
      try {
        setCameraError(null)

        const stream =
          await navigator.mediaDevices.getUserMedia({
            video: {
              width: 640,
              height: 480,
              facingMode: 'user',
            },
            audio: false,
          })

        if (cancelled) {
          stream
            .getTracks()
            .forEach((track) => track.stop())

          return
        }

        activeStream = stream
        streamRef.current = stream

        if (videoRef.current) {
          videoRef.current.srcObject = stream

          await videoRef.current.play()
        }

        await ensureExtractor()

        if (!cancelled) {
          startWatchLoop()
        }
      } catch (err) {
        if (!cancelled) {
          setCameraError(
            err instanceof Error
              ? err.message
              : '카메라를 연결할 수 없습니다.',
          )
        }
      }
    }

    startCamera()

    return () => {
      cancelled = true

      if (activeStream) {
        activeStream
          .getTracks()
          .forEach((track) => track.stop())
      }

      stopCameraStream()
    }
  }, [isActive, inputMode])

  // ------------------------------------------------------------
  // 업로드 영상 URL 정리
  // ------------------------------------------------------------
  useEffect(() => {
    return () => {
      if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl)
      }
    }
  }, [videoObjectUrl])

  // ------------------------------------------------------------
  // Canvas 렌더링 루프
  // setState를 사용하지 않아 불필요한 리렌더링 방지
  // ------------------------------------------------------------
  useEffect(() => {
    if (
      !isActive ||
      inputMode !== 'camera'
    ) {
      return
    }

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (!canvas || !ctx) {
      return
    }

    let animationId = 0
    let frame = 0

    const render = () => {
      frame += 1

      const width =
        (canvas.width =
          canvas.parentElement?.clientWidth ||
          640)

      const height =
        (canvas.height =
          canvas.parentElement?.clientHeight ||
          400)

      ctx.clearRect(
        0,
        0,
        width,
        height,
      )

      ctx.strokeStyle =
        'rgba(15, 194, 158, 0.15)'

      ctx.beginPath()

      ctx.moveTo(
        0,
        (frame * 3) % height,
      )

      ctx.lineTo(
        width,
        (frame * 3) % height,
      )

      ctx.stroke()

      ctx.fillStyle =
        'rgba(15, 194, 158, 0.7)'

      ctx.font = '10px monospace'

      ctx.fillText(
        `${
          cameraCaptureStatus ===
          'recording'
            ? 'REC 30FPS'
            : 'WATCHING'
        } | CAM: ${
          cameraFrameIdRef.current
        } | SENT: ${
          sentFramesRef.current
        }`,
        16,
        height - 20,
      )

      animationId =
        requestAnimationFrame(render)
    }

    render()

    return () =>
      cancelAnimationFrame(animationId)
  }, [
    cameraCaptureStatus,
    isActive,
    inputMode,
  ])

  // ------------------------------------------------------------
  // 업로드 영상 이벤트
  // ------------------------------------------------------------
  const handleVideoLoadedMetadata = () => {
    const video = videoRef.current

    if (!video) {
      return
    }

    setVideoDuration(video.duration || 0)
  }

  const handleVideoTimeUpdate = () => {
    const video = videoRef.current

    if (!video) {
      return
    }

    setVideoCurrentTime(video.currentTime)
  }

  const handleVideoPlay = () => {
    setIsVideoPaused(false)
  }

  const handleVideoPause = () => {
    setIsVideoPaused(true)
  }

  const handleVideoEnded = () => {
    setIsVideoPaused(true)
  }

  // ------------------------------------------------------------
  // 업로드 영상에서 camera srcObject 제거
  // ------------------------------------------------------------
  useEffect(() => {
    if (
      inputMode === 'video' &&
      videoRef.current &&
      videoObjectUrl
    ) {
      videoRef.current.srcObject = null
      videoRef.current.src = videoObjectUrl
      videoRef.current.load()
    }
  }, [inputMode, videoObjectUrl])

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]

          if (!file) {
            return
          }

          handleVideoFile(file)

          // 같은 파일을 다시 선택할 수 있도록 초기화
          e.currentTarget.value = ''
        }}
      />

      <div
        className="relative w-full aspect-video min-h-[280px] sm:min-h-[420px] bg-brand-emerald rounded-2xl flex flex-col items-center justify-center overflow-hidden border border-emerald-950/20 shadow-inner group cursor-pointer"
        onClick={() => {
          if (
            !isActive &&
            inputMode !== 'video'
          ) {
            onToggleActive()
          }
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={
            handleVideoLoadedMetadata
          }
          onTimeUpdate={
            handleVideoTimeUpdate
          }
          onPlay={handleVideoPlay}
          onPause={handleVideoPause}
          onEnded={handleVideoEnded}
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
            isActive ||
            inputMode === 'video'
              ? 'opacity-100'
              : 'hidden'
          }`}
        />

        {/* -------------------------------------------------- */}
        {/* 카메라 Canvas */}
        {/* -------------------------------------------------- */}
        {isActive &&
          inputMode === 'camera' && (
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full z-10"
            />
          )}

        {/* -------------------------------------------------- */}
        {/* 카메라 시작 화면 */}
        {/* -------------------------------------------------- */}
        {!isActive &&
          inputMode !== 'video' && (
            <div className="flex flex-col items-center justify-center p-6 text-center select-none z-20">
              <div className="w-20 h-20 rounded-full bg-white/10 text-white flex items-center justify-center border border-white/20 shadow-lg">
                <Hand className="w-10 h-10 text-white -rotate-12 animate-bounce" />
              </div>

              <h3 className="mt-5 text-xl font-bold text-white">
                화면을 눌러 수어 번역 시작하기
              </h3>

              <p className="text-sm text-neutral-300 max-w-sm mt-2 font-medium">
                카메라를 통해 환자의 수어 동작을 인식하고 실시간으로 번역합니다.
              </p>

              {cameraError && (
                <p className="mt-3 text-sm text-red-200 max-w-md">
                  {cameraError}
                </p>
              )}
            </div>
          )}

        {/* -------------------------------------------------- */}
        {/* 카메라 HUD */}
        {/* -------------------------------------------------- */}
        {isActive &&
          inputMode === 'camera' && (
            <>
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none">
                <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      cameraCaptureStatus ===
                      'recording'
                        ? 'bg-red-500 animate-ping'
                        : 'bg-brand-green'
                    }`}
                  />

                  <span className="text-xs font-bold text-white">
                    {cameraCaptureStatus ===
                    'recording'
                      ? 'LIVE 30fps 인식 중'
                      : '수어 감지 대기 중'}
                  </span>
                </div>

                <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                  <Sparkles className="w-4 h-4 text-brand-green animate-pulse" />

                  <span className="text-xs font-bold text-white">
                    프레임: {sentFramesDisplay}
                  </span>
                </div>
              </div>
              {/* ---------------------------------------------- */}
              {/* 오류 표시 */}
              {/* ---------------------------------------------- */}
              {cameraError && (
                <div className="absolute top-16 left-4 right-4 z-30 pointer-events-none">
                  <div className="mx-auto max-w-md rounded-lg bg-red-500/80 backdrop-blur-md px-4 py-2 text-center text-xs font-semibold text-white">
                    {cameraError}
                  </div>
                </div>
              )}

              {/* ---------------------------------------------- */}
              {/* 카메라 끄기 버튼 */}
              {/* ---------------------------------------------- */}
              <button
                type="button"
                onClick={async (e) => {
                  e.preventDefault()
                  e.stopPropagation()

                  // AI 스트림이 살아 있는 경우에만 종료
                  await endSignStream()

                  // 그 다음 카메라 종료
                  stopCameraStream()

                  // 내부 상태 초기화
                  resetStreamingState()

                  // 부모의 카메라 상태 변경
                  onToggleActive()
                }}
                className="absolute bottom-4 right-4 z-50 flex items-center gap-2 bg-black/70 hover:bg-red-500/90 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/20 text-white transition-colors cursor-pointer"
              >
                <VideoOff className="w-4 h-4" />

                <span className="text-xs font-bold">
                  카메라 끄기
                </span>
              </button>
            </>
          )}

        {/* -------------------------------------------------- */}
        {/* 영상 업로드 모드 */}
        {/* -------------------------------------------------- */}
        {inputMode === 'video' &&
          videoObjectUrl && (
            <>
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-30 pointer-events-none">
                <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                  <Film className="w-4 h-4 text-brand-green" />

                  <span className="text-xs font-bold text-white">
                    업로드 영상
                  </span>
                </div>

                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    clearVideoFile()
                  }}
                  className="pointer-events-auto flex items-center justify-center w-8 h-8 rounded-full bg-black/60 hover:bg-red-500/90 text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="absolute bottom-4 left-4 right-4 z-30 flex items-center gap-3">
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    toggleVideoPlayback()
                  }}
                  className="flex items-center justify-center w-9 h-9 rounded-full bg-black/70 hover:bg-black/90 text-white"
                >
                  {isVideoPaused ? (
                    <Play className="w-4 h-4" />
                  ) : (
                    <Pause className="w-4 h-4" />
                  )}
                </button>

                <div className="flex-1 h-1.5 rounded-full bg-white/20 overflow-hidden">
                  <div
                    className="h-full bg-brand-green transition-all"
                    style={{
                      width:
                        videoDuration > 0
                          ? `${
                              (videoCurrentTime /
                                videoDuration) *
                              100
                            }%`
                          : '0%',
                    }}
                  />
                </div>

                <span className="text-[10px] font-mono text-white whitespace-nowrap">
                  {Math.floor(videoCurrentTime)}
                  /
                  {Math.floor(videoDuration)}
                </span>
              </div>
            </>
          )}

        {/* -------------------------------------------------- */}
        {/* 영상 선택 버튼 */}
        {/* -------------------------------------------------- */}
        {!isActive &&
          inputMode !== 'video' && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
              className="absolute bottom-4 left-4 z-30 flex items-center gap-2 bg-black/60 hover:bg-black/80 backdrop-blur-md px-3.5 py-2 rounded-full border border-white/20 text-white transition-colors"
            >
              <Upload className="w-4 h-4" />

              <span className="text-xs font-bold">
                영상 업로드
              </span>
            </button>
          )}

        {/* -------------------------------------------------- */}
        {/* 카메라가 꺼져 있을 때 오류 메시지 */}
        {/* -------------------------------------------------- */}
        {!isActive &&
          inputMode !== 'video' &&
          cameraError && (
            <div className="absolute bottom-16 left-4 right-4 z-30">
              <div className="mx-auto max-w-md rounded-lg bg-red-500/80 backdrop-blur-md px-4 py-2 text-center text-xs font-semibold text-white">
                {cameraError}
              </div>
            </div>
          )}
      </div>

      {/* ---------------------------------------------------- */}
      {/* 업로드 영상 파일명 */}
      {/* ---------------------------------------------------- */}
      {videoFile &&
        inputMode === 'video' && (
          <div className="mt-3 flex items-center justify-between px-2">
            <div className="flex items-center gap-2 min-w-0">
              <Film className="w-4 h-4 text-brand-green shrink-0" />

              <span className="text-sm text-neutral-600 truncate">
                {videoFile.name}
              </span>
            </div>

            <button
              type="button"
              onClick={() => {
                clearVideoFile()
              }}
              className="text-xs text-neutral-500 hover:text-red-500 transition-colors"
            >
              제거
            </button>
          </div>
        )}
    </div>
  )
}
