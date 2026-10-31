import React, { useEffect, useRef, useState } from 'react'
import {
  Camera,
  Film,
  Hand,
  Pause,
  Play,
  Square,
  Upload,
  VideoOff,
  Wand2,
  X,
  ArrowDown,
} from 'lucide-react'
import { GESTURE_PRESETS } from '../data'
import { BrowserKeypointExtractor } from '../keypointExtractor'
import { GesturePreset } from '../types'
import { FaceFrame } from '../nms'

interface CameraViewProps {
  isActive: boolean
  onToggleActive: () => void
  onTriggerTranslation: (text: string, category: string) => void
  onStreamStart: () => Promise<boolean>
  onKeypointFrame: (
    keypoints: number[],
    frameId: number,
    still: boolean,
    face?: FaceFrame | null,
  ) => Promise<void>
  onWordBoundary: () => Promise<void>
  onStreamEnd: () => Promise<void>
  isPredicting?: boolean
}

type InputMode = 'camera' | 'video'
type CameraCaptureStatus = 'idle' | 'recording'
type MotionCalibrationStatus =
  | 'preparing'
  | 'calibrating'
  | 'unstable'
  | 'ready'

const CAMERA_EXTRACTION_FPS = 30
const CAMERA_FRAME_INTERVAL_MS = 1000 / CAMERA_EXTRACTION_FPS
const UPLOAD_EXTRACTION_FPS = 30

// 최소 샘플 수와 실제 경과 시간을 모두 만족한 뒤 보정을 완료한다.
// 프레임 속도가 빨라도 안내를 읽고 정지할 시간을 확보한다.
const CALIBRATION_FRAMES = 20
const CALIBRATION_MIN_DURATION_MS = 2000
// keypointExtractor의 포즈 블록: 33개 랜드마크 × 3개 좌표.
// 이 블록이 전부 0이면 사람이 검출되지 않은 프레임이다.
const POSE_KEYPOINT_VALUES = 99
// 손 데이터는 손당 78개
// 21개 랜드마크 × xyz(63개) + 관절각(15개)
const HAND_KEYPOINT_VALUES = 78
const HAND_COORDINATE_VALUES = 21 * 3
const KEYPOINT_VECTOR_LENGTH = 261

const LEFT_HAND_START = 99
const RIGHT_HAND_START = LEFT_HAND_START + HAND_KEYPOINT_VALUES
// 추정 초기값. 실제 카메라/추출기에서 측정한 정지 잡음에 맞춰 조정한다.
// baseline(기존 20백분위값)이 이 값을 초과하면 보정을 완료하지 않는다.
const MAX_CALIBRATION_BASELINE = 0.02
const START_THRESHOLD_MULTIPLIER = 6
const END_THRESHOLD_MULTIPLIER = 2.5
const MIN_START_THRESHOLD = 0.0015
const MIN_END_THRESHOLD = 0.0006

const MOTION_START_FRAMES = 3
const WORD_END_FRAMES = 8
const SENTENCE_END_FRAMES = 30
const POST_END_COOLDOWN_FRAMES = 8

// 수어 종료 방식
//  true : 모션이 완전히 멈춰도 문장을 끝내지 않는다. 사용자가 "수어 종료" 버튼을 눌렀을 때만
//         스트림을 종료하고 녹화 전체를 번역한다.
//         (정지 중 프레임 전송 중단과 단어 경계 전송은 기존 그대로 유지한다)
//  false: 기존처럼 SENTENCE_END_FRAMES만큼 정지하면 자동 종료한다.
const MANUAL_STOP_MODE = true
// 수동 모드에서 버튼을 잊고 방치했을 때의 안전장치. 이 프레임 수만큼 계속 정지하면 자동 종료한다
// (30fps 기준 30초). 0이면 안전장치를 끄고 오직 버튼으로만 종료한다.
const IDLE_SAFETY_END_FRAMES = 900
// 한 세션에서 보낸 프레임이 이 수에 도달하면 자동 종료한다. 서버의 MAX_RECORD_FRAMES(5400)와 같은 값.
const MAX_MANUAL_RECORD_FRAMES = 5400
// 버튼을 누르러 손을 뻗는 동작이 새 수어 세션으로 오인되지 않게 하는 쿨다운(30fps 기준 1.5초)
const MANUAL_STOP_COOLDOWN_FRAMES = 45

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

/**
 * 손 랜드마크 검출 여부 확인
 *
 * keypointExtractor.ts에서 손이 검출되지 않으면
 * 해당 손의 78개 데이터가 모두 0으로 반환된다.
 *
 * 좌표 63개를 검사하여 손 검출 여부를 판단한다.
 */
function hasDetectedHand(
  keypoints: number[],
  startIndex: number,
): boolean {
  const coordinates = keypoints.slice(
    startIndex,
    startIndex + HAND_COORDINATE_VALUES,
  )

  if (coordinates.length !== HAND_COORDINATE_VALUES) {
    return false
  }

  return coordinates.some(
    (value) => Math.abs(value) > 1e-6,
  )
}

/**
 * 왼손 또는 오른손 중 하나라도 검출되었는지 확인
 */
function hasDetectedAnyHand(keypoints: number[]): boolean {
  return (
    hasDetectedHand(keypoints, LEFT_HAND_START) ||
    hasDetectedHand(keypoints, RIGHT_HAND_START)
  )
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
  // 비수지(NMS)용: 직전에 추출한 프레임의 얼굴 랜드마크
  const lastFaceRef = useRef<FaceFrame | null>(null)
  const faceWarnedRef = useRef(false)
  const loopRef = useRef<number | null>(null)
  // 카메라를 끈 뒤 늦게 끝난 비동기 작업이 안내를 다시 띄우지 않도록 한다.
  const watchLoopRunRef = useRef(0)
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
  const calibrationStartedAtRef = useRef<number | null>(null)
  // RAF 루프는 시작 시점의 state를 캡처하므로 실패 여부는 Ref로도 유지한다.
  // 실패 후에는 사용자가 다시 보정을 누르기 전까지 샘플을 수집하지 않는다.
  const calibrationFailedRef = useRef(false)
  const motionStartThresholdRef = useRef(MIN_START_THRESHOLD)
  const motionEndThresholdRef = useRef(MIN_END_THRESHOLD)
  const isCalibratedRef = useRef(false)
  const prevKeypointsRef = useRef<number[] | null>(null)
  const streamStartingRef = useRef(false)

  // ---- 스트림 종료 중복 방지 ----
  const streamEndingRef = useRef(false)

  const [inputMode, setInputMode] = useState<InputMode>('camera')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [calibrationStatus, setCalibrationStatus] =
    useState<MotionCalibrationStatus>('preparing')
  const [calibrationProgress, setCalibrationProgress] = useState(0)
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
  // 보정 샘플/시간만 초기화. 실패 안내의 잠금 상태는 유지한다.
  // ------------------------------------------------------------
  const resetCalibrationWindow = () => {
    calibrationSamplesRef.current = []
    calibrationStartedAtRef.current = null
    setCalibrationProgress(0)
  }

  // ------------------------------------------------------------
  // 카메라는 유지한 채 모션 보정만 처음부터 다시 시작한다.
  // ------------------------------------------------------------
  const resetMotionCalibration = () => {
    resetCalibrationWindow()
    calibrationFailedRef.current = false
    motionStartThresholdRef.current = MIN_START_THRESHOLD
    motionEndThresholdRef.current = MIN_END_THRESHOLD
    isCalibratedRef.current = false
    prevKeypointsRef.current = null
    startCountRef.current = 0
    setCalibrationStatus('preparing')
  }

  const handleRecalibrate = () => {
    // 진행 중인 수어 세션이나 종료 처리는 건드리지 않는다.
    if (
      !isActive ||
      inputMode !== 'camera' ||
      motionActiveRef.current ||
      streamStartingRef.current ||
      streamEndingRef.current ||
      isPredicting
    ) {
      return
    }

    resetMotionCalibration()
    endCountRef.current = 0
    wordBoundarySentRef.current = false
    cooldownRef.current = 0
  }

  // ------------------------------------------------------------
  // 모션 게이트 초기화
  // ------------------------------------------------------------
  const resetMotionGate = () => {
    motionActiveRef.current = false
    startCountRef.current = 0
    endCountRef.current = 0
    cooldownRef.current = 0
    wordBoundarySentRef.current = false

    resetMotionCalibration()
    streamStartingRef.current = false

    debugFrameCounterRef.current = 0
    cameraFrameIdRef.current = 0
    sentFramesRef.current = 0
  }

  // ------------------------------------------------------------
  // 프레임 루프 종료
  // ------------------------------------------------------------
  const stopFrameLoop = () => {
    watchLoopRunRef.current += 1

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

    try {
      await onStreamEnd()
    } catch (error) {
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
  // 사용자가 "수어 종료" 버튼으로 녹화를 끝낸다.
  //
  // 자동 종료(정지 감지)와 같은 정리를 하되, 상태 플래그를 onStreamEnd()보다
  // 먼저 내려서 이후 프레임이 종료된 세션으로 전송되지 않게 하고,
  // 버튼 쪽으로 손을 뻗는 동작이 새 세션 시작으로 오인되지 않게 쿨다운을 건다.
  // ------------------------------------------------------------
  const handleManualStop = async () => {
    if (
      streamEndingRef.current ||
      streamStartingRef.current ||
      !motionActiveRef.current
    ) {
      return
    }

    streamEndingRef.current = true
    motionActiveRef.current = false
    startCountRef.current = 0
    endCountRef.current = 0
    wordBoundarySentRef.current = false
    cooldownRef.current = MANUAL_STOP_COOLDOWN_FRAMES

    setCameraCaptureStatus('idle')
    setSentFramesDisplay(sentFramesRef.current)

    console.log('[CAMERA] manual stop - close stream', {
      sentFrames: sentFramesRef.current,
    })

    try {
      await onStreamEnd()
    } catch (error) {
      console.error('[CAMERA] manual stop error:', error)
    } finally {
      streamEndingRef.current = false
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

      const keypoints = await extractor.extract(video)

      // 비수지(NMS)용 얼굴 랜드마크(얼굴 미검출이면 null).
      // NMS는 보조 기능이므로 여기서 어떤 문제가 생겨도 수지 keypoint 추출/전송은
      // 절대 막지 않는다(extractor가 구버전이어도 예외 없이 NMS만 꺼진다).
      let landmarks: number[] | null = null
      try {
        if (typeof extractor.getLastFaceLandmarks === 'function') {
          landmarks = extractor.getLastFaceLandmarks()
        } else if (!faceWarnedRef.current) {
          faceWarnedRef.current = true
          console.warn(
            '[CAMERA] extractor에 getLastFaceLandmarks가 없습니다 — 비수지(NMS) 비활성. ' +
              'keypointExtractor.ts가 최신인지, 페이지를 강력 새로고침(Ctrl+Shift+R)했는지 확인하세요.',
          )
        }
      } catch (err) {
        console.warn('[CAMERA] 얼굴 랜드마크 읽기 실패(NMS만 건너뜀):', err)
      }

      lastFaceRef.current = landmarks
        ? { landmarks, width: video.videoWidth, height: video.videoHeight }
        : null

      return keypoints
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
    const runId = watchLoopRunRef.current

    lastFrameAtRef.current = 0

    resetMotionGate()

    setCameraCaptureStatus('idle')

    const tick = async (time: number) => {
      if (runId !== watchLoopRunRef.current) {
        return
      }

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

            if (runId !== watchLoopRunRef.current) {
              return
            }

            // 이번 프레임의 얼굴(이후 await 동안 다음 프레임으로 덮이지 않도록 지역 상수로 고정)
            const face = lastFaceRef.current

            if (
              keypoints &&
              keypoints.length >= KEYPOINT_VECTOR_LENGTH &&
              keypoints.every((value) => Number.isFinite(value))
            ) {
              // 실제 카메라 프레임 번호.
              // 전송 여부와 관계없이 증가한다.
              const cameraFrameId =
                cameraFrameIdRef.current

              cameraFrameIdRef.current += 1

              // 이전 프레임을 덮어쓰기 전에 포즈 검출 여부를 함께 확인한다.
              const prev = prevKeypointsRef.current
              const hasPreviousFrame =
                prev !== null && prev.length === keypoints.length

              const prevHasPose =
                prev !== null &&
                prev.length >= POSE_KEYPOINT_VALUES &&
                prev.slice(0, POSE_KEYPOINT_VALUES).some((value) => value !== 0)
              const hasPose = keypoints
                .slice(0, POSE_KEYPOINT_VALUES)
                .some((value) => value !== 0)
              const prevHasHand =
                prev !== null && 
                prev.length >= KEYPOINT_VECTOR_LENGTH && 
                hasDetectedAnyHand(prev)

              const hasHand = hasDetectedAnyHand(keypoints)

              const score = computeMotionScore(prev, keypoints)

              prevKeypointsRef.current = keypoints

              // ------------------------------------------------
              // 자동 캘리브레이션
              // 이 분기에서는 스트림을 시작하거나 프레임을 전송하지 않는다.
              // ------------------------------------------------
              if (!isCalibratedRef.current) {
                if (!hasPose || !hasHand) {
                  // 상체가 사라지면 연속 샘플과 경과 시간을 모두 다시 센다.
                  resetCalibrationWindow()
                  if (!calibrationFailedRef.current) {
                    setCalibrationStatus('preparing')
                  }
                } else if (!calibrationFailedRef.current) {
                  setCalibrationStatus('calibrating')

                  // 첫 프레임과 미검출 → 재검출 프레임의 점프 값은 제외한다.
                  if (hasPreviousFrame && prevHasPose && prevHasHand) {
                    const now = performance.now()

                    if (calibrationStartedAtRef.current === null) {
                      calibrationStartedAtRef.current = now
                    }

                    calibrationSamplesRef.current.push(score)

                    const elapsed = now - calibrationStartedAtRef.current
                    const sampleProgress =
                      calibrationSamplesRef.current.length / CALIBRATION_FRAMES
                    const timeProgress = elapsed / CALIBRATION_MIN_DURATION_MS
                    const progress = Math.min(
                      99,
                      Math.floor(Math.min(sampleProgress, timeProgress) * 100),
                    )

                    setCalibrationProgress((current) =>
                      current === progress ? current : progress,
                    )

                    if (
                      calibrationSamplesRef.current.length >= CALIBRATION_FRAMES &&
                      elapsed >= CALIBRATION_MIN_DURATION_MS
                    ) {
                      const samples = [...calibrationSamplesRef.current].sort(
                        (a, b) => a - b,
                      )
                      const percentileIndex = Math.floor(samples.length * 0.2)
                      const baseline = samples[percentileIndex]

                      if (baseline > MAX_CALIBRATION_BASELINE) {
                        // 높은 baseline을 임계값으로 채택하지 않는다.
                        // 경고는 다음 프레임에도 유지하며 재시도 버튼으로만 해제한다.
                        calibrationFailedRef.current = true
                        resetCalibrationWindow()
                        setCalibrationStatus('unstable')
                        console.warn('[CAMERA] motion calibration rejected:', {
                          baseline,
                          maxBaseline: MAX_CALIBRATION_BASELINE,
                        })
                      } else {
                        motionStartThresholdRef.current = Math.max(
                          MIN_START_THRESHOLD,
                          baseline * START_THRESHOLD_MULTIPLIER,
                        )
                        motionEndThresholdRef.current = Math.max(
                          MIN_END_THRESHOLD,
                          baseline * END_THRESHOLD_MULTIPLIER,
                        )

                        // 완료 상태와 실제 모션 게이트를 함께 전환한다.
                        isCalibratedRef.current = true
                        startCountRef.current = 0
                        setCalibrationProgress(100)
                        setCalibrationStatus('ready')

                        console.log('[CAMERA] motion calibration complete:', {
                          baseline,
                          startThreshold: motionStartThresholdRef.current,
                          endThreshold: motionEndThresholdRef.current,
                        })
                      }
                    }
                  }
                }
              } else {
                debugFrameCounterRef.current += 1

                // ----------------------------------------------
                // 아직 수어 세션이 시작되지 않은 상태
                // ----------------------------------------------
                if (!motionActiveRef.current) {
                  if (cooldownRef.current > 0) {
                    cooldownRef.current -= 1
                    startCountRef.current = 0
                  } else if (!hasPose || !prevHasPose || !prevHasHand || !hasPreviousFrame) {
                    // 보정 후에도 퇴장/재입장 점프가 새 세션을 시작하지 않게 한다.
                    startCountRef.current = 0
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

                    if (runId !== watchLoopRunRef.current) {
                      return
                    }

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
                        false,
                        face,
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
                    !hasHand || score <= motionEndThresholdRef.current

                  if (isStill) {
                    endCountRef.current += 1
                  } else {
                    endCountRef.current = 0
                    wordBoundarySentRef.current =
                      false
                  }

                  const shouldSendFrame =
                    hasHand &&  
                  (!isStill ||
                    endCountRef.current <=
                      WORD_END_FRAMES
                    ) 
                  // --------------------------------------------
                  // frame 전송
                  // --------------------------------------------
                  if (shouldSendFrame) {
                    await onKeypointFrame(
                      keypoints,
                      cameraFrameId,
                      isStill,
                      face,
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
                    MANUAL_STOP_MODE
                      ? (IDLE_SAFETY_END_FRAMES > 0 &&
                          endCountRef.current >=
                            IDLE_SAFETY_END_FRAMES) ||
                        sentFramesRef.current >=
                          MAX_MANUAL_RECORD_FRAMES
                      : endCountRef.current >=
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

                      try {
                        await onStreamEnd()
                      } catch (error) {
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
            } else {
              // null/빈 배열/짧은 배열/NaN 프레임도 연속 보정을 끊는다.
              // 추출 실패 전후의 프레임을 직접 비교하지 않는다.
              prevKeypointsRef.current = null
              startCountRef.current = 0
              if (!isCalibratedRef.current) {
                resetCalibrationWindow()
                if (!calibrationFailedRef.current) {
                  setCalibrationStatus('preparing')
                }
              }
            }
          } catch (err) {
            if (runId !== watchLoopRunRef.current) {
              return
            }

            // 예외가 난 구간을 정상 보정 시간/연속 샘플로 계산하지 않는다.
            prevKeypointsRef.current = null
            startCountRef.current = 0
            if (!isCalibratedRef.current) {
              resetCalibrationWindow()
              if (!calibrationFailedRef.current) {
                setCalibrationStatus('preparing')
              }
            }

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
            if (runId === watchLoopRunRef.current) {
              isSendingFrameRef.current = false
            }
          }
        }
      }

      if (runId === watchLoopRunRef.current) {
        loopRef.current = requestAnimationFrame(tick)
      }
    }

    loopRef.current =
      requestAnimationFrame(tick)
  }

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
            : !isCalibratedRef.current
              ? 'CALIBRATING'
              : 'READY'
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
        className={`relative w-full aspect-video ${
          isActive && inputMode === 'camera'
            ? 'min-h-[400px]'
            : 'min-h-[280px]'
        } sm:min-h-[420px] bg-brand-emerald rounded-2xl flex flex-col items-center justify-center overflow-hidden border border-emerald-950/20 shadow-inner group cursor-pointer`}
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
              <div className="pointer-events-none absolute inset-x-4 top-4 z-20 flex items-center justify-between gap-3">
                <div className="inline-flex min-w-0 items-center gap-2 rounded-lg bg-neutral-950/80 px-3 py-2 backdrop-blur-sm">
                  <span
                    aria-hidden="true"
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      cameraCaptureStatus === 'recording'
                        ? 'bg-red-400'
                        : calibrationStatus === 'ready'
                          ? 'bg-emerald-400'
                          : calibrationStatus === 'unstable'
                            ? 'bg-amber-300'
                            : 'bg-white/60'
                    }`}
                  />
                  <span className="truncate text-xs font-medium leading-4 text-white/90">
                    {cameraCaptureStatus === 'recording'
                      ? MANUAL_STOP_MODE
                        ? '인식 중 · 끝나면 수어 종료'
                        : '수어 인식 중'
                      : isPredicting
                        ? '번역 중'
                        : calibrationStatus === 'preparing'
                          ? '위치 맞추기'
                          : calibrationStatus === 'calibrating'
                            ? '보정 중'
                            : calibrationStatus === 'unstable'
                              ? '보정 필요'
                              : '인식 대기'}
                  </span>
                </div>

                {cameraCaptureStatus === 'recording' && (
                  <span className="shrink-0 rounded-lg bg-neutral-950/80 px-3 py-2 text-[11px] leading-4 tabular-nums text-white/70 backdrop-blur-sm">
                    {sentFramesDisplay} 프레임
                  </span>
                )}
              </div>

              {/* 상체 검출은 포즈 앞 99값으로 확인한다. 모서리 선은 구도 안내용이다. */}
              {cameraCaptureStatus === 'idle' &&
                !cameraError &&
                !isPredicting && (
                  <div className="pointer-events-none absolute inset-x-4 bottom-16 top-16 z-20 flex flex-col gap-5 sm:inset-x-6 sm:gap-6">
                    {calibrationStatus !== 'ready' && (
                      <div
                        aria-hidden="true"
                        className="flex min-h-0 flex-1 items-center justify-center"
                      >
                        <div className="relative h-full max-h-[440px] w-[88%] max-w-[560px]">
                          <span className="absolute left-0 top-0 h-6 w-6 rounded-tl-md border-l border-t border-white/80" />
                          <span className="absolute right-0 top-0 h-6 w-6 rounded-tr-md border-r border-t border-white/80" />
                          <span className="absolute bottom-0 left-0 h-6 w-6 rounded-bl-md border-b border-l border-white/80" />
                          <span className="absolute bottom-0 right-0 h-6 w-6 rounded-br-md border-b border-r border-white/80" />
                        </div>
                      </div>
                    )}

                    <div className="mx-auto mt-auto w-full max-w-md shrink-0 rounded-xl border border-white/10 bg-neutral-950/90 px-4 py-3 backdrop-blur-sm">
                      <div className="flex items-start justify-between gap-3">
                        <div
                          role="status"
                          aria-live="polite"
                          aria-atomic="true"
                          className="min-w-0 flex-1"
                        >
                          <div className="flex items-center gap-2">
                            <span
                              aria-hidden="true"
                              className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                                calibrationStatus === 'ready'
                                  ? 'bg-emerald-400'
                                  : calibrationStatus === 'unstable'
                                    ? 'bg-amber-300'
                                    : 'bg-white/60'
                              }`}
                            />
                            <p className="text-[13px] font-medium leading-5 text-white">
                              {calibrationStatus === 'ready'
                                ? '보정 완료'
                                : calibrationStatus === 'unstable'
                                  ? '다시 보정해 주세요'
                                  : calibrationStatus === 'calibrating'
                                    ? '잠시 멈춰 주세요'
                                    : '상체를 화면에 맞춰 주세요'}
                            </p>
                          </div>
                          <p className="mt-1 text-xs leading-5 text-white/70">
                            {calibrationStatus === 'ready'
                              ? '이제 수어를 시작해 주세요.'
                              : calibrationStatus === 'unstable'
                                ? '움직임이 감지되었습니다. 잠시 멈춘 뒤 다시 시도해 주세요.'
                                : calibrationStatus === 'calibrating'
                                  ? '손과 상체를 편안하게 고정해 주세요.'
                                  : '얼굴과 양손이 프레임 안에 보이게 맞춰 주세요.'}
                          </p>
                        </div>

                        {calibrationStatus === 'calibrating' && (
                          <span
                            aria-hidden="true"
                            className="shrink-0 pt-0.5 text-xs leading-5 tabular-nums text-white/70"
                          >
                            {calibrationProgress}%
                          </span>
                        )}

                        {(calibrationStatus === 'unstable' ||
                        calibrationStatus === 'ready') && (
                          <button
                            type="button"
                            aria-label="모션 캘리브레이션 다시 시작"
                            onClick={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              handleRecalibrate()
                            }}
                            className="pointer-events-auto inline-flex min-h-[44px] shrink-0 items-center justify-center rounded-lg bg-white/10 px-3 text-xs font-medium text-white transition-colors hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white motion-reduce:transition-none"
                          >
                            다시 보정
                          </button>
                        )}
                      </div>

                      {calibrationStatus === 'calibrating' && (
                        <div
                          role="progressbar"
                          aria-label="모션 캘리브레이션 진행률"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={calibrationProgress}
                          className="mt-3 h-0.5 overflow-hidden rounded-full bg-white/20"
                        >
                          <div
                            className="h-full rounded-full bg-white/90 transition-[width] duration-150 motion-reduce:transition-none"
                            style={{ width: `${calibrationProgress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}

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
              {/* 수어 종료 버튼: 녹화 중에만 표시. 누르면 녹화 전체를 번역한다 */}
              {/* ---------------------------------------------- */}
              {cameraCaptureStatus === 'recording' && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    void handleManualStop()
                  }}
                  className="absolute bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 bg-red-500 hover:bg-red-600 px-5 py-2 rounded-full border border-white/30 text-white shadow-lg transition-colors cursor-pointer"
                >
                  <Square className="w-4 h-4 fill-current" />

                  <span className="text-sm font-bold">수어 종료</span>
                </button>
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