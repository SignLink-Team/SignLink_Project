import React, { useEffect, useRef, useState } from 'react'
import { Camera, Film, Hand, Pause, Play, Square, Upload, VideoOff, X } from 'lucide-react'
import { BrowserKeypointExtractor } from '../keypointExtractor'
import { FaceFrame } from '../nms'
import { getErrorMessage } from '../utils'
import {
  GuideSteps,
  HowToCard,
  MissingPart,
  MotionCalibrationStatus,
  PositionGuideOverlay,
  getCurrentStep,
  getGuideText,
  getStatusText,
} from './camera/CameraGuide'
import {
  CALIBRATION_FRAMES,
  CALIBRATION_MIN_DURATION_MS,
  CAMERA_FRAME_INTERVAL_MS,
  END_THRESHOLD_MULTIPLIER,
  KEYPOINT_VECTOR_LENGTH,
  MANUAL_STOP_COOLDOWN_FRAMES,
  MAX_CALIBRATION_BASELINE,
  MIN_END_THRESHOLD,
  MIN_START_THRESHOLD,
  MISSING_NOTICE_FRAMES,
  MOTION_START_FRAMES,
  POST_END_COOLDOWN_FRAMES,
  START_THRESHOLD_MULTIPLIER,
  WORD_END_FRAMES,
  computeCalibrationBaseline,
  computeMotionScore,
  hasDetectedAnyHand,
  hasDetectedPose,
  isValidKeypointVector,
  shouldAutoEndSentence,
  stopMediaStream,
} from './camera/cameraMotion'

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
  /** 값이 바뀔 때마다 보정부터 다시 시작한다(번역 결과의 '다시 인식'). 생략하면 사용하지 않는다. */
  retryRequestId?: number
  /** 카메라 화면 좌우 반전. 화면 표시만 뒤집으므로 키포인트 추출과 인식에는 영향이 없다. */
  isMirrored?: boolean
}

type InputMode = 'camera' | 'video'
type CameraCaptureStatus = 'idle' | 'recording'

/** 한 프레임을 모션 게이트에서 판단하는 데 필요한 값 */
interface AnalyzedFrame {
  keypoints: number[]
  face: FaceFrame | null
  cameraFrameId: number
  score: number
  hasPose: boolean
  hasHand: boolean
  hasPreviousFrame: boolean
  prevHasPose: boolean
  prevHasHand: boolean
}

const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: 640,
    height: 480,
    facingMode: 'user',
  },
  audio: false,
}

const ERROR_BOX_CLASS =
  'mx-auto max-w-md rounded-lg bg-red-500/85 px-4 py-3 text-center text-sm font-semibold leading-relaxed text-white backdrop-blur-md sm:text-base'

/** 영상 영역 클릭(카메라 켜기)으로 전파되지 않는 버튼 클릭 핸들러를 만든다. */
function isolatedClick(handler: () => void) {
  return (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    handler()
  }
}

// onTriggerTranslation은 외부 인터페이스 유지를 위해 props에 남겨두지만 현재 화면에서는 사용하지 않는다.
export const CameraView: React.FC<CameraViewProps> = ({
  isActive,
  onToggleActive,
  onStreamStart,
  onKeypointFrame,
  onWordBoundary,
  onStreamEnd,
  isPredicting = false,
  retryRequestId = 0,
  isMirrored = false,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
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

  // 실제 카메라에서 추출한 프레임 번호.
  // 서버로 전송하지 않는 정지 프레임도 계속 증가한다.
  const cameraFrameIdRef = useRef(0)

  // ---- 자동 캘리브레이션 상태 ----
  const calibrationSamplesRef = useRef<number[]>([])
  const calibrationStartedAtRef = useRef<number | null>(null)
  // RAF 루프는 시작 시점의 state를 캡처하므로 실패 여부는 Ref로도 유지한다.
  // 실패 후에는 사용자가 다시 보정을 누르기 전까지 샘플을 수집하지 않는다.
  const calibrationFailedRef = useRef(false)
  // 한 문장을 마친 뒤 '다시 맞추기'를 누르기 전까지 새 수어 인식을 시작하지 않는다.
  // (결과를 보는 동안 손을 움직여 새 녹화가 시작되고 결과가 지워지는 것을 막는다)
  const sentenceDoneRef = useRef(false)
  const motionStartThresholdRef = useRef(MIN_START_THRESHOLD)
  const motionEndThresholdRef = useRef(MIN_END_THRESHOLD)
  const isCalibratedRef = useRef(false)
  const prevKeypointsRef = useRef<number[] | null>(null)
  const streamStartingRef = useRef(false)

  // ---- 스트림 종료 중복 방지 ----
  const streamEndingRef = useRef(false)

  const [inputMode, setInputMode] = useState<InputMode>('camera')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [calibrationStatus, setCalibrationStatus] = useState<MotionCalibrationStatus>('preparing')
  const [calibrationProgress, setCalibrationProgress] = useState(0)
  // 보정이 시작되지 못하는 이유(상반신/손 미검출). 안내 문구를 구체적으로 바꾸는 데만 쓴다.
  const [missingPart, setMissingPart] = useState<MissingPart>(null)
  const missingFramesRef = useRef(0)
  const [cameraCaptureStatus, setCameraCaptureStatus] = useState<CameraCaptureStatus>('idle')

  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)

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

  /** 보정 실패 안내가 떠 있지 않을 때만 '자리 맞추기' 상태로 되돌린다. */
  const markCalibrationPreparing = () => {
    if (!calibrationFailedRef.current && !sentenceDoneRef.current) {
      setCalibrationStatus('preparing')
    }
  }

  // ------------------------------------------------------------
  // 카메라는 유지한 채 모션 보정만 처음부터 다시 시작한다.
  // ------------------------------------------------------------
  const resetMotionCalibration = () => {
    resetCalibrationWindow()
    calibrationFailedRef.current = false
    sentenceDoneRef.current = false
    motionStartThresholdRef.current = MIN_START_THRESHOLD
    motionEndThresholdRef.current = MIN_END_THRESHOLD
    isCalibratedRef.current = false
    prevKeypointsRef.current = null
    startCountRef.current = 0
    missingFramesRef.current = 0
    setMissingPart(null)
    setCalibrationStatus('preparing')
  }

  // ------------------------------------------------------------
  // 한 문장의 수어가 끝나면 보정 상태를 풀고 '번역 완료' 안내만 남긴다.
  // 다음 수어는 '다시 맞추기'(handleRecalibrate)를 눌러 자리 맞추기부터 다시 시작한다.
  // ------------------------------------------------------------
  const lockUntilRecalibration = () => {
    sentenceDoneRef.current = true
    isCalibratedRef.current = false
    prevKeypointsRef.current = null
    startCountRef.current = 0
    missingFramesRef.current = 0
    resetCalibrationWindow()
    setMissingPart(null)
    setCalibrationStatus('done')
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

    cameraFrameIdRef.current = 0
    sentFramesRef.current = 0
  }

  /**
   * 추출 실패/예외 프레임 이후에는 연속 보정을 끊는다.
   * 추출 실패 전후의 프레임을 직접 비교하지 않는다.
   */
  const resetFrameContinuity = () => {
    prevKeypointsRef.current = null
    startCountRef.current = 0
    if (!isCalibratedRef.current) {
      resetCalibrationWindow()
      markCalibrationPreparing()
    }
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
    setCameraCaptureStatus('idle')
    setCameraError(null)
  }

  /** onStreamEnd 호출. 실패해도 카메라 흐름은 계속되도록 오류는 로그만 남긴다. */
  const callStreamEnd = async (logLabel: string) => {
    try {
      await onStreamEnd()
    } catch (error) {
      console.error(`[CAMERA] ${logLabel} error:`, error)
    }
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

    if (!motionActiveRef.current && !streamStartingRef.current) {
      return
    }

    streamEndingRef.current = true

    try {
      await callStreamEnd('stream end')
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
    if (streamEndingRef.current || streamStartingRef.current || !motionActiveRef.current) {
      return
    }

    streamEndingRef.current = true
    motionActiveRef.current = false
    startCountRef.current = 0
    endCountRef.current = 0
    wordBoundarySentRef.current = false
    cooldownRef.current = MANUAL_STOP_COOLDOWN_FRAMES

    setCameraCaptureStatus('idle')
    lockUntilRecalibration()

    console.log('[CAMERA] manual stop - close stream', {
      sentFrames: sentFramesRef.current,
    })

    try {
      await callStreamEnd('manual stop')
    } finally {
      streamEndingRef.current = false
    }
  }

  // ------------------------------------------------------------
  // 카메라 MediaStream 종료
  // ------------------------------------------------------------
  const stopCameraStream = () => {
    stopFrameLoop()

    if (streamRef.current) {
      stopMediaStream(streamRef.current)
      streamRef.current = null
    }

    if (videoRef.current && inputMode === 'camera') {
      videoRef.current.srcObject = null
    }

    resetMotionGate()
    setCameraCaptureStatus('idle')
  }

  // ------------------------------------------------------------
  // "카메라 끄기" 버튼
  // ------------------------------------------------------------
  const handleCameraOff = async () => {
    // AI 스트림이 살아 있는 경우에만 종료
    await endSignStream()

    // 그 다음 카메라 종료
    stopCameraStream()

    // 내부 상태 초기화
    resetStreamingState()

    // 부모의 카메라 상태 변경
    onToggleActive()
  }

  // ------------------------------------------------------------
  // MediaPipe / Keypoint Extractor 초기화
  // ------------------------------------------------------------
  const ensureExtractor = async () => {
    if (!extractorRef.current) {
      extractorRef.current = new BrowserKeypointExtractor()

      await extractorRef.current.init()
    }

    return extractorRef.current
  }

  /**
   * 비수지(NMS)용 얼굴 랜드마크(얼굴 미검출이면 null).
   * NMS는 보조 기능이므로 여기서 어떤 문제가 생겨도 수지 keypoint 추출/전송은
   * 절대 막지 않는다(extractor가 구버전이어도 예외 없이 NMS만 꺼진다).
   */
  const readFaceLandmarks = (extractor: BrowserKeypointExtractor): number[] | null => {
    try {
      if (typeof extractor.getLastFaceLandmarks === 'function') {
        return extractor.getLastFaceLandmarks()
      }
      if (!faceWarnedRef.current) {
        faceWarnedRef.current = true
        console.warn(
          '[CAMERA] extractor에 getLastFaceLandmarks가 없습니다 — 비수지(NMS) 비활성. ' +
            'keypointExtractor.ts가 최신인지, 페이지를 강력 새로고침(Ctrl+Shift+R)했는지 확인하세요.',
        )
      }
    } catch (err) {
      console.warn('[CAMERA] 얼굴 랜드마크 읽기 실패(NMS만 건너뜀):', err)
    }
    return null
  }

  // ------------------------------------------------------------
  // 현재 video에서 keypoint 추출
  // ------------------------------------------------------------
  const extractCurrentVideoKeypoints = async (): Promise<number[] | null> => {
    const video = videoRef.current

    if (!video || video.readyState < 2) {
      return null
    }

    const extractor = await ensureExtractor()
    const keypoints = await extractor.extract(video)
    const landmarks = readFaceLandmarks(extractor)

    lastFaceRef.current = landmarks
      ? { landmarks, width: video.videoWidth, height: video.videoHeight }
      : null

    return keypoints
  }

  // ------------------------------------------------------------
  // 업로드 영상 처리
  // ------------------------------------------------------------
  const resetVideoPlayback = () => {
    setIsVideoPaused(false)
    setVideoDuration(0)
    setVideoCurrentTime(0)
  }

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
    resetVideoPlayback()
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
    resetVideoPlayback()
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
      console.error('[VIDEO] playback error:', error)
    }
  }

  // ------------------------------------------------------------
  // 프레임 분석: 직전 프레임과 비교해 검출 여부와 움직임 점수를 계산한다.
  // ------------------------------------------------------------
  const analyzeFrame = (keypoints: number[], face: FaceFrame | null): AnalyzedFrame => {
    // 실제 카메라 프레임 번호.
    // 전송 여부와 관계없이 증가한다.
    const cameraFrameId = cameraFrameIdRef.current
    cameraFrameIdRef.current += 1

    // 이전 프레임을 덮어쓰기 전에 포즈 검출 여부를 함께 확인한다.
    const prev = prevKeypointsRef.current
    const analyzed: AnalyzedFrame = {
      keypoints,
      face,
      cameraFrameId,
      score: computeMotionScore(prev, keypoints),
      hasPose: hasDetectedPose(keypoints),
      hasHand: hasDetectedAnyHand(keypoints),
      hasPreviousFrame: prev !== null && prev.length === keypoints.length,
      prevHasPose: prev !== null && hasDetectedPose(prev),
      prevHasHand: prev !== null && prev.length >= KEYPOINT_VECTOR_LENGTH && hasDetectedAnyHand(prev),
    }

    prevKeypointsRef.current = keypoints

    return analyzed
  }

  // ------------------------------------------------
  // 자동 캘리브레이션
  // 이 단계에서는 스트림을 시작하거나 프레임을 전송하지 않는다.
  // ------------------------------------------------
  const processCalibrationFrame = (frame: AnalyzedFrame) => {
    const { hasPose, hasHand, hasPreviousFrame, prevHasPose, prevHasHand, score } = frame

    // 문장을 마친 뒤에는 '다시 맞추기'를 누를 때까지 아무것도 하지 않는다.
    if (sentenceDoneRef.current) {
      return
    }

    if (!hasPose || !hasHand) {
      // 상체가 사라지면 연속 샘플과 경과 시간을 모두 다시 센다.
      resetCalibrationWindow()

      // 어느 쪽이 안 보이는지 안내하기 위해 기록한다(잠깐 끊기는 정도로는 문구를 바꾸지 않는다).
      missingFramesRef.current += 1
      if (missingFramesRef.current >= MISSING_NOTICE_FRAMES) {
        setMissingPart(!hasPose ? 'body' : 'hand')
      }

      markCalibrationPreparing()
      return
    }

    if (calibrationFailedRef.current) {
      return
    }

    missingFramesRef.current = 0
    setMissingPart(null)
    setCalibrationStatus('calibrating')

    // 첫 프레임과 미검출 → 재검출 프레임의 점프 값은 제외한다.
    if (!hasPreviousFrame || !prevHasPose || !prevHasHand) {
      return
    }

    const now = performance.now()

    if (calibrationStartedAtRef.current === null) {
      calibrationStartedAtRef.current = now
    }

    calibrationSamplesRef.current.push(score)

    const elapsed = now - calibrationStartedAtRef.current
    const sampleProgress = calibrationSamplesRef.current.length / CALIBRATION_FRAMES
    const timeProgress = elapsed / CALIBRATION_MIN_DURATION_MS
    const progress = Math.min(99, Math.floor(Math.min(sampleProgress, timeProgress) * 100))

    setCalibrationProgress((current) => (current === progress ? current : progress))

    if (calibrationSamplesRef.current.length >= CALIBRATION_FRAMES && elapsed >= CALIBRATION_MIN_DURATION_MS) {
      finishCalibration(computeCalibrationBaseline(calibrationSamplesRef.current))
    }
  }

  const finishCalibration = (baseline: number) => {
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
      return
    }

    motionStartThresholdRef.current = Math.max(MIN_START_THRESHOLD, baseline * START_THRESHOLD_MULTIPLIER)
    motionEndThresholdRef.current = Math.max(MIN_END_THRESHOLD, baseline * END_THRESHOLD_MULTIPLIER)

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

  // ----------------------------------------------
  // 아직 수어 세션이 시작되지 않은 상태
  // ----------------------------------------------
  const processIdleFrame = async (frame: AnalyzedFrame, runId: number) => {
    const { hasPose, hasPreviousFrame, prevHasPose, prevHasHand, score } = frame

    if (cooldownRef.current > 0) {
      cooldownRef.current -= 1
      startCountRef.current = 0
    } else if (!hasPose || !prevHasPose || !prevHasHand || !hasPreviousFrame) {
      // 보정 후에도 퇴장/재입장 점프가 새 세션을 시작하지 않게 한다.
      startCountRef.current = 0
    } else if (score >= motionStartThresholdRef.current) {
      startCountRef.current += 1
    } else {
      startCountRef.current = 0
    }

    // 모션이 일정 프레임 이상 감지되면 AI 스트림 시작
    if (cooldownRef.current !== 0 || startCountRef.current < MOTION_START_FRAMES || streamStartingRef.current) {
      return
    }

    streamStartingRef.current = true
    motionActiveRef.current = true

    startCountRef.current = 0
    endCountRef.current = 0
    wordBoundarySentRef.current = false

    sentFramesRef.current = 0

    setCameraCaptureStatus('recording')

    const started = await onStreamStart()

    if (runId !== watchLoopRunRef.current) {
      return
    }

    streamStartingRef.current = false

    if (!started) {
      motionActiveRef.current = false
      setCameraCaptureStatus('idle')
      return
    }

    console.log('[CAMERA] sending frame', frame.cameraFrameId, 'motion:', score)

    await onKeypointFrame(frame.keypoints, frame.cameraFrameId, false, frame.face)

    sentFramesRef.current += 1
  }

  // --------------------------------------------
  // 수어 세션 중
  //
  // - 움직이면 즉시 frame 전송
  // - 멈추기 시작한 초반 WORD_END_FRAMES 동안 정지 프레임 전송
  // - WORD_END_FRAMES 이후에는 정지 프레임 전송하지 않음
  // - 다시 움직이면 즉시 전송 재개
  // --------------------------------------------
  const processRecordingFrame = async (frame: AnalyzedFrame) => {
    const { keypoints, face, cameraFrameId, hasHand, score } = frame

    const isStill = !hasHand || score <= motionEndThresholdRef.current

    if (isStill) {
      endCountRef.current += 1
    } else {
      endCountRef.current = 0
      wordBoundarySentRef.current = false
    }

    const shouldSendFrame = hasHand && (!isStill || endCountRef.current <= WORD_END_FRAMES)

    if (shouldSendFrame) {
      // frame 전송
      await onKeypointFrame(keypoints, cameraFrameId, isStill, face)

      sentFramesRef.current += 1

      console.log('[CAMERA] frame sent', {
        cameraFrameId,
        motion: score,
        still: isStill,
        stillCount: endCountRef.current,
      })
    } else if (!wordBoundarySentRef.current) {
      // 단어 경계: WORD_END_FRAMES만큼 마지막 정지 프레임을
      // 보낸 뒤 더 이상 정지 프레임을 보내지 않는다.
      wordBoundarySentRef.current = true

      console.log('[CAMERA] motion paused - stop sending frames', {
        cameraFrameId,
        motion: score,
        stillCount: endCountRef.current,
      })

      try {
        await onWordBoundary()
      } catch (error) {
        console.error('[CAMERA] word boundary error:', error)
      }
    }

    // 문장 종료
    //
    // 일정 시간 계속 정지하면 스트림 종료.
    // 정지 구간에서는 frame을 보내지 않고 WebSocket 연결만 유지한다.
    if (!shouldAutoEndSentence(endCountRef.current, sentFramesRef.current)) {
      return
    }

    motionActiveRef.current = false
    endCountRef.current = 0
    wordBoundarySentRef.current = false
    cooldownRef.current = POST_END_COOLDOWN_FRAMES

    setCameraCaptureStatus('idle')
    lockUntilRecalibration()

    console.log('[CAMERA] sentence end - close stream', {
      lastCameraFrameId: cameraFrameId,
      sentFrames: sentFramesRef.current,
    })

    if (!streamEndingRef.current) {
      streamEndingRef.current = true

      try {
        await callStreamEnd('stream end')
      } finally {
        streamEndingRef.current = false
      }
    }
  }

  /** 유효한 keypoint 한 프레임을 현재 단계(보정/대기/녹화)에 맞게 처리한다. */
  const processFrame = async (keypoints: number[], face: FaceFrame | null, runId: number) => {
    const frame = analyzeFrame(keypoints, face)

    if (!isCalibratedRef.current) {
      processCalibrationFrame(frame)
    } else if (!motionActiveRef.current) {
      await processIdleFrame(frame, runId)
    } else {
      await processRecordingFrame(frame)
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

    const isCurrentRun = () => runId === watchLoopRunRef.current

    const handleFrame = async () => {
      try {
        const keypoints = await extractCurrentVideoKeypoints()

        if (!isCurrentRun()) {
          return
        }

        // 이번 프레임의 얼굴(이후 await 동안 다음 프레임으로 덮이지 않도록 지역 상수로 고정)
        const face = lastFaceRef.current

        if (isValidKeypointVector(keypoints)) {
          await processFrame(keypoints, face, runId)
        } else {
          // null/빈 배열/짧은 배열/NaN 프레임도 연속 보정을 끊는다.
          resetFrameContinuity()
        }
      } catch (err) {
        if (!isCurrentRun()) {
          return
        }

        // 예외가 난 구간을 정상 보정 시간/연속 샘플로 계산하지 않는다.
        resetFrameContinuity()

        console.error('[CAMERA] keypoint extraction/send error:', err)

        setCameraError(getErrorMessage(err, '키포인트 추출/전송 실패'))

        // 프레임 전송 중 오류가 발생해도
        // 다음 frame loop는 계속 동작하도록 한다.
      } finally {
        if (isCurrentRun()) {
          isSendingFrameRef.current = false
        }
      }
    }

    const tick = async (time: number) => {
      if (!isCurrentRun()) {
        return
      }

      if (time - lastFrameAtRef.current >= CAMERA_FRAME_INTERVAL_MS && !isSendingFrameRef.current) {
        lastFrameAtRef.current = time
        isSendingFrameRef.current = true
        await handleFrame()
      }

      if (isCurrentRun()) {
        loopRef.current = requestAnimationFrame(tick)
      }
    }

    loopRef.current = requestAnimationFrame(tick)
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

        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('이 브라우저에서는 카메라를 사용할 수 없습니다.')
        }

        const stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS)

        if (cancelled) {
          stopMediaStream(stream)
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
          setCameraError(getErrorMessage(err, '카메라를 연결할 수 없습니다.'))
        }
      }
    }

    startCamera()

    return () => {
      cancelled = true
      stopMediaStream(activeStream)
      stopCameraStream()
    }
  }, [isActive, inputMode])

  // ------------------------------------------------------------
  // 번역 결과의 '다시 인식': 수어가 끝난(번역 완료) 잠금을 풀고 자리 맞추기부터 다시 시작한다.
  // 카메라가 꺼져 있었다면 부모가 카메라를 켜면서 처음부터 보정이 시작된다.
  // ------------------------------------------------------------
  const lastRetryRequestRef = useRef(retryRequestId)

  useEffect(() => {
    if (retryRequestId === lastRetryRequestRef.current) {
      return
    }
    lastRetryRequestRef.current = retryRequestId
    handleRecalibrate()
  }, [retryRequestId])

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

  // ------------------------------------------------------------
  // 업로드 영상에서 camera srcObject 제거
  // ------------------------------------------------------------
  useEffect(() => {
    if (inputMode === 'video' && videoRef.current && videoObjectUrl) {
      videoRef.current.srcObject = null
      videoRef.current.src = videoObjectUrl
      videoRef.current.load()
    }
  }, [inputMode, videoObjectUrl])

  // ------------------------------------------------------------
  // 화면 표시용 파생 값
  // ------------------------------------------------------------
  const isCameraMode = isActive && inputMode === 'camera'
  const isIdleScreen = !isActive && inputMode !== 'video'
  const isRecording = cameraCaptureStatus === 'recording'
  const guideState = { isRecording, isPredicting, calibrationStatus, missingPart }


  const canRecalibrate =
    !isRecording &&
    !isPredicting &&
    (calibrationStatus === 'unstable' || calibrationStatus === 'ready' || calibrationStatus === 'done')

  const progress = Number.isFinite(calibrationProgress) ? Math.min(100, Math.max(0, calibrationProgress)) : 0

  const currentStep = getCurrentStep(guideState)
  const guideText = getGuideText(guideState)
  const statusText = getStatusText(guideState)

  const statusDotClass = isRecording
    ? 'bg-red-400'
    : calibrationStatus === 'ready' || calibrationStatus === 'done'
      ? 'bg-emerald-400'
      : calibrationStatus === 'unstable'
        ? 'bg-amber-300'
        : 'bg-white/60'

  const videoProgressPercent = videoDuration > 0 ? (videoCurrentTime / videoDuration) * 100 : 0

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
        className={`relative w-full ${
          isCameraMode ? 'aspect-[4/3] bg-neutral-950' : 'aspect-video min-h-[340px] bg-brand-emerald sm:min-h-[420px]'
        } rounded-2xl flex flex-col items-center justify-center overflow-hidden border border-emerald-950/20 shadow-inner group ${
          isIdleScreen ? 'cursor-pointer' : ''
        }`}
        onClick={() => {
          if (isIdleScreen) {
            onToggleActive()
          }
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={handleVideoLoadedMetadata}
          onTimeUpdate={handleVideoTimeUpdate}
          onPlay={() => setIsVideoPaused(false)}
          onPause={() => setIsVideoPaused(true)}
          onEnded={() => setIsVideoPaused(true)}
          className={`absolute inset-0 w-full h-full transition-opacity duration-300 ${
            inputMode === 'camera' ? `object-contain${isMirrored ? ' -scale-x-100' : ''}` : 'object-cover'
          } ${isActive || inputMode === 'video' ? 'opacity-100' : 'hidden'}`}
        />

        {/* -------------------------------------------------- */}
        {/* 카메라 시작 화면 */}
        {/* -------------------------------------------------- */}
        {isIdleScreen && (
          <div className="flex flex-col items-center justify-center gap-3 p-6 text-center select-none z-20">
            <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/20 bg-white/10 text-white shadow-lg sm:h-20 sm:w-20">
              <Hand className="h-8 w-8 -rotate-12 animate-bounce text-white motion-reduce:animate-none sm:h-10 sm:w-10" />
            </div>

            <h3 className="break-keep text-2xl font-bold text-white sm:text-3xl">화면을 눌러 수어 번역 시작하기</h3>

            <p className="max-w-md break-keep text-base leading-relaxed text-white/90 sm:text-lg">
              카메라가 켜지면 화면의 안내를 따라 주세요. 아래에 사용 방법이 있어요.
            </p>

            <span className="mt-1 inline-flex min-h-[48px] items-center gap-2 rounded-full bg-white px-6 py-3 text-base font-bold text-neutral-950 shadow-lg sm:text-lg">
              <Camera aria-hidden="true" className="h-5 w-5" />
              카메라 켜기
            </span>

            {cameraError && (
              <div className="max-w-md rounded-lg bg-red-500/85 px-4 py-3 text-left text-sm font-semibold leading-relaxed text-white sm:text-base">
                <p>{cameraError}</p>
                <p className="mt-1 font-normal text-white/90">
                  브라우저의 카메라 권한을 허용했는지, 다른 프로그램이 카메라를 쓰고 있지 않은지 확인해 주세요.
                </p>
              </div>
            )}
          </div>
        )}

        {/* -------------------------------------------------- */}
        {/* 카메라 HUD */}
        {/* -------------------------------------------------- */}
        {isCameraMode && (
          <>
            <div className="pointer-events-none absolute inset-x-4 top-4 z-20 flex items-center justify-between gap-3">
              <div className="inline-flex min-w-0 items-center gap-2 rounded-lg bg-neutral-950/80 px-3 py-2 backdrop-blur-sm">
                <span aria-hidden="true" className={`h-2.5 w-2.5 shrink-0 rounded-full ${statusDotClass}`} />
                <span className="truncate text-sm font-semibold leading-5 text-white sm:text-base">{statusText}</span>
              </div>

              <span className="shrink-0 rounded-lg bg-neutral-950/80 px-3 py-2 text-xs font-medium leading-4 text-white/90 backdrop-blur-sm sm:text-sm">
                상반신 + 두 손
              </span>
            </div>

            {/* 상반신 가이드는 카메라가 켜져 있는 동안 항상 표시한다. */}
            <PositionGuideOverlay state={guideState} />

            {/* 준비 완료/인식 중에는 가이드 안내 문구 대신 짧은 안내만 보여 수어 동작을 가리지 않는다. */}
            {!cameraError && !isPredicting && (calibrationStatus === 'ready' || isRecording) && (
              <div className="pointer-events-none absolute inset-x-4 bottom-20 z-20 flex justify-center">
                <p className="rounded-full bg-neutral-950/80 px-4 py-2.5 text-center text-sm font-semibold leading-relaxed text-white backdrop-blur-sm sm:text-base">
                  {isRecording ? '두 손을 화면 안에 유지해 주세요' : '손을 움직이면 인식이 시작돼요'}
                </p>
              </div>
            )}

            {/* 오류 표시 */}
            {cameraError && (
              <div className="absolute top-16 left-4 right-4 z-30 pointer-events-none">
                <div className={ERROR_BOX_CLASS}>{cameraError}</div>
              </div>
            )}

            {/* 종료 버튼은 한 줄에 배치해 작은 화면에서도 겹치지 않게 한다. */}
            <div className="absolute inset-x-4 bottom-4 z-50 flex items-center justify-end gap-3">
              {isRecording && (
                <button
                  type="button"
                  onClick={isolatedClick(() => void handleManualStop())}
                  className="inline-flex min-h-[52px] flex-1 items-center justify-center gap-2 rounded-full border border-white/30 bg-red-500 px-5 py-3 text-white shadow-lg ring-4 ring-red-500/40 transition-colors hover:bg-red-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white motion-reduce:transition-none sm:flex-none sm:px-8"
                >
                  <Square aria-hidden="true" className="h-5 w-5 fill-current" />
                  <span className="text-base font-bold sm:text-lg">수어 종료</span>
                </button>
              )}

              <button
                type="button"
                onClick={isolatedClick(() => void handleCameraOff())}
                className="inline-flex min-h-[48px] shrink-0 items-center justify-center gap-2 rounded-full border border-white/20 bg-black/70 px-4 py-2.5 text-white backdrop-blur-md transition-colors hover:bg-red-500/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white motion-reduce:transition-none"
              >
                <VideoOff aria-hidden="true" className="h-5 w-5" />
                <span className="text-sm font-bold sm:text-base">카메라 끄기</span>
              </button>
            </div>
          </>
        )}

        {/* -------------------------------------------------- */}
        {/* 영상 업로드 모드 */}
        {/* -------------------------------------------------- */}
        {inputMode === 'video' && videoObjectUrl && (
          <>
            <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-30 pointer-events-none">
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <Film className="w-4 h-4 text-brand-green" />
                <span className="text-sm font-bold text-white">업로드 영상</span>
              </div>

              <button
                type="button"
                onClick={isolatedClick(clearVideoFile)}
                className="pointer-events-auto flex items-center justify-center w-8 h-8 rounded-full bg-black/60 hover:bg-red-500/90 text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="absolute bottom-4 left-4 right-4 z-30 flex items-center gap-3">
              <button
                type="button"
                onClick={isolatedClick(() => void toggleVideoPlayback())}
                className="flex items-center justify-center w-9 h-9 rounded-full bg-black/70 hover:bg-black/90 text-white"
              >
                {isVideoPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              </button>

              <div className="flex-1 h-1.5 rounded-full bg-white/20 overflow-hidden">
                <div className="h-full bg-brand-green transition-all" style={{ width: `${videoProgressPercent}%` }} />
              </div>

              <span className="text-xs font-mono text-white whitespace-nowrap">
                {Math.floor(videoCurrentTime)}/{Math.floor(videoDuration)}
              </span>
            </div>
          </>
        )}

        {/* -------------------------------------------------- */}
        {/* 영상 선택 버튼 */}
        {/* -------------------------------------------------- */}
        {isIdleScreen && (
          <button
            type="button"
            onClick={isolatedClick(() => fileInputRef.current?.click())}
            className="absolute bottom-4 left-4 z-30 flex items-center gap-2 bg-black/60 hover:bg-black/80 backdrop-blur-md px-3.5 py-2 rounded-full border border-white/20 text-white transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span className="text-sm font-bold">영상 업로드</span>
          </button>
        )}

        {/* -------------------------------------------------- */}
        {/* 카메라가 꺼져 있을 때 오류 메시지 */}
        {/* -------------------------------------------------- */}
        {isIdleScreen && cameraError && (
          <div className="absolute bottom-16 left-4 right-4 z-30">
            <div className={ERROR_BOX_CLASS}>{cameraError}</div>
          </div>
        )}
      </div>

      {/* 안내는 영상 아래에 배치해 얼굴과 손을 가리지 않는다. */}
      {isCameraMode && !cameraError && (
        <div className="mt-3 rounded-2xl border border-neutral-800 bg-neutral-950 px-5 py-5 text-white sm:px-7 sm:py-6">
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row">
            <div role="status" aria-live="polite" aria-atomic="true" className="min-w-0 flex-1">
              <h3 className="break-keep text-xl font-bold leading-snug sm:text-2xl">{guideText.title}</h3>
              <p className="mt-2 break-keep text-base leading-relaxed text-white/90 sm:text-lg">
                {guideText.description}
              </p>
            </div>

            {canRecalibrate && (
              <button
                type="button"
                aria-label={calibrationStatus === 'done' ? '다음 수어 다시 하기' : '카메라 움직임 다시 맞추기'}
                onClick={isolatedClick(handleRecalibrate)}
                className={`inline-flex min-h-[48px] shrink-0 items-center justify-center rounded-lg px-5 text-base font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white motion-reduce:transition-none ${
                  calibrationStatus === 'unstable'
                    ? 'bg-amber-400 text-neutral-950 hover:bg-amber-300'
                    : calibrationStatus === 'done'
                      ? 'bg-white text-neutral-950 hover:bg-white/90'
                      : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                {calibrationStatus === 'done' ? '다시 수어하기' : '다시 맞추기'}
              </button>
            )}
          </div>

          {calibrationStatus === 'calibrating' && !isRecording && !isPredicting && (
            <div className="mt-5">
              <div className="mb-2 flex items-center justify-between text-sm font-medium text-white/85 sm:text-base">
                <span>가만히 계세요</span>
                <span aria-hidden="true">{Math.round(progress)}%</span>
              </div>
              <div
                role="progressbar"
                aria-label="수어 인식 준비 진행률"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(progress)}
                className="h-2.5 overflow-hidden rounded-full bg-white/20"
              >
                <div
                  className="h-full rounded-full bg-white/90 transition-[width] duration-150 motion-reduce:transition-none"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* 단계 표시는 설명 아래에 둔다. */}
          <div className="mt-5 border-t border-white/10 pt-5">
            <GuideSteps current={currentStep} />
          </div>
        </div>
      )}

      {/* 카메라가 꺼져 있을 때는 처음 쓰는 사람을 위한 사용 방법을 보여준다. */}
      {isIdleScreen && <HowToCard />}

      {/* 업로드 영상 파일명 */}
      {videoFile && inputMode === 'video' && (
        <div className="mt-3 flex items-center justify-between px-2">
          <div className="flex items-center gap-2 min-w-0">
            <Film className="w-4 h-4 text-brand-green shrink-0" />
            <span className="text-base text-neutral-700 truncate">{videoFile.name}</span>
          </div>

          <button
            type="button"
            onClick={clearVideoFile}
            className="text-sm text-neutral-600 hover:text-red-500 transition-colors"
          >
            제거
          </button>
        </div>
      )}
    </div>
  )
}
