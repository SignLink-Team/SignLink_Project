import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { api } from './api'
import { CameraView } from './components/CameraView'
import { Header } from './components/Header'
import { HistoryView } from './components/HistoryView'
import { InstructionsModal } from './components/InstructionsModal'
import { LoginView } from './components/LoginView'
import { TranslationResult } from './components/TranslationResult'
import { FaceFrame, NmsFrame, NmsSegment } from './nms'
import {
  ApiSession,
  ApiTranslation,
  ApiUser,
  CameraGuideVariant,
  TranslationCandidate,
  TranslationLog,
  UserState,
  ViewState,
} from './types'
import { asArray, getErrorMessage, safeStorage } from './utils'

const emptyUser: UserState = {
  isLoggedIn: false,
  id: null,
  medical_id: null,
  email: null,
  name: null,
  role: null,
}

const WS_BASE = import.meta.env.VITE_WS_BASE || `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`
// WebSocket 연결 / 스트림 시작 응답 대기 시간
const WS_RESPONSE_TIMEOUT_MS = 8000

const STORAGE_KEY_AUTOSAVE = 'signlink_autosave'
const STORAGE_KEY_NMS_COLLAPSED = 'signlink_nms_collapsed'
const STORAGE_KEY_CAMERA_MIRRORED = 'signlink_camera_mirrored'
const STORAGE_KEY_CAMERA_GUIDE_VARIANT = 'signlink_camera_guide_variant'

const SESSION_TITLE = '실시간 수어 진료'
const DEFAULT_PATIENT_ID = 'none'

const CATEGORY_RULES: Array<[string, string[]]> = [
  ['두통', ['머리', '두통', '어지']],
  ['호흡기', ['기침', '숨', '호흡', '가래']],
  ['전신/감기', ['열', '감기', '몸살', '춥']],
  ['소화기', ['배', '복부', '속', '소화', '구토']],
  ['근골격계', ['허리', '팔', '다리', '관절', '통증']],
  ['알레르기', ['알레르기', '두드러기', '가려']],
  ['이비인후과', ['목', '귀', '코', '삼키']],
]

function mapApiLog(item: ApiTranslation): TranslationLog {
  return {
    log_id: item.log_id,
    medical_id: item.medical_id,
    patient_id: item.patient_id || DEFAULT_PATIENT_ID,
    input_time: item.input_time,
    gloss_result: item.gloss_result || '',
    translated_text: item.translated_text,
    confidence: item.confidence ?? 0,
    category: item.category,
  }
}

function toUserState(me: ApiUser): UserState {
  return {
    isLoggedIn: true,
    id: me.id,
    medical_id: me.medical_id,
    email: me.email,
    name: me.name,
    role: me.role,
  }
}

function inferCategory(text: string) {
  return CATEGORY_RULES.find(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))?.[0] || '기타'
}

/** 서버 WebSocket 메시지. 형식이 보장되지 않으므로 사용하는 쪽에서 방어적으로 읽는다. */
type SocketMessage = Record<string, any>

function parseSocketMessage(raw: unknown): SocketMessage | null {
  if (typeof raw !== 'string') return null
  try {
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : null
  } catch {
    return null
  }
}

export default function App() {
  const [currentView, setCurrentView] = useState<ViewState>('main')
  const [user, setUser] = useState<UserState>(emptyUser)
  const [autoSave, setAutoSave] = useState(() => safeStorage.get(STORAGE_KEY_AUTOSAVE) !== 'false')
  const [logs, setLogs] = useState<TranslationLog[]>([])
  const [session, setSession] = useState<ApiSession | null>(null)
  const [activeTranslation, setActiveTranslation] = useState<string | null>(null)
  const [activeGlossResult, setActiveGlossResult] = useState('')
  const [translationCandidates, setTranslationCandidates] = useState<TranslationCandidate[]>([])
  const [activeConfidence, setActiveConfidence] = useState(0)
  const [lastLogId, setLastLogId] = useState<number | null>(null)
  const [cameraActive, setCameraActive] = useState(false)
  // '다시 인식'을 누를 때마다 증가시켜 CameraView가 보정부터 다시 시작하게 한다.
  const [retryRequestId, setRetryRequestId] = useState(0)
  // 카메라 화면 좌우 반전(도움말 창에서 변경). 새로고침해도 유지
  const [isCameraMirrored, setIsCameraMirrored] = useState(() => safeStorage.get(STORAGE_KEY_CAMERA_MIRRORED) === 'true')
  // 카메라 상반신 가이드 모양(도움말 창에서 변경). 새로고침해도 유지
  const [cameraGuideVariant, setCameraGuideVariant] = useState<CameraGuideVariant>(() =>
    safeStorage.get(STORAGE_KEY_CAMERA_GUIDE_VARIANT) === 'silhouette' ? 'silhouette' : 'basic',
  )
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // 비수지(NMS) 실시간 결과와 세션 종료 시 구간. (현재 화면에는 표시하지 않지만 수신 상태는 유지한다)
  const [, setNmsFrame] = useState<NmsFrame | null>(null)
  const [, setNmsSegments] = useState<NmsSegment[]>([])
  // 녹화 중 실시간 인식 단어(확정 단어 + 아직 확정 전인 후보). 최종 번역이 오면 비운다.
  const [, setLiveWords] = useState<string[]>([])
  const [, setLivePartial] = useState<string | null>(null)
  // 비수지 패널 접힘(표시만 숨김. 인식/LLM 전달은 그대로). 새로고침해도 유지
  const [nmsCollapsed] = useState(() => safeStorage.get(STORAGE_KEY_NMS_COLLAPSED) === 'true')
  const wsRef = useRef<WebSocket | null>(null)
  const activeSessionRef = useRef<ApiSession | null>(null)
  const pendingStreamStartRef = useRef<((ok: boolean) => void) | null>(null)

  const isDoctor = user.isLoggedIn && user.role === 'doctor'

  useEffect(() => {
    async function loadMe() {
      if (!api.getToken()) return
      try {
        setUser(toUserState(await api.me()))
      } catch {
        api.clearToken()
        setUser(emptyUser)
      }
    }
    loadMe()
  }, [])

  useEffect(() => {
    safeStorage.set(STORAGE_KEY_AUTOSAVE, String(autoSave))
  }, [autoSave])

  useEffect(() => {
    safeStorage.set(STORAGE_KEY_NMS_COLLAPSED, String(nmsCollapsed))
  }, [nmsCollapsed])

  useEffect(() => {
    safeStorage.set(STORAGE_KEY_CAMERA_MIRRORED, String(isCameraMirrored))
  }, [isCameraMirrored])

  useEffect(() => {
    safeStorage.set(STORAGE_KEY_CAMERA_GUIDE_VARIANT, cameraGuideVariant)
  }, [cameraGuideVariant])

  // ------------------------------------------------------------
  // 공통 상태 초기화
  // ------------------------------------------------------------

  /** 화면에 표시 중인 번역 결과(문장/글로스/후보)를 비운다. */
  const clearDisplayedTranslation = () => {
    setActiveTranslation(null)
    setActiveGlossResult('')
    setTranslationCandidates([])
  }

  /** 녹화 중 실시간 인식 단어 영역을 비운다. */
  const clearLiveGloss = () => {
    setLiveWords([])
    setLivePartial(null)
  }

  /** "start" 응답을 기다리는 쪽이 있으면 결과를 알려주고 대기를 해제한다. */
  const resolvePendingStreamStart = (ok: boolean) => {
    const resolve = pendingStreamStartRef.current
    if (!resolve) return
    pendingStreamStartRef.current = null
    resolve(ok)
  }

  /** 열린 소켓과 진행 중 세션이 모두 있을 때만 돌려준다. */
  const getOpenStream = () => {
    const socket = wsRef.current
    const activeSession = activeSessionRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN || !activeSession) return null
    return { socket, activeSession }
  }

  // ------------------------------------------------------------
  // 번역 기록
  // ------------------------------------------------------------

  const loadLogs = useCallback(async () => {
    if (!isDoctor) {
      setLogs([])
      return
    }
    const data = await api.listTranslations({ limit: 100 })
    setLogs(data.map(mapApiLog))
  }, [isDoctor])

  const reloadLogs = useCallback(() => {
    loadLogs().catch((err) => setError(getErrorMessage(err, '번역 기록을 불러오지 못했습니다.')))
  }, [loadLogs])

  useEffect(() => {
    reloadLogs()
  }, [reloadLogs])

  const ensureSession = useCallback(async () => {
    if (activeSessionRef.current) return activeSessionRef.current
    if (session) {
      activeSessionRef.current = session
      return session
    }
    const created = await api.createSession({ title: SESSION_TITLE })
    activeSessionRef.current = created
    setSession(created)
    return created
  }, [session])

  const saveTranslationIfNeeded = useCallback(
    async (text: string, category: string, glossResult: string, confidence: number) => {
      setLastLogId(null)
      if (!autoSave || !isDoctor) return null

      const activeSession = await ensureSession()
      const saved = await api.createTranslation({
        session_id: activeSession.id,
        patient_id: DEFAULT_PATIENT_ID,
        gloss_result: glossResult,
        translated_text: text,
        confidence,
        category,
      })
      const mapped = mapApiLog(saved)
      setLogs((prev) => [mapped, ...prev.filter((log) => log.log_id !== mapped.log_id)])
      setLastLogId(mapped.log_id)
      return mapped
    },
    [autoSave, ensureSession, isDoctor],
  )

  const handleCompletedTranslation = useCallback(
    async (text: string, category = inferCategory(text), glossResult = text, confidence = 0.98) => {
      setError(null)
      setActiveTranslation(text)
      setActiveGlossResult(glossResult)
      setTranslationCandidates([])
      setActiveConfidence(confidence)
      try {
        await saveTranslationIfNeeded(text, category, glossResult, confidence)
      } catch (err) {
        setError(getErrorMessage(err, '번역 기록 저장에 실패했습니다.'))
      }
    },
    [saveTranslationIfNeeded],
  )

  // ------------------------------------------------------------
  // /ws/translate 소켓
  // ------------------------------------------------------------

  const handleSocketMessage = (message: SocketMessage) => {
    switch (message.type) {
      case 'error':
        setError(message.message || 'WebSocket 처리 중 오류가 발생했습니다.')
        setIsPredicting(false)
        clearLiveGloss()
        resolvePendingStreamStart(false)
        return

      case 'stream_started':
        setError(null)
        resolvePendingStreamStart(true)
        return

      // 비수지(NMS) 실시간 결과: 8개 키의 smoothed 확률과 임계값 초과 여부
      case 'nms':
        setNmsFrame({
          frameId: Number(message.frame_id ?? 0),
          probs: asArray<number>(message.probs),
          smoothed: asArray<number>(message.smoothed),
          active: asArray<boolean>(message.active),
        })
        return

      // word_boundary 응답: 아직 문장이 끝난 게 아니라 단어 하나가 막 인식된 상태.
      // (백엔드가 이 타입을 안 보내면 이 블록은 그냥 안 타므로 무해함)
      case 'partial': {
        const words = asArray(message.words)
        setActiveGlossResult(words.join(' '))
        // 실시간 표시는 TranslationResult(최종 문장이 있어야 그려짐)와 별개로 전용 영역에 그린다.
        setLiveWords(words.map((word) => String(word)))
        setLivePartial(typeof message.partial_gloss === 'string' && message.partial_gloss ? message.partial_gloss : null)
        return
      }

      case 'translation': {
        const words = asArray(message.words)
        const glossResult = message.gloss_result || words.join(' ')
        const text = message.text || glossResult || '인식된 수어가 없습니다.'

        setActiveTranslation(text)
        setActiveGlossResult(glossResult)
        setTranslationCandidates(asArray<TranslationCandidate>(message.translation_candidates))
        setActiveConfidence(Number(message.confidence ?? 0))
        setIsPredicting(false)
        setNmsSegments(asArray<NmsSegment>(message.nms_segments))
        // 최종 문장이 도착했으니 실시간 단어 영역은 비운다(전체 문장으로 교체)
        clearLiveGloss()

        if (message.log_id) {
          setLastLogId(Number(message.log_id))
          reloadLogs()
        }
        return
      }

      // 최초 접속 시 백엔드가 보내는 {"type": "connected"} 등은 별도 처리 없이 무시한다.
      default:
        return
    }
  }

  const ensureTranslationSocket = useCallback(async (): Promise<WebSocket | null> => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current
    }

    if (!isDoctor) {
      setCurrentView('login')
      setError('AI 예측을 사용하려면 의료진 계정으로 로그인해야 합니다.')
      return null
    }

    const token = api.getToken()
    if (!token) {
      setCurrentView('login')
      return null
    }

    wsRef.current?.close()

    const socket = new WebSocket(`${WS_BASE}/ws/translate?token=${encodeURIComponent(token)}`)
    wsRef.current = socket

    // 소켓 생명주기 동안 딱 한 번만 붙는 영구 핸들러.
    // 문장이 몇 번을 반복되든 이 핸들러 하나로 계속 처리한다.
    socket.onmessage = (event) => {
      const message = parseSocketMessage(event.data)
      if (message) handleSocketMessage(message)
    }

    socket.onclose = () => {
      if (wsRef.current === socket) {
        wsRef.current = null
      }
      resolvePendingStreamStart(false)
    }

    return await new Promise<WebSocket | null>((resolve) => {
      const timeout = window.setTimeout(() => {
        setError('백엔드 WebSocket 연결 시간이 초과되었습니다.')
        resolve(null)
      }, WS_RESPONSE_TIMEOUT_MS)

      socket.onopen = () => {
        window.clearTimeout(timeout)
        resolve(socket)
      }

      socket.onerror = () => {
        window.clearTimeout(timeout)
        setError('백엔드 WebSocket에 연결할 수 없습니다.')
        resolve(null)
      }
    })
    // handleSocketMessage는 state setter와 ref, reloadLogs만 사용하므로 reloadLogs만 의존성에 둔다.
  }, [reloadLogs, isDoctor])

  // 소켓은 이미 열려 있다는 전제 하에, 이번 문장을 시작한다(session_id 확보 + "start" 전송
  // + "stream_started" 응답 대기). 소켓 자체를 새로 만들지 않으므로 여러 문장에 걸쳐 재사용 가능.
  const startStream = useCallback(async (): Promise<boolean> => {
    const socket = await ensureTranslationSocket()
    const activeSession = await ensureSession()
    if (!socket) return false
    activeSessionRef.current = activeSession

    return await new Promise<boolean>((resolve) => {
      const timeout = window.setTimeout(() => {
        pendingStreamStartRef.current = null
        setError('세션 시작 응답 시간이 초과되었습니다.')
        resolve(false)
      }, WS_RESPONSE_TIMEOUT_MS)

      pendingStreamStartRef.current = (ok: boolean) => {
        window.clearTimeout(timeout)
        resolve(ok)
      }

      socket.send(JSON.stringify({ type: 'start', session_id: activeSession.id }))
    })
  }, [ensureSession, ensureTranslationSocket])

  // ------------------------------------------------------------
  // CameraView 콜백
  // ------------------------------------------------------------

  const handleStreamStart = async () => {
    setIsPredicting(false)
    setNmsFrame(null)
    setNmsSegments([])
    // 새 녹화가 시작되면 이전 결과와 실시간 표시를 비워, 이번 녹화의 실시간 단어가 먼저 보이게 한다
    clearLiveGloss()
    clearDisplayedTranslation()
    setLastLogId(null)
    return await startStream()
  }

  const handleKeypointFrame = async (keypoints: number[], frameId: number, still: boolean, face?: FaceFrame | null) => {
    const stream = getOpenStream()
    if (!stream) return
    stream.socket.send(
      JSON.stringify({
        type: 'frame',
        session_id: stream.activeSession.id,
        frame_id: frameId,
        keypoints,
        still,
        // 비수지(NMS)용 얼굴 랜드마크. 없으면 필드 자체를 보내지 않는다.
        ...(face ? { face: face.landmarks, face_w: face.width, face_h: face.height } : {}),
      }),
    )
  }

  const handleWordBoundary = async () => {
    const stream = getOpenStream()
    if (!stream) return
    stream.socket.send(JSON.stringify({ type: 'word_boundary', session_id: stream.activeSession.id }))
  }

  const handleStreamEnd = async () => {
    const stream = getOpenStream()
    if (!stream) return
    setIsPredicting(true)
    stream.socket.send(JSON.stringify({ type: 'end', session_id: stream.activeSession.id, auto_save: autoSave }))
  }

  const handleToggleCamera = () => {
    setCameraActive((active) => {
      const next = !active
      if (!next) {
        // 카메라를 끄는 순간에만 /ws/translate 연결도 정리한다.
        // (카메라 켜져있는 동안에는 문장이 몇 번 반복되든 소켓을 계속 재사용함)
        wsRef.current?.close()
        wsRef.current = null
        activeSessionRef.current = null
      }
      return next
    })
    clearDisplayedTranslation()
    clearLiveGloss()
  }

  const handleTriggerTranslation = async (text: string, category: string) => {
    await handleCompletedTranslation(text, category, text)
  }

  // ------------------------------------------------------------
  // 인증
  // ------------------------------------------------------------

  const handleLoginSuccess = async (email: string, password: string) => {
    setError(null)
    const { access_token: token } = await api.login({ email, password })
    api.setToken(token)
    setUser(toUserState(await api.me()))
    setAutoSave(true)
    setCurrentView('main')
  }

  const handleRegister = async (name: string, email: string, password: string) => {
    setError(null)
    await api.register({ name, email, password, role: 'doctor' })
    await handleLoginSuccess(email, password)
  }

  const handleLogout = () => {
    wsRef.current?.close()
    api.clearToken()
    setUser(emptyUser)
    setAutoSave(false)
    clearDisplayedTranslation()
    setCameraActive(false)
    setSession(null)
    activeSessionRef.current = null
    setLogs([])
    setCurrentView('main')
  }

  // ------------------------------------------------------------
  // TranslationResult 콜백
  // ------------------------------------------------------------

  const handleSelectTranslation = async (text: string, category: string, confidence = activeConfidence) => {
    await handleCompletedTranslation(text, category, activeGlossResult || text, confidence)
  }

  const handleRetryTranslation = async () => {
    if (lastLogId) {
      try {
        await api.deleteTranslation(lastLogId)
        setLogs((prev) => prev.filter((log) => log.log_id !== lastLogId))
      } catch {
        // The history screen still allows manual cleanup if this request fails.
      }
      setLastLogId(null)
    }
    clearDisplayedTranslation()
    setCameraActive(true)
    setRetryRequestId((id) => id + 1)
  }

  const handleClearTranslation = () => {
    clearDisplayedTranslation()
    setLastLogId(null)
  }

  // ------------------------------------------------------------
  // HistoryView 콜백
  // ------------------------------------------------------------

  const handleDeleteLog = async (logId: number) => {
    await api.deleteTranslation(logId)
    setLogs((prev) => prev.filter((log) => log.log_id !== logId))
  }

  const handleClearLogs = async () => {
    await api.clearTranslations()
    setLogs([])
  }

  const handleAddManualLog = async (text: string, category: string) => {
    const saved = await api.createTranslation({
      patient_id: DEFAULT_PATIENT_ID,
      translated_text: text,
      gloss_result: '직접 입력',
      confidence: 1,
      category,
    })
    setLogs((prev) => [mapApiLog(saved), ...prev])
  }

  const handleUpdatePatientId = async (logId: number, patientId: string) => {
    const saved = await api.updateTranslation(logId, { patient_id: patientId || DEFAULT_PATIENT_ID })
    const mapped = mapApiLog(saved)
    setLogs((prev) => prev.map((log) => (log.log_id === logId ? mapped : log)))
  }

  return (
    <div className="min-h-screen bg-bg-base text-on-surface flex flex-col justify-between antialiased selection:bg-primary-light selection:text-primary font-sans">
      <Header
        user={user}
        onLoginClick={() => setCurrentView('login')}
        onLogoutClick={handleLogout}
        autoSave={autoSave}
        onAutoSaveToggle={() => user.isLoggedIn && setAutoSave((value) => !value)}
        onHelpClick={() => setIsHelpOpen(true)}
        onLogoClick={() => setCurrentView('main')}
      />

      <main className="flex-1 w-full flex justify-center p-4 sm:p-6 md:p-8">
        <div className="w-full max-w-7xl">
          {error && (
            <div className="max-w-3xl mx-auto mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-bold text-red-700">
              {error}
            </div>
          )}

          <AnimatePresence mode="wait">
            {currentView === 'main' && (
              <motion.div
                key="main-view"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
                className="w-full flex flex-col items-center gap-6"
              >
                <div className="w-full max-w-3xl flex flex-col items-center">
                  <CameraView
                    isActive={cameraActive}
                    onToggleActive={handleToggleCamera}
                    onTriggerTranslation={handleTriggerTranslation}
                    onStreamStart={handleStreamStart}
                    onKeypointFrame={handleKeypointFrame}
                    onWordBoundary={handleWordBoundary}
                    onStreamEnd={handleStreamEnd}
                    isPredicting={isPredicting}
                    retryRequestId={retryRequestId}
                    isMirrored={isCameraMirrored}
                    guideVariant={cameraGuideVariant}
                  />

                  <TranslationResult
                    text={activeTranslation}
                    glossResult={activeGlossResult}
                    candidates={translationCandidates}
                    onViewHistory={() => setCurrentView(user.isLoggedIn ? 'history' : 'login')}
                    isLoggedIn={user.isLoggedIn}
                    autoSave={autoSave}
                    onRetry={handleRetryTranslation}
                    onSelectTranslation={handleSelectTranslation}
                    onClearTranslation={handleClearTranslation}
                  />
                </div>
              </motion.div>
            )}

            {currentView === 'history' && (
              <HistoryView
                key="history-view"
                logs={logs}
                onBack={() => setCurrentView('main')}
                onClearLogs={handleClearLogs}
                onDeleteLog={handleDeleteLog}
                onAddManualLog={handleAddManualLog}
                onUpdatePatientId={handleUpdatePatientId}
              />
            )}

            {(currentView === 'login' || currentView === 'register') && (
              <LoginView
                key="login-view"
                mode={currentView}
                onLoginSuccess={handleLoginSuccess}
                onRegister={handleRegister}
                onSwitchMode={() => setCurrentView(currentView === 'login' ? 'register' : 'login')}
                onBackToMain={() => setCurrentView('main')}
              />
            )}
          </AnimatePresence>
        </div>
      </main>

      <footer className="w-full py-4 border-t border-neutral-250 bg-white text-center text-[11px] text-neutral-400 font-medium">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2.5">
          <p>© 2026 SignLink. All rights reserved.</p>
        </div>
      </footer>

      <InstructionsModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        isLoggedIn={user.isLoggedIn}
        isCameraMirrored={isCameraMirrored}
        onToggleCameraMirrored={() => setIsCameraMirrored((value) => !value)}
        cameraGuideVariant={cameraGuideVariant}
        onChangeCameraGuideVariant={setCameraGuideVariant}
      />
    </div>
  )
}
