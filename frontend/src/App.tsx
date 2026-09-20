import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Activity, Info } from 'lucide-react'
import { api } from './api'
import { CameraView } from './components/CameraView'
import { Header } from './components/Header'
import { HistoryView } from './components/HistoryView'
import { InstructionsModal } from './components/InstructionsModal'
import { LoginView } from './components/LoginView'
import { TranslationResult } from './components/TranslationResult'
import { ApiSession, ApiTranslation, TranslationCandidate, TranslationLog, UserState, ViewState } from './types'

const emptyUser: UserState = {
  isLoggedIn: false,
  id: null,
  medical_id: null,
  email: null,
  name: null,
  role: null,
}

const WS_BASE = import.meta.env.VITE_WS_BASE || `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`

function mapApiLog(item: ApiTranslation): TranslationLog {
  return {
    log_id: item.log_id,
    medical_id: item.medical_id,
    patient_id: item.patient_id || 'none',
    input_time: item.input_time,
    gloss_result: item.gloss_result || '',
    translated_text: item.translated_text,
    confidence: item.confidence ?? 0,
    category: item.category,
  }
}

function inferCategory(text: string) {
  const rules: Array<[string, string[]]> = [
    ['두통', ['머리', '두통', '어지']],
    ['호흡기', ['기침', '숨', '호흡', '가래']],
    ['전신/감기', ['열', '감기', '몸살', '춥']],
    ['소화기', ['배', '복부', '속', '소화', '구토']],
    ['근골격계', ['허리', '팔', '다리', '관절', '통증']],
    ['알레르기', ['알레르기', '두드러기', '가려']],
    ['이비인후과', ['목', '귀', '코', '삼키']],
  ]
  return rules.find(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))?.[0] || '기타'
}

export default function App() {
  const [currentView, setCurrentView] = useState<ViewState>('main')
  const [user, setUser] = useState<UserState>(emptyUser)
  const [autoSave, setAutoSave] = useState(() => localStorage.getItem('signlink_autosave') !== 'false')
  const [logs, setLogs] = useState<TranslationLog[]>([])
  const [session, setSession] = useState<ApiSession | null>(null)
  const [activeTranslation, setActiveTranslation] = useState<string | null>(null)
  const [activeGlossResult, setActiveGlossResult] = useState('')
  const [translationCandidates, setTranslationCandidates] = useState<TranslationCandidate[]>([])
  const [activeConfidence, setActiveConfidence] = useState(0)
  const [lastLogId, setLastLogId] = useState<number | null>(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const activeSessionRef = useRef<ApiSession | null>(null)
  const pendingStreamStartRef = useRef<((ok: boolean) => void) | null>(null)
  useEffect(() => {
    async function loadMe() {
      if (!api.getToken()) return
      try {
        const me = await api.me()
        setUser({
          isLoggedIn: true,
          id: me.id,
          medical_id: me.medical_id,
          email: me.email,
          name: me.name,
          role: me.role,
        })
      } catch {
        api.clearToken()
        setUser(emptyUser)
      }
    }
    loadMe()
  }, [])

  useEffect(() => {
    localStorage.setItem('signlink_autosave', String(autoSave))
  }, [autoSave])

  const loadLogs = useCallback(async () => {
    if (!user.isLoggedIn || user.role !== 'doctor') {
      setLogs([])
      return
    }
    const data = await api.listTranslations({ limit: 100 })
    setLogs(data.map(mapApiLog))
  }, [user.isLoggedIn, user.role])

  useEffect(() => {
    loadLogs().catch((err) => setError(err.message))
  }, [loadLogs])

  const ensureSession = useCallback(async () => {
    if (activeSessionRef.current) return activeSessionRef.current
    if (session) {
      activeSessionRef.current = session
      return session
    }
    const created = await api.createSession({ title: '실시간 수어 진료' })
    activeSessionRef.current = created
    setSession(created)
    return created
  }, [session])

  const saveTranslationIfNeeded = useCallback(
    async (text: string, category: string, glossResult: string, confidence: number) => {
      setLastLogId(null)
      if (!autoSave || !user.isLoggedIn || user.role !== 'doctor') return null

      const activeSession = await ensureSession()
      const saved = await api.createTranslation({
        session_id: activeSession.id,
        patient_id: 'none',
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
    [autoSave, ensureSession, user.isLoggedIn, user.role],
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
        setError(err instanceof Error ? err.message : '번역 기록 저장에 실패했습니다.')
      }
    },
    [saveTranslationIfNeeded],
  )

  const ensureTranslationSocket = useCallback(async (): Promise<WebSocket | null> => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        return wsRef.current
      }
 
      if (!user.isLoggedIn || user.role !== 'doctor') {
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
        const message = JSON.parse(event.data)
 
        if (message.type === 'error') {
          setError(message.message || 'WebSocket 처리 중 오류가 발생했습니다.')
          setIsPredicting(false)
          if (pendingStreamStartRef.current) {
            pendingStreamStartRef.current(false)
            pendingStreamStartRef.current = null
          }
          return
        }
 
        if (message.type === 'stream_started') {
          setError(null)
          if (pendingStreamStartRef.current) {
            pendingStreamStartRef.current(true)
            pendingStreamStartRef.current = null
          }
          return
        }
 
        // word_boundary 응답: 아직 문장이 끝난 게 아니라 단어 하나가 막 인식된 상태.
        // (백엔드가 이 타입을 안 보내면 이 블록은 그냥 안 타므로 무해함)
        if (message.type === 'partial') {
          const words = Array.isArray(message.words) ? message.words : []
          setActiveGlossResult(words.join(' '))
          return
        }
 
        if (message.type === 'translation') {
          const words = Array.isArray(message.words) ? message.words : []
          const candidates = Array.isArray(message.translation_candidates)
            ? (message.translation_candidates as TranslationCandidate[])
            : []
          const glossResult = message.gloss_result || words.join(' ')
          const text = message.text || glossResult || '인식된 수어가 없습니다.'
          const confidence = Number(message.confidence ?? 0)
 
          setActiveTranslation(text)
          setActiveGlossResult(glossResult)
          setTranslationCandidates(candidates)
          setActiveConfidence(confidence)
          setIsPredicting(false)
 
          if (message.log_id) {
            setLastLogId(Number(message.log_id))
            loadLogs().catch((err) => setError(err.message))
          }
        }
      }
 
      socket.onclose = () => {
        if (wsRef.current === socket) {
          wsRef.current = null
        }
        if (pendingStreamStartRef.current) {
          pendingStreamStartRef.current(false)
          pendingStreamStartRef.current = null
        }
      }
 
      return await new Promise<WebSocket | null>((resolve) => {
        const timeout = window.setTimeout(() => {
          setError('백엔드 WebSocket 연결 시간이 초과되었습니다.')
          resolve(null)
        }, 8000)
 
        socket.onopen = () => {
          window.clearTimeout(timeout)
          // 최초 접속 시 백엔드가 보내는 {"type": "connected"} 메시지는
          // 위 onmessage에서 별도 처리 없이 그냥 무시됨 (해가 없음).
          resolve(socket)
        }
 
        socket.onerror = () => {
          window.clearTimeout(timeout)
          setError('백엔드 WebSocket에 연결할 수 없습니다.')
          resolve(null)
        }
      })
    }, [loadLogs, user.isLoggedIn, user.role])
 
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
        }, 8000)
 
        pendingStreamStartRef.current = (ok: boolean) => {
          window.clearTimeout(timeout)
          resolve(ok)
        }
 
        socket.send(JSON.stringify({ type: 'start', session_id: activeSession.id }))
      })
    }, [ensureSession, ensureTranslationSocket])
  const handleStreamStart = async () => {
    setIsPredicting(false)
    return await startStream()
  }

  const handleKeypointFrame = async (keypoints: number[],frameId: number) => {
      const socket = wsRef.current
      const activeSession = activeSessionRef.current
      if (!socket || socket.readyState !== WebSocket.OPEN || !activeSession) return
      socket.send(JSON.stringify({ type: 'frame', session_id: activeSession.id, frame_id: frameId, keypoints}))
    }

  const handleWordBoundary = async () => {
      const socket = wsRef.current
      const activeSession = activeSessionRef.current
      if (!socket || socket.readyState !== WebSocket.OPEN || !activeSession) return
      socket.send(JSON.stringify({ type: 'word_boundary', session_id: activeSession.id }))
    }

  const handleStreamEnd = async () => {
    const socket = wsRef.current
    const activeSession = activeSessionRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN || !activeSession) return
    setIsPredicting(true)
    socket.send(JSON.stringify({ type: 'end', session_id: activeSession.id, auto_save: autoSave }))
  }

  const handleLoginSuccess = async (email: string, password: string) => {
    setError(null)
    const { access_token: token } = await api.login({ email, password })
    api.setToken(token)
    const me = await api.me()
    setUser({
      isLoggedIn: true,
      id: me.id,
      medical_id: me.medical_id,
      email: me.email,
      name: me.name,
      role: me.role,
    })
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
    setActiveTranslation(null)
    setActiveGlossResult('')
    setTranslationCandidates([])
    setCameraActive(false)
    setSession(null)
    activeSessionRef.current = null
    setLogs([])
    setCurrentView('main')
  }

  const handleTriggerTranslation = async (text: string, category: string) => {
    await handleCompletedTranslation(text, category, text)
  }

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
    setActiveTranslation(null)
    setActiveGlossResult('')
    setTranslationCandidates([])
    setCameraActive(true)
  }

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
      patient_id: 'none',
      translated_text: text,
      gloss_result: '직접 입력',
      confidence: 1,
      category,
    })
    setLogs((prev) => [mapApiLog(saved), ...prev])
  }

  const handleUpdatePatientId = async (logId: number, patientId: string) => {
    const saved = await api.updateTranslation(logId, { patient_id: patientId || 'none' })
    const mapped = mapApiLog(saved)
    setLogs((prev) => prev.map((log) => (log.log_id === logId ? mapped : log)))
  }

  const handleClearTranslation = () => {
    setActiveTranslation(null)
    setActiveGlossResult('')
    setTranslationCandidates([])
    setLastLogId(null)
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
                    onToggleActive={() => {
                      setCameraActive((active) => {
                        const next = !active
                        if (!next){
                          // 카메라를 끄는 순간에만 /ws/translate 연결도 정리한다.
                          // (카메라 켜져있는 동안에는 문장이 몇 번 반복되든 소켓을 계속 재사용함)
                          wsRef.current?.close()
                          wsRef.current = null
                          activeSessionRef.current = null
                        }
                        return next
                      })
                      setActiveTranslation(null)
                      setActiveGlossResult('')
                      setTranslationCandidates([])
                    }}
                    onTriggerTranslation={handleTriggerTranslation}
                    onStreamStart={handleStreamStart}
                    onKeypointFrame={handleKeypointFrame}
                    onWordBoundary={handleWordBoundary}
                    onStreamEnd={handleStreamEnd}
                    isPredicting={isPredicting}
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

      <InstructionsModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} isLoggedIn={user.isLoggedIn} />
    </div>
  )
}
