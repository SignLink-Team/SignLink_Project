import { FormEvent, ReactNode, useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { AlertCircle, ArrowLeft, Eye, EyeOff, Heart, Lock, LogIn, LucideIcon, Mail, ShieldCheck, User } from 'lucide-react'
import { ApiError } from '../api'
import { ViewState } from '../types'

interface LoginViewProps {
  mode: Extract<ViewState, 'login' | 'register'>
  onLoginSuccess: (email: string, password: string) => Promise<void>
  onRegister: (name: string, email: string, password: string) => Promise<void>
  onSwitchMode: () => void
  onBackToMain: () => void
}

const INPUT_BASE_CLASS =
  'w-full py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface'
const INPUT_CLASS = `${INPUT_BASE_CLASS} px-3.5`
const PASSWORD_INPUT_CLASS = `${INPUT_BASE_CLASS} pl-3.5 pr-10`

function FormField({ icon: Icon, label, children }: { icon: LucideIcon; label: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
        <Icon className="w-3.5 h-3.5 text-neutral-400" />
        {label}
      </label>
      {children}
    </div>
  )
}

interface AuthErrorText {
  title: string
  description: string
}

/**
 * 로그인/회원가입 실패를 사용자가 이해하고 바로 조치할 수 있는 문장으로 바꾼다.
 * 서버 원문(영문 또는 개발자용 문구)은 화면에 그대로 보여주지 않는다.
 */
function toAuthErrorText(err: unknown, isRegister: boolean): AuthErrorText {
  const status = err instanceof ApiError ? err.status : -1
  const fields = err instanceof ApiError ? err.fields : []

  if (status === 0) {
    return {
      title: '서버에 연결하지 못했어요',
      description: '네트워크 연결을 확인한 뒤 잠시 후 다시 시도해 주세요. 계속 안 되면 관리자에게 문의해 주세요.',
    }
  }
  if (status === 401) {
    return {
      title: '이메일 또는 비밀번호가 맞지 않아요',
      description: '입력한 내용을 다시 확인해 주세요. 키보드의 Caps Lock이 켜져 있지 않은지도 확인해 보세요.',
    }
  }
  if (status === 409) {
    return {
      title: '이미 가입된 이메일이에요',
      description: '이 이메일로 로그인하거나, 다른 이메일로 가입해 주세요.',
    }
  }
  if (status === 422) {
    if (fields.includes('email')) {
      return {
        title: '이메일 형식이 올바르지 않아요',
        description: 'name@hospital.com 처럼 @가 들어간 이메일 주소를 입력해 주세요.',
      }
    }
    if (fields.includes('password')) {
      return { title: '비밀번호가 너무 짧아요', description: '비밀번호는 8자 이상 입력해 주세요.' }
    }
    if (fields.includes('name')) {
      return { title: '이름을 확인해 주세요', description: '이름은 1~100자로 입력해 주세요.' }
    }
    return { title: '입력한 내용을 확인해 주세요', description: '빠진 항목이나 잘못 입력한 항목이 없는지 확인해 주세요.' }
  }
  if (status >= 500) {
    return {
      title: '서버에 일시적인 문제가 생겼어요',
      description: '잠시 후 다시 시도해 주세요. 계속 안 되면 관리자에게 문의해 주세요.',
    }
  }
  return {
    title: isRegister ? '계정을 만들지 못했어요' : '로그인하지 못했어요',
    description: '잠시 후 다시 시도해 주세요. 계속 안 되면 관리자에게 문의해 주세요.',
  }
}

export const LoginView = ({
  mode,
  onLoginSuccess,
  onRegister,
  onSwitchMode,
  onBackToMain,
}: LoginViewProps) => {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('name@hospital.com')
  const [password, setPassword] = useState('password123')
  const [showPassword, setShowPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<AuthErrorText | null>(null)
  const isRegister = mode === 'register'

  // 입력을 고치거나 로그인/회원가입을 전환하면 이전 안내는 지운다.
  useEffect(() => {
    setError(null)
  }, [name, email, password, mode])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (isSubmitting) return
    setError(null)
    setIsSubmitting(true)

    try {
      if (isRegister) await onRegister(name.trim(), email.trim(), password)
      else await onLoginSuccess(email.trim(), password)
    } catch (err) {
      setError(toAuthErrorText(err, isRegister))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      className="max-w-md mx-auto px-4 py-12 font-sans"
    >
      <div className="flex flex-col items-center">
        <motion.div
          className="w-20 h-20 rounded-full bg-secondary/15 flex flex-col items-center justify-center border border-secondary-container/50 relative shadow-sm"
          whileHover={{ y: -3 }}
        >
          <div className="w-14 h-14 rounded-full bg-secondary flex items-center justify-center text-white shadow-md">
            <Heart className="w-6 h-6 text-white fill-white/20 animate-pulse" />
          </div>
          <div className="absolute -bottom-2 capitalize">
            <span className="bg-white border border-neutral-200 text-secondary text-[11px] font-extrabold px-2.5 py-1 rounded-full flex items-center gap-1 leading-none shadow-2xs font-hyper uppercase tracking-wider">
              <ShieldCheck className="w-3 h-3" />
              의료진 전용
            </span>
          </div>
        </motion.div>

        <h2 className="mt-8 text-3xl font-black text-on-surface tracking-tight font-sans">
          {isRegister ? 'SignLink 계정 생성' : '의료진 로그인'}
        </h2>
        <p className="text-sm text-neutral-500 mt-1.5 font-semibold text-center max-w-sm leading-relaxed">
          의료 수어 번역 기록 조회 및 EMR 연동을 위해 의료진 계정으로 접속하세요.
        </p>

        <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs mt-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div role="alert" className="flex gap-2.5 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg">
                <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
                <div className="text-sm leading-relaxed">
                  <p className="font-bold">{error.title}</p>
                  <p className="mt-0.5 font-medium text-red-700/90">{error.description}</p>
                </div>
              </div>
            )}

            {isRegister && (
              <FormField icon={User} label="이름">
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="의료진 이름"
                  className={INPUT_CLASS}
                />
              </FormField>
            )}

            <FormField icon={Mail} label="이메일">
              <input
                type="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="name@hospital.com"
                className={INPUT_CLASS}
              />
            </FormField>

            <FormField icon={Lock} label="비밀번호">
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  minLength={8}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="8자 이상"
                  className={PASSWORD_INPUT_CLASS}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 text-neutral-400 hover:text-neutral-600 rounded"
                  title={showPassword ? '비밀번호 숨기기' : '비밀번호 보기'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </FormField>

            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              type="submit"
              disabled={isSubmitting}
              className="w-full mt-2 py-3 bg-secondary hover:bg-secondary-dark text-white rounded-lg text-base font-bold flex items-center justify-center gap-2 transition-all shadow-sm cursor-pointer disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <span className="w-4 h-4 rounded-full border border-t-white border-secondary-container animate-spin" />
                  <span>처리 중...</span>
                </>
              ) : (
                <>
                  <LogIn className="w-4 h-4" />
                  <span>{isRegister ? '계정 생성' : '로그인'}</span>
                </>
              )}
            </motion.button>
          </form>

          <div className="flex justify-between mt-4">
            <button
              type="button"
              onClick={onSwitchMode}
              className="text-xs font-semibold text-secondary hover:text-secondary-dark transition-colors cursor-pointer"
            >
              {isRegister ? '이미 계정이 있나요? 로그인' : '계정이 없나요? 회원가입'}
            </button>
            {!isRegister && (
              <button type="button" className="text-xs font-semibold text-neutral-450 hover:text-primary transition-colors cursor-pointer">
                비밀번호 찾기
              </button>
            )}
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onBackToMain}
          className="mt-6 flex items-center gap-1.5 text-neutral-500 hover:text-on-surface text-sm font-bold transition-colors duration-150 cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>메인으로 돌아가기</span>
        </motion.button>
      </div>
    </motion.div>
  )
}