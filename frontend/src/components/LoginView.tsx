import { FormEvent, useState } from 'react'
import { motion } from 'motion/react'
import { ArrowLeft, Eye, EyeOff, Heart, Lock, LogIn, Mail, ShieldCheck, User } from 'lucide-react'
import { ViewState } from '../types'

interface LoginViewProps {
  mode: Extract<ViewState, 'login' | 'register'>
  onLoginSuccess: (email: string, password: string) => Promise<void>
  onRegister: (name: string, email: string, password: string) => Promise<void>
  onSwitchMode: () => void
  onBackToMain: () => void
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
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const isRegister = mode === 'register'

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setErrorMessage(null)
    setIsSubmitting(true)

    try {
      if (isRegister) await onRegister(name.trim(), email.trim(), password)
      else await onLoginSuccess(email.trim(), password)
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : '인증 처리에 실패했습니다.')
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
            {errorMessage && (
              <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-semibold leading-relaxed">
                {errorMessage}
              </div>
            )}

            {isRegister && (
              <div className="space-y-1.5">
                <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
                  <User className="w-3.5 h-3.5 text-neutral-400" />
                  이름
                </label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="의료진 이름"
                  className="w-full px-3.5 py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface"
                />
              </div>
            )}

            <div className="space-y-1.5">
              <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
                <Mail className="w-3.5 h-3.5 text-neutral-400" />
                이메일
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="name@hospital.com"
                className="w-full px-3.5 py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-neutral-400" />
                비밀번호
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  minLength={8}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="8자 이상"
                  className="w-full pl-3.5 pr-10 py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface"
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
            </div>

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
