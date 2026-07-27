import { motion } from 'motion/react'
import { HelpCircle, LogOut, Save, SaveAll, User } from 'lucide-react'
import { UserState } from '../types'
import signlinkIcon from '../assets/images/signlink_icon_1780075452377.png'

interface HeaderProps {
  user: UserState
  onLoginClick: () => void
  onLogoutClick: () => void
  autoSave: boolean
  onAutoSaveToggle: () => void
  onHelpClick: () => void
  onLogoClick: () => void
}

export const Header = ({
  user,
  onLoginClick,
  onLogoutClick,
  autoSave,
  onAutoSaveToggle,
  onHelpClick,
  onLogoClick,
}: HeaderProps) => {
  return (
    <header className="sticky top-0 z-40 w-full bg-white border-b border-neutral-100 shadow-xs px-4 md:px-8 py-4">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <button
          type="button"
          onClick={onLogoClick}
          className="flex items-center gap-3 cursor-pointer group select-none self-start text-left"
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="w-10 h-10 rounded-xl overflow-hidden shadow-md shadow-secondary/15"
          >
            <img src={signlinkIcon} alt="SignLink Logo" className="w-full h-full object-cover scale-[1.18]" />
          </motion.div>
          <div>
            <h1 className="text-2xl font-bold font-sans text-on-surface tracking-tight flex items-center gap-1.5 leading-none">
              SignLink
              <span className="text-xs bg-secondary-container text-secondary px-1.5 py-0.5 rounded-sm font-semibold font-hyper">
                PRO v1.2
              </span>
            </h1>
            <p className="text-sm text-on-surface-variant font-medium mt-1">
              의료진을 위한 실시간 수어 번역 서비스
            </p>
          </div>
        </button>

        <div className="flex items-center justify-end gap-3 self-end sm:self-center">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onHelpClick}
            className="p-2 text-on-surface-variant hover:text-primary hover:bg-neutral-50 rounded-full transition-colors duration-150"
            title="사용 방법"
          >
            <HelpCircle className="w-5 h-5" />
          </motion.button>

          {user.isLoggedIn ? (
            <div className="flex items-center gap-2">
              <div className="hidden md:flex flex-col text-right">
                <span className="text-sm font-semibold text-on-surface font-hyper">
                  {user.name || 'Name'} 전문의
                </span>
                <span className="text-xs text-primary/80 font-medium">{user.email}</span>
              </div>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onLogoutClick}
                className="flex items-center gap-2 px-3.5 py-1.5 border border-neutral-200 hover:border-red-200 hover:bg-red-50 text-on-surface hover:text-red-600 rounded-lg text-base font-semibold font-hyper transition-all duration-200"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">로그아웃</span>
              </motion.button>
            </div>
          ) : (
            <motion.button
              whileHover={{ scale: 1.02, y: -1 }}
              whileTap={{ scale: 0.98 }}
              onClick={onLoginClick}
              className="flex items-center gap-2 px-4 py-1.5 bg-neutral-50 border border-neutral-200 text-on-surface hover:bg-neutral-100 rounded-lg text-base font-semibold font-hyper transition-all duration-200 shadow-xs"
            >
              <User className="w-4 h-4 text-neutral-500" />
              <span>로그인</span>
            </motion.button>
          )}

          <motion.button
            layout
            whileHover={user.isLoggedIn ? { scale: 1.03 } : { scale: 1 }}
            whileTap={user.isLoggedIn ? { scale: 0.97 } : { scale: 1 }}
            onClick={() => {
              if (user.isLoggedIn) onAutoSaveToggle()
              else alert('자동 저장 기능은 의료진 로그인 후 사용할 수 있습니다.')
            }}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-bold font-hyper tracking-wide transition-all duration-300 shadow-sm border ${
              !user.isLoggedIn
                ? 'bg-neutral-50 text-neutral-300 border-neutral-200 cursor-not-allowed opacity-60'
                : autoSave
                  ? 'bg-brand-green/10 text-brand-green border-brand-green/30 cursor-pointer'
                  : 'bg-neutral-100 text-neutral-500 border-neutral-200 cursor-pointer'
            }`}
          >
            {autoSave ? <SaveAll className="w-4 h-4 animate-pulse" /> : <Save className="w-4 h-4" />}
            <span>자동 저장 {autoSave ? 'ON' : 'OFF'}</span>
          </motion.button>
        </div>
      </div>
    </header>
  )
}
