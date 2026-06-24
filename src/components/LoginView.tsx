/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Heart, Mail, Lock, Eye, EyeOff, LogIn, ArrowLeft, ShieldCheck } from 'lucide-react';
import { UserState } from '../types';

interface LoginViewProps {
  onLoginSuccess: (email: string, name: string) => void;
  onBackToMain: () => void;
}

export const LoginView: React.FC<LoginViewProps> = ({
  onLoginSuccess,
  onBackToMain,
}) => {
  const [email, setEmail] = useState('name@hospital.com');
  const [password, setPassword] = useState('password123');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setIsSubmitting(true);

    // Simulate authentic validation delays
    setTimeout(() => {
      // Basic validation check
      if (!email.includes('@')) {
        setErrorMessage('유효한 병원 도메인 이메일 주소를 입력해 주세요.');
        setIsSubmitting(false);
        return;
      }
      if (password.length < 4) {
        setErrorMessage('비밀번호는 최소 4글자 이상이어야 합니다.');
        setIsSubmitting(false);
        return;
      }

      // Simulated matching success path
      // Extract doctor name from prefix or assign a professional title
      const userPrefix = email.split('@')[0];
      const nameCapitalized = userPrefix.charAt(0).toUpperCase() + userPrefix.slice(1);
      const docName = `${nameCapitalized} 전문의`;

      onLoginSuccess(email, docName);
      setIsSubmitting(false);
    }, 1000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      className="max-w-md mx-auto px-4 py-12 font-sans"
    >
      <div className="flex flex-col items-center">
        
        {/* Brand Shield Guard Top Icon */}
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

        {/* Title */}
        <h2 className="mt-8 text-3xl font-black text-on-surface tracking-tight font-sans">
          의료진 로그인
        </h2>
        <p className="text-sm text-neutral-500 mt-1.5 font-semibold text-center max-w-sm leading-relaxed">
          의료 수어 번역 기록 조회 및 전자의무기록(EMR) 자동 연동을 위해 권한 등급 이메일로 로그인하세요.
        </p>

        {/* Form panel Card elements matcher screen 3 */}
        <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs mt-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            
            {/* Action Alert Banner */}
            {errorMessage && (
              <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-semibold leading-relaxed">
                {errorMessage}
              </div>
            )}

            {/* Email field */}
            <div className="space-y-1.5">
              <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
                <Mail className="w-3.5 h-3.5 text-neutral-400" />
                이메일
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@hospital.com"
                className="w-full px-3.5 py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface"
              />
            </div>

            {/* Password field */}
            <div className="space-y-1.5">
              <label className="text-sm font-bold text-neutral-500 font-hyper flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-neutral-400" />
                비밀번호
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-3.5 pr-10 py-2.5 bg-neutral-50/50 hover:bg-neutral-50 focus:bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm font-medium outline-none transition-all duration-150 text-on-surface"
                />
                
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 p-0.5 text-neutral-400 hover:text-neutral-600 rounded"
                  title={showPassword ? '비밀번호 숨기기' : '비밀번호 보이기'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Submit Matcher Screen */}
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
                  <span>인증 시스템 조회 중...</span>
                </>
              ) : (
                <>
                  <LogIn className="w-4 h-4" />
                  <span>로그인</span>
                </>
              )}
            </motion.button>

          </form>

          {/* auxiliary link password find */}
          <div className="flex justify-start mt-4">
            <button
              onClick={() => alert('진료 계약 기관의 관리 정보 전산실 혹은 보안 부서에 비밀번호 재발급 신청을 문의하십시오.')}
              className="text-xs font-semibold text-neutral-450 hover:text-primary transition-colors cursor-pointer"
            >
              ⊙ 비밀번호 찾기
            </button>
          </div>
        </div>

        {/* Back Action matching image */}
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
  );
};
