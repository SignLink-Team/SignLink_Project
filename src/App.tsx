import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { TranslationLog, UserState, ViewState } from './types';
import { INITIAL_TRANSLATION_LOGS, GESTURE_PRESETS } from './data';
import { Header } from './components/Header';
import { CameraView } from './components/CameraView';
import { TranslationResult } from './components/TranslationResult';
import { UnkSolver } from './components/UnkSolver';
import { HistoryView } from './components/HistoryView';
import { LoginView } from './components/LoginView';
import { InstructionsModal } from './components/InstructionsModal';
import { Activity, Languages, Info, ExternalLink } from 'lucide-react';

export default function App() {
  // Navigation states
  const [currentView, setCurrentView] = useState<ViewState>('main');
  
  // Login Profile State with local storage persistence
  const [user, setUser] = useState<UserState>(() => {
    const saved = localStorage.getItem('signlink_user');
    if (saved) {
      try { return JSON.parse(saved); } catch (e) { /* ignore */ }
    }
    return { isLoggedIn: false, email: null, name: null };
  });

  // Auto Save State - preselects on-login as per screenshot
  const [autoSave, setAutoSave] = useState<boolean>(() => {
    return localStorage.getItem('signlink_autosave') === 'true';
  });

  // Translation history list
  const [logs, setLogs] = useState<TranslationLog[]>(() => {
    const saved = localStorage.getItem('signlink_logs');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0 && ('id' in parsed[0] || 'text' in parsed[0])) {
          // Old schema detected, reset to new initial logs
          return INITIAL_TRANSLATION_LOGS;
        }
        return parsed;
      } catch (e) { /* ignore */ }
    }
    return INITIAL_TRANSLATION_LOGS;
  });

  // Current active live translation outcome
  const [activeTranslation, setActiveTranslation] = useState<string | null>(null);

  // Active status of the camera viewfinder
  const [cameraActive, setCameraActive] = useState<boolean>(false);

  // Help modal popup toggle
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  // Persist user and auto-save options
  useEffect(() => {
    localStorage.setItem('signlink_user', JSON.stringify(user));
    // When logging out, force turn off autosave to remain secure
    if (!user.isLoggedIn) {
      setAutoSave(false);
      localStorage.setItem('signlink_autosave', 'false');
    }
  }, [user]);

  useEffect(() => {
    localStorage.setItem('signlink_autosave', String(autoSave));
  }, [autoSave]);

  // Persist translation logs database
  useEffect(() => {
    localStorage.setItem('signlink_logs', JSON.stringify(logs));
  }, [logs]);

  // Handle manual/automatic save insertions
  const handleAddNewTranslationLog = (
    text: string, 
    category: string, 
    pId?: string, 
    gloss?: string, 
    conf?: number
  ) => {
    const now = new Date();
    const isoString = now.toISOString();

    const maxLogId = logs.reduce((max, log) => log.log_id > max ? log.log_id : max, 0);

    const newLogItem: TranslationLog = {
      log_id: maxLogId + 1,
      medical_id: user.isLoggedIn && user.userId ? user.userId : 1001,
      patient_id: pId || 'none',
      input_time: isoString,
      gloss_result: gloss || `${category} 제스처`,
      translated_text: text,
      confidence: conf !== undefined ? conf : Math.round(85 + Math.random() * 14),
      category: category,
      isCustom: true
    };

    setLogs(prev => [newLogItem, ...prev]);
  };

  // Custom login dispatcher
  const handleLoginSuccess = (email: string, name: string) => {
    setUser({
      isLoggedIn: true,
      email: email,
      name: name,
      userId: 1001
    });
    // Auto Save is automatically pre-toggles to "ON" upon clinic login as in the design!
    setAutoSave(true);
    setCurrentView('main');
  };

  const handleLogout = () => {
    setUser({ isLoggedIn: false, email: null, name: null });
    setAutoSave(false);
    setActiveTranslation(null);
    setCameraActive(false);
  };

  // Active translation dispatcher from simulated gesture selection
  const handleTriggerTranslation = (text: string, category: string, gloss?: string, conf?: number) => {
    setActiveTranslation(text);

    // If auto-save is enabled on translation, stream into the history state instantly!
    if (autoSave) {
      let derivedGloss = gloss;
      if (!derivedGloss) {
        const preset = GESTURE_PRESETS.find(p => p.translationText === text);
        derivedGloss = preset ? preset.gestureName : `${category} 인식`;
      }
      handleAddNewTranslationLog(text, category, 'none', derivedGloss, conf);
    }
  };

  const handleDeleteLog = (logId: number) => {
    setLogs(prev => prev.filter(log => log.log_id !== logId));
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleAddManualLog = (text: string, category: string, pId?: string) => {
    handleAddNewTranslationLog(text, category, pId || 'none', '수동 입력', 100);
  };

  const handleUpdatePatientId = (logId: number, patientId: string) => {
    setLogs(prev => prev.map(log => log.log_id === logId ? { ...log, patient_id: patientId } : log));
  };

  return (
    <div className="min-h-screen bg-bg-base text-on-surface flex flex-col justify-between antialiased selection:bg-primary-light selection:text-primary font-sans">
      
      {/* Top Application header */}
      <Header
        user={user}
        onLoginClick={() => setCurrentView('login')}
        onLogoutClick={handleLogout}
        autoSave={autoSave}
        onAutoSaveToggle={() => setAutoSave(!autoSave)}
        onHelpClick={() => setIsHelpOpen(true)}
        onLogoClick={() => setCurrentView('main')}
      />

      {/* Main Views Panel */}
      <main className="flex-1 w-full flex items-center justify-center p-4 sm:p-6 md:p-8">
        <div className="w-full max-w-7xl">
          <AnimatePresence mode="wait">
            
            {/* View State 1: Dashboard Flow */}
            {currentView === 'main' && (
              <motion.div
                key="main-view"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
                className="w-full flex flex-col items-center gap-6"
              >
                {/* Clinical Context info banner */}
                <div className="w-full max-w-3xl flex flex-col sm:flex-row sm:items-center sm:justify-between px-4 py-3 bg-white border border-neutral-100 rounded-xl shadow-2xs gap-3">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-full bg-secondary-container/10 flex items-center justify-center text-secondary">
                      <Activity className="w-4 h-4" />
                    </div>
                    <div>
                      <span className="text-sm font-bold text-on-surface flex items-center gap-1.5 leading-none">
                        분당서울대병원 EMR 연동 솔루션
                      </span>
                      <p className="text-xs text-neutral-400 font-medium mt-1">
                        본 터미널은 의사 자격 면허 소령 계정과 1:1 암호화 페어링됩니다.
                      </p>
                    </div>
                  </div>
                  
                  {/* Interactive Status banner */}
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="text-xs bg-neutral-150 border border-neutral-250 text-neutral-600 font-bold px-2.5 py-1.5 rounded-sm">
                      {logs.length}개 누적 번역
                    </span>
                    <button
                      onClick={() => setIsHelpOpen(true)}
                      className="text-xs text-secondary hover:underline flex items-center gap-0.5 font-bold cursor-pointer"
                    >
                      <Info className="w-3.5 h-3.5" />
                      도움말 가이드
                    </button>
                  </div>
                </div>

                {/* Central viewfinder column */}
                <div id="signlink-viewport-container" className="w-full max-w-3xl flex flex-col items-center">
                  <CameraView
                    isActive={cameraActive}
                    onToggleActive={() => {
                      setCameraActive(!cameraActive);
                      if (!cameraActive) {
                        // Let's pre-populate the subtitle text with general welcome or reset
                        setActiveTranslation(null);
                      } else {
                        // Trigger a demo welcome prompt upon turning the active stream on
                        setActiveTranslation("환자분, 어디가 불편하신가요?");
                      }
                    }}
                    onTriggerTranslation={handleTriggerTranslation}
                  />

                  {/* Dynamic output section */}
                  <TranslationResult
                    text={activeTranslation}
                    onViewHistory={() => setCurrentView('history')}
                    isLoggedIn={user.isLoggedIn}
                    autoSave={autoSave}
                    onSelectTranslation={(text, category) => {
                      handleTriggerTranslation(text, category);
                    }}
                    onClearTranslation={() => {
                      setActiveTranslation(null);
                    }}
                  />
                </div>
              </motion.div>
            )}

            {/* View State 2: Logs list history */}
            {currentView === 'history' && (
              user.isLoggedIn ? (
                <HistoryView
                  key="history-view"
                  logs={logs}
                  onBack={() => setCurrentView('main')}
                  onClearLogs={handleClearLogs}
                  onDeleteLog={handleDeleteLog}
                  onAddManualLog={handleAddManualLog}
                  onUpdatePatientId={handleUpdatePatientId}
                />
              ) : (
                <LoginView
                  key="login-view"
                  onLoginSuccess={handleLoginSuccess}
                  onBackToMain={() => setCurrentView('main')}
                />
              )
            )}

            {/* View State 3: Secure Medical Staff login portal */}
            {currentView === 'login' && (
              <LoginView
                key="login-view"
                onLoginSuccess={handleLoginSuccess}
                onBackToMain={() => setCurrentView('main')}
              />
            )}

          </AnimatePresence>
        </div>
      </main>

      {/* Auxiliary informative callout for clinical test environments */}
      <footer className="w-full py-4 border-t border-neutral-250 bg-white text-center text-[11px] text-neutral-400 font-medium">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2.5">
          <p>© 2026 SignLink. All rights reserved.</p>
          
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
              EMR 보안서버 수어망 대기 중
            </span>
            <span>|</span>
            <span className="text-[10px] bg-neutral-100 text-neutral-500 border border-neutral-200 px-1.5 py-0.5 rounded">
              Atkinson Accessibility Standards Approved
            </span>
          </div>
        </div>
      </footer>

      {/* Global Information Help Overlay */}
      <InstructionsModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
      />

    </div>
  );
}