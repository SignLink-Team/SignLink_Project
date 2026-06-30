export type ViewState = 'main' | 'history' | 'login' | 'register'

export type UserRole = 'doctor' | 'patient'

export interface TranslationLog {
  id: string
  text: string
  timestamp: string
  category?: string
  isCustom?: boolean
}

export interface UserState {
  isLoggedIn: boolean
  id: string | null
  email: string | null
  name: string | null
  role: UserRole | null
}

export interface GesturePreset {
  id: string
  gestureName: string
  translationText: string
  category: string
  description: string
}

export interface ApiUser {
  id: string
  email: string
  name: string
  role: UserRole
  created_at: string
}

export interface ApiSession {
  id: string
  doctor_id: string
  patient_id?: string | null
  title: string
  status: 'active' | 'closed'
  created_at: string
  closed_at?: string | null
}

export interface ApiTranslation {
  id: string
  log_id: string
  medical_id: string
  session_id?: string | null
  patient_id?: string | null
  gloss_result: string
  translated_text: string
  confidence?: number | null
  category: string
  input_time: string
  timestamp: string
}

export interface AiPredictionResult {
  type: 'prediction'
  text: string
  words: string[]
  gloss_result: string
  confidence: number
  frame_count?: number
  npy_saved?: boolean
}
