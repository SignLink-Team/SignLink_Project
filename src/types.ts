export type ViewState = 'main' | 'history' | 'login' | 'register'

export type UserRole = 'doctor' | 'patient'

export interface TranslationCandidate {
  text: string
  confidence: number
}

export interface TranslationLog {
  log_id: number
  medical_id: number
  patient_id: string
  input_time: string
  gloss_result: string
  translated_text: string
  confidence: number
  category?: string
}

export interface UserState {
  isLoggedIn: boolean
  id: string | null
  medical_id: number | null
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
  medical_id: number
  email: string
  name: string
  role: UserRole
  created_at: string
}

export interface ApiSession {
  id: string
  doctor_id: string
  medical_id?: number | null
  patient_id?: string | null
  title: string
  status: 'active' | 'closed'
  created_at: string
  closed_at?: string | null
}

export interface ApiTranslation extends TranslationLog {}

export interface AiPredictionResult {
  type: 'prediction'
  text: string
  words: string[]
  gloss_result: string
  confidence: number
  frame_count?: number
  npy_saved?: boolean
  translation_candidates?: TranslationCandidate[]
}
