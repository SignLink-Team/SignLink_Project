import { ApiSession, ApiTranslation, ApiUser, UserRole } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

function getToken() {
  return localStorage.getItem('token')
}

function setToken(token: string) {
  localStorage.setItem('token', token)
}

function clearToken() {
  localStorage.removeItem('token')
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers)
  headers.set('Content-Type', 'application/json')

  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  } catch {
    throw new Error('백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.')
  }

  const text = await response.text()
  const data = text ? JSON.parse(text) : null

  if (!response.ok) {
    const detail = typeof data?.detail === 'string'
      ? data.detail
      : Array.isArray(data?.detail)
        ? data.detail.map((item: { msg: string }) => item.msg).join(', ')
        : '요청 처리에 실패했습니다.'
    throw new Error(detail)
  }

  return data as T
}

export const api = {
  getToken,
  setToken,
  clearToken,

  async register(body: { name: string; email: string; password: string; role: UserRole }) {
    return request<ApiUser>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  async login(body: { email: string; password: string }) {
    return request<{ access_token: string; token_type: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  async me() {
    return request<ApiUser>('/auth/me')
  },

  async createSession(body: { title: string; patient_id?: string | null }) {
    return request<ApiSession>('/sessions', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  async listSessions() {
    return request<ApiSession[]>('/sessions')
  },

  async createTranslation(body: {
    session_id?: string | null
    patient_id?: string | null
    gloss_result?: string
    translated_text: string
    confidence?: number | null
    category?: string
  }) {
    return request<ApiTranslation>('/translations', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  async listTranslations(params: { search?: string; category?: string; limit?: number } = {}) {
    const query = new URLSearchParams()
    if (params.search) query.set('search', params.search)
    if (params.category && params.category !== '전체') query.set('category', params.category)
    if (params.limit) query.set('limit', String(params.limit))
    const suffix = query.toString() ? `?${query.toString()}` : ''
    return request<ApiTranslation[]>(`/translations${suffix}`)
  },

  async deleteTranslation(id: string) {
    return request<void>(`/translations/${id}`, { method: 'DELETE' })
  },

  async clearTranslations() {
    return request<void>('/translations', { method: 'DELETE' })
  },
}
