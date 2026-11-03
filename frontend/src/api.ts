import { ApiSession, ApiTranslation, ApiUser, UserRole } from './types'
import { safeStorage } from './utils'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'
const TOKEN_STORAGE_KEY = 'token'

const NETWORK_ERROR_MESSAGE = '백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.'
const DEFAULT_ERROR_MESSAGE = '요청 처리에 실패했습니다.'
const INVALID_RESPONSE_MESSAGE = '서버 응답을 해석할 수 없습니다.'

function getToken() {
  return safeStorage.get(TOKEN_STORAGE_KEY)
}

function setToken(token: string) {
  safeStorage.set(TOKEN_STORAGE_KEY, token)
}

function clearToken() {
  safeStorage.remove(TOKEN_STORAGE_KEY)
}

/** 응답 본문을 JSON으로 해석한다. 빈 본문은 null, 해석 실패는 undefined. */
function parseJson(text: string): unknown {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

/** FastAPI 오류 본문(detail: string | {msg}[])에서 메시지를 뽑는다. */
function extractErrorDetail(data: unknown): string {
  const detail = (data as { detail?: unknown } | null | undefined)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item: { msg?: unknown }) => String(item?.msg ?? '')).join(', ')
  }
  return DEFAULT_ERROR_MESSAGE
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
    throw new Error(NETWORK_ERROR_MESSAGE)
  }

  const data = parseJson(await response.text())

  if (!response.ok) throw new Error(extractErrorDetail(data))
  if (data === undefined) throw new Error(INVALID_RESPONSE_MESSAGE)

  return data as T
}

/** JSON 본문을 가진 요청 옵션 */
function withJson(method: 'POST' | 'PATCH', body: unknown): RequestInit {
  return { method, body: JSON.stringify(body) }
}

export const api = {
  getToken,
  setToken,
  clearToken,

  async register(body: { name: string; email: string; password: string; role: UserRole }) {
    return request<ApiUser>('/auth/register', withJson('POST', body))
  },

  async login(body: { email: string; password: string }) {
    return request<{ access_token: string; token_type: string }>('/auth/login', withJson('POST', body))
  },

  async me() {
    return request<ApiUser>('/auth/me')
  },

  async createSession(body: { title: string; patient_id?: string | null }) {
    return request<ApiSession>('/sessions', withJson('POST', body))
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
    return request<ApiTranslation>('/translations', withJson('POST', body))
  },

  async updateTranslation(logId: number, body: { patient_id?: string | null }) {
    return request<ApiTranslation>(`/translations/${logId}`, withJson('PATCH', body))
  },

  async listTranslations(params: { search?: string; category?: string; limit?: number } = {}) {
    const query = new URLSearchParams()
    if (params.search) query.set('search', params.search)
    if (params.category && params.category !== '전체') query.set('category', params.category)
    if (params.limit) query.set('limit', String(params.limit))
    const suffix = query.toString() ? `?${query.toString()}` : ''
    return request<ApiTranslation[]>(`/translations${suffix}`)
  },

  async deleteTranslation(logId: number) {
    return request<void>(`/translations/${logId}`, { method: 'DELETE' })
  },

  async clearTranslations() {
    return request<void>('/translations', { method: 'DELETE' })
  },
}