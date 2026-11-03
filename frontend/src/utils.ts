// 여러 컴포넌트에서 공통으로 쓰는 작은 유틸리티.

/** 예외 객체에서 사용자에게 보여줄 메시지를 꺼낸다. Error가 아니면 fallback을 쓴다. */
export function getErrorMessage(err: unknown, fallback: string): string {
  return err instanceof Error ? err.message : fallback
}

// localStorage는 사생활 보호 모드/저장소 차단 환경에서 예외를 던질 수 있으므로 항상 감싸서 접근한다.
export const safeStorage = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(key, value)
    } catch {
      // 저장소를 쓸 수 없는 환경이면 조용히 무시한다.
    }
  },
  remove(key: string) {
    try {
      localStorage.removeItem(key)
    } catch {
      // 저장소를 쓸 수 없는 환경이면 조용히 무시한다.
    }
  },
}

/** 클립보드 복사. 권한 거부 등으로 실패해도 처리되지 않은 Promise 거부가 남지 않게 한다. */
export function copyToClipboard(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {
    // 복사 실패는 UI 흐름을 막지 않는다.
  })
}

/** 값이 배열이면 그대로, 아니면 빈 배열을 돌려준다(서버 메시지 방어용). */
export function asArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : []
}