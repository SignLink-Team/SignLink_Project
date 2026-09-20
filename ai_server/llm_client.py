from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx


def _load_project_env() -> None:
  env_path = Path(__file__).resolve().parents[1] / ".env"
  if not env_path.exists():
    return
  for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    if not key.startswith("SIGNLINK_LLM_") or key in os.environ:
      continue
    os.environ[key] = value.strip().strip('"').strip("'")


_load_project_env()

LLM_BASE_URL = os.getenv("SIGNLINK_LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("SIGNLINK_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LLM_TIMEOUT_SECONDS = float(os.getenv("SIGNLINK_LLM_TIMEOUT_SECONDS", "120"))


def llm_enabled() -> bool:
  return bool(LLM_BASE_URL)


def fallback_translation(words: list[str], gloss_result: str, confidence: float) -> dict[str, Any]:
  text = gloss_result or " ".join(words)
  return {
    "text": text,
    "translation_candidates": [],
    "llm_used": False,
    "llm_error": None,
    "confidence": confidence,
  }


def _build_messages(words: list[str], gloss_result: str) -> list[dict[str, str]]:
  input_words = gloss_result or " ".join(words)
  system = """당신은 수어 번역기이다.

규칙

입력 단어와 비수지 신호 사전 정보만 사용한다.

없는 정보를 추가하지 않는다.

자연스러운 한국어로만 바꾼다.

출력은 문장 하나만 작성한다.

[비수지 신호 사전]

- Mo1 : 입 벌리기("마", "팍")
  → 끝맺음, 가능, 문장 종결, 부정 등의 의미를 강조할 수 있다.

- Mno : 미소(입술 양쪽 올림)
  → 긍정적 표현, 칭찬, 기쁨, 친근함을 나타낸다.

- Mctr : 입꼬리가 움직이며 입을 꽉 다물기
  → 참다, 기다리다, 열심히 하다, 인사(안녕하세요) 등의 의미를 표현한다.

- Hno : 고개 끄덕임
  → 긍정, 동의, 부탁, 필요, 나열, 문장 종결 등을 나타낸다.

- Hs : 고개를 좌우로 흔들기
  → 부정, 불가능, 못하다, 안 된다를 의미한다.

- Ebf : 눈썹 찌푸리기
  → 심각함, 위험함, 안 됨, 경고를 나타낸다.

- Ci : 볼 부풀리기
  → 상황이나 상태를 묘사하거나 강조한다.

- Ebu : 눈썹을 위로 올리기
  → 의문문, 놀람, 강조를 나타낸다."""

  user = f"""입력 단어:
{input_words}

자연스러운 한국어 문장:"""
  return [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
  ]


def _clean_sentence(content: str) -> str:
  content = content.strip()
  if content.startswith("```"):
    content = re.sub(r"^```(?:json|text)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
  content = re.sub(r"^\s*(답변|출력|번역|문장)\s*[:：]\s*", "", content).strip()
  lines = [line.strip() for line in content.splitlines() if line.strip()]
  if not lines:
    return ""
  return lines[0].strip().strip('"').strip("'").strip()


async def translate_gloss(words: list[str], gloss_result: str, confidence: float) -> dict[str, Any]:
  fallback = fallback_translation(words, gloss_result, confidence)
  if not llm_enabled():
    return fallback

  endpoint = f"{LLM_BASE_URL}/v1/chat/completions"
  payload = {
    "model": LLM_MODEL,
    "messages": _build_messages(words, gloss_result),
    "temperature": 0.2,
    "top_p": 0.8,
    "max_tokens": 128,
  }

  try:
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
      response = await client.post(endpoint, json=payload)
      response.raise_for_status()
      data = response.json()
    content = data["choices"][0]["message"]["content"]
    translated_text = _clean_sentence(content) or fallback["text"]
    return {
      "text": translated_text,
      "translation_candidates": [],
      "llm_used": True,
      "llm_error": None,
      "confidence": confidence,
    }
  except Exception as exc:
    fallback["llm_error"] = str(exc)
    print("[ai-llm] failed", {"endpoint": endpoint, "model": LLM_MODEL, "error": str(exc)}, flush=True)
    return fallback
