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
    if (
      not key.startswith("SIGNLINK_LLM_")
      and key != "SIGNLINK_SKIP_LLM"
    ) or key in os.environ:
      continue
    os.environ[key] = value.strip().strip('"').strip("'")


_load_project_env()

LLM_BASE_URL = os.getenv("SIGNLINK_LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("SIGNLINK_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LLM_TIMEOUT_SECONDS = float(os.getenv("SIGNLINK_LLM_TIMEOUT_SECONDS", "120"))
LLM_API_KEY = os.getenv("SIGNLINK_LLM_API_KEY", "")
LLM_NUM_GPU = os.getenv("SIGNLINK_LLM_NUM_GPU", "").strip()

SYSTEM_PROMPT = """당신은 수어 사용 환자의 말을 한국어로 번역하는 의료 수어 번역기이다. 글로스의 순서와 비수지 구간 정보를 근거로 자연스러운 한국어 문장으로 번역한다. 입력에 없는 증상, 검사, 진단, 치료 정보는 추가하지 않는다.

비수지 태그:
Mo1=입 벌림/강조, Mmo=미소/긍정, Mctr=입 다묾/참음·노력, Hno=고개 끄덕임/긍정·문장종결, Hs=고개 저음/부정, Ebf=눈썹 찌푸림/심각·경고, Ci=볼 부풀림/상태강조, Ebu=눈썹 올림/의문·강조."""

NONMANUAL_TAG_ORDER = {
  tag: index
  for index, tag in enumerate(("Mo1", "Mmo", "Mctr", "Hno", "Hs", "Ebf", "Ci", "Ebu"))
}
NONMANUAL_TAG_BY_LOWER = {
  tag.lower(): tag
  for tag in NONMANUAL_TAG_ORDER
}


def llm_enabled() -> bool:
  return bool(LLM_BASE_URL)


def chat_completions_url() -> str:
  if LLM_BASE_URL.endswith("/v1"):
    return f"{LLM_BASE_URL}/chat/completions"
  return f"{LLM_BASE_URL}/v1/chat/completions"


def fallback_translation(words: list[str], gloss_result: str, confidence: float) -> dict[str, Any]:
  text = gloss_result or " ".join(words)
  return {
    "text": text,
    "translation_candidates": [],
    "llm_used": False,
    "llm_error": None,
    "confidence": confidence,
  }


def _canonical_nonmanual_tag(raw_tag: Any) -> str:
  tag = str(raw_tag or "").strip()
  return NONMANUAL_TAG_BY_LOWER.get(tag.lower(), tag)


def _contiguous_ranges(indices: set[int]) -> list[tuple[int, int]]:
  ordered = sorted(indices)
  if not ordered:
    return []

  ranges: list[tuple[int, int]] = []
  start = end = ordered[0]
  for index in ordered[1:]:
    if index == end + 1:
      end = index
      continue
    ranges.append((start, end))
    start = end = index
  ranges.append((start, end))
  return ranges


def _nonmanual_spans(nms_context: dict[str, Any] | None) -> list[dict[str, Any]]:
  if not nms_context:
    return []

  provided_spans = nms_context.get("nonmanual_spans")
  if isinstance(provided_spans, list):
    spans = []
    for span in provided_spans:
      if not isinstance(span, dict):
        continue
      try:
        start = int(span["start_index"])
        end = int(span["end_index"])
      except (KeyError, TypeError, ValueError):
        continue
      tag = _canonical_nonmanual_tag(span.get("tag"))
      if tag and start >= 0 and end >= start:
        spans.append({"tag": tag, "start_index": start, "end_index": end})
    return sorted(
      spans,
      key=lambda span: (
        span["start_index"],
        span["end_index"],
        NONMANUAL_TAG_ORDER.get(span["tag"], len(NONMANUAL_TAG_ORDER)),
      ),
    )

  # main.py가 동일 비수지 segment 객체를 겹치는 모든 단어의 nms 목록에
  # 넣어 주므로 (태그, 시작, 끝)별 단어 인덱스를 모으면 학습 데이터와
  # 같은 start_index/end_index 구간으로 되돌릴 수 있다.
  segment_indices: dict[tuple[str, Any, Any], set[int]] = {}
  aligned_words = nms_context.get("words")
  if not isinstance(aligned_words, list):
    return []

  for word_index, word in enumerate(aligned_words):
    if not isinstance(word, dict) or not isinstance(word.get("nms"), list):
      continue
    for segment in word["nms"]:
      if not isinstance(segment, dict):
        continue
      tag = _canonical_nonmanual_tag(segment.get("key") or segment.get("tag"))
      if not tag:
        continue
      identity = (tag, segment.get("start"), segment.get("end"))
      segment_indices.setdefault(identity, set()).add(word_index)

  spans = []
  for (tag, _start_time, _end_time), indices in segment_indices.items():
    for start, end in _contiguous_ranges(indices):
      spans.append({"tag": tag, "start_index": start, "end_index": end})

  return sorted(
    spans,
    key=lambda span: (
      span["start_index"],
      span["end_index"],
      NONMANUAL_TAG_ORDER.get(span["tag"], len(NONMANUAL_TAG_ORDER)),
    ),
  )


def build_user_prompt(
  words: list[str],
  gloss_result: str,
  nms_context: dict[str, Any] | None = None,
) -> str:
  prompt_words = list(words)
  if not prompt_words and gloss_result:
    prompt_words = gloss_result.split()
  if not prompt_words and nms_context and isinstance(nms_context.get("words"), list):
    prompt_words = [
      str(word.get("gloss", "")).strip()
      for word in nms_context["words"]
      if isinstance(word, dict) and str(word.get("gloss", "")).strip()
    ]

  indexed_words = " | ".join(
    f"{index}={word}"
    for index, word in enumerate(prompt_words)
  )
  spans = _nonmanual_spans(nms_context)
  nonmanual_lines = [
    (
      f"{span['start_index']}: {span['tag']}"
      if span["start_index"] == span["end_index"]
      else f"{span['start_index']}~{span['end_index']}: {span['tag']}"
    )
    for span in spans
    if span["end_index"] < len(prompt_words)
  ]
  nonmanual_text = "\n".join(nonmanual_lines) or "없음"

  return f"""입력 단어:
{indexed_words}

비수지:
{nonmanual_text}

자연스러운 한국어 문장:"""


def _build_messages(
  words: list[str],
  gloss_result: str,
  nms_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
  return [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_user_prompt(words, gloss_result, nms_context)},
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


async def translate_gloss(
  words: list[str],
  gloss_result: str,
  confidence: float,
  nms_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
  fallback = fallback_translation(words, gloss_result, confidence)
  if not llm_enabled():
    return fallback

  endpoint = chat_completions_url()
  payload = {
    "model": LLM_MODEL,
    "messages": _build_messages(words, gloss_result, nms_context),
    "temperature": 0.1,
    "top_p": 0.8,
    "max_tokens": 96,
    "stream": False,
  }
  if LLM_NUM_GPU:
    payload["options"] = {
      "num_gpu": int(LLM_NUM_GPU),
    }
  headers = (
    {"Authorization": f"Bearer {LLM_API_KEY}"}
    if LLM_API_KEY
    else None
  )

  try:
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
      response = await client.post(endpoint, json=payload, headers=headers)
      response.raise_for_status()
      data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
      raise ValueError("sLLM response content must be a string")
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
