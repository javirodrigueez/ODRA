import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from langchain.messages import AIMessage, HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]
from langchain_aws import ChatBedrock  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # pyright: ignore[reportMissingImports]
from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
from openai import OpenAI  # pyright: ignore[reportMissingImports]

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

MessageRole = Literal["system", "user", "assistant"]
GENERATION_CONFIG_KEYS = (
  "temperature",
  "top_p",
  "top_k",
  "min_p",
  "presence_penalty",
  "frequency_penalty",
  "repetition_penalty",
  "max_tokens",
)


class ChatMessage(TypedDict):
  role: MessageRole
  content: str


def load_prompt(filename: str | Path) -> str:
  path = filename if isinstance(filename, Path) else PROMPTS_DIR / filename
  return path.read_text(encoding="utf-8").strip()

def render_prompt(filename: str, **values: Any) -> str:
  template = PromptTemplate.from_template(load_prompt(filename))
  return template.format(
    **{key: "" if value is None else str(value) for key, value in values.items()}
  )


def _to_langchain_messages(messages: list[ChatMessage]) -> list[Any]:
  """Convert dict messages to LangChain message objects."""
  lc_messages: list[Any] = []
  for m in messages:
    if m["role"] == "system":
      lc_messages.append(SystemMessage(content=m["content"]))
    elif m["role"] == "user":
      lc_messages.append(HumanMessage(content=m["content"]))
    else:
      lc_messages.append(AIMessage(content=m["content"]))
  return lc_messages


class StructuredOutputError(RuntimeError):
  pass


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def coerce_float(value: Any, default: float, low: float | None = None, high: float | None = None) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    parsed = default
  if low is not None:
    parsed = max(low, parsed)
  if high is not None:
    parsed = min(high, parsed)
  return parsed


def coerce_bool(value: Any, default: bool = False) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1"}:
      return True
    if lowered in {"false", "no", "0"}:
      return False
  return default


def ensure_str(value: Any, default: str = "") -> str:
  if isinstance(value, str):
    return value.strip()
  if value is None:
    return default
  return str(value).strip()


def ensure_str_list(value: Any) -> list[str]:
  if isinstance(value, list):
    return [ensure_str(v) for v in value if ensure_str(v)]
  if isinstance(value, str) and value.strip():
    return [value.strip()]
  return []


def parse_json_response(raw_text: str) -> tuple[Any, dict[str, Any]]:
  text = (raw_text or "").strip()
  if not text:
    raise StructuredOutputError("empty response")

  try:
    return json.loads(text), {"parsed": True, "mode": "direct"}
  except json.JSONDecodeError as exc:
    raise StructuredOutputError(f"invalid JSON: {exc.msg}") from exc


def _to_config_mapping(config: Any) -> dict:
  if config is None:
    return {}
  raw_config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
  if not isinstance(raw_config, dict):
    raise ValueError(f"LLM config block must be a mapping, got {type(raw_config).__name__}.")
  return raw_config


def extract_generation_config(config: Any) -> dict[str, Any]:
  raw_config = _to_config_mapping(config)
  generation_config: dict[str, Any] = {}
  for key in GENERATION_CONFIG_KEYS:
    value = raw_config.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
      continue
    if key in {"top_k", "max_tokens"}:
      generation_config[key] = int(value)
    else:
      generation_config[key] = float(value)
  return generation_config

# LLM interfaces
class LLM(ABC):
  @abstractmethod
  def generate(self, messages: list[ChatMessage], **kwargs: Any) -> str:
    pass

class HuggingFaceLLM(LLM):
  def __init__(self, model_id: str, generation_config: dict[str, Any] | None = None):
    generation_config = generation_config or {}
    pipeline_kwargs: dict[str, Any] = {
      "temperature": generation_config.get("temperature", 0.7),
      "do_sample": True,
      "return_full_text": False,
      "max_new_tokens": generation_config.get("max_tokens", 1024),
    }
    for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
      if key in generation_config:
        pipeline_kwargs[key] = generation_config[key]
    self.llm = HuggingFacePipeline.from_model_id(
      model_id=model_id,
      task="text-generation",
      device_map="cuda",
      pipeline_kwargs=pipeline_kwargs,
      model_kwargs={
        "dtype": "auto",
      }
    )
    self.chat_model = ChatHuggingFace(llm=self.llm)

  def generate(self, messages: list[ChatMessage], **kwargs: Any) -> str:
    response = self.chat_model.invoke(_to_langchain_messages(messages))
    return str(response.content)

class BedrockLLM(LLM):
  def __init__(self, model_id: str, region_name: str = "eu-west-2"):
    self.model_id = model_id
    self.region_name = region_name
    self.chat_model = ChatBedrock(
      model=model_id,
      model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1024,
      },
    )

  def generate(self, messages: list[ChatMessage], **kwargs: Any) -> str:
    lc_messages = _to_langchain_messages(messages)
    if kwargs:
      model_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in GENERATION_CONFIG_KEYS and value is not None
      }
      model_kwargs.setdefault("max_tokens", 1024)
      chat = ChatBedrock(
        model=self.model_id,
        model_kwargs=model_kwargs,
      )
      response = chat.invoke(lc_messages)
    else:
      response = self.chat_model.invoke(lc_messages)
    return str(response.content)

class LocalLLM(LLM):
  def __init__(self, host: str, model_id: str):
    self.chat_model = OpenAI(api_key="test", base_url=host)
    self.model_id = model_id

  def generate(self, messages: list[ChatMessage], **kwargs: Any) -> str:
    create_kwargs: dict[str, Any] = {
      "model": self.model_id,
      "messages": messages,
    }
    for key in GENERATION_CONFIG_KEYS:
      if key in kwargs and kwargs[key] is not None:
        create_kwargs[key] = kwargs[key]
    if "enable_thinking" in kwargs:
      create_kwargs["enable_thinking"] = kwargs["enable_thinking"]
    if "response_format" in kwargs:
      create_kwargs["response_format"] = kwargs["response_format"]
    response = self.chat_model.chat.completions.create(**create_kwargs)
    response = response.choices[0].message.content
    if '</think>' in response:
      response = response.split('</think>')[-1].strip()
    return response

class DeepseekLLM(LLM):
  def __init__(self, model_id: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1"):
    self.model_id = model_id
    self.chat_model = OpenAI(api_key=os.getenv("OR_API_KEY") or "", base_url=base_url, timeout=60, max_retries=3)

  def generate(self, messages: list[ChatMessage], **kwargs: Any) -> str:
    create_kwargs: dict[str, Any] = {
      "model": self.model_id,
      "messages": messages,
      "extra_body": {
        "provider": {
          "order": ["deepseek", "atlas-cloud/fp8", "novita/fp8"],
          "allow_fallbacks": False
        }
      }
    }
    for key in GENERATION_CONFIG_KEYS:
      if key in kwargs and kwargs[key] is not None:
        create_kwargs[key] = kwargs[key]
    response = self.chat_model.chat.completions.create(**create_kwargs)
    return str(response.choices[0].message.content).replace('\n', '')

class LLMFactory:
    @staticmethod
    def create(llm_type: str, base_url: str | None = None, model_id: str | None = None, **kwargs: Any) -> LLM:
        if llm_type == "huggingface":
          if model_id is None:
            raise ValueError("huggingface LLM requires a model_id")
          return HuggingFaceLLM(model_id, generation_config=kwargs.get("generation_config"))
        elif llm_type == "bedrock":
          if model_id is None:
            raise ValueError("bedrock LLM requires a model_id")
          return BedrockLLM(model_id, **{k: v for k, v in kwargs.items() if k == "region_name"})
        elif llm_type == "local":
          if model_id is None or base_url is None:
            raise ValueError("local LLM requires a host model_id/base URL")
          return LocalLLM(base_url, model_id)
        elif llm_type == "deepseek":
          return DeepseekLLM(model_id or "deepseek-chat", base_url=base_url or "https://api.deepseek.com/v1")
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    @staticmethod
    def create_from_config(config: Any) -> LLM:
        if config is None:
          raise ValueError("Missing LLM config block.")
        raw_config = _to_config_mapping(config)
        llm_type = ensure_str(raw_config.get("type"))
        base_url = ensure_str(raw_config.get("base_url")) or None
        model_id = ensure_str(raw_config.get("model_id")) or None
        extra = {
          key: value
          for key, value in raw_config.items()
          if key not in {"type", "base_url", "model_id", "profiles", *GENERATION_CONFIG_KEYS} and value is not None
        }
        generation_config = extract_generation_config(raw_config)
        if generation_config:
          extra["generation_config"] = generation_config
        return LLMFactory.create(llm_type, base_url=base_url, model_id=model_id, **extra)


# Agent interface
class Agent(ABC):
  def __init__(
    self,
    llm: LLM,
    generation_defaults: dict[str, Any] | None = None,
  ):
    self.llm = llm
    self.generation_defaults = extract_generation_config(generation_defaults)

  def _build_generation_kwargs(
    self,
    *,
    default_values: dict[str, Any] | None = None,
    override_values: dict[str, Any] | None = None,
    include_generation_defaults: bool = True,
  ) -> dict[str, Any]:
    generation_kwargs = {
      key: value
      for key, value in (default_values or {}).items()
      if value is not None
    }
    if include_generation_defaults:
      generation_kwargs.update(self.generation_defaults)
    generation_kwargs.update(
      {
        key: value
        for key, value in (override_values or {}).items()
        if value is not None
      }
    )
    return generation_kwargs

  # @abstractmethod
  # def generate(self, context: str) -> str:
  #   pass

  # @abstractmethod
  # def generateOpening(self, context: str) -> str:
  #   pass

  def generate_utterance(self, *args: Any, **kwargs: Any) -> str:
    raise NotImplementedError

  def generate_structured(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    raise NotImplementedError
