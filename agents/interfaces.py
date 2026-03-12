import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from openai import OpenAI

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
  return (PROMPTS_DIR / filename).read_text().strip()

def _to_langchain_messages(messages: list[dict]) -> list:
  """Convert dict messages to LangChain message objects."""
  lc_messages = []
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


def coerce_float(value: Any, default: float, low: Optional[float] = None, high: Optional[float] = None) -> float:
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


def parse_json_response(raw_text: str) -> tuple[Any, dict]:
  text = (raw_text or "").strip()
  if not text:
    raise StructuredOutputError("empty response")

  try:
    return json.loads(text), {"parsed": True, "mode": "direct"}
  except json.JSONDecodeError as exc:
    raise StructuredOutputError(f"invalid JSON: {exc.msg}") from exc

# LLM interfaces
class LLM(ABC):
  @abstractmethod
  def generate(self, messages: list[dict], **kwargs) -> str:
    pass

class HuggingFaceLLM(LLM):
  def __init__(self, model_id: str):
    self.llm = HuggingFacePipeline.from_model_id(
      model_id=model_id,
      task="text-generation",
      device_map="cuda",
      pipeline_kwargs={
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text": False,
        "max_new_tokens": 1024,
      },
      model_kwargs={
        "dtype": "auto",
      }
    )
    self.chat_model = ChatHuggingFace(llm=self.llm)

  def generate(self, messages: list[dict], **kwargs) -> str:
    response = self.chat_model.invoke(_to_langchain_messages(messages))
    return str(response.content)

class BedrockLLM(LLM):
  def __init__(self, model_id: str, region_name: str = "eu-west-2"):
    self.model_id = model_id
    self.region_name = region_name
    self.chat_model = ChatBedrock(
      model_id=model_id,
      region_name=region_name,
      model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1024,
      },
    )

  def generate(self, messages: list[dict], **kwargs) -> str:
    lc_messages = _to_langchain_messages(messages)
    if kwargs:
      model_kwargs = {"max_tokens": kwargs.get("max_tokens", 1024)}
      if "temperature" in kwargs:
        model_kwargs["temperature"] = kwargs["temperature"]
      if "top_p" in kwargs:
        model_kwargs["top_p"] = kwargs["top_p"]
      chat = ChatBedrock(
        model_id=self.model_id,
        region_name=self.region_name,
        model_kwargs=model_kwargs,
      )
      response = chat.invoke(lc_messages)
    else:
      response = self.chat_model.invoke(lc_messages)
    return str(response.content)

class LocalLLM(LLM):
  def __init__(self, host: str):
    self.chat_model = OpenAI(api_key="test", base_url=host)

  def generate(self, messages: list[dict], **kwargs) -> str:
    create_kwargs = {
      "model": "local",
      "messages": messages,
      "temperature": kwargs.get("temperature", 0.7),
      "max_tokens": kwargs.get("max_tokens", 20000),
    }
    if "top_p" in kwargs:
      create_kwargs["top_p"] = kwargs["top_p"]
    if "response_format" in kwargs:
      create_kwargs["response_format"] = kwargs["response_format"]
    response = self.chat_model.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content

class DeepseekLLM(LLM):
  def __init__(self, model_id: str = "deepseek-chat"):
    self.model_id = model_id
    self.chat_model = OpenAI(api_key=os.getenv("DS_API_KEY"), base_url="https://api.deepseek.com/v1")

  def generate(self, messages: list[dict], **kwargs) -> str:
    create_kwargs = {
      "model": self.model_id,
      "messages": messages,
      "temperature": kwargs.get("temperature", 0.7),
      "max_tokens": kwargs.get("max_tokens", 8192),
    }
    response = self.chat_model.chat.completions.create(**create_kwargs)
    return str(response.choices[0].message.content)

class LLMFactory:
    @staticmethod
    def create(llm_type: str, model_id: str = None, **kwargs):
        if llm_type == "huggingface":
          return HuggingFaceLLM(model_id)
        elif llm_type == "bedrock":
          return BedrockLLM(model_id, **{k: v for k, v in kwargs.items() if k == "region_name"})
        elif llm_type == "local":
          return LocalLLM(model_id)
        elif llm_type == "deepseek":
          return DeepseekLLM(model_id or "deepseek-chat")
        raise ValueError(f"Unsupported LLM type: {llm_type}")


# Agent interface
class Agent(ABC):
  def __init__(self, llm: LLM):
    self.llm = llm

  # @abstractmethod
  # def generate(self, context: str) -> str:
  #   pass

  # @abstractmethod
  # def generateOpening(self, context: str) -> str:
  #   pass

  def generate_utterance(self, *args, **kwargs) -> str:
    raise NotImplementedError

  def generate_structured(self, *args, **kwargs):
    raise NotImplementedError
