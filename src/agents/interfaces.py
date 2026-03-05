import os

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from abc import ABC, abstractmethod
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from langchain_aws import ChatBedrock
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
  return (PROMPTS_DIR / filename).read_text().strip()

# LLM interfaces
class LLM(ABC):
  @abstractmethod
  def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
    """Generate a response from system + user prompt.
    Optional kwargs: temperature, top_p, max_tokens, response_format
    """
    pass

  def generate_chat(self, messages: list[dict], **kwargs) -> str:
    """Generate from a full message list (for multi-turn / CAMI STAR pipeline).
    Default implementation converts to system+user form; subclasses may override.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] != "system"]
    system_prompt = "\n".join(system_parts) if system_parts else ""
    user_message = "\n".join(user_parts) if user_parts else ""
    return self.generate(system_prompt, user_message, **kwargs)

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

  def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
    messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_message),
    ]
    response = self.chat_model.invoke(messages)
    return str(response.content)

  def generate_chat(self, messages: list[dict], **kwargs) -> str:
    lc_messages = []
    for m in messages:
      if m["role"] == "system":
        lc_messages.append(SystemMessage(content=m["content"]))
      elif m["role"] == "user":
        lc_messages.append(HumanMessage(content=m["content"]))
      else:
        from langchain.messages import AIMessage
        lc_messages.append(AIMessage(content=m["content"]))
    response = self.chat_model.invoke(lc_messages)
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

  def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
    messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_message),
    ]
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
      response = chat.invoke(messages)
    else:
      response = self.chat_model.invoke(messages)
    return str(response.content)

  def generate_chat(self, messages: list[dict], **kwargs) -> str:
    lc_messages = []
    for m in messages:
      if m["role"] == "system":
        lc_messages.append(SystemMessage(content=m["content"]))
      elif m["role"] == "user":
        lc_messages.append(HumanMessage(content=m["content"]))
      else:
        from langchain.messages import AIMessage
        lc_messages.append(AIMessage(content=m["content"]))
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

  def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    return self._call(messages, **kwargs)

  def generate_chat(self, messages: list[dict], **kwargs) -> str:
    return self._call(messages, **kwargs)

  def _call(self, messages: list[dict], **kwargs) -> str:
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

  def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    return self._call(messages, **kwargs)

  def generate_chat(self, messages: list[dict], **kwargs) -> str:
    return self._call(messages, **kwargs)

  def _call(self, messages: list[dict], **kwargs) -> str:
    create_kwargs = {
      "model": self.model_id,
      "messages": messages,
      "temperature": kwargs.get("temperature", 0.7),
      "max_tokens": kwargs.get("max_tokens", 8192),
    }
    if "top_p" in kwargs:
      create_kwargs["top_p"] = kwargs["top_p"]
    if "response_format" in kwargs:
      create_kwargs["response_format"] = kwargs["response_format"]
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

  @abstractmethod
  def generate(self, context: str) -> str:
    pass

  @abstractmethod
  def generateOpening(self, context: str) -> str:
    pass