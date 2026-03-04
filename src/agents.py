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
  def generate(self, system_prompt: str, user_message: str) -> str:
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

  def generate(self, system_prompt: str, user_message: str) -> str:
    messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_message),
    ]
    response = self.chat_model.invoke(messages)
    return str(response.content)

class BedrockLLM(LLM):
  def __init__(self, model_id: str, region_name: str = "eu-west-2"):
    self.chat_model = ChatBedrock(
      model_id=model_id,
      region_name=region_name,
      model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1024,
      },
    )

  def generate(self, system_prompt: str, user_message: str) -> str:
    messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_message),
    ]
    response = self.chat_model.invoke(messages)
    return str(response.content)

class LocalLLM(LLM):
  def __init__(self, host: str):
    self.chat_model = OpenAI(api_key="test", base_url=host)

  def generate(self, system_prompt: str, user_message: str) -> str:
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    
    response = self.chat_model.chat.completions.create(
      model="local",
      messages=messages,
      temperature=0.7,
      max_tokens=20000,
    )
    reasoning = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    # if reasoning:
    #   print(f"  [LocalLLM reasoning] {reasoning}")
    return content

class DeepseekLLM(LLM):
  def __init__(self):
    self.chat_model = OpenAI(api_key=os.getenv("DS_API_KEY"), base_url="https://api.deepseek.com/v1")

  def generate(self, system_prompt: str, user_message: str) -> str:
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    #response = self.chat_model.invoke(messages)
    response = self.chat_model.chat.completions.create(
      model="deepseek-chat",
      messages=messages,
      temperature=0.7,
      max_tokens=8192,
    )
    return str(response.choices[0].message.content)

class LLMFactory:
    @staticmethod
    def create(llm_type: str, model_id: str):
        if llm_type == "huggingface":
          return HuggingFaceLLM(model_id)
        elif llm_type == "bedrock":
          return BedrockLLM(model_id)
        elif llm_type == "local":
          return LocalLLM(model_id)
        elif llm_type == "deepseek":
          return DeepseekLLM(model_id)
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

class TherapistAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    self._opening_system = self._render_prompt("therapist_opening.txt")
    self._system_prompt = self._render_prompt("therapist_system.txt")

  def _render_prompt(self, filename: str) -> str:
    template = load_prompt(filename)
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(
      goal=self.profile["topic"],
      behavior=self.profile["Behavior"],
    )

  def generateOpening(self, context: str = "") -> str:
    user_message = context if context else "Begin the therapy session."
    return self.llm.generate(self._opening_system, user_message)
  
  def generate(self, context: str) -> str:
    return self.llm.generate(self._system_prompt, context)

class PatientAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    self._opening_system = self._render_prompt("patient_opening.txt")
    self._system_prompt = self._render_prompt("patient_system.txt")

  def _render_prompt(self, filename: str) -> str:
    template = load_prompt(filename)
    prompt_template = PromptTemplate.from_template(template)
    personas = "\n".join(f"- {p}" for p in self.profile["Personas"])
    reference = self._format_reference()
    return prompt_template.format(
      goal=self.profile["topic"],
      behavior=self.profile["Behavior"],
      personas=personas,
      reference=reference,
    )

  def _format_reference(self) -> str:
    speakers = self.profile["speakers"]
    utterances = self.profile["utterances"]
    lines = []
    for speaker, utterance in zip(speakers, utterances):
      role = "Therapist" if speaker == "counselor" else "Client"
      lines.append(f"{role}: {utterance}")
    return "\n".join(lines)

  def generateOpening(self, context: str) -> str:
    return self.llm.generate(self._opening_system, context)
  
  def generate(self, context: str) -> str:
    return self.llm.generate(self._system_prompt, context)