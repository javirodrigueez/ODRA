from .interfaces import LLM, Agent, load_prompt
from langchain_core.prompts import PromptTemplate

class TherapistAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    self._system_template = self._build_template("therapist_system.txt")

  def _build_template(self, filename: str) -> PromptTemplate:
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

  def _get_system_prompt(self, conversation: str) -> str:
    return self._system_template.format(conversation=conversation)

  def generate(self, context: str, wrap_up: bool = False) -> str:
    system_prompt = self._get_system_prompt(context)
    user_msg = "Continue the conversation. Produce only your next utterance as the therapist."
    if wrap_up:
      user_msg += " The session has been going on for a while. Start wrapping up the session naturally. When you decide to end, include the </end> tag at the very end of your response."
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_msg},
    ]
    return self.llm.generate(messages)

  def generateOpening(self, context: str = "") -> str:
    system_prompt = self._get_system_prompt(context)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "Produce a brief opening greeting to welcome your patient at the start of the therapy session. Keep it to 1 sentence."},
    ]
    return self.llm.generate(messages)