from interfaces import LLM, Agent, load_prompt
from langchain_core.prompts import PromptTemplate

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