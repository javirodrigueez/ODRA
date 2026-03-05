from interfaces import LLM, Agent, load_prompt
from langchain_core.prompts import PromptTemplate

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