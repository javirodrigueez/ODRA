from .interfaces import LLM, Agent, load_prompt
from langchain_core.prompts import PromptTemplate

ATTITUDE_DESCRIPTIONS = {
  "negative": "Express reluctance, skepticism, or emotional resistance toward the therapeutic process",
  "positive": "Be open, cooperative, and willing to engage with the therapeutic process",
  "neutral": "Be balanced and measured in your emotional responses",
}

class PatientAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    self._system_template = self._build_template("patient_system.txt")

  def _build_template(self, filename: str) -> PromptTemplate:
    template = load_prompt(filename)
    prompt_template = PromptTemplate.from_template(template)
    attitude = self.profile.get("attitude", "neutral")
    return prompt_template.partial(
      persona=self._format_persona(),
      more_info=self._format_more_info(),
      patterns=self._format_patterns(),
      presenting_problem=self.profile["presenting_problem"],
      reason_seeking_counseling=self.profile["reason_seeking_counseling"],
      attitude=attitude,
      attitude_description=ATTITUDE_DESCRIPTIONS.get(attitude, "Respond naturally"),
    )

  def _format_persona(self) -> str:
    info = self.profile["personal_info"]
    lines = []
    for key, val in info.items():
      label = key.replace("_", " ").title()
      lines.append(f"{label}: {val}")
    return "\n".join(lines)

  def _format_more_info(self) -> str:
    info = self.profile["more_information"]
    sections = [
      ("Past History", info.get("past_history", "")),
      ("Academic/Occupational Functioning", info.get("academic_occupational_functioning", "")),
      ("Social Support System", info.get("social_support_system", "")),
    ]
    return "\n\n".join(f"{title}:\n{content}" for title, content in sections if content)

  def _format_patterns(self) -> str:
    return "\n".join(f"- {p}" for p in self.profile["patterns"])

  def _get_system_prompt(self, conversation: str) -> str:
    return self._system_template.format(conversation=conversation)

  def generate(self, context: str) -> str:
    system_prompt = self._get_system_prompt(context)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "Continue the conversation. Produce only your next utterance as the patient."},
    ]
    return self.llm.generate(messages)

  def generateOpening(self, context: str) -> str:
    system_prompt = self._get_system_prompt(context)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "Your therapist just greeted you. Respond with a brief opening utterance (1-2 sentences)."},
    ]
    return self.llm.generate(messages)

  def generateClosing(self, context: str) -> str:
    system_prompt = self._get_system_prompt(context)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "The therapist is ending the session. Produce a brief closing response (1-2 sentences) to wrap up the conversation."},
    ]
    return self.llm.generate(messages)