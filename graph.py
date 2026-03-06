from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import AnyMessage
from agents.therapist import TherapistAgent
from agents.patient import PatientAgent
from typing import Annotated, Optional
import operator
import re
from omegaconf import OmegaConf
from agents.interfaces import LLMFactory

# State
class State(MessagesState):
  current_turn: Annotated[int, operator.add]
  max_turns: int
  min_turns: int 
  session_ended: Optional[bool]

class TherapyGraphBuilder:
  def __init__(self, config_path, profile: dict):
    self.cfg = OmegaConf.load(config_path)
    self.profile = profile
    self._init_agents()

  def _init_agents(self):
    therapist_llm = LLMFactory.create(self.cfg.therapist.type, self.cfg.therapist.model_id)
    patient_llm = LLMFactory.create(self.cfg.patient.type, self.cfg.patient.model_id)
    self.therapist = TherapistAgent(therapist_llm, self.profile)
    self.patient = PatientAgent(patient_llm, self.profile)
  
  def _format_context(self, messages: list[AnyMessage]) -> str:
      return '\n'.join(list(map(lambda m: m.content, messages)))

  # Nodes
  def generateOpeningTherapist(self, state: State):
    opening = 'Therapist: ' + self.therapist.generateOpening()
    return {'messages': opening, 'current_turn': 1}

  def generateOpeningPatient(self, state: State):
    context = self._format_context(state['messages'])
    opening = 'Patient: ' + self.patient.generateOpening(context)
    return {'messages': opening, 'current_turn': 1}

  def turnTherapist(self, state: State):
    context = self._format_context(state['messages'])
    should_wrap_up = state['current_turn'] >= state['min_turns']
    response = self.therapist.generate(context, wrap_up=should_wrap_up)
    ended = bool(re.search(r'</?\s*end\s*>', response, re.IGNORECASE))
    if ended:
      response = re.sub(r'</?\s*end\s*>', '', response, flags=re.IGNORECASE).strip()
    message = 'Therapist: ' + response
    return {'messages': message, 'current_turn': 1, 'session_ended': ended}

  def turnPatient(self, state: State):
    context = self._format_context(state['messages'])
    message = 'Patient: ' + self.patient.generate(context)
    return {'messages': message, 'current_turn': 1}

  def finalPatient(self, state: State):
    context = self._format_context(state['messages'])
    message = 'Patient: ' + self.patient.generateClosing(context)
    return {'messages': message, 'current_turn': 1}

  # Conditions
  def shouldContinue(self, state: State):
    if state.get('session_ended', False) or state['current_turn'] >= state['max_turns']:
      return 'End'
    return 'Continue'

  # Graph
  def createChain(self):
    workflow = StateGraph(State)

    workflow.add_node('openTherapist', self.generateOpeningTherapist)
    workflow.add_node('openPatient', self.generateOpeningPatient)
    workflow.add_node('turnTherapist', self.turnTherapist)
    workflow.add_node('turnPatient', self.turnPatient)
    workflow.add_node('finalPatient', self.finalPatient)

    workflow.add_edge(START, 'openTherapist')
    workflow.add_edge('openTherapist', 'openPatient')
    workflow.add_edge('openPatient', 'turnTherapist')
    workflow.add_conditional_edges('turnTherapist', self.shouldContinue, {'Continue': 'turnPatient', 'End': 'finalPatient'})
    workflow.add_edge('turnPatient', 'turnTherapist')
    workflow.add_edge('finalPatient', END)

    chain = workflow.compile()
    return chain