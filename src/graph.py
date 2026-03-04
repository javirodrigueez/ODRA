from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import AnyMessage
from agents import TherapistAgent, PatientAgent
from typing import Annotated
import operator
from omegaconf import OmegaConf
from agents import LLMFactory

# State
class State(MessagesState):
  current_turn: Annotated[int, operator.add]
  max_turns: int 

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
    message = 'Therapist: ' + self.therapist.generate(context)
    return {'messages': message, 'current_turn': 1}

  def turnPatient(self, state: State):
    context = self._format_context(state['messages'])
    message = 'Patient: ' + self.patient.generate(context)
    return {'messages': message, 'current_turn': 1}

  # Conditions
  def shouldContinue(self, state: State):
    if state['current_turn'] >= state['max_turns']:
      return 'Fail'
    if "goodbye" in state['messages'][-2].content.lower() or "goodbye" in state['messages'][-1].content.lower() or "good bye" in state['messages'][-2].content.lower() or "good bye" in state['messages'][-1].content.lower():  # check last therapist/client message
      return 'Fail'
    return 'Pass'

  # Graph
  def createChain(self):
    workflow = StateGraph(State)

    workflow.add_node('openTherapist', self.generateOpeningTherapist)
    workflow.add_node('openPatient', self.generateOpeningPatient)
    workflow.add_node('turnTherapist', self.turnTherapist)
    workflow.add_node('turnPatient', self.turnPatient)

    workflow.add_edge(START, 'openTherapist')
    workflow.add_edge('openTherapist', 'openPatient')
    workflow.add_edge('openPatient', 'turnTherapist')
    workflow.add_edge('turnTherapist', 'turnPatient')
    workflow.add_conditional_edges('turnPatient', self.shouldContinue, {'Fail': END, 'Pass': 'turnTherapist'})  # patient is always the last one

    chain = workflow.compile()
    return chain