"""
This module provides the main classes and functions for DeepPersona's  agents.

Agents are the key abstraction used in DeepPersona. An agent is a simulated person or entity that can interact with other agents and the environment, by
receiving stimuli and producing actions. Agents have cognitive states, which are updated as they interact with the environment and other agents. 
Agents can also store and retrieve information from memory, and can perform actions in the environment. Different from agents whose objective is to
provide support for AI-based assistants or other such productivity tools, **DeepPersona agents aim at representing human-like behavior**, which includes
idiossincracies, emotions, and other human-like traits, that one would not expect from a productivity tool.

The overall underlying design is inspired mainly by Cognitive Psychology, which is why agents have various internal cognitive states, such as attention, emotions, and goals.
It is also why agent memory, differently from other LLM-based agent platforms, has subtle internal divisions, notably between episodic and semantic memory. 
Some behaviorist concepts are also present, such as the explicit and decoupled concepts of "stimulus" and "response" in the `listen` and `act` methods, which are key abstractions
to understand how agents interact with the environment and other agents.
"""

import deeppersona.utils as utils
from pydantic import BaseModel

import logging
logger = logging.getLogger("deeppersona")

from deeppersona import default

###########################################################################
# Types and constants
###########################################################################
from typing import TypeVar, Union
Self = TypeVar("Self", bound="DeepPersona")
AgentOrWorld = Union[Self, "DeepWorld"]


###########################################################################
# Data structures to enforce output format during LLM API call.
###########################################################################
class Action(BaseModel):
    type: str
    content: str
    target: str

class CognitiveState(BaseModel):
    goals: str
    context: list[str]
    attention: str
    emotions: str

class CognitiveActionModel(BaseModel):
    action: Action
    cognitive_state: CognitiveState

class CognitiveActionModelWithReasoning(BaseModel):
    reasoning: str
    action: Action
    cognitive_state: CognitiveState


###########################################################################
# Exposed API
###########################################################################
# from. grounding ... ---> not exposing this, clients should not need to know about detailed grounding mechanisms
from .memory import SemanticMemory, EpisodicMemory, EpisodicConsolidator, ReflectionConsolidator
from .mental_faculty import CustomMentalFaculty, RecallFaculty, FilesAndWebGroundingFaculty, DeepToolUse
from .deep_persona import DeepPersona

__all__ = ["SemanticMemory", "EpisodicMemory", "EpisodicConsolidator", "ReflectionConsolidator",
           "CustomMentalFaculty", "RecallFaculty", "FilesAndWebGroundingFaculty", "DeepToolUse",
           "DeepPersona"]