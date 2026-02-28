from typing import List, Dict, Any
from deeppersona.integrations.linkedin_api import LinkedInAPI
from deeppersona.agent import DeepPersona
from deeppersona.factory.deep_persona_factory import DeepPersonaFactory

class LinkedInAudienceAnalyzer:
    def __init__(self, linkedin_api: LinkedInAPI):
        self.api = linkedin_api
        self.factory = DeepPersonaFactory()

    def create_audience_personas(self, count: int = 10) -> List[DeepPersona]:
        connections = self.api.get_connections(count=count)
        personas = []
        for conn in connections:
            persona = self.factory.generate_from_linkedin_profile(conn)
            personas.append(persona)
        return personas
