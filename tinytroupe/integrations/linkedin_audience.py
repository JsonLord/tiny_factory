from typing import List, Dict, Any
from tinytroupe.integrations.linkedin_api import LinkedInAPI
from tinytroupe.agent import TinyPerson
from tinytroupe.factory.tiny_person_factory import TinyPersonFactory

class LinkedInAudienceAnalyzer:
    def __init__(self, linkedin_api: LinkedInAPI):
        self.api = linkedin_api
        self.factory = TinyPersonFactory()

    def create_audience_personas(self, count: int = 10) -> List[TinyPerson]:
        connections = self.api.get_connections(count=count)
        personas = []
        for conn in connections:
            persona = self.factory.generate_from_linkedin_profile(conn)
            personas.append(persona)
        return personas
