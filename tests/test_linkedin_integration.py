import pytest
from unittest.mock import MagicMock, patch
from deeppersona.integrations.linkedin_audience import LinkedInAudienceAnalyzer
from deeppersona.integrations.linkedin_api import LinkedInAPI
from deeppersona.agent import DeepPersona

def test_linkedin_audience_analysis():
    mock_api = MagicMock(spec=LinkedInAPI)
    mock_api.get_connections.return_value = [
        {"id": "c1", "headline": "CEO", "industry": "Tech", "location": "SF", "career_level": "Senior"},
        {"id": "c2", "headline": "Dev", "industry": "Tech", "location": "NY", "career_level": "Mid"}
    ]

    analyzer = LinkedInAudienceAnalyzer(mock_api)

    # Mock DeepPersonaFactory.generate_person to avoid LLM calls
    with patch("deeppersona.factory.deep_persona_factory.DeepPersonaFactory.generate_person") as mock_gen:
        def side_effect(agent_particularities=None, **kwargs):
            name = f"Persona_{mock_gen.call_count}"
            p = DeepPersona(name)
            return p
        mock_gen.side_effect = side_effect

        personas = analyzer.create_audience_personas(count=2)

        assert len(personas) == 2
        assert mock_api.get_connections.called
        assert mock_gen.call_count == 2
