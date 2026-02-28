import pytest
from unittest.mock import MagicMock, patch
from app import identify_personas

@pytest.fixture
def mock_llm():
    with patch("deeppersona.openai_utils.client") as mock:
        client = MagicMock()
        mock.return_value = client
        # Mock indices returned by select_relevant_personas_utility
        # The llm decorator with justification enabled expects a specific JSON structure
        client.send_message.return_value = {
            "content": '{"justification": "test", "value": [0, 2], "confidence": 1.0}'
        }
        yield client

def test_identify_personas(mock_llm):
    # Mock load_persona_base and load_example_personas
    with patch("app.load_persona_base") as mock_tresor, \
         patch("app.load_example_personas") as mock_examples:

        mock_tresor.return_value = [{"name": "Tresor 1"}, {"name": "Tresor 2"}]
        mock_examples.return_value = [{"name": "Example 1"}, {"name": "Example 2"}]

        # total 4 personas: 0: Tresor 1, 1: Tresor 2, 2: Example 1, 3: Example 2

        result = identify_personas("testing context")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Tresor 1"
        assert result[1]["name"] == "Example 1"
