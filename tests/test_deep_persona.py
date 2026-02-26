import pytest
from unittest.mock import MagicMock, patch
from tinytroupe.factory.tiny_person_factory import TinyPersonFactory

@pytest.fixture
def mock_client():
    with patch("tinytroupe.openai_utils.client") as mock:
        client = MagicMock()
        mock.return_value = client
        # Return a simple persona JSON
        client.send_message.return_value = {
            "role": "assistant",
            "content": '{"name": "Deep Persona", "age": 30, "gender": "male", "nationality": "German", "residence": "Berlin", "occupation": "Architect", "education": "Master", "style": "formal", "personality": "meticulous", "preferences": ["coffee"], "beliefs": ["quality"], "skills": ["design"], "behaviors": ["working"], "health": "good", "relationships": [], "other_facts": []}'
        }
        yield client

def test_deep_persona_sequential_calls(mock_client):
    factory = TinyPersonFactory(context="Context")

    # We expect generate_person to be called,
    # which internally calls several LLM steps.
    # Total send_message calls should be 4:
    # 1. Name generation (_unique_full_name)
    # 2. Initial profile generation (generate_person -> _aux_model_call, approx 100 depth)
    # 3. Enrichment step (_generate_deep_persona_internal -> _aux_model_call, approx 350 depth)
    # 4. Extended summary generation (person.minibio())

    person = factory.generate_person(deep_persona=True)

    assert person is not None
    assert person.name == "Deep Persona"
    assert mock_client.send_message.call_count == 4
