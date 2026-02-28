import pytest
from unittest.mock import patch, MagicMock
import os

import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import extract_persona_parameters, generate_personas

def test_extract_persona_parameters_fallback():
    # If the LLM call fails or returns empty, the fallback should return
    with patch('tinytroupe.openai_utils.client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.send_message.return_value = None
        mock_client.return_value = mock_instance

        result = extract_persona_parameters("Test Business", "Test Customer")
        assert "age" in result
        assert result["age"] == 30

def test_extract_persona_parameters_success():
    with patch('tinytroupe.openai_utils.client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.send_message.return_value = {
            "content": '{"age": 25, "gender": "Female", "occupation": "Engineer", "city": "NYC", "country": "USA", "custom_values": "Innovation", "custom_life_attitude": "Positive", "life_story": "A story", "interests_hobbies": "Coding", "attribute_count": 200}'
        }
        mock_client.return_value = mock_instance

        result = extract_persona_parameters("Tech Startup", "Young professionals")
        assert result["age"] == 25
        assert result["gender"] == "Female"
        assert result["city"] == "NYC"

@patch('gradio_client.Client') # Mocking gradio_client Client
def test_generate_personas(mock_client_class):
    mock_client_instance = MagicMock()
    mock_client_instance.predict.return_value = "Generated persona profile text"
    mock_client_class.return_value = mock_client_instance

    with patch('app.extract_persona_parameters') as mock_extract:
        mock_extract.return_value = {
            "age": 25, "gender": "Female", "occupation": "Engineer",
            "city": "NYC", "country": "USA", "custom_values": "Innovation",
            "custom_life_attitude": "Positive", "life_story": "A story",
            "interests_hobbies": "Coding", "attribute_count": 200
        }

        # We need an API key to pass the check
        result = generate_personas("Tech Startup", "Young professionals", 1, blablador_api_key="TEST_KEY")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "parameters_used" in result[0]
        assert "persona_profile" in result[0]
        assert result[0]["persona_profile"] == "Generated persona profile text"
