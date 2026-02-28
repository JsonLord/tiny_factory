import pytest
from unittest.mock import MagicMock, patch
from deeppersona.agent import DeepPersona
from deeppersona.agent.social_types import Content, Reaction

@pytest.fixture
def mock_llm():
    with patch("deeppersona.openai_utils.client") as mock:
        client = MagicMock()
        mock.return_value = client
        # Mock send_message for LLMPredictor
        client.send_message.return_value = {
            "content": '{"will_engage": true, "probability": 0.8, "reasoning": "test", "reaction_type": "like", "comment": "nice"}'
        }
        yield client

def test_deep_persona_social_extension(mock_llm):
    DeepPersona.clear_agents()
    person = DeepPersona("Alice")
    person._persona.update({
        "age": 30,
        "occupation": "Engineer",
        "nationality": "German",
        "residence": "Berlin"
    })
    assert hasattr(person, "social_connections")
    assert hasattr(person, "engagement_patterns")

    content = Content(text="Hello world", topics=["tech"], format="text")

    # We need to mock EngagementPredictor.predict as well if we don't want real LLM calls there
    with patch("deeppersona.ml_models.EngagementPredictor.predict") as mock_pred:
        mock_pred.return_value = 0.7
        prob = person.calculate_engagement_probability(content)
        assert prob == 0.7

    reaction = person.predict_reaction(content)
    assert reaction.will_engage is True
    assert reaction.probability == 0.8

def test_simulation_manager(mock_llm):
    from deeppersona.simulation_manager import SimulationManager, SimulationConfig
    manager = SimulationManager()

    # Mock create_simulation to avoid real persona generation
    with patch("deeppersona.factory.deep_persona_factory.DeepPersonaFactory.generate_people") as mock_gen:
        mock_gen.return_value = [DeepPersona("P1"), DeepPersona("P2")]
        config = SimulationConfig(name="Test Sim", persona_count=2)
        sim = manager.create_simulation(config)

    assert sim.id is not None
    assert len(sim.personas) == 2
    assert sim.world is not None

    content = Content(text="Social media post")

    # Mock simulation spread
    with patch("deeppersona.environment.social_deep_world.SocialDeepWorld.simulate_content_spread") as mock_spread:
        from deeppersona.environment.social_deep_world import SimulationResult
        from datetime import datetime
        res = SimulationResult(content, datetime.now())
        res.total_reach = 10
        mock_spread.return_value = res

        result = manager.run_simulation(sim.id, content)
        assert result.total_reach == 10
