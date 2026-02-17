import pytest
import time
from unittest.mock import MagicMock, patch
from tinytroupe.agent import TinyPerson
from tinytroupe.simulation_manager import SimulationManager, SimulationConfig
from tinytroupe.agent.social_types import Content

@pytest.fixture
def mock_llm():
    with patch("tinytroupe.openai_utils.client") as mock:
        client = MagicMock()
        mock.return_value = client
        client.send_message.return_value = {"content": "{}"}
        yield client

def test_async_simulation_and_chat(mock_llm):
    TinyPerson.clear_agents()
    manager = SimulationManager()

    with patch("tinytroupe.factory.tiny_person_factory.TinyPersonFactory.generate_people") as mock_gen:
        mock_gen.return_value = [TinyPerson("P1"), TinyPerson("P2")]
        config = SimulationConfig(name="Async Test", persona_count=2)
        sim = manager.create_simulation(config)

    content = Content(text="Async test post")

    with patch("tinytroupe.environment.social_tiny_world.SocialTinyWorld.simulate_content_spread") as mock_spread:
        from tinytroupe.environment.social_tiny_world import SimulationResult
        from datetime import datetime
        res = SimulationResult(content, datetime.now())
        mock_spread.return_value = res

        manager.run_simulation(sim.id, content, background=True)

    assert sim.status in ["running", "completed"]

    # Test chat
    msg = manager.send_chat_message(sim.id, "User", "Hello personas")
    assert msg["sender"] == "User"

    history = manager.get_chat_history(sim.id)
    assert len(history) >= 1

    # Wait for background task
    max_wait = 5
    while sim.status != "completed" and max_wait > 0:
        time.sleep(0.5)
        max_wait -= 0.5

    assert sim.status == "completed"
    assert sim.progress == 1.0

def test_focus_groups(mock_llm):
    TinyPerson.clear_agents()
    manager = SimulationManager()

    with patch("tinytroupe.factory.tiny_person_factory.TinyPersonFactory.generate_people") as mock_gen:
        mock_gen.return_value = [TinyPerson("P1"), TinyPerson("P2")]
        config = SimulationConfig(name="Base Sim", persona_count=2)
        sim = manager.create_simulation(config)

    # Save focus group
    manager.save_focus_group("MyGroup", sim.personas)
    assert "MyGroup" in manager.list_focus_groups()

    # Create new sim from focus group
    config2 = SimulationConfig(name="Sim 2")
    sim2 = manager.create_simulation(config2, focus_group_name="MyGroup")
    assert len(sim2.personas) == 2
    assert sim2.personas == sim.personas
