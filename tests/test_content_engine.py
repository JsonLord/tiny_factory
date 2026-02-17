import pytest
from unittest.mock import MagicMock, patch
from tinytroupe.content_generation import ContentVariantGenerator
from tinytroupe.variant_optimizer import VariantOptimizer
from tinytroupe.ml_models import EngagementPredictor
from tinytroupe.agent.social_types import Content
from tinytroupe.agent import TinyPerson

@pytest.fixture
def mock_llm():
    with patch("tinytroupe.openai_utils.client") as mock:
        client = MagicMock()
        mock.return_value = client
        client.send_message.return_value = {"content": "Rewritten content variant"}
        yield client

def test_content_variant_generation(mock_llm):
    generator = ContentVariantGenerator()
    original = "This is a test post."
    variants = generator.generate_variants(original, num_variants=3)

    assert len(variants) == 3
    assert variants[0].text == "Rewritten content variant"
    assert variants[0].original_content == original

def test_variant_optimization(mock_llm):
    TinyPerson.clear_agents()
    predictor = EngagementPredictor()
    optimizer = VariantOptimizer(predictor)

    personas = [TinyPerson("User1"), TinyPerson("User2")]
    for p in personas:
        p._persona.update({"age": 25, "occupation": "Tester", "nationality": "US", "residence": "NY"})

    from tinytroupe.content_generation import ContentVariant
    variants = [
        ContentVariant("Variant 1", "strategy", {}, "original"),
        ContentVariant("Variant 2", "strategy", {}, "original")
    ]

    # Mock predictor to return different scores for different variants
    with patch.object(EngagementPredictor, 'predict') as mock_predict:
        mock_predict.side_effect = [0.8, 0.7, 0.4, 0.5] # V1 for P1, P2; V2 for P1, P2

        from tinytroupe.social_network import NetworkTopology
        network = NetworkTopology()

        ranked = optimizer.rank_variants_for_audience(variants, personas, network)

        assert len(ranked) == 2
        assert ranked[0].variant.text == "Variant 1"
        assert ranked[0].score == 0.75 # (0.8 + 0.7) / 2
        assert ranked[1].variant.text == "Variant 2"
        assert ranked[1].score == 0.45 # (0.4 + 0.5) / 2
