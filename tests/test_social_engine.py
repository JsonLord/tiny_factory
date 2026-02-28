import pytest
from deeppersona.social_network import NetworkTopology
from deeppersona.network_generator import NetworkGenerator
from deeppersona.influence import InfluencePropagator
from deeppersona.agent import DeepPersona
from deeppersona.agent.social_types import Content

def test_network_topology():
    DeepPersona.clear_agents()
    topo = NetworkTopology()
    p1 = DeepPersona("Alice")
    p2 = DeepPersona("Bob")

    topo.add_persona(p1)
    topo.add_persona(p2)

    topo.add_connection("Alice", "Bob", strength=0.9, relationship_type="friend")

    assert "Alice" in topo.nodes
    assert "Bob" in topo.nodes
    assert len(topo.edges) == 1
    assert "Bob" in p1.social_connections
    assert p1.social_connections["Bob"].strength == 0.9

def test_network_generation():
    DeepPersona.clear_agents()
    personas = [DeepPersona(f"P{i}") for i in range(10)]
    gen = NetworkGenerator(personas)

    sf_net = gen.generate_scale_free_network(10, 2)
    assert len(sf_net.nodes) == 10
    assert len(sf_net.edges) > 0

    sw_net = gen.generate_small_world_network(10, 4, 0.1)
    assert len(sw_net.nodes) == 10
    assert len(sw_net.edges) > 0

def test_influence_propagation():
    DeepPersona.clear_agents()
    topo = NetworkTopology()
    personas = [DeepPersona(f"P{i}") for i in range(5)]
    for p in personas:
        topo.add_persona(p)
        # Give them high engagement probability to ensure propagation in test
        p.engagement_patterns["overall_rate"] = 1.0
        p._persona.update({"age": 30, "occupation": "User", "nationality": "US", "residence": "CA"})

    # Create a line of connections: P0 -> P1 -> P2 -> P3 -> P4
    for i in range(4):
        topo.add_connection(f"P{i}", f"P{i+1}", strength=1.0)

    propagator = InfluencePropagator(topo)
    content = Content(text="Viral message", topics=["test"])

    # Mock calculate_engagement_probability to always return high value
    from unittest.mock import patch
    with patch.object(DeepPersona, 'calculate_engagement_probability', return_value=0.8):
        result = propagator.propagate(["P0"], content)

        assert result.total_reach > 1
        assert "P0" in result.activated_personas
        # Since steps are limited and it's probabilistic (though we mocked it high), check we reached some depth
        assert result.cascade_depth >= 1
