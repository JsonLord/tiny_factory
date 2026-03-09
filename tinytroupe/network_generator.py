from typing import List, Dict, Any
import random
from tinytroupe.social_network import NetworkTopology
from tinytroupe.agent import TinyPerson

class NetworkGenerator:
    def __init__(self, personas: List[TinyPerson]):
        self.personas = personas

    def generate_scale_free_network(self, n: int, m: int) -> NetworkTopology:
        """Barabási-Albert model"""
        network = NetworkTopology()
        for p in self.personas:
            network.add_persona(p)

        # Simplified BA model
        # For each new node, connect it to m existing nodes with probability proportional to their degree
        # For now, a very simple version
        persona_names = [p.name for p in self.personas]
        for i, name in enumerate(persona_names):
            if i == 0: continue
            # Connect to some random previous nodes
            targets = random.sample(persona_names[:i], min(i, m))
            for target in targets:
                network.add_connection(name, target, strength=random.random())

        return network

    def generate_small_world_network(self, n: int, k: int, p: float) -> NetworkTopology:
        """Watts-Strogatz model"""
        network = NetworkTopology()
        for persona in self.personas:
            network.add_persona(persona)

        # Simplified WS model
        persona_names = [p.name for p in self.personas]
        num_nodes = len(persona_names)
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                target = persona_names[(i + j) % num_nodes]
                network.add_connection(persona_names[i], target, strength=random.random())

        # Rewiring...
        return network

    def generate_community_network(self, num_communities: int, community_sizes: List[int]) -> NetworkTopology:
        network = NetworkTopology()
        # ...
        return network

    def generate_professional_network(self, personas: List[TinyPerson]) -> NetworkTopology:
        """LinkedIn-style network based on industry, company, role"""
        network = NetworkTopology()
        for p in personas:
            network.add_persona(p)

        for i, p1 in enumerate(personas):
            for p2 in personas[i+1:]:
                # Connect if same industry or similar roles
                if p1._persona.get("occupation") == p2._persona.get("occupation"):
                    if random.random() < 0.3:
                        network.add_connection(p1.name, p2.name, strength=random.random(), relationship_type="colleague")
        return network
