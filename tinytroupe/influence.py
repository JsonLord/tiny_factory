from typing import List, Set, Dict, Any, Tuple
from dataclasses import dataclass
from tinytroupe.social_network import NetworkTopology
from tinytroupe.agent.social_types import Content

@dataclass
class PropagationResult:
    activated_personas: Set[str]
    activation_times: Dict[str, int]
    total_reach: int
    cascade_depth: int
    engagement_by_time: List[int]

class InfluencePropagator:
    def __init__(self, network: NetworkTopology, model: str = "cascade"):
        self.network = network
        self.model = model
        self.max_steps = 10

    def propagate(self, seed_personas: List[str], content: Content) -> PropagationResult:
        """Main propagation simulation"""
        activated = set(seed_personas)
        activation_times = {pid: 0 for pid in seed_personas}

        for time_step in range(1, self.max_steps + 1):
            newly_activated = self._propagate_step(activated, content, time_step)
            if not newly_activated:
                break
            for pid in newly_activated:
                activation_times[pid] = time_step
            activated.update(newly_activated)

        return PropagationResult(
            activated_personas=activated,
            activation_times=activation_times,
            total_reach=len(activated),
            cascade_depth=max(activation_times.values()) if activation_times else 0,
            engagement_by_time=[] # TODO
        )

    def _propagate_step(self, activated: Set[str], content: Content, time: int) -> Set[str]:
        """Single step of propagation"""
        newly_activated = set()
        for pid in activated:
            # Check neighbors of activated personas
            neighbors = self.network.get_neighbors(pid)
            for neighbor in neighbors:
                if neighbor.name not in activated and neighbor.name not in newly_activated:
                    # Decide if neighbor activates
                    prob = neighbor.calculate_engagement_probability(content)
                    if prob > 0.7: # Higher threshold for viral spread
                        newly_activated.add(neighbor.name)
        return newly_activated

    def calculate_influence_score(self, persona_id: str) -> float:
        """Calculate overall influence of a persona"""
        if persona_id not in self.network.nodes: return 0.0
        # Combine: centrality, follower quality
        return 0.5
