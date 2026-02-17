from typing import List, Dict, Any, Set, Optional
import random
from datetime import datetime
from tinytroupe.environment.tiny_world import TinyWorld
from tinytroupe.social_network import NetworkTopology
from tinytroupe.agent.social_types import Content, Reaction
from tinytroupe.agent import TinyPerson
from tinytroupe.agent import logger

class SimulationResult:
    def __init__(self, content: Content, start_time: datetime):
        self.content = content
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.engagements: List[Dict[str, Any]] = []
        self.step_metrics: List[Dict[str, Any]] = []
        self.total_reach = 0
        self.engagement_rate = 0.0
        self.expected_likes = 0
        self.expected_comments = 0
        self.expected_shares = 0
        self.cascade_depth = 0
        self.execution_time = 0.0
        self.avg_sentiment = 0.0
        self.feedback_summary: List[str] = []

    def add_engagement(self, persona_id: str, engagement_type: str, step: int, sentiment: float = 0.0, feedback: str = None):
        self.engagements.append({
            "persona_id": persona_id,
            "type": engagement_type,
            "step": step,
            "sentiment": sentiment,
            "feedback": feedback
        })
        if engagement_type == "like": self.expected_likes += 1
        elif engagement_type == "comment": self.expected_comments += 1
        elif engagement_type == "share": self.expected_shares += 1

        if feedback:
            self.feedback_summary.append(feedback)

    def add_step_metrics(self, step: int, reach: int, engagements: int):
        self.step_metrics.append({
            "step": step,
            "reach": reach,
            "engagements": engagements
        })

    def finalize(self, end_time: datetime):
        self.end_time = end_time
        self.execution_time = (end_time - self.start_time).total_seconds()
        self.total_reach = len(set(e["persona_id"] for e in self.engagements)) # Simplified
        # ... more metrics

class SocialTinyWorld(TinyWorld):
    """Extended TinyWorld with social network capabilities"""

    def __init__(self, name: str, network: NetworkTopology = None, **kwargs):
        super().__init__(name, **kwargs)
        self.network = network or NetworkTopology()
        self.content_items: List[Content] = []
        self.simulation_history: List[SimulationResult] = []
        self.time_step = 0

    def add_content(self, content: Content) -> None:
        """Add content to the world for personas to interact with"""
        self.content_items.append(content)
        self.broadcast(f"New content available: {content.text[:100]}...")

    def simulate_content_spread(self, content: Content,
                               initial_viewers: List[str],
                               max_steps: int = 10) -> SimulationResult:
        """Simulate how content spreads through the network"""

        result = SimulationResult(content=content, start_time=datetime.now())
        viewed = set(initial_viewers)
        engaged = set()

        for step in range(max_steps):
            self.time_step = step
            new_viewers = set()

            for viewer_id in viewed - engaged:
                if viewer_id not in self.network.nodes: continue
                persona = self.network.nodes[viewer_id]

                # Predict reaction (simplified)
                reaction = persona.predict_reaction(content)

                if reaction.will_engage:
                    engaged.add(viewer_id)
                    result.add_engagement(
                        viewer_id,
                        reaction.reaction_type,
                        step,
                        sentiment=reaction.sentiment,
                        feedback=reaction.comment
                    )

                    if reaction.will_share:
                        neighbors = self.network.get_neighbors(viewer_id)
                        new_viewers.update([n.name for n in neighbors])

            viewed.update(new_viewers)
            result.add_step_metrics(step, len(viewed), len(engaged))

            if not new_viewers:
                break

        result.finalize(datetime.now())
        self.simulation_history.append(result)
        return result
