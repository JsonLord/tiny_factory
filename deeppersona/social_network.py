from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np
from datetime import datetime
from deeppersona.agent import DeepPersona
from deeppersona.agent.social_types import ConnectionEdge

class Community:
    """Represents a cluster of closely connected personas"""
    def __init__(self, community_id: str, members: List[str]):
        self.community_id = community_id
        self.members = members
        self.density: float = 0.0
        self.central_personas: List[str] = []
        self.shared_interests: List[str] = []
        self.avg_engagement_rate: float = 0.0

class NetworkTopology:
    """Represents the entire social network structure"""
    def __init__(self):
        self.nodes: Dict[str, DeepPersona] = {}  # persona_id -> persona
        self.edges: List[ConnectionEdge] = []
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.influence_matrix: Optional[np.ndarray] = None
        self.communities: List[Community] = []

    def add_persona(self, persona: DeepPersona) -> None:
        self.nodes[persona.name] = persona
        # Update adjacency matrix if necessary

    def add_connection(self, source_id: str, target_id: str, **kwargs) -> ConnectionEdge:
        connection = ConnectionEdge(connection_id=f"{source_id}_{target_id}", **kwargs)
        self.edges.append(connection)

        # Also update the persona's internal social_connections
        if source_id in self.nodes:
            self.nodes[source_id].social_connections[target_id] = connection

        return connection

    def remove_connection(self, source_id: str, target_id: str) -> None:
        self.edges = [e for e in self.edges if not (e.connection_id == f"{source_id}_{target_id}")]
        if source_id in self.nodes:
            self.nodes[source_id].social_connections.pop(target_id, None)

    def get_neighbors(self, persona_id: str, depth: int = 1) -> List[DeepPersona]:
        if depth <= 0: return []

        neighbors = []
        if persona_id in self.nodes:
            neighbor_ids = list(self.nodes[persona_id].social_connections.keys())
            neighbors = [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

            if depth > 1:
                for nid in neighbor_ids:
                    neighbors.extend(self.get_neighbors(nid, depth - 1))

        return list(set(neighbors))

    def calculate_centrality_metrics(self) -> Dict[str, float]:
        # Placeholder for centrality calculation
        return {name: 0.0 for name in self.nodes}

    def detect_communities(self) -> List[Community]:
        # Placeholder for community detection
        return self.communities

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "density": len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0
        }
