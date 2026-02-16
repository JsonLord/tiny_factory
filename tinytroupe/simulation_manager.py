from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from tinytroupe.agent import TinyPerson
from tinytroupe.social_network import NetworkTopology
from tinytroupe.environment.social_tiny_world import SocialTinyWorld, SimulationResult
from tinytroupe.agent.social_types import Content
from tinytroupe.ml_models import EngagementPredictor
from tinytroupe.content_generation import ContentVariantGenerator
from tinytroupe.network_generator import NetworkGenerator

class SimulationConfig:
    def __init__(self, name: str, persona_count: int = 10, network_type: str = "scale_free", **kwargs):
        self.name = name
        self.persona_count = persona_count
        self.network_type = network_type
        self.user_id = kwargs.get("user_id")

class Simulation:
    def __init__(self, id: str, config: SimulationConfig, world: SocialTinyWorld, personas: List[TinyPerson], network: NetworkTopology):
        self.id = id
        self.config = config
        self.world = world
        self.personas = personas
        self.network = network
        self.status = "ready"
        self.created_at = datetime.now()
        self.last_result: Optional[SimulationResult] = None

class SimulationManager:
    """Manages simulation lifecycle and execution"""

    def __init__(self):
        self.simulations: Dict[str, Simulation] = {}
        self.predictor = EngagementPredictor()
        self.variant_generator = ContentVariantGenerator()

    def create_simulation(self, config: SimulationConfig) -> Simulation:
        # For now, we use existing agents or create simple ones
        personas = [TinyPerson(f"Person_{i}") for i in range(config.persona_count)]

        # Generate network
        net_gen = NetworkGenerator(personas)
        if config.network_type == "scale_free":
            network = net_gen.generate_scale_free_network(config.persona_count, 2)
        else:
            network = net_gen.generate_small_world_network(config.persona_count, 4, 0.1)

        # Create world
        world = SocialTinyWorld(config.name, network=network)
        for persona in personas:
            world.add_agent(persona)

        sim_id = str(uuid.uuid4())
        simulation = Simulation(sim_id, config, world, personas, network)
        self.simulations[sim_id] = simulation
        return simulation

    def run_simulation(self, simulation_id: str, content: Content, mode: str = "full") -> SimulationResult:
        if simulation_id not in self.simulations:
            raise ValueError(f"Simulation {simulation_id} not found.")

        simulation = self.simulations[simulation_id]
        simulation.status = "running"

        initial_viewers = [p.name for p in simulation.personas[:5]] # Seed with first 5
        result = simulation.world.simulate_content_spread(content, initial_viewers)

        simulation.status = "completed"
        simulation.last_result = result
        return result

    def get_simulation(self, simulation_id: str, user_id: str = None) -> Optional[Simulation]:
        return self.simulations.get(simulation_id)

    def list_simulations(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": sim.id,
                "name": sim.config.name,
                "status": sim.status,
                "persona_count": len(sim.personas),
                "created_at": sim.created_at.isoformat()
            }
            for sim in self.simulations.values()
        ]

    def get_persona(self, simulation_id: str, persona_name: str) -> Optional[Dict[str, Any]]:
        sim = self.get_simulation(simulation_id)
        if not sim: return None
        for p in sim.personas:
            if p.name == persona_name:
                return p._persona
        return None

    def list_personas(self, simulation_id: str) -> List[Dict[str, Any]]:
        sim = self.get_simulation(simulation_id)
        if not sim: return []
        return [p._persona for p in sim.personas]

    def delete_simulation(self, simulation_id: str) -> bool:
        if simulation_id in self.simulations:
            del self.simulations[simulation_id]
            return True
        return False

    def export_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        sim = self.get_simulation(simulation_id)
        if not sim: return None
        return {
            "id": sim.id,
            "config": {
                "name": sim.config.name,
                "persona_count": sim.config.persona_count,
                "network_type": sim.config.network_type
            },
            "status": sim.status,
            "created_at": sim.created_at.isoformat(),
            "personas": [p._persona for p in sim.personas],
            "network": sim.network.get_metrics()
        }
