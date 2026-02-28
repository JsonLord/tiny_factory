from typing import List, Dict, Any, Optional
import uuid
import threading
from datetime import datetime
from deeppersona.agent import DeepPersona
from deeppersona.social_network import NetworkTopology
from deeppersona.environment.social_deep_world import SocialDeepWorld, SimulationResult
from deeppersona.agent.social_types import Content
from deeppersona.ml_models import EngagementPredictor
from deeppersona.content_generation import ContentVariantGenerator
from deeppersona.network_generator import NetworkGenerator

class SimulationConfig:
    def __init__(self, name: str, persona_count: int = 10, network_type: str = "scale_free", **kwargs):
        self.name = name
        self.persona_count = persona_count
        self.network_type = network_type
        self.user_id = kwargs.get("user_id")

class Simulation:
    def __init__(self, id: str, config: SimulationConfig, world: SocialDeepWorld, personas: List[DeepPersona], network: NetworkTopology):
        self.id = id
        self.config = config
        self.world = world
        self.personas = personas
        self.network = network
        self.status = "ready"
        self.created_at = datetime.now()
        self.last_result: Optional[SimulationResult] = None
        self.chat_history: List[Dict[str, Any]] = []
        self.progress = 0.0

class SimulationManager:
    """Manages simulation lifecycle and execution"""

    def __init__(self):
        self.simulations: Dict[str, Simulation] = {}
        self.focus_groups: Dict[str, List[DeepPersona]] = {}
        self.predictor = EngagementPredictor()
        self.variant_generator = ContentVariantGenerator()

    def create_simulation(self, config: SimulationConfig, focus_group_name: str = None) -> Simulation:
        if focus_group_name and focus_group_name in self.focus_groups:
            personas = self.focus_groups[focus_group_name]
        else:
            from deeppersona.factory import DeepPersonaFactory
            factory = DeepPersonaFactory(
                context=config.name,
                total_population_size=config.persona_count
            )
            personas = factory.generate_people(number_of_people=config.persona_count)

        # Generate network
        net_gen = NetworkGenerator(personas)
        if config.network_type == "scale_free":
            network = net_gen.generate_scale_free_network(config.persona_count, 2)
        else:
            network = net_gen.generate_small_world_network(config.persona_count, 4, 0.1)

        # Create world
        world = SocialDeepWorld(config.name, network=network)
        for persona in personas:
            world.add_agent(persona)

        sim_id = str(uuid.uuid4())
        simulation = Simulation(sim_id, config, world, personas, network)
        self.simulations[sim_id] = simulation
        return simulation

    def run_simulation(self, simulation_id: str, content: Content, mode: str = "full", background: bool = False) -> Optional[SimulationResult]:
        if simulation_id not in self.simulations:
            raise ValueError(f"Simulation {simulation_id} not found.")

        simulation = self.simulations[simulation_id]

        if background:
            thread = threading.Thread(target=self._run_simulation_task, args=(simulation, content))
            thread.start()
            return None
        else:
            return self._run_simulation_task(simulation, content)

    def _run_simulation_task(self, simulation: Simulation, content: Content) -> SimulationResult:
        simulation.status = "running"
        simulation.progress = 0.1

        initial_viewers = [p.name for p in simulation.personas[:5]] # Seed with first 5

        # In a real async scenario, simulate_content_spread would update progress
        result = simulation.world.simulate_content_spread(content, initial_viewers)

        simulation.status = "completed"
        simulation.progress = 1.0
        simulation.last_result = result
        return result

    def send_chat_message(self, simulation_id: str, sender: str, message: str) -> Dict[str, Any]:
        sim = self.get_simulation(simulation_id)
        if not sim: raise ValueError(f"Simulation {simulation_id} not found.")

        msg = {
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        sim.chat_history.append(msg)

        # Trigger persona responses if it's a "User" message
        if sender == "User":
            # For now, pick a random persona to respond
            import random
            responder = random.choice(sim.personas)
            # In a real implementation, the persona would "think" and "act"
            response_text = f"As a {responder._persona.get('occupation')}, I think: {message[:10]}... sounds interesting!"

            response_msg = {
                "sender": responder.name,
                "message": response_text,
                "timestamp": datetime.now().isoformat()
            }
            sim.chat_history.append(response_msg)

        return msg

    def get_chat_history(self, simulation_id: str) -> List[Dict[str, Any]]:
        sim = self.get_simulation(simulation_id)
        if not sim: return []
        return sim.chat_history

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

    def save_focus_group(self, name: str, personas: List[DeepPersona]):
        self.focus_groups[name] = personas

    def list_focus_groups(self) -> List[str]:
        return list(self.focus_groups.keys())

    def get_focus_group(self, name: str) -> Optional[List[DeepPersona]]:
        return self.focus_groups.get(name)

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
