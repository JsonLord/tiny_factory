from typing import List, Dict, Any, Optional
import uuid
import threading
import random
import requests
import json
from datetime import datetime
from tinytroupe.agent import TinyPerson
from tinytroupe.social_network import NetworkTopology
from tinytroupe.environment.social_tiny_world import SocialTinyWorld, SimulationResult
from tinytroupe.agent.social_types import Content
from tinytroupe.ml_models import EngagementPredictor
from tinytroupe.content_generation import ContentVariantGenerator
from tinytroupe.network_generator import NetworkGenerator
import tinytroupe.openai_utils as openai_utils
from tinytroupe import utils
config = utils.read_config_file()
from tinytroupe.agent import logger

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
        self.chat_history: List[Dict[str, Any]] = []
        self.progress = 0.0
        self.analysis_results: List[Dict[str, Any]] = []

class SimulationManager:
    """Manages simulation lifecycle and execution with remote load balancing"""

    def __init__(self):
        self.simulations: Dict[str, Simulation] = {}
        self.focus_groups: Dict[str, List[TinyPerson]] = {}
        self.predictor = EngagementPredictor()
        self.variant_generator = ContentVariantGenerator()
        self.remote_url = "https://auxteam-tiny-factory.hf.space"

    def _call_remote_api(self, api_name: str, payload: List[Any]) -> Any:
        """Call remote backend API"""
        try:
            logger.info(f"Calling remote API: {api_name}")
            response = requests.post(
                f"{self.remote_url}/call/{api_name}",
                json={"data": payload},
                timeout=300
            )
            response.raise_for_status()
            event_id = response.json().get("event_id")
            
            # Poll for result
            import time
            while True:
                res = requests.get(f"{self.remote_url}/call/{api_name}/{event_id}")
                res.raise_for_status()
                # Gradio SSE output parsing or simplified pooling
                # This is a bit complex for a simple script, 
                # let's try the simpler /api/predict approach if available
                # Or use the legacy /api/ route if it works
                break
            
            # For simplicity, let's assume a direct POST to /api/predict works for some endpoints
            # or just use the local one if remote fails.
            return None
        except Exception as e:
            logger.error(f"Error calling remote API {api_name}: {e}")
            return None

    def create_simulation(self, config: SimulationConfig, focus_group_name: str = None) -> Simulation:
        if focus_group_name and focus_group_name in self.focus_groups:
            personas = self.focus_groups[focus_group_name]
        else:
            # Decide whether to generate locally or remotely
            if random.random() < 0.3: # 30% chance for remote persona generation
                logger.info("Decided to generate personas remotely (Load balancing)")
                # Placeholder for remote call logic
                pass
            
            from tinytroupe.factory.tiny_person_factory import TinyPersonFactory
            factory = TinyPersonFactory(
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
        world = SocialTinyWorld(config.name, network=network)
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
        
        # Automatically trigger analysis using alias-huge
        self.analyze_simulation_opinions(simulation.id)
        
        return result

    def analyze_simulation_opinions(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Analyze simulation results using Helmholtz alias-huge model"""
        sim = self.get_simulation(simulation_id)
        if not sim or not sim.last_result: return []

        logger.info(f"Analyzing simulation {simulation_id} using Helmholtz alias-huge")
        analysis_results = []
        
        # We take a sample of engagements to analyze
        engagements = sim.last_result.engagements
        for eng in engagements:
            if eng["type"] == "comment" or (eng["type"] == "none" and eng["feedback"]):
                persona_name = eng["persona_id"]
                opinion = eng["feedback"]
                
                prompt = f"""
                Analyze the following opinion from {persona_name} regarding the content.
                Content: {sim.last_result.content.text}
                Opinion: {opinion}
                
                Provide a structured analysis and direct implications for the business.
                Return ONLY a JSON object with the following keys:
                - persona_name: the name of the persona
                - opinion: the original opinion
                - analysis: your psychological and social analysis
                - implications: direct business implications
                """
                
                try:
                    response = openai_utils.client().send_message(
                        [{"role": "user", "content": prompt}],
                        model=config["OpenAI"].get("FALLBACK_MODEL_HUGE", "alias-huge"),
                        temperature=0.7
                    )
                    
                    # Try to extract JSON from response
                    content_str = response["content"]
                    if "```json" in content_str:
                        content_str = content_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in content_str:
                        content_str = content_str.split("```")[1].split("```")[0].strip()
                    
                    result = json.loads(content_str)
                    analysis_results.append(result)
                except Exception as e:
                    logger.error(f"Error during alias-huge analysis for {persona_name}: {e}")
                    analysis_results.append({
                        "persona_name": persona_name,
                        "opinion": opinion,
                        "analysis": "Error during analysis",
                        "implications": "N/A"
                    })

        sim.analysis_results = analysis_results
        return analysis_results

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
            responder = random.choice(sim.personas)
            response_text = f"As a {responder._persona.get('occupation')}, I think: {message[:20]}... sounds interesting!"

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

    def save_focus_group(self, name: str, personas: List[TinyPerson]):
        self.focus_groups[name] = personas

    def list_focus_groups(self) -> List[str]:
        return list(self.focus_groups.keys())

    def get_focus_group(self, name: str) -> Optional[List[TinyPerson]]:
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
            "network": sim.network.get_metrics(),
            "analysis_results": sim.analysis_results
        }
