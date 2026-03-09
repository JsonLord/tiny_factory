from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass, field
import tinytroupe.openai_utils as openai_utils
from tinytroupe.agent.social_types import Content

@dataclass
class TraitProfile:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    controversiality_tolerance: float = 0.5
    information_seeking_behavior: float = 0.5
    visual_content_preference: float = 0.5

class TraitBasedBehaviorModel:
    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def compute_action_probability(self, persona: Any, action_type: str, content: Optional[Content] = None) -> float:
        """
        Compute probability of an action based on persona traits.
        """
        traits = persona.behavioral_traits
        if not traits:
            return 0.5

        base_prob = 0.5

        if action_type == "engage":
            # Example logic
            if content and content.format == "video":
                base_prob = self.apply_trait_modifiers(base_prob, {"visual_content_preference": traits.get("visual_content_preference", 0.5)})

            base_prob = self.apply_trait_modifiers(base_prob, {"openness": traits.get("openness", 0.5)})

        return base_prob

    def apply_trait_modifiers(self, base_probability: float, traits: Dict[str, float]) -> float:
        """
        Apply trait modifiers to a base probability.
        """
        prob = base_probability
        for trait, value in traits.items():
            # Simple linear adjustment for now
            # values > 0.5 increase probability, < 0.5 decrease it
            modifier = (value - 0.5) * 0.2
            prob += modifier

        return max(0.0, min(1.0, prob))

    def generate_trait_profile_from_description(self, description: str) -> Dict[str, float]:
        """
        Use LLM to infer traits from persona descriptions.
        """
        prompt = f"""
        Analyze the following persona description and infer their behavioral traits on a scale of 0.0 to 1.0.

        Description: {description}

        Traits to infer:
        - openness (Openness to new ideas/novel content)
        - conscientiousness (Posting regularity, thoughtfulness)
        - extraversion (Sharing frequency, network activity)
        - agreeableness (Commenting positivity, conflict avoidance)
        - controversiality_tolerance (Engagement with divisive topics)
        - information_seeking_behavior (Long-form vs short-form preference)
        - visual_content_preference (Image/video vs text preference)

        Provide the result in JSON format.
        """

        response = openai_utils.client().send_message(
            [
                {"role": "system", "content": "You are an expert psychologist and persona modeler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        try:
            traits = json.loads(response["content"])
            return traits
        except Exception:
            return TraitProfile().__dict__
