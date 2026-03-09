import json
from typing import Dict, Any, Optional
from tinytroupe.agent import TinyPerson
from tinytroupe.agent.social_types import Content, Reaction
import tinytroupe.openai_utils as openai_utils

class LLMPredictor:
    """Use LLM reasoning for engagement prediction"""

    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def predict(self, persona: TinyPerson, content: Content) -> Reaction:
        """Use LLM to predict engagement"""

        prompt = f"""
        You are predicting how a specific persona will react to content on a professional social network.

        PERSONA PROFILE:
        Name: {persona.name}
        Bio: {persona.minibio()}

        CONTENT TO EVALUATE:
        {content.text}

        TASK:
        Analyze whether this persona would engage with this content.
        Provide your prediction in JSON format:
        {{
            "will_engage": true/false,
            "probability": 0.0-1.0,
            "reasoning": "detailed explanation",
            "reaction_type": "like|comment|share|none",
            "comment": "predicted comment text if applicable"
        }}
        """

        response = openai_utils.client().send_message(
            [
                {"role": "system", "content": "You are an expert in social psychology and behavioral prediction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        prediction = json.loads(response["content"])

        return Reaction(
            will_engage=prediction["will_engage"],
            probability=prediction["probability"],
            reasoning=prediction["reasoning"],
            reaction_type=prediction["reaction_type"],
            comment=prediction.get("comment")
        )
