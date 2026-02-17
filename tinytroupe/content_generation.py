from typing import List, Dict, Any, Optional
import random
from tinytroupe.agent import TinyPerson
from tinytroupe.agent.social_types import Content
import tinytroupe.openai_utils as openai_utils

class ContentVariant:
    def __init__(self, text: str, strategy: str, parameters: Dict[str, Any], original_content: str):
        self.text = text
        self.strategy = strategy
        self.parameters = parameters
        self.original_content = original_content

class ContentVariantGenerator:
    """Generate multiple variants of input content"""

    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def generate_variants(self, original_content: str, num_variants: int = 5,
                         target_personas: List[TinyPerson] = None) -> List[ContentVariant]:
        """Generate diverse variants of content"""
        variants = []

        # In a real implementation, we would use different prompts for different strategies
        # Here we use a simplified approach

        for i in range(num_variants):
            prompt = f"Rewrite the following content in a different style or tone:\n\n{original_content}"

            response = openai_utils.client().send_message(
                [{"role": "user", "content": prompt}]
            )

            variant_text = response["content"].strip()
            variants.append(ContentVariant(
                text=variant_text,
                strategy="style_variation",
                parameters={"variant_index": i},
                original_content=original_content
            ))

        return variants
