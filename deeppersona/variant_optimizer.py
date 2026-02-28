from typing import List, Dict, Any
import numpy as np
from deeppersona.content_generation import ContentVariant
from deeppersona.agent.social_types import Content
from deeppersona.agent import DeepPersona
from deeppersona.social_network import NetworkTopology
from deeppersona.ml_models import EngagementPredictor

class RankedVariant:
    def __init__(self, variant: ContentVariant, score: float):
        self.variant = variant
        self.score = score

class VariantOptimizer:
    """Optimize and rank content variants"""

    def __init__(self, predictor: EngagementPredictor):
        self.predictor = predictor

    def rank_variants_for_audience(self, variants: List[ContentVariant],
                                   target_personas: List[DeepPersona],
                                   network: NetworkTopology) -> List[RankedVariant]:
        """Rank variants by predicted performance"""
        ranked = []
        for variant in variants:
            # Predict engagement for each persona
            scores = []
            for persona in target_personas:
                prob = self.predictor.predict(persona, Content(text=variant.text), network)
                scores.append(prob)

            avg_score = np.mean(scores) if scores else 0.0
            ranked.append(RankedVariant(variant, avg_score))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked
