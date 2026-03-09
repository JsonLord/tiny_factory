from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from tinytroupe.agent import TinyPerson
from tinytroupe.agent.social_types import Content
from tinytroupe.social_network import NetworkTopology

class ContentFeatureExtractor:
    def extract(self, content: Content) -> Dict[str, float]:
        """Extract all content features"""
        return {
            "word_count": float(len(content.text.split())),
            "has_image": 1.0 if content.images else 0.0,
            "has_video": 1.0 if content.video_url else 0.0,
            "has_link": 1.0 if content.external_links else 0.0,
            "sentiment_score": content.sentiment,
            "num_hashtags": float(len(content.hashtags)),
            "is_weekend": 1.0 if content.timestamp.weekday() >= 5 else 0.0,
        }

class PersonaFeatureExtractor:
    def extract(self, persona: TinyPerson) -> Dict[str, float]:
        """Extract persona features"""
        return {
            "age": float(persona._persona.get("age") or 30),
            "num_connections": float(len(persona.social_connections)),
            "influence_score": persona.influence_metrics.authority,
            "engagement_rate": persona.engagement_patterns.get("overall_rate", 0.0),
        }

class InteractionFeatureExtractor:
    def extract(self, persona: TinyPerson, content: Content, network: NetworkTopology) -> Dict[str, float]:
        """Extract features from persona-content interaction context"""
        return {
            "topic_alignment": persona.get_content_affinity(content),
            # "num_friends_engaged": ...
        }
