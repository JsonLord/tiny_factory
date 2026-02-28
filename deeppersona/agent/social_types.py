from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

@dataclass
class ConnectionEdge:
    connection_id: str
    strength: float = 0.0  # 0.0-1.0
    influence_score: float = 0.0
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    relationship_type: str = "follower" # "follower", "friend", "colleague", "family"
    last_interaction: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BehavioralEvent:
    timestamp: datetime
    action_type: str
    content_id: str
    outcome: Any
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InfluenceProfile:
    reach: int = 0
    authority: float = 0.0
    expertise_domains: List[str] = field(default_factory=list)
    follower_to_following_ratio: float = 0.0
    engagement_rate: float = 0.0

@dataclass
class Content:
    text: str
    content_id: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    format: str = "text" # "article", "video", "poll", "survey", "ux_test", "email", "ad", etc.
    length: int = 0
    tone: str = "neutral"
    author_name: Optional[str] = None
    author_title: Optional[str] = None
    sentiment: float = 0.0
    images: List[str] = field(default_factory=list)
    video_url: Optional[str] = None
    external_links: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    platform: str = "LinkedIn"

@dataclass
class Reaction:
    reaction_type: str # "like", "love", "insightful", "celebrate", "none", "positive", "negative", "neutral"
    will_engage: bool
    probability: float
    reasoning: Optional[str] = None
    comment: Optional[str] = None
    will_share: bool = False
    virality_coefficient: float = 0.0
    sentiment: float = 0.0 # -1.0 to 1.0
    detailed_feedback: Dict[str, Any] = field(default_factory=dict) # For surveys/UX tests
