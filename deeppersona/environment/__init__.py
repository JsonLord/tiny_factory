"""
Environments provide a structured way to define the world in which the
agents interact with each other as well as external entities (e.g., search engines).
"""

import logging
logger = logging.getLogger("deeppersona")

from deeppersona import default

###########################################################################
# Exposed API
###########################################################################
from deeppersona.environment.deep_world import DeepWorld
from deeppersona.environment.deep_social_network import DeepSocialNetwork

__all__ = ["DeepWorld", "DeepSocialNetwork"]