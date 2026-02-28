import logging
logger = logging.getLogger("deeppersona")

from deeppersona import default

###########################################################################
# Exposed API
###########################################################################
from deeppersona.enrichment.deep_enricher import DeepEnricher

__all__ = ["DeepEnricher"]