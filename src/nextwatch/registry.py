"""Every recommender, registered by importing its module. Import MODELS from here."""

import nextwatch.cf.als  # noqa: F401
import nextwatch.features.content  # noqa: F401
import nextwatch.ranking.ranker  # noqa: F401
from nextwatch.evaluation.models import MODELS

__all__ = ["MODELS"]
