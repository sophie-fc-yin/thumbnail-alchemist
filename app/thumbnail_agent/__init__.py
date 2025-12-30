"""Thumbnail selection agent for intelligent frame selection.

This package provides context-aware thumbnail selection using niche-specific
scoring weights and psychology-driven evaluation.
"""

from app.thumbnail_agent.contextual_scoring import (
    ContextualScoringCriteria,
    get_contextual_scores,
)
from app.thumbnail_agent.scoring_weights import (
    ScoringWeights,
    get_scoring_weights,
    list_available_niches,
    preview_niche_weights,
)
from app.thumbnail_agent.thumbnail_selector import ThumbnailSelector

__all__ = [
    # Main agent
    "ThumbnailSelector",
    # Scoring weights
    "ScoringWeights",
    "get_scoring_weights",
    "list_available_niches",
    "preview_niche_weights",
    # Contextual scoring
    "ContextualScoringCriteria",
    "get_contextual_scores",
]
