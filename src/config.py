"""
Configuration for Self-Edit Generation

Defines preservation tier configurations that control edit intensity.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Configuration for self-edit generation"""
    
    # Preservation weights (0.0-1.0)
    semantic_preservation_weight: float
    lexical_preservation_weight: float
    syntactic_preservation_weight: float
    
    # Strategy name
    formulation_strategy: str  # "high_preservation", "medium_preservation", "low_preservation"
    
    # Quality constraints
    max_semantic_drift: float
    min_intent_preservation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "preservation_weights": {
                "semantic": self.semantic_preservation_weight,
                "lexical": self.lexical_preservation_weight,
                "syntactic": self.syntactic_preservation_weight
            },
            "strategy": self.formulation_strategy,
            "constraints": {
                "max_drift": self.max_semantic_drift,
                "min_intent": self.min_intent_preservation
            }
        }


def high_preservation_config() -> Config:
    """
    High preservation tier: Minimal changes
    Target drift: <15%
    """
    return Config(
        semantic_preservation_weight=0.95,
        lexical_preservation_weight=0.90,
        syntactic_preservation_weight=0.95,
        formulation_strategy="high_preservation",
        max_semantic_drift=0.15,
        min_intent_preservation=0.95
    )


def medium_preservation_config() -> Config:
    """
    Medium preservation tier: Balanced changes
    Target drift: 15-35%
    """
    return Config(
        semantic_preservation_weight=0.85,
        lexical_preservation_weight=0.70,
        syntactic_preservation_weight=0.70,
        formulation_strategy="medium_preservation",
        max_semantic_drift=0.35,
        min_intent_preservation=0.85
    )


def low_preservation_config() -> Config:
    """
    Low preservation tier: Maximum diversity (RECOMMENDED)
    Target drift: 20-30%
    Empirically validated as optimal.
    """
    return Config(
        semantic_preservation_weight=0.60,
        lexical_preservation_weight=0.40,
        syntactic_preservation_weight=0.40,
        formulation_strategy="low_preservation",
        max_semantic_drift=0.60,
        min_intent_preservation=0.75
    )


def get_default_config() -> Config:
    """Get default configuration (low preservation)"""
    return low_preservation_config()
