"""
Quality Assurance Module

Evaluates generated images and filters out failures.
"""

from .metrics import (
    QualityMetrics,
    CLIPScorer,
    BlurDetector,
    EdgeConsistencyChecker,
    ColorHarmonyChecker,
    compute_all_metrics
)

from .filtering import (
    QualityThresholds,
    FilterResult,
    QualityFilter,
    QualityLogger
)

# Aliases for orchestrator compatibility
QualityAssurance = QualityFilter


# Add evaluate method wrapper for compatibility
class QualityAssurance(QualityFilter):
    """Wrapper around QualityFilter with evaluate() method."""
    
    def evaluate(
        self,
        image,
        background=None,
        mask=None,
        prompt: str = ""
    ):
        """
        Evaluate image quality.
        
        Args:
            image: Generated image
            background: Original background (optional)
            mask: Mask of modified region
            prompt: Text prompt
            
        Returns:
            Result with passed flag and scores
        """
        if mask is None:
            import numpy as np
            mask = np.ones(image.shape[:2], dtype=np.float32)
        
        result = self.check(image, mask, prompt, background)
        
        # Add scores attribute for compatibility
        result.scores = result.metrics.to_dict()
        
        return result


QAConfig = QualityThresholds

__all__ = [
    # Metrics
    'QualityMetrics',
    'CLIPScorer',
    'BlurDetector',
    'EdgeConsistencyChecker',
    'ColorHarmonyChecker',
    'compute_all_metrics',
    
    # Filtering
    'QualityThresholds',
    'FilterResult',
    'QualityFilter',
    'QualityLogger',
    
    # Aliases
    'QualityAssurance',
    'QAConfig'
]
