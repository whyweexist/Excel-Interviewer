"""
Learning System Package for AI Excel Interviewer

This package implements continuous improvement and active learning capabilities
to enhance the AI interviewer's performance over time.
"""

from .continuous_improvement_engine import (
    ImprovementType,
    LearningSession,
    ModelPerformance,
    ImprovementRecommendation,
    ContinuousImprovementEngine
)

from .active_learning_system import (
    ActiveLearningStrategy,
    QueryType,
    ActiveLearningSample,
    StrategicQuery,
    LearningAdaptation,
    UncertaintySampler,
    DiversitySampler,
    StrategicQueryGenerator,
    LearningAdaptationEngine,
    ActiveLearningSystem
)

__all__ = [
    # Continuous Improvement Components
    'ImprovementType',
    'LearningSession',
    'ModelPerformance',
    'ImprovementRecommendation',
    'ContinuousImprovementEngine',
    
    # Active Learning Components
    'ActiveLearningStrategy',
    'QueryType',
    'ActiveLearningSample',
    'StrategicQuery',
    'LearningAdaptation',
    'UncertaintySampler',
    'DiversitySampler',
    'StrategicQueryGenerator',
    'LearningAdaptationEngine',
    'ActiveLearningSystem'
]