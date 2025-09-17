"""
Active Learning System for AI Excel Interviewer

This module implements active learning capabilities that enable the system to
intelligently select the most informative samples for training, query candidates
strategically, and adapt its learning strategy based on performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

from .continuous_improvement_engine import LearningSession, ModelPerformance
from ..evaluation.answer_evaluator import ComprehensiveEvaluation, EvaluationDimension
from ..interview.conversation_state import ConversationState, DifficultyLevel
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ActiveLearningStrategy(Enum):
    """Active learning strategies"""
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    DIVERSITY_SAMPLING = "diversity_sampling"
    QUERY_BY_COMMITTEE = "query_by_committee"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    HYBRID_STRATEGY = "hybrid_strategy"

class QueryType(Enum):
    """Types of strategic queries"""
    KNOWLEDGE_PROBE = "knowledge_probe"
    SKILL_ASSESSMENT = "skill_assessment"
    CONFIDENCE_CHECK = "confidence_check"
    EDGE_CASE_EXPLORATION = "edge_case_exploration"
    ADAPTIVE_FOLLOW_UP = "adaptive_follow_up"

@dataclass
class ActiveLearningSample:
    """Represents a sample selected for active learning"""
    sample_id: str
    session_id: str
    candidate_response: Dict[str, Any]
    current_prediction: float
    prediction_uncertainty: float
    sample_diversity_score: float
    strategic_value: float
    query_type: QueryType
    learning_strategy: ActiveLearningStrategy
    selection_reasoning: str
    timestamp: datetime

@dataclass
class StrategicQuery:
    """Represents a strategically designed query"""
    query_id: str
    query_text: str
    query_type: QueryType
    target_skill_area: str
    expected_difficulty: str
    strategic_purpose: str
    alternative_formulations: List[str]
    follow_up_questions: List[str]
    evaluation_criteria: Dict[str, Any]

@dataclass
class LearningAdaptation:
    """Represents an adaptation to the learning strategy"""
    adaptation_type: str
    description: str
    trigger_conditions: List[str]
    implementation_details: Dict[str, Any]
    expected_outcome: str
    success_metrics: List[str]
    timestamp: datetime

class UncertaintySampler:
    """Implements uncertainty-based active learning sampling"""
    
    def __init__(self, uncertainty_threshold: float = 0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_history = []
    
    def calculate_prediction_uncertainty(self, model_predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction uncertainty using ensemble disagreement"""
        if len(model_predictions.shape) == 1:
            # Single model - use prediction variance
            return np.ones_like(model_predictions) * 0.5  # Placeholder uncertainty
        
        # Multiple models - use prediction variance
        mean_pred = np.mean(model_predictions, axis=0)
        variance = np.var(model_predictions, axis=0)
        uncertainty = variance / (mean_pred + 1e-8)  # Normalized variance
        
        return uncertainty
    
    def select_uncertain_samples(self, 
                               candidate_samples: List[Dict[str, Any]], 
                               model_predictions: np.ndarray,
                               max_samples: int = 10) -> List[ActiveLearningSample]:
        """Select samples with high prediction uncertainty"""
        uncertainties = self.calculate_prediction_uncertainty(model_predictions)
        
        uncertain_samples = []
        for i, (sample, uncertainty) in enumerate(zip(candidate_samples, uncertainties)):
            if uncertainty > self.uncertainty_threshold:
                al_sample = ActiveLearningSample(
                    sample_id=f"uncertainty_{i}_{datetime.now().timestamp()}",
                    session_id=sample.get('session_id', 'unknown'),
                    candidate_response=sample,
                    current_prediction=np.mean(model_predictions[:, i]) if len(model_predictions.shape) > 1 else model_predictions[i],
                    prediction_uncertainty=uncertainty,
                    sample_diversity_score=0.0,  # Will be calculated by diversity sampler
                    strategic_value=uncertainty * 0.8,  # High strategic value for uncertain samples
                    query_type=QueryType.CONFIDENCE_CHECK,
                    learning_strategy=ActiveLearningStrategy.UNCERTAINTY_SAMPLING,
                    selection_reasoning=f"High prediction uncertainty ({uncertainty:.3f})",
                    timestamp=datetime.now()
                )
                uncertain_samples.append(al_sample)
        
        # Sort by uncertainty and select top samples
        uncertain_samples.sort(key=lambda x: x.prediction_uncertainty, reverse=True)
        return uncertain_samples[:max_samples]

class DiversitySampler:
    """Implements diversity-based active learning sampling"""
    
    def __init__(self, diversity_threshold: float = 0.7):
        self.diversity_threshold = diversity_threshold
        self.selected_samples = []
    
    def calculate_sample_representation(self, 
                                      candidate_features: np.ndarray,
                                      selected_features: np.ndarray) -> np.ndarray:
        """Calculate how well samples represent the feature space"""
        if len(selected_features) == 0:
            return np.ones(len(candidate_features))
        
        # Calculate minimum distance to selected samples
        min_distances = np.zeros(len(candidate_features))
        for i, candidate in enumerate(candidate_features):
            distances = np.linalg.norm(selected_features - candidate, axis=1)
            min_distances[i] = np.min(distances)
        
        # Normalize distances
        max_distance = np.max(min_distances) if np.max(min_distances) > 0 else 1
        diversity_scores = min_distances / max_distance
        
        return diversity_scores
    
    def select_diverse_samples(self,
                             candidate_samples: List[Dict[str, Any]],
                             candidate_features: np.ndarray,
                             max_samples: int = 10) -> List[ActiveLearningSample]:
        """Select samples that maximize diversity in the feature space"""
        # Convert selected samples to feature matrix
        if self.selected_samples:
            selected_features = np.array([
                self._extract_features(sample.candidate_response) 
                for sample in self.selected_samples
            ])
        else:
            selected_features = np.array([]).reshape(0, candidate_features.shape[1])
        
        diverse_samples = []
        candidate_indices = list(range(len(candidate_samples)))
        
        # Greedy selection of diverse samples
        for _ in range(min(max_samples, len(candidate_samples))):
            if not candidate_indices:
                break
            
            # Calculate diversity scores for remaining candidates
            current_candidates = candidate_features[candidate_indices]
            diversity_scores = self.calculate_sample_representation(
                current_candidates, selected_features
            )
            
            # Select sample with highest diversity score
            best_idx = np.argmax(diversity_scores)
            best_original_idx = candidate_indices[best_idx]
            
            sample = candidate_samples[best_original_idx]
            al_sample = ActiveLearningSample(
                sample_id=f"diversity_{best_original_idx}_{datetime.now().timestamp()}",
                session_id=sample.get('session_id', 'unknown'),
                candidate_response=sample,
                current_prediction=0.0,  # Will be set by model
                prediction_uncertainty=0.0,
                sample_diversity_score=diversity_scores[best_idx],
                strategic_value=diversity_scores[best_idx] * 0.6,
                query_type=QueryType.EDGE_CASE_EXPLORATION,
                learning_strategy=ActiveLearningStrategy.DIVERSITY_SAMPLING,
                selection_reasoning=f"High diversity score ({diversity_scores[best_idx]:.3f})",
                timestamp=datetime.now()
            )
            
            diverse_samples.append(al_sample)
            self.selected_samples.append(al_sample)
            
            # Update selected features
            if selected_features.size == 0:
                selected_features = candidate_features[best_original_idx:best_original_idx+1]
            else:
                selected_features = np.vstack([
                    selected_features, 
                    candidate_features[best_original_idx:best_original_idx+1]
                ])
            
            # Remove selected index from candidates
            candidate_indices.pop(best_idx)
        
        return diverse_samples
    
    def _extract_features(self, response: Dict[str, Any]) -> np.ndarray:
        """Extract features from candidate response for diversity calculation"""
        # Simple feature extraction - can be enhanced
        features = []
        
        # Response length
        response_text = response.get('response', '')
        features.append(len(response_text))
        
        # Evaluation scores
        scores = response.get('evaluation_scores', {})
        features.extend([
            scores.get('technical_accuracy', 0.0),
            scores.get('problem_solving', 0.0),
            scores.get('communication_clarity', 0.0),
            scores.get('practical_application', 0.0),
            scores.get('excel_expertise', 0.0)
        ])
        
        # Response time
        features.append(response.get('response_time', 0.0))
        
        # Question difficulty
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        difficulty = response.get('question_difficulty', 'medium')
        features.append(difficulty_map.get(difficulty, 2))
        
        return np.array(features)

class StrategicQueryGenerator:
    """Generates strategic queries for active learning"""
    
    def __init__(self):
        self.query_templates = self._initialize_query_templates()
        self.generated_queries = []
    
    def _initialize_query_templates(self) -> Dict[QueryType, List[str]]:
        """Initialize strategic query templates"""
        return {
            QueryType.KNOWLEDGE_PROBE: [
                "Can you explain the difference between {concept1} and {concept2} in Excel?",
                "What would happen if you applied {function} to {scenario}?",
                "How would you approach solving {problem_type} using Excel?"
            ],
            QueryType.SKILL_ASSESSMENT: [
                "Walk me through your process for {task_type}.",
                "Describe a time when you had to {challenge_type} in Excel.",
                "What Excel features do you consider essential for {use_case}?"
            ],
            QueryType.CONFIDENCE_CHECK: [
                "On a scale of 1-10, how confident are you about {topic}?",
                "What aspects of {skill_area} do you find most challenging?",
                "How would you rate your expertise in {excel_feature}?"
            ],
            QueryType.EDGE_CASE_EXPLORATION: [
                "What would you do if {unusual_scenario} occurred?",
                "How would you handle {edge_case} in Excel?",
                "Can you think of a creative solution for {complex_problem}?"
            ],
            QueryType.ADAPTIVE_FOLLOW_UP: [
                "Based on your previous answer about {previous_topic}, what about {follow_up}?",
                "You mentioned {previous_point}. Can you elaborate on {specific_aspect}?",
                "How does {previous_concept} relate to {new_concept}?"
            ]
        }
    
    def generate_strategic_query(self,
                               target_skill_area: str,
                               query_type: QueryType,
                               context: Dict[str, Any] = None) -> StrategicQuery:
        """Generate a strategic query for active learning"""
        templates = self.query_templates.get(query_type, [])
        if not templates:
            # Fallback to knowledge probe
            templates = self.query_templates[QueryType.KNOWLEDGE_PROBE]
        
        # Select template and fill in variables
        import random
        template = random.choice(templates)
        
        # Simple template filling - can be enhanced with more sophisticated NLP
        query_text = template.format(
            concept1=target_skill_area,
            concept2="advanced techniques",
            function="VLOOKUP",
            scenario="a large dataset",
            problem_type=f"{target_skill_area} challenges",
            task_type=f"working with {target_skill_area}",
            challenge_type=f"complex {target_skill_area} problems",
            use_case=f"{target_skill_area} analysis",
            topic=target_skill_area,
            skill_area=target_skill_area,
            excel_feature=target_skill_area,
            unusual_scenario=f"unexpected {target_skill_area} behavior",
            edge_case=f"rare {target_skill_area} scenarios",
            complex_problem=f"advanced {target_skill_area} challenges",
            previous_topic=context.get('previous_topic', target_skill_area) if context else target_skill_area,
            follow_up="the practical applications" if context else "practical applications",
            previous_point=context.get('previous_point', 'your approach') if context else 'your approach',
            specific_aspect="the implementation details" if context else 'implementation details',
            previous_concept=context.get('previous_concept', 'basic concepts') if context else 'basic concepts',
            new_concept="advanced techniques" if context else 'advanced techniques'
        )
        
        # Generate alternative formulations
        alternative_formulations = [
            query_text.replace("Can you", "Please"),
            query_text.replace("What", "How"),
            query_text.replace("Explain", "Describe")
        ]
        
        # Generate follow-up questions
        follow_up_questions = [
            f"Can you provide a specific example of {target_skill_area}?",
            f"What challenges have you faced with {target_skill_area}?",
            f"How do you stay updated with {target_skill_area} best practices?"
        ]
        
        query = StrategicQuery(
            query_id=f"strategic_{query_type.value}_{datetime.now().timestamp()}",
            query_text=query_text,
            query_type=query_type,
            target_skill_area=target_skill_area,
            expected_difficulty=self._estimate_difficulty(target_skill_area, query_type),
            strategic_purpose=f"Active learning exploration of {target_skill_area}",
            alternative_formulations=alternative_formulations[:2],
            follow_up_questions=follow_up_questions,
            evaluation_criteria={
                'depth_of_knowledge': 0.4,
                'clarity_of_explanation': 0.3,
                'practical_application': 0.3
            }
        )
        
        self.generated_queries.append(query)
        return query
    
    def _estimate_difficulty(self, skill_area: str, query_type: QueryType) -> str:
        """Estimate the difficulty of a strategic query"""
        # Simple difficulty estimation - can be enhanced
        base_difficulty = {
            'basic_excel': 'easy',
            'advanced_formulas': 'hard',
            'data_analysis': 'medium',
            'automation': 'hard',
            'visualization': 'medium'
        }
        
        difficulty_multiplier = {
            QueryType.KNOWLEDGE_PROBE: 1,
            QueryType.SKILL_ASSESSMENT: 1.2,
            QueryType.CONFIDENCE_CHECK: 0.8,
            QueryType.EDGE_CASE_EXPLORATION: 1.5,
            QueryType.ADAPTIVE_FOLLOW_UP: 1.1
        }
        
        base = base_difficulty.get(skill_area, 'medium')
        multiplier = difficulty_multiplier.get(query_type, 1)
        
        # Adjust difficulty based on multiplier
        difficulty_levels = ['easy', 'medium', 'hard']
        base_idx = difficulty_levels.index(base)
        adjusted_idx = min(len(difficulty_levels) - 1, int(base_idx * multiplier))
        
        return difficulty_levels[adjusted_idx]

class LearningAdaptationEngine:
    """Manages adaptations to the learning strategy based on performance"""
    
    def __init__(self):
        self.adaptations = []
        self.performance_history = []
        self.adaptation_triggers = self._initialize_adaptation_triggers()
    
    def _initialize_adaptation_triggers(self) -> Dict[str, Any]:
        """Initialize adaptation triggers"""
        return {
            'low_model_accuracy': {
                'threshold': 0.7,
                'window_size': 10,
                'adaptation_type': 'increase_training_data',
                'description': 'Model accuracy below threshold'
            },
            'high_uncertainty_samples': {
                'threshold': 0.4,
                'window_size': 5,
                'adaptation_type': 'adjust_uncertainty_threshold',
                'description': 'Too many high uncertainty samples'
            },
            'poor_diversity_coverage': {
                'threshold': 0.5,
                'window_size': 8,
                'adaptation_type': 'enhance_diversity_sampling',
                'description': 'Insufficient diversity in selected samples'
            },
            'slow_learning_convergence': {
                'threshold': 0.05,
                'window_size': 15,
                'adaptation_type': 'modify_learning_rate',
                'description': 'Slow improvement in model performance'
            }
        }
    
    def monitor_and_adapt(self, 
                         recent_performance: Dict[str, float],
                         active_learning_metrics: Dict[str, Any]) -> List[LearningAdaptation]:
        """Monitor performance and generate adaptations"""
        self.performance_history.append(recent_performance)
        
        # Keep only recent history
        max_history = 20
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        adaptations = []
        
        # Check each trigger
        for trigger_name, trigger_config in self.adaptation_triggers.items():
            if self._check_trigger(trigger_name, trigger_config):
                adaptation = self._create_adaptation(trigger_name, trigger_config, active_learning_metrics)
                if adaptation:
                    adaptations.append(adaptation)
        
        self.adaptations.extend(adaptations)
        return adaptations
    
    def _check_trigger(self, trigger_name: str, trigger_config: Dict[str, Any]) -> bool:
        """Check if an adaptation trigger is activated"""
        window_size = trigger_config['window_size']
        threshold = trigger_config['threshold']
        
        if len(self.performance_history) < window_size:
            return False
        
        recent_history = self.performance_history[-window_size:]
        
        if trigger_name == 'low_model_accuracy':
            recent_accuracies = [h.get('model_accuracy', 1.0) for h in recent_history]
            avg_accuracy = np.mean(recent_accuracies)
            return avg_accuracy < threshold
        
        elif trigger_name == 'high_uncertainty_samples':
            recent_uncertainties = [h.get('avg_uncertainty', 0.0) for h in recent_history]
            avg_uncertainty = np.mean(recent_uncertainties)
            return avg_uncertainty > threshold
        
        elif trigger_name == 'poor_diversity_coverage':
            recent_diversities = [h.get('diversity_score', 1.0) for h in recent_history]
            avg_diversity = np.mean(recent_diversities)
            return avg_diversity < threshold
        
        elif trigger_name == 'slow_learning_convergence':
            if len(self.performance_history) >= window_size * 2:
                older_performance = np.mean([h.get('model_accuracy', 0.0) for h in self.performance_history[-window_size*2:-window_size]])
                recent_performance = np.mean([h.get('model_accuracy', 0.0) for h in recent_history])
                improvement = recent_performance - older_performance
                return improvement < threshold
        
        return False
    
    def _create_adaptation(self, trigger_name: str, trigger_config: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[LearningAdaptation]:
        """Create a learning adaptation based on trigger"""
        adaptation_type = trigger_config['adaptation_type']
        
        if adaptation_type == 'increase_training_data':
            return LearningAdaptation(
                adaptation_type='data_augmentation',
                description='Increase training data collection and augmentation',
                trigger_conditions=[trigger_config['description']],
                implementation_details={
                    'target_additional_samples': 100,
                    'augmentation_strategies': ['paraphrasing', 'difficulty_adjustment'],
                    'collection_frequency': 'increased'
                },
                expected_outcome='Improved model accuracy through more diverse training data',
                success_metrics=['model_accuracy', 'validation_loss', 'generalization_error'],
                timestamp=datetime.now()
            )
        
        elif adaptation_type == 'adjust_uncertainty_threshold':
            current_threshold = metrics.get('uncertainty_threshold', 0.3)
            new_threshold = max(0.1, current_threshold - 0.05)
            
            return LearningAdaptation(
                adaptation_type='threshold_optimization',
                description=f'Adjust uncertainty threshold from {current_threshold} to {new_threshold}',
                trigger_conditions=[trigger_config['description']],
                implementation_details={
                    'old_threshold': current_threshold,
                    'new_threshold': new_threshold,
                    'adjustment_strategy': 'gradual_reduction'
                },
                expected_outcome='Better balance between exploration and exploitation',
                success_metrics=['sample_selection_efficiency', 'learning_rate', 'model_convergence'],
                timestamp=datetime.now()
            )
        
        elif adaptation_type == 'enhance_diversity_sampling':
            return LearningAdaptation(
                adaptation_type='diversity_enhancement',
                description='Enhance diversity sampling mechanisms',
                trigger_conditions=[trigger_config['description']],
                implementation_details={
                    'clustering_strategies': ['kmeans', 'hierarchical'],
                    'feature_engineering': 'enhanced_representation',
                    'sampling_weights': 'adaptive_adjustment'
                },
                expected_outcome='Better coverage of the feature space',
                success_metrics=['diversity_score', 'coverage_metric', 'cluster_distribution'],
                timestamp=datetime.now()
            )
        
        elif adaptation_type == 'modify_learning_rate':
            return LearningAdaptation(
                adaptation_type='learning_rate_adjustment',
                description='Modify learning rate and optimization strategy',
                trigger_conditions=[trigger_config['description']],
                implementation_details={
                    'learning_rate_multiplier': 1.5,
                    'optimization_algorithm': 'adaptive_optimizer',
                    'convergence_criteria': 'relaxed'
                },
                expected_outcome='Faster convergence and better optimization',
                success_metrics=['convergence_speed', 'final_accuracy', 'training_efficiency'],
                timestamp=datetime.now()
            )
        
        return None

class ActiveLearningSystem:
    """Main active learning system that coordinates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.uncertainty_sampler = UncertaintySampler()
        self.diversity_sampler = DiversitySampler()
        self.query_generator = StrategicQueryGenerator()
        self.adaptation_engine = LearningAdaptationEngine()
        
        self.active_samples = []
        self.learning_history = []
        self.performance_metrics = {}
        
        self.is_active = True
        self.learning_cycle = 0
        
        logger.info("Initialized Active Learning System")
    
    def select_active_learning_samples(self,
                                     candidate_sessions: List[LearningSession],
                                     current_models: Dict[str, Any],
                                     strategy: ActiveLearningStrategy = ActiveLearningStrategy.HYBRID_STRATEGY,
                                     max_samples: int = 10) -> List[ActiveLearningSample]:
        """Select samples for active learning"""
        if not self.is_active:
            return []
        
        try:
            # Extract candidate responses and features
            candidate_responses = []
            candidate_features = []
            
            for session in candidate_sessions:
                for response in session.candidate_responses:
                    candidate_responses.append(response)
                    features = self._extract_session_features(session, response)
                    candidate_features.append(features)
            
            candidate_features = np.array(candidate_features)
            
            selected_samples = []
            
            if strategy == ActiveLearningStrategy.UNCERTAINTY_SAMPLING:
                selected_samples = self.uncertainty_sampler.select_uncertain_samples(
                    candidate_responses, self._get_model_predictions(candidate_responses, current_models), max_samples
                )
            
            elif strategy == ActiveLearningStrategy.DIVERSITY_SAMPLING:
                selected_samples = self.diversity_sampler.select_diverse_samples(
                    candidate_responses, candidate_features, max_samples
                )
            
            elif strategy == ActiveLearningStrategy.HYBRID_STRATEGY:
                # Combine uncertainty and diversity sampling
                uncertain_samples = self.uncertainty_sampler.select_uncertain_samples(
                    candidate_responses, self._get_model_predictions(candidate_responses, current_models), max_samples // 2
                )
                
                diverse_samples = self.diversity_sampler.select_diverse_samples(
                    candidate_responses, candidate_features, max_samples // 2
                )
                
                selected_samples = uncertain_samples + diverse_samples
            
            # Generate strategic queries for selected samples
            for sample in selected_samples:
                strategic_query = self.query_generator.generate_strategic_query(
                    target_skill_area=sample.candidate_response.get('skill_area', 'general'),
                    query_type=sample.query_type,
                    context={'previous_topic': sample.candidate_response.get('topic', '')}
                )
                sample.strategic_query = strategic_query
            
            self.active_samples.extend(selected_samples)
            self.learning_cycle += 1
            
            logger.info(f"Selected {len(selected_samples)} active learning samples using {strategy.value}")
            return selected_samples
            
        except Exception as e:
            logger.error(f"Error in active learning sample selection: {str(e)}")
            return []
    
    def generate_strategic_queries(self,
                                   skill_areas: List[str],
                                   query_types: List[QueryType],
                                   context: Dict[str, Any] = None) -> List[StrategicQuery]:
        """Generate strategic queries for active learning"""
        strategic_queries = []
        
        for skill_area in skill_areas:
            for query_type in query_types:
                query = self.query_generator.generate_strategic_query(
                    target_skill_area=skill_area,
                    query_type=query_type,
                    context=context
                )
                strategic_queries.append(query)
        
        return strategic_queries
    
    def adapt_learning_strategy(self,
                              recent_performance: Dict[str, float],
                              active_learning_metrics: Dict[str, Any]) -> List[LearningAdaptation]:
        """Adapt the learning strategy based on performance"""
        adaptations = self.adaptation_engine.monitor_and_adapt(
            recent_performance, active_learning_metrics
        )
        
        if adaptations:
            logger.info(f"Generated {len(adaptations)} learning adaptations")
            self._apply_adaptations(adaptations)
        
        return adaptations
    
    def _extract_session_features(self, session: LearningSession, response: Dict[str, Any]) -> np.ndarray:
        """Extract features from session and response for active learning"""
        features = []
        
        # Session-level features
        features.extend([
            session.evaluation_scores.get('technical_accuracy', 0.0),
            session.evaluation_scores.get('problem_solving', 0.0),
            session.evaluation_scores.get('communication_clarity', 0.0),
            session.evaluation_scores.get('practical_application', 0.0),
            session.evaluation_scores.get('excel_expertise', 0.0)
        ])
        
        # Response-level features
        features.extend([
            response.get('evaluation_score', 0.0),
            response.get('response_time', 0.0),
            len(response.get('response', '')),
            response.get('follow_up_count', 0)
        ])
        
        # Question difficulty
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        difficulty = response.get('question_difficulty', 'medium')
        features.append(difficulty_map.get(difficulty, 2))
        
        return np.array(features)
    
    def _get_model_predictions(self, candidate_responses: List[Dict[str, Any]], current_models: Dict[str, Any]) -> np.ndarray:
        """Get model predictions for candidate responses"""
        # Placeholder - would use actual models to generate predictions
        # For now, return random predictions for demonstration
        import random
        return np.array([[random.uniform(0.3, 0.9) for _ in range(len(candidate_responses))] for _ in range(3)])
    
    def _apply_adaptations(self, adaptations: List[LearningAdaptation]):
        """Apply learning adaptations"""
        for adaptation in adaptations:
            logger.info(f"Applying adaptation: {adaptation.description}")
            
            # Apply specific adaptations based on type
            if adaptation.adaptation_type == 'threshold_optimization':
                new_threshold = adaptation.implementation_details.get('new_threshold', 0.3)
                self.uncertainty_sampler.uncertainty_threshold = new_threshold
                
            elif adaptation.adaptation_type == 'diversity_enhancement':
                # Update diversity sampling parameters
                self.diversity_sampler.diversity_threshold = 0.8
                
            elif adaptation.adaptation_type == 'data_augmentation':
                # Increase data collection frequency
                pass  # Would implement enhanced data collection
    
    def get_active_learning_status(self) -> Dict[str, Any]:
        """Get current active learning system status"""
        return {
            'is_active': self.is_active,
            'learning_cycle': self.learning_cycle,
            'active_samples_count': len(self.active_samples),
            'total_queries_generated': len(self.query_generator.generated_queries),
            'adaptations_applied': len(self.adaptation_engine.adaptations),
            'recent_performance': self.performance_metrics,
            'sampler_status': {
                'uncertainty_threshold': self.uncertainty_sampler.uncertainty_threshold,
                'diversity_threshold': self.diversity_sampler.diversity_threshold
            }
        }
    
    def export_learning_insights(self) -> Dict[str, Any]:
        """Export comprehensive active learning insights"""
        return {
            'active_learning_status': self.get_active_learning_status(),
            'sample_selection_history': [
                {
                    'sample_id': sample.sample_id,
                    'strategic_value': sample.strategic_value,
                    'learning_strategy': sample.learning_strategy.value,
                    'query_type': sample.query_type.value,
                    'timestamp': sample.timestamp.isoformat()
                }
                for sample in self.active_samples
            ],
            'adaptation_history': [
                {
                    'adaptation_type': adaptation.adaptation_type,
                    'description': adaptation.description,
                    'expected_outcome': adaptation.expected_outcome,
                    'timestamp': adaptation.timestamp.isoformat()
                }
                for adaptation in self.adaptation_engine.adaptations
            ],
            'strategic_queries': [
                {
                    'query_id': query.query_id,
                    'query_type': query.query_type.value,
                    'target_skill_area': query.target_skill_area,
                    'expected_difficulty': query.expected_difficulty,
                    'strategic_purpose': query.strategic_purpose
                }
                for query in self.query_generator.generated_queries
            ]
        }