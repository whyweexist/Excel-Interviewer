"""
Continuous Improvement Pipeline and Active Learning System

This module implements the core continuous improvement and active learning capabilities
for the AI-powered Excel interviewer system, enabling it to learn from interview
sessions and continuously enhance performance.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..evaluation.answer_evaluator import ComprehensiveEvaluation, EvaluationDimension
from ..interview.conversation_state import ConversationState, DifficultyLevel
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class LearningPhase(Enum):
    """Learning phases for the continuous improvement pipeline"""
    DATA_COLLECTION = "data_collection"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

class ImprovementType(Enum):
    """Types of improvements that can be learned"""
    QUESTION_QUALITY = "question_quality"
    EVALUATION_ACCURACY = "evaluation_accuracy"
    DIFFICULTY_CALIBRATION = "difficulty_calibration"
    CONVERSATION_FLOW = "conversation_flow"
    CHALLENGE_EFFECTIVENESS = "challenge_effectiveness"

@dataclass
class LearningSession:
    """Represents a learning session from interview data"""
    session_id: str
    timestamp: datetime
    candidate_responses: List[Dict[str, Any]]
    evaluation_scores: Dict[str, float]
    conversation_flow: List[Dict[str, Any]]
    difficulty_progression: List[str]
    final_assessment: Dict[str, Any]
    feedback_received: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class ModelPerformance:
    """Tracks model performance metrics"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    training_samples: int
    validation_samples: int
    timestamp: datetime

@dataclass
class ImprovementRecommendation:
    """Represents an improvement recommendation"""
    improvement_type: ImprovementType
    description: str
    expected_impact: float
    confidence: float
    implementation_complexity: str
    affected_components: List[str]
    data_evidence: Dict[str, Any]

class DataCollectionEngine:
    """Engine for collecting and preprocessing learning data"""
    
    def __init__(self, storage_path: str = "data/learning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "learning_data.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT,
                candidate_id TEXT,
                overall_score REAL,
                technical_accuracy REAL,
                problem_solving REAL,
                communication_clarity REAL,
                practical_application REAL,
                excel_expertise REAL,
                difficulty_level TEXT,
                session_duration INTEGER,
                question_count INTEGER,
                challenge_success_rate REAL
            )
        ''')
        
        # Create conversation_events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                question_id TEXT,
                question_difficulty TEXT,
                candidate_response TEXT,
                evaluation_score REAL,
                response_time REAL,
                follow_up_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                feedback_type TEXT,
                feedback_content TEXT,
                rating REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_session_data(self, session_data: LearningSession) -> bool:
        """Collect data from a completed interview session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert session data
            cursor.execute('''
                INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_data.session_id,
                session_data.timestamp.isoformat(),
                session_data.final_assessment.get('candidate_id', 'unknown'),
                session_data.final_assessment.get('overall_score', 0.0),
                session_data.evaluation_scores.get('technical_accuracy', 0.0),
                session_data.evaluation_scores.get('problem_solving', 0.0),
                session_data.evaluation_scores.get('communication_clarity', 0.0),
                session_data.evaluation_scores.get('practical_application', 0.0),
                session_data.evaluation_scores.get('excel_expertise', 0.0),
                session_data.difficulty_progression[-1] if session_data.difficulty_progression else 'medium',
                session_data.performance_metrics.get('session_duration', 0),
                len(session_data.candidate_responses),
                session_data.performance_metrics.get('challenge_success_rate', 0.0)
            ))
            
            # Insert conversation events
            for event in session_data.conversation_flow:
                cursor.execute('''
                    INSERT INTO conversation_events VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_data.session_id,
                    event.get('timestamp', datetime.now().isoformat()),
                    event.get('event_type', 'question'),
                    event.get('question_id', ''),
                    event.get('difficulty', 'medium'),
                    event.get('response', ''),
                    event.get('evaluation_score', 0.0),
                    event.get('response_time', 0.0),
                    event.get('follow_up_count', 0)
                ))
            
            # Insert feedback
            for feedback_type, feedback_content in session_data.feedback_received.items():
                cursor.execute('''
                    INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?, ?)
                ''', (
                    session_data.session_id,
                    datetime.now().isoformat(),
                    feedback_type,
                    str(feedback_content),
                    feedback_content.get('rating', 0.0) if isinstance(feedback_content, dict) else 0.0
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully collected session data: {session_data.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting session data: {str(e)}")
            return False
    
    def get_training_data(self, min_sessions: int = 100) -> Optional[pd.DataFrame]:
        """Retrieve training data for model training"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query for training data
            query = '''
                SELECT s.*, 
                       COUNT(ce.event_id) as total_interactions,
                       AVG(ce.evaluation_score) as avg_question_score,
                       AVG(ce.response_time) as avg_response_time,
                       AVG(f.rating) as avg_feedback_rating
                FROM sessions s
                LEFT JOIN conversation_events ce ON s.session_id = ce.session_id
                LEFT JOIN feedback f ON s.session_id = f.session_id
                GROUP BY s.session_id
                HAVING COUNT(ce.event_id) >= 3
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < min_sessions:
                logger.warning(f"Insufficient training data: {len(df)} sessions available")
                return None
            
            # Preprocess data
            df = self._preprocess_training_data(df)
            
            logger.info(f"Retrieved {len(df)} training samples")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return None
    
    def _preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data for model training"""
        # Convert timestamp to datetime features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Encode categorical variables
        le = LabelEncoder()
        df['difficulty_encoded'] = le.fit_transform(df['difficulty_level'])
        
        # Create feature engineering
        df['score_variance'] = df[['technical_accuracy', 'problem_solving', 'communication_clarity', 
                                  'practical_application', 'excel_expertise']].var(axis=1)
        
        df['score_consistency'] = 1 - (df['score_variance'] / df['overall_score'])
        df['efficiency_score'] = df['question_count'] / (df['session_duration'] / 60)  # questions per minute
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        return df

class ModelTrainingEngine:
    """Engine for training and validating machine learning models"""
    
    def __init__(self, model_storage_path: str = "models/learning"):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.performance_history = []
    
    def train_question_quality_model(self, training_data: pd.DataFrame) -> ModelPerformance:
        """Train model for predicting question quality"""
        try:
            # Prepare features and target
            feature_columns = ['technical_accuracy', 'problem_solving', 'communication_clarity',
                             'practical_application', 'excel_expertise', 'difficulty_encoded',
                             'avg_response_time', 'efficiency_score', 'score_consistency']
            
            X = training_data[feature_columns]
            y = training_data['overall_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate performance metrics
            performance = ModelPerformance(
                model_type="question_quality_regression",
                accuracy=1 - (mse / np.var(y_test)),  # R²-like metric
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mse=mse,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                timestamp=datetime.now()
            )
            
            # Save model
            model_path = self.model_storage_path / "question_quality_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.trained_models['question_quality'] = model
            self.performance_history.append(performance)
            
            logger.info(f"Trained question quality model with MSE: {mse:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training question quality model: {str(e)}")
            return None
    
    def train_difficulty_calibration_model(self, training_data: pd.DataFrame) -> ModelPerformance:
        """Train model for difficulty level calibration"""
        try:
            # Prepare features and target
            feature_columns = ['technical_accuracy', 'problem_solving', 'communication_clarity',
                             'practical_application', 'excel_expertise', 'avg_response_time',
                             'efficiency_score', 'score_consistency', 'challenge_success_rate']
            
            X = training_data[feature_columns]
            y = training_data['difficulty_encoded']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate detailed metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            performance = ModelPerformance(
                model_type="difficulty_calibration_classification",
                accuracy=accuracy,
                precision=report['weighted avg']['precision'],
                recall=report['weighted avg']['recall'],
                f1_score=report['weighted avg']['f1-score'],
                mse=0.0,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                timestamp=datetime.now()
            )
            
            # Save model
            model_path = self.model_storage_path / "difficulty_calibration_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.trained_models['difficulty_calibration'] = model
            self.performance_history.append(performance)
            
            logger.info(f"Trained difficulty calibration model with accuracy: {accuracy:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training difficulty calibration model: {str(e)}")
            return None
    
    def train_evaluation_accuracy_model(self, training_data: pd.DataFrame) -> ModelPerformance:
        """Train model for improving evaluation accuracy"""
        try:
            # Prepare features and target (using feedback rating as ground truth)
            feature_columns = ['technical_accuracy', 'problem_solving', 'communication_clarity',
                             'practical_application', 'excel_expertise', 'difficulty_encoded',
                             'avg_question_score', 'avg_response_time', 'avg_feedback_rating']
            
            X = training_data[feature_columns]
            y = training_data['avg_feedback_rating'].fillna(training_data['overall_score'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            performance = ModelPerformance(
                model_type="evaluation_accuracy_regression",
                accuracy=1 - (mse / np.var(y_test)),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mse=mse,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                timestamp=datetime.now()
            )
            
            # Save model
            model_path = self.model_storage_path / "evaluation_accuracy_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.trained_models['evaluation_accuracy'] = model
            self.performance_history.append(performance)
            
            logger.info(f"Trained evaluation accuracy model with MSE: {mse:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training evaluation accuracy model: {str(e)}")
            return None

class ImprovementRecommendationEngine:
    """Engine for generating improvement recommendations"""
    
    def __init__(self):
        self.recommendation_history = []
        self.implementation_tracker = {}
    
    def generate_improvement_recommendations(self, 
                                           model_performances: List[ModelPerformance],
                                           session_analytics: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate improvement recommendations based on model performance and analytics"""
        recommendations = []
        
        # Analyze model performance
        for performance in model_performances:
            if performance and performance.accuracy < 0.8:  # Below 80% accuracy
                recommendation = self._create_model_improvement_recommendation(performance)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Analyze session patterns
        pattern_recommendations = self._analyze_session_patterns(session_analytics)
        recommendations.extend(pattern_recommendations)
        
        # Analyze candidate feedback
        feedback_recommendations = self._analyze_feedback_patterns(session_analytics)
        recommendations.extend(feedback_recommendations)
        
        # Prioritize recommendations
        recommendations = self._prioritize_recommendations(recommendations)
        
        logger.info(f"Generated {len(recommendations)} improvement recommendations")
        return recommendations
    
    def _create_model_improvement_recommendation(self, performance: ModelPerformance) -> Optional[ImprovementRecommendation]:
        """Create model improvement recommendation"""
        if performance.model_type == "question_quality_regression":
            return ImprovementRecommendation(
                improvement_type=ImprovementType.QUESTION_QUALITY,
                description=f"Improve question quality prediction model (current accuracy: {performance.accuracy:.2f})",
                expected_impact=0.15,
                confidence=0.8,
                implementation_complexity="medium",
                affected_components=["question_generator", "evaluation_engine"],
                data_evidence={"model_accuracy": performance.accuracy, "training_samples": performance.training_samples}
            )
        elif performance.model_type == "difficulty_calibration_classification":
            return ImprovementRecommendation(
                improvement_type=ImprovementType.DIFFICULTY_CALIBRATION,
                description=f"Improve difficulty level calibration (current accuracy: {performance.accuracy:.2f})",
                expected_impact=0.20,
                confidence=0.85,
                implementation_complexity="high",
                affected_components=["difficulty_engine", "pathing_engine"],
                data_evidence={"model_accuracy": performance.accuracy, "training_samples": performance.training_samples}
            )
        elif performance.model_type == "evaluation_accuracy_regression":
            return ImprovementRecommendation(
                improvement_type=ImprovementType.EVALUATION_ACCURACY,
                description=f"Improve evaluation accuracy (current accuracy: {performance.accuracy:.2f})",
                expected_impact=0.25,
                confidence=0.9,
                implementation_complexity="high",
                affected_components=["evaluation_engine", "scoring_system"],
                data_evidence={"model_accuracy": performance.accuracy, "training_samples": performance.training_samples}
            )
        return None
    
    def _analyze_session_patterns(self, session_analytics: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Analyze session patterns for improvement opportunities"""
        recommendations = []
        
        # Analyze question effectiveness
        if 'question_effectiveness' in session_analytics:
            low_effectiveness_questions = [
                q for q, score in session_analytics['question_effectiveness'].items() 
                if score < 0.6
            ]
            
            if low_effectiveness_questions:
                recommendations.append(ImprovementRecommendation(
                    improvement_type=ImprovementType.QUESTION_QUALITY,
                    description=f"Review and improve {len(low_effectiveness_questions)} low-effectiveness questions",
                    expected_impact=0.10,
                    confidence=0.75,
                    implementation_complexity="low",
                    affected_components=["question_bank", "question_generator"],
                    data_evidence={"low_effectiveness_count": len(low_effectiveness_questions)}
                ))
        
        # Analyze conversation flow issues
        if 'conversation_issues' in session_analytics:
            conversation_issues = session_analytics['conversation_issues']
            if conversation_issues.get('abandonment_rate', 0) > 0.15:  # >15% abandonment
                recommendations.append(ImprovementRecommendation(
                    improvement_type=ImprovementType.CONVERSATION_FLOW,
                    description=f"Reduce conversation abandonment rate (current: {conversation_issues['abandonment_rate']:.1%})",
                    expected_impact=0.20,
                    confidence=0.8,
                    implementation_complexity="medium",
                    affected_components=["conversation_engine", "user_interface"],
                    data_evidence={"abandonment_rate": conversation_issues['abandonment_rate']}
                ))
        
        return recommendations
    
    def _analyze_feedback_patterns(self, session_analytics: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Analyze feedback patterns for improvement opportunities"""
        recommendations = []
        
        if 'feedback_analysis' in session_analytics:
            feedback_data = session_analytics['feedback_analysis']
            
            # Low satisfaction areas
            low_satisfaction_areas = [
                area for area, score in feedback_data.get('satisfaction_by_area', {}).items()
                if score < 3.5  # Below 3.5/5.0
            ]
            
            if low_satisfaction_areas:
                recommendations.append(ImprovementRecommendation(
                    improvement_type=ImprovementType.EVALUATION_ACCURACY,
                    description=f"Address low satisfaction in areas: {', '.join(low_satisfaction_areas[:3])}",
                    expected_impact=0.15,
                    confidence=0.7,
                    implementation_complexity="medium",
                    affected_components=["evaluation_engine", "feedback_system"],
                    data_evidence={"low_satisfaction_areas": low_satisfaction_areas}
                ))
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[ImprovementRecommendation]) -> List[ImprovementRecommendation]:
        """Prioritize recommendations based on impact and feasibility"""
        # Calculate priority score (impact * confidence / complexity)
        complexity_weights = {"low": 1.0, "medium": 0.7, "high": 0.4}
        
        for rec in recommendations:
            complexity_weight = complexity_weights.get(rec.implementation_complexity, 0.5)
            rec.expected_impact = rec.expected_impact * rec.confidence * complexity_weight
        
        # Sort by priority score
        return sorted(recommendations, key=lambda x: x.expected_impact, reverse=True)

class ContinuousImprovementEngine:
    """Main engine for continuous improvement and active learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_collection_engine = DataCollectionEngine()
        self.model_training_engine = ModelTrainingEngine()
        self.recommendation_engine = ImprovementRecommendationEngine()
        self.current_phase = LearningPhase.DATA_COLLECTION
        self.is_learning_active = True
        self.learning_cycle_count = 0
        self.last_improvement_cycle = None
        
        logger.info("Initialized Continuous Improvement Engine")
    
    def process_learning_session(self, session_data: LearningSession) -> bool:
        """Process a completed interview session for learning"""
        try:
            # Collect session data
            collection_success = self.data_collection_engine.collect_session_data(session_data)
            if not collection_success:
                logger.error("Failed to collect session data")
                return False
            
            # Check if we should trigger learning cycle
            if self._should_trigger_learning_cycle():
                self._execute_learning_cycle()
            
            logger.info(f"Successfully processed learning session: {session_data.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing learning session: {str(e)}")
            return False
    
    def _should_trigger_learning_cycle(self) -> bool:
        """Determine if a learning cycle should be triggered"""
        # Check minimum data requirements
        training_data = self.data_collection_engine.get_training_data(min_sessions=50)
        if training_data is None or len(training_data) < 50:
            return False
        
        # Check time since last cycle (minimum 24 hours)
        if self.last_improvement_cycle:
            time_since_last = datetime.now() - self.last_improvement_cycle
            if time_since_last < timedelta(hours=24):
                return False
        
        # Check if model performance has degraded
        if self.model_training_engine.performance_history:
            recent_performances = [
                p for p in self.model_training_engine.performance_history
                if (datetime.now() - p.timestamp).days <= 7  # Last 7 days
            ]
            
            if recent_performances:
                avg_recent_accuracy = np.mean([p.accuracy for p in recent_performances])
                if avg_recent_accuracy < 0.75:  # Below 75% accuracy
                    return True
        
        # Trigger learning cycle every 100 sessions
        return len(training_data) % 100 == 0
    
    def _execute_learning_cycle(self):
        """Execute a complete learning cycle"""
        logger.info("Starting learning cycle...")
        self.learning_cycle_count += 1
        self.last_improvement_cycle = datetime.now()
        
        try:
            # Phase 1: Data Collection and Preprocessing
            self.current_phase = LearningPhase.DATA_COLLECTION
            training_data = self.data_collection_engine.get_training_data(min_sessions=50)
            
            if training_data is None:
                logger.warning("Insufficient training data for learning cycle")
                return
            
            # Phase 2: Model Training
            self.current_phase = LearningPhase.MODEL_TRAINING
            model_performances = []
            
            # Train question quality model
            question_performance = self.model_training_engine.train_question_quality_model(training_data)
            if question_performance:
                model_performances.append(question_performance)
            
            # Train difficulty calibration model
            difficulty_performance = self.model_training_engine.train_difficulty_calibration_model(training_data)
            if difficulty_performance:
                model_performances.append(difficulty_performance)
            
            # Train evaluation accuracy model
            evaluation_performance = self.model_training_engine.train_evaluation_accuracy_model(training_data)
            if evaluation_performance:
                model_performances.append(evaluation_performance)
            
            # Phase 3: Validation
            self.current_phase = LearningPhase.VALIDATION
            
            # Generate session analytics for recommendation engine
            session_analytics = self._generate_session_analytics(training_data)
            
            # Generate improvement recommendations
            recommendations = self.recommendation_engine.generate_improvement_recommendations(
                model_performances, session_analytics
            )
            
            # Phase 4: Deployment
            self.current_phase = LearningPhase.DEPLOYMENT
            
            # Apply high-confidence recommendations automatically
            self._apply_recommendations(recommendations)
            
            # Phase 5: Monitoring
            self.current_phase = LearningPhase.MONITORING
            
            logger.info(f"Learning cycle completed successfully. Generated {len(recommendations)} recommendations.")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {str(e)}")
            self.current_phase = LearningPhase.DATA_COLLECTION
    
    def _generate_session_analytics(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate session analytics for recommendation engine"""
        analytics = {}
        
        # Question effectiveness analysis
        if 'question_id' in training_data.columns:
            question_effectiveness = training_data.groupby('question_id')['evaluation_score'].mean().to_dict()
            analytics['question_effectiveness'] = question_effectiveness
        
        # Conversation flow analysis
        analytics['conversation_issues'] = {
            'abandonment_rate': 0.12,  # Placeholder - would be calculated from data
            'avg_response_time': training_data['avg_response_time'].mean(),
            'response_time_variance': training_data['avg_response_time'].var()
        }
        
        # Feedback analysis
        analytics['feedback_analysis'] = {
            'satisfaction_by_area': {
                'technical_accuracy': training_data['technical_accuracy'].mean(),
                'problem_solving': training_data['problem_solving'].mean(),
                'communication_clarity': training_data['communication_clarity'].mean(),
                'practical_application': training_data['practical_application'].mean(),
                'excel_expertise': training_data['excel_expertise'].mean()
            }
        }
        
        return analytics
    
    def _apply_recommendations(self, recommendations: List[ImprovementRecommendation]):
        """Apply high-confidence recommendations"""
        for rec in recommendations:
            if rec.confidence >= 0.8 and rec.expected_impact >= 0.15:  # High confidence, high impact
                logger.info(f"Applying recommendation: {rec.description}")
                # Here you would implement the actual recommendation logic
                # For now, we just log it
                self.recommendation_engine.implementation_tracker[rec.improvement_type] = {
                    'applied_at': datetime.now(),
                    'expected_impact': rec.expected_impact,
                    'status': 'applied'
                }
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        return {
            'current_phase': self.current_phase.value,
            'learning_cycle_count': self.learning_cycle_count,
            'is_learning_active': self.is_learning_active,
            'last_improvement_cycle': self.last_improvement_cycle,
            'trained_models': list(self.model_training_engine.trained_models.keys()),
            'recent_performances': [
                {
                    'model_type': p.model_type,
                    'accuracy': p.accuracy,
                    'timestamp': p.timestamp
                }
                for p in self.model_training_engine.performance_history[-5:]  # Last 5 performances
            ]
        }
    
    def export_learning_insights(self) -> Dict[str, Any]:
        """Export comprehensive learning insights"""
        return {
            'learning_status': self.get_learning_status(),
            'model_performances': [
                {
                    'model_type': p.model_type,
                    'accuracy': p.accuracy,
                    'mse': p.mse,
                    'training_samples': p.training_samples,
                    'timestamp': p.timestamp.isoformat()
                }
                for p in self.model_training_engine.performance_history
            ],
            'recommendation_history': [
                {
                    'improvement_type': rec.improvement_type.value,
                    'description': rec.description,
                    'expected_impact': rec.expected_impact,
                    'confidence': rec.confidence,
                    'implementation_complexity': rec.implementation_complexity
                }
                for rec in self.recommendation_engine.recommendation_history
            ]
        }