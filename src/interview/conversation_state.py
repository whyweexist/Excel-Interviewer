from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

class InterviewState(Enum):
    """Enumeration of possible interview states."""
    INITIALIZING = "initializing"
    INTRODUCTION = "introduction"
    TOPIC_ASSESSMENT = "topic_assessment"
    PRACTICAL_CHALLENGE = "practical_challenge"
    FOLLOW_UP_QUESTIONS = "follow_up_questions"
    SUMMARY = "summary"
    COMPLETED = "completed"
    ERROR = "error"

class QuestionType(Enum):
    """Types of questions that can be asked."""
    TECHNICAL = "technical"
    PRACTICAL = "practical"
    SCENARIO_BASED = "scenario_based"
    FOLLOW_UP = "follow_up"
    CHALLENGE = "challenge"

class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ResponseQuality(Enum):
    """Quality assessment of candidate responses."""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

@dataclass
class Question:
    """Represents a single interview question."""
    id: str
    text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    topic: str
    expected_keywords: List[str] = field(default_factory=list)
    scoring_criteria: Dict[str, Any] = field(default_factory=dict)
    follow_up_questions: List[str] = field(default_factory=list)
    max_time_seconds: int = 180
    hints: List[str] = field(default_factory=list)

@dataclass
class Answer:
    """Represents a candidate's answer to a question."""
    question_id: str
    text: str
    timestamp: datetime
    time_taken_seconds: int
    confidence_level: Optional[float] = None
    self_assessment: Optional[str] = None

@dataclass
class EvaluationResult:
    """Results of evaluating a candidate's answer."""
    question_id: str
    technical_score: float
    problem_solving_score: float
    communication_score: float
    practical_score: float
    overall_score: float
    quality_rating: ResponseQuality
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    detailed_feedback: str = ""

@dataclass
class InterviewContext:
    """Context information for the current interview session."""
    candidate_name: str
    interview_id: str
    current_state: InterviewState
    current_question_index: int = 0
    questions_asked: List[Question] = field(default_factory=list)
    answers_given: List[Answer] = field(default_factory=list)
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    overall_score: Optional[float] = None
    skill_assessments: Dict[str, float] = field(default_factory=dict)
    difficulty_progression: List[DifficultyLevel] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_number: int
    speaker: str  # "interviewer" or "candidate"
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationStateMachine:
    """State machine for managing interview conversation flow."""
    
    def __init__(self):
        self.current_state = InterviewState.INITIALIZING
        self.state_history = []
        self.context = None
        self.transition_rules = self._define_transition_rules()
    
    def _define_transition_rules(self) -> Dict[InterviewState, List[InterviewState]]:
        """Define valid state transitions."""
        return {
            InterviewState.INITIALIZING: [InterviewState.INTRODUCTION],
            InterviewState.INTRODUCTION: [InterviewState.TOPIC_ASSESSMENT],
            InterviewState.TOPIC_ASSESSMENT: [
                InterviewState.PRACTICAL_CHALLENGE,
                InterviewState.FOLLOW_UP_QUESTIONS,
                InterviewState.SUMMARY
            ],
            InterviewState.PRACTICAL_CHALLENGE: [
                InterviewState.FOLLOW_UP_QUESTIONS,
                InterviewState.TOPIC_ASSESSMENT,
                InterviewState.SUMMARY
            ],
            InterviewState.FOLLOW_UP_QUESTIONS: [
                InterviewState.TOPIC_ASSESSMENT,
                InterviewState.PRACTICAL_CHALLENGE,
                InterviewState.SUMMARY
            ],
            InterviewState.SUMMARY: [InterviewState.COMPLETED],
            InterviewState.COMPLETED: [],
            InterviewState.ERROR: [InterviewState.INTRODUCTION]
        }
    
    def initialize_interview(self, candidate_name: str, interview_id: str) -> InterviewContext:
        """Initialize a new interview session."""
        self.context = InterviewContext(
            candidate_name=candidate_name,
            interview_id=interview_id,
            current_state=InterviewState.INTRODUCTION,
            start_time=datetime.now()
        )
        self.current_state = InterviewState.INTRODUCTION
        self.state_history = [InterviewState.INITIALIZING, InterviewState.INTRODUCTION]
        return self.context
    
    def transition_to(self, new_state: InterviewState, reason: str = "") -> bool:
        """Transition to a new state if valid."""
        if new_state in self.transition_rules[self.current_state]:
            self.state_history.append(self.current_state)
            self.current_state = new_state
            if self.context:
                self.context.current_state = new_state
                self.context.conversation_history.append({
                    "type": "state_transition",
                    "from_state": self.state_history[-1].value,
                    "to_state": new_state.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
            return True
        return False
    
    def get_current_state(self) -> InterviewState:
        """Get the current state."""
        return self.current_state
    
    def get_state_history(self) -> List[InterviewState]:
        """Get the history of states."""
        return self.state_history.copy()
    
    def get_context(self) -> Optional[InterviewContext]:
        """Get the current interview context."""
        return self.context
    
    def is_interview_complete(self) -> bool:
        """Check if the interview is complete."""
        return self.current_state == InterviewState.COMPLETED
    
    def get_next_valid_states(self) -> List[InterviewState]:
        """Get list of valid next states."""
        return self.transition_rules[self.current_state].copy()
    
    def record_conversation_turn(self, speaker: str, message: str, metadata: Dict[str, Any] = None):
        """Record a conversation turn."""
        if self.context:
            turn = ConversationTurn(
                turn_number=len(self.context.conversation_history) + 1,
                speaker=speaker,
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.context.conversation_history.append({
                "speaker": speaker,
                "message": message,
                "timestamp": turn.timestamp.isoformat(),
                "metadata": turn.metadata
            })
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation so far."""
        if not self.context:
            return {}
        
        return {
            "total_turns": len(self.context.conversation_history),
            "current_state": self.current_state.value,
            "state_history": [state.value for state in self.state_history],
            "questions_asked": len(self.context.questions_asked),
            "answers_given": len(self.context.answers_given),
            "start_time": self.context.start_time.isoformat() if self.context.start_time else None,
            "duration_minutes": self._calculate_duration_minutes()
        }
    
    def _calculate_duration_minutes(self) -> float:
        """Calculate interview duration in minutes."""
        if not self.context or not self.context.start_time:
            return 0.0
        
        end_time = self.context.end_time or datetime.now()
        duration = end_time - self.context.start_time
        return duration.total_seconds() / 60.0