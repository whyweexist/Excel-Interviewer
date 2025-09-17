from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..utils.config import Config
from ..utils.logger import get_logger
from .conversation_state import InterviewContext, Question, Answer, EvaluationResult, DifficultyLevel, QuestionType

logger = get_logger(__name__)

class AgentPersonality(Enum):
    """Different personality modes for the AI interviewer."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CHALLENGING = "challenging"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"

class ConversationStrategy(Enum):
    """Different conversation strategies."""
    LINEAR_PROGRESSION = "linear_progression"
    ADAPTIVE_BRANCHING = "adaptive_branching"
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    MIXED_APPROACH = "mixed_approach"

@dataclass
class ConversationNode:
    """Represents a node in the conversation tree."""
    id: str
    question: Question
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    conditions: Dict[str, Any] = None
    probability_weight: float = 1.0
    depth_level: int = 0
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.conditions is None:
            self.conditions = {}

@dataclass
class PathDecision:
    """Represents a decision point in the conversation path."""
    decision_type: str
    criteria: Dict[str, Any]
    outcomes: List[Tuple[float, str]]  # (probability, next_node_id)
    fallback_action: str = "continue"

class DynamicPathingEngine:
    """Engine for dynamic conversation pathing based on candidate responses."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.openai_model,
            temperature=0.3,  # Lower temperature for more consistent pathing decisions
            max_tokens=500
        )
        self.conversation_tree = {}
        self.current_node_id = "root"
        self.decision_history = []
        self.personality_mode = AgentPersonality.PROFESSIONAL
        self.strategy = ConversationStrategy.ADAPTIVE_BRANCHING
        
    def analyze_response_quality(self, question: Question, answer: Answer, evaluation: EvaluationResult) -> Dict[str, Any]:
        """Analyze the quality and characteristics of a candidate's response."""
        
        # Calculate response metrics
        response_metrics = {
            "technical_depth": self._assess_technical_depth(answer.text, question),
            "clarity_score": evaluation.communication_score,
            "confidence_indicators": self._extract_confidence_indicators(answer.text),
            "knowledge_gaps": self._identify_knowledge_gaps(answer.text, question),
            "problem_solving_approach": self._assess_problem_solving_approach(answer.text),
            "response_time_ratio": answer.time_taken_seconds / question.max_time_seconds,
            "keyword_coverage": self._calculate_keyword_coverage(answer.text, question.expected_keywords),
            "elaboration_level": self._assess_elaboration_level(answer.text),
            "practical_examples": self._count_practical_examples(answer.text),
            "technical_accuracy": evaluation.technical_score
        }
        
        # Determine response category
        response_category = self._categorize_response(response_metrics)
        response_metrics["category"] = response_category
        
        return response_metrics
    
    def _assess_technical_depth(self, answer_text: str, question: Question) -> float:
        """Assess the technical depth of the response."""
        # Count technical terms and concepts
        technical_indicators = [
            "function", "formula", "syntax", "parameter", "argument",
            "reference", "range", "array", "logical", "nested",
            "optimization", "efficiency", "performance", "best practice"
        ]
        
        technical_count = sum(1 for indicator in technical_indicators if indicator.lower() in answer_text.lower())
        return min(technical_count / 5, 1.0) * 100  # Normalize to 0-100
    
    def _extract_confidence_indicators(self, answer_text: str) -> Dict[str, float]:
        """Extract confidence indicators from the response."""
        confidence_markers = {
            "high_confidence": ["definitely", "certainly", "absolutely", "always", "never", "must"],
            "uncertainty": ["maybe", "perhaps", "possibly", "might", "could be", "not sure"],
            "experience_based": ["in my experience", "I have found", "typically", "usually", "generally"],
            "authoritative": ["best practice", "recommended", "standard approach", "industry standard"]
        }
        
        results = {}
        answer_lower = answer_text.lower()
        
        for category, markers in confidence_markers.items():
            count = sum(1 for marker in markers if marker.lower() in answer_lower)
            results[category] = count / len(markers) if markers else 0
        
        return results
    
    def _identify_knowledge_gaps(self, answer_text: str, question: Question) -> List[str]:
        """Identify potential knowledge gaps in the response."""
        gaps = []
        
        # Check for missing expected keywords
        if question.expected_keywords:
            missing_keywords = [kw for kw in question.expected_keywords if kw.lower() not in answer_text.lower()]
            if missing_keywords:
                gaps.append(f"Missing key concepts: {', '.join(missing_keywords[:3])}")
        
        # Check for vague or generic responses
        vague_indicators = ["something like", "kind of", "basically", "just", "simply"]
        if any(indicator in answer_text.lower() for indicator in vague_indicators):
            gaps.append("Response may be too vague or oversimplified")
        
        # Check for lack of examples
        if "example" not in answer_text.lower() and question.question_type.value in ["practical", "scenario_based"]:
            gaps.append("No practical examples provided")
        
        return gaps
    
    def _assess_problem_solving_approach(self, answer_text: str) -> Dict[str, float]:
        """Assess the problem-solving approach demonstrated."""
        approach_indicators = {
            "systematic": ["step by step", "first", "then", "next", "finally", "approach", "method"],
            "analytical": ["analyze", "consider", "evaluate", "compare", "assess"],
            "creative": ["alternative", "different way", "workaround", "creative", "innovative"],
            "efficiency_focused": ["efficient", "optimize", "faster", "better", "improve"]
        }
        
        results = {}
        answer_lower = answer_text.lower()
        
        for approach, indicators in approach_indicators.items():
            count = sum(1 for indicator in indicators if indicator.lower() in answer_lower)
            results[approach] = min(count / 3, 1.0)  # Normalize to 0-1
        
        return results
    
    def _calculate_keyword_coverage(self, answer_text: str, expected_keywords: List[str]) -> float:
        """Calculate how well the answer covers expected keywords."""
        if not expected_keywords:
            return 0.5  # Neutral score if no keywords specified
        
        answer_lower = answer_text.lower()
        covered_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        return len(covered_keywords) / len(expected_keywords)
    
    def _assess_elaboration_level(self, answer_text: str) -> float:
        """Assess the level of elaboration in the response."""
        word_count = len(answer_text.split())
        
        if word_count < 30:
            return 0.2  # Too brief
        elif word_count < 100:
            return 0.5  # Moderate
        elif word_count < 200:
            return 0.8  # Good elaboration
        else:
            return 1.0  # Very detailed
    
    def _count_practical_examples(self, answer_text: str) -> int:
        """Count practical examples mentioned in the response."""
        example_indicators = ["for example", "for instance", "such as", "like when", "suppose", "imagine"]
        return sum(1 for indicator in example_indicators if indicator.lower() in answer_text.lower())
    
    def _categorize_response(self, metrics: Dict[str, Any]) -> str:
        """Categorize the response based on metrics."""
        technical_score = metrics.get("technical_accuracy", 0)
        clarity_score = metrics.get("clarity_score", 0)
        keyword_coverage = metrics.get("keyword_coverage", 0)
        
        if technical_score >= 80 and clarity_score >= 80 and keyword_coverage >= 0.7:
            return "excellent_comprehensive"
        elif technical_score >= 70 and clarity_score >= 70:
            return "good_understanding"
        elif technical_score >= 60 or keyword_coverage >= 0.5:
            return "basic_understanding"
        elif technical_score < 40:
            return "poor_understanding"
        else:
            return "needs_improvement"
    
    def determine_next_action(self, context: InterviewContext, response_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the next action based on response analysis and context."""
        
        decision_prompt = f"""
        Based on the following interview context and candidate response metrics, determine the optimal next action:
        
        Interview Context:
        - Current State: {context.current_state.value}
        - Questions Asked: {len(context.questions_asked)}
        - Average Score: {sum(r.overall_score for r in context.evaluation_results) / len(context.evaluation_results) if context.evaluation_results else 0:.1f}
        - Topics Covered: {list(context.skill_assessments.keys())}
        
        Response Metrics:
        - Category: {response_metrics.get('category', 'unknown')}
        - Technical Depth: {response_metrics.get('technical_depth', 0):.1f}
        - Clarity Score: {response_metrics.get('clarity_score', 0):.1f}
        - Keyword Coverage: {response_metrics.get('keyword_coverage', 0):.2f}
        - Response Time Ratio: {response_metrics.get('response_time_ratio', 0):.2f}
        
        Available Actions:
        1. continue_same_topic - Ask another question on the same topic
        2. increase_difficulty - Move to more advanced questions
        3. decrease_difficulty - Move to easier questions or provide hints
        4. ask_followup - Ask a specific follow-up question
        5. practical_challenge - Give a hands-on Excel challenge
        6. change_topic - Move to a different topic area
        7. summarize_conclude - Start wrapping up the interview
        
        Consider:
        - Candidate's confidence level and knowledge gaps
        - Interview progression and time constraints
        - Balance between different skill areas
        - Maintaining candidate engagement
        
        Provide your decision in JSON format:
        {{
            "action": "chosen_action",
            "reasoning": "detailed explanation of why this action is optimal",
            "confidence": 0.8,
            "alternative_actions": ["action1", "action2"]
        }}
        """
        
        try:
            response = self.llm.invoke(decision_prompt)
            decision_data = json.loads(response.content)
            
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "context": {
                    "questions_asked": len(context.questions_asked),
                    "average_score": sum(r.overall_score for r in context.evaluation_results) / len(context.evaluation_results) if context.evaluation_results else 0,
                    "topics_covered": list(context.skill_assessments.keys())
                },
                "response_metrics": response_metrics,
                "decision": decision_data
            })
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Error in determine_next_action: {e}")
            # Fallback to simple rule-based decision
            return self._fallback_decision(context, response_metrics)
    
    def _fallback_decision(self, context: InterviewContext, response_metrics: str) -> Dict[str, Any]:
        """Fallback decision logic when AI decision fails."""
        category = response_metrics.get('category', 'unknown')
        questions_asked = len(context.questions_asked)
        
        if category == "excellent_comprehensive":
            action = "increase_difficulty" if questions_asked < 5 else "change_topic"
        elif category == "good_understanding":
            action = "continue_same_topic" if questions_asked < 3 else "practical_challenge"
        elif category == "basic_understanding":
            action = "ask_followup"
        elif category == "poor_understanding":
            action = "decrease_difficulty"
        else:
            action = "ask_followup"
        
        return {
            "action": action,
            "reasoning": f"Fallback decision based on response category: {category}",
            "confidence": 0.5,
            "alternative_actions": ["continue_same_topic", "ask_followup"]
        }
    
    def generate_dynamic_response(self, question: Question, candidate_answer: Answer, 
                                  evaluation: EvaluationResult, personality: AgentPersonality = None) -> str:
        """Generate a dynamic response based on the candidate's answer and evaluation."""
        
        if personality is None:
            personality = self.personality_mode
        
        response_templates = {
            AgentPersonality.PROFESSIONAL: {
                "excellent": [
                    "Excellent response! Your understanding of {topic} is very thorough.",
                    "That's a comprehensive answer. You clearly understand the concepts.",
                    "Very well explained. Your technical knowledge is impressive."
                ],
                "good": [
                    "Good answer! You demonstrate solid understanding of {topic}.",
                    "That's correct. You have a good grasp of the fundamentals.",
                    "Well done. Your answer shows practical understanding."
                ],
                "needs_improvement": [
                    "Thank you for your response. Let me ask a follow-up question to explore this further.",
                    "I see your point. Could you elaborate on {aspect}?",
                    "That's a start. What about considering {additional_point}?"
                ],
                "poor": [
                    "I appreciate your attempt. Let me provide some context to help you.",
                    "Thank you for trying. This is a complex topic, let me guide you through it.",
                    "I understand this might be challenging. Let's break it down step by step."
                ]
            },
            AgentPersonality.SUPPORTIVE: {
                "excellent": [
                    "Fantastic! You're doing great with {topic}.",
                    "Amazing work! Your knowledge really shines through.",
                    "Perfect! You've clearly mastered these concepts."
                ],
                "good": [
                    "Nice job! You're on the right track with {topic}.",
                    "Good work! I can see you're building a solid understanding.",
                    "Well done! Your approach to {topic} is very good."
                ],
                "needs_improvement": [
                    "You're getting there! Let's explore this a bit more.",
                    "Good effort! I can help you develop this further.",
                    "Nice try! Let me ask you something that might clarify this."
                ],
                "poor": [
                    "No worries! Everyone learns at their own pace.",
                    "That's okay! This is tricky stuff, let's work through it together.",
                    "Don't worry! I'm here to help you understand this better."
                ]
            }
        }
        
        # Determine response category based on evaluation
        if evaluation.overall_score >= 90:
            category = "excellent"
        elif evaluation.overall_score >= 70:
            category = "good"
        elif evaluation.overall_score >= 50:
            category = "needs_improvement"
        else:
            category = "poor"
        
        # Select appropriate template
        templates = response_templates.get(personality, response_templates[AgentPersonality.PROFESSIONAL])
        category_templates = templates.get(category, templates["needs_improvement"])
        
        template = random.choice(category_templates)
        
        # Fill in template variables
        response = template.format(
            topic=question.topic.replace("_", " "),
            aspect="this concept" if category == "needs_improvement" else "your approach",
            additional_point="the practical applications" if category == "needs_improvement" else "additional considerations"
        )
        
        return response
    
    def adapt_conversation_style(self, context: InterviewContext) -> AgentPersonality:
        """Adapt the conversation style based on candidate performance and engagement."""
        if not context.evaluation_results:
            return AgentPersonality.PROFESSIONAL
        
        # Analyze recent performance
        recent_scores = [result.overall_score for result in context.evaluation_results[-3:]]
        average_recent_score = sum(recent_scores) / len(recent_scores)
        
        # Analyze response patterns
        response_times = []
        answer_lengths = []
        
        for answer in context.answers_given[-3:]:
            response_times.append(answer.time_taken_seconds)
            answer_lengths.append(len(answer.text.split()))
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        
        # Determine optimal personality mode
        if average_recent_score < 50:
            return AgentPersonality.SUPPORTIVE
        elif average_recent_score > 85 and avg_answer_length > 100:
            return AgentPersonality.CHALLENGING
        elif avg_response_time > 300:  # Slow responses might indicate difficulty
            return AgentPersonality.SUPPORTIVE
        else:
            return AgentPersonality.PROFESSIONAL
    
    def generate_personalized_recommendations(self, context: InterviewContext) -> List[str]:
        """Generate personalized recommendations based on the entire interview."""
        recommendations = []
        
        # Analyze overall performance patterns
        all_scores = [result.overall_score for result in context.evaluation_results]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Topic-specific analysis
        topic_performance = {}
        for result in context.evaluation_results:
            topic = result.question_id.split("_")[0]
            if topic not in topic_performance:
                topic_performance[topic] = []
            topic_performance[topic].append(result.overall_score)
        
        # Generate recommendations based on analysis
        if avg_score < 70:
            recommendations.append("Focus on strengthening your foundational Excel knowledge before advancing to complex topics.")
        
        if avg_score > 85:
            recommendations.append("Consider exploring advanced Excel features and automation techniques to further enhance your skills.")
        
        # Topic-specific recommendations
        for topic, scores in topic_performance.items():
            topic_avg = sum(scores) / len(scores)
            topic_name = topic.replace("_", " ").title()
            
            if topic_avg < 60:
                recommendations.append(f"Dedicate time to studying {topic_name} concepts and practicing related exercises.")
            elif topic_avg > 90:
                recommendations.append(f"Your {topic_name} skills are excellent - consider mentoring others or exploring related advanced topics.")
        
        # Communication recommendations
        communication_scores = [result.communication_score for result in context.evaluation_results]
        avg_communication = sum(communication_scores) / len(communication_scores) if communication_scores else 0
        
        if avg_communication < 70:
            recommendations.append("Work on clearly articulating your thought process and providing more structured explanations.")
        
        return recommendations[:5]  # Limit to top 5 recommendations