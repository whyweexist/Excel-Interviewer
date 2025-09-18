import openai
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from ..utils.config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class EvaluationDimension(Enum):
    TECHNICAL_ACCURACY = "technical_accuracy"
    PROBLEM_SOLVING = "problem_solving"
    COMMUNICATION_CLARITY = "communication_clarity"
    PRACTICAL_APPLICATION = "practical_application"
    EXCEL_SPECIFIC = "excel_specific"

class ScoreLevel(Enum):
    EXCELLENT = (90, 100)
    GOOD = (75, 89)
    AVERAGE = (60, 74)
    BELOW_AVERAGE = (40, 59)
    POOR = (0, 39)

@dataclass
class DimensionScore:
    dimension: EvaluationDimension
    score: float
    max_score: float = 25.0
    feedback: str = ""
    strengths: List[str] = None
    weaknesses: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class ComprehensiveEvaluation:
    total_score: float
    dimension_scores: Dict[EvaluationDimension, DimensionScore]
    overall_feedback: str
    skill_level: str
    recommended_next_topics: List[str]
    confidence_score: float
    evaluation_timestamp: datetime
    question_context: str
    answer_text: str
    
    @property
    def percentage_score(self) -> float:
        return (self.total_score / 100.0) * 100
    
    @property
    def skill_category(self) -> str:
        score = self.percentage_score
        for level in ScoreLevel:
            min_score, max_score = level.value
            if min_score <= score <= max_score:
                return level.name.replace("_", " ").title()
        return "Unknown"

class AnswerEvaluator:
    # class AnswerEvaluator:
    def __init__(self, config=None):
        # Use passed config if available, else fallback to global settings
        self.config = config or settings

        self.llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=self.config.openai_temperature,
            api_key=self.config.openai_api_key
            # temperature=self.config.OPENAI_TEMPERATURE,
            # api_key=self.config.OPENAI_API_KEY
        )
        self.evaluation_prompts = self._create_evaluation_prompts()
        self.excel_knowledge_base = self._load_excel_knowledge_base()

    # def __init__(self):
    #     self.llm = ChatOpenAI(
    #         model=settings.OPENAI_MODEL,
    #         temperature=settings.OPENAI_TEMPERATURE,
    #         api_key=settings.OPENAI_API_KEY
    #     )
    #     self.evaluation_prompts = self._create_evaluation_prompts()
    #     self.excel_knowledge_base = self._load_excel_knowledge_base()
        
    def _create_evaluation_prompts(self) -> Dict[EvaluationDimension, ChatPromptTemplate]:
        prompts = {}
        
        # Technical Accuracy Evaluation
        prompts[EvaluationDimension.TECHNICAL_ACCURACY] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert Excel technical evaluator. Assess the technical accuracy of the candidate's response.
            
Scoring Criteria (out of 25 points):
- Correctness of Excel concepts mentioned (10 points)
- Accuracy of formulas/functions (10 points)
- Technical depth and precision (5 points)

Provide specific feedback on what was correct and what was incorrect."""),
            HumanMessage(content="""Question: {question}
Candidate's Answer: {answer}

Evaluate the technical accuracy of this response. Consider:
1. Are the Excel concepts mentioned correct?
2. Are formulas/functions accurate?
3. Is the technical information precise?

Provide a score (0-25) with detailed justification.""")
        ])
        
        # Problem Solving Evaluation
        prompts[EvaluationDimension.PROBLEM_SOLVING] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are evaluating problem-solving methodology in Excel contexts. Assess how well the candidate approaches and solves problems.
            
Scoring Criteria (out of 25 points):
- Logical approach to problem (10 points)
- Methodology clarity (8 points)
- Efficiency of solution (7 points)

Focus on the thinking process, not just the final answer."""),
            HumanMessage(content="""Question: {question}
Candidate's Answer: {answer}

Evaluate the problem-solving approach:
1. Is there a clear, logical methodology?
2. How efficient is the proposed solution?
3. Does the candidate show good analytical thinking?

Provide a score (0-25) with detailed feedback.""")
        ])
        
        # Communication Clarity Evaluation
        prompts[EvaluationDimension.COMMUNICATION_CLARITY] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Evaluate how clearly the candidate communicates their Excel knowledge and solutions.
            
Scoring Criteria (out of 25 points):
- Clarity of explanation (10 points)
- Use of appropriate terminology (8 points)
- Structure and organization (7 points)

Consider both technical accuracy and ease of understanding."""),
            HumanMessage(content="""Question: {question}
Candidate's Answer: {answer}

Assess communication effectiveness:
1. How clear and understandable is the explanation?
2. Is Excel terminology used appropriately?
3. Is the response well-structured and organized?

Provide a score (0-25) with specific feedback.""")
        ])
        
        # Practical Application Evaluation
        prompts[EvaluationDimension.PRACTICAL_APPLICATION] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Evaluate the candidate's understanding of practical Excel applications and real-world scenarios.
            
Scoring Criteria (out of 25 points):
- Real-world relevance (10 points)
- Practical implementation knowledge (10 points)
- Understanding of business contexts (5 points)

Focus on applicability to actual work scenarios."""),
            HumanMessage(content="""Question: {question}
Candidate's Answer: {answer}

Evaluate practical application knowledge:
1. Does the answer show real-world understanding?
2. Is there knowledge of practical implementation?
3. Does the candidate understand business contexts?

Provide a score (0-25) with detailed feedback.""")
        ])
        
        return prompts
    
    def _load_excel_knowledge_base(self) -> Dict:
        return {
            "formulas": {
                "basic": ["SUM", "AVERAGE", "COUNT", "MAX", "MIN"],
                "intermediate": ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "IF", "COUNTIF", "SUMIF"],
                "advanced": ["ARRAY_FORMULAS", "INDIRECT", "OFFSET", "VLOOKUP_COMBINATIONS", "NESTED_IF"]
            },
            "functions": {
                "text": ["LEFT", "RIGHT", "MID", "CONCATENATE", "TEXT"],
                "date_time": ["TODAY", "NOW", "DATEDIF", "EOMONTH", "WORKDAY"],
                "logical": ["IF", "AND", "OR", "NOT", "IFERROR"],
                "lookup": ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "XLOOKUP"]
            },
            "features": {
                "data_analysis": ["PIVOT_TABLES", "DATA_VALIDATION", "CONDITIONAL_FORMATTING"],
                "visualization": ["CHARTS", "SPARKLINES", "DASHBOARD_CREATION"],
                "automation": ["MACROS", "VBA", "POWER_QUERY"]
            }
        }
    
    async def evaluate_answer(self, question: str, answer: str, question_type: str = "general", context: Optional[Dict] = None) -> ComprehensiveEvaluation:
        try:
            logger.info(f"Starting evaluation for question: {question[:50]}...")
            
            dimension_scores = {}
            total_score = 0.0
            
            # Evaluate each dimension
            for dimension in EvaluationDimension:
                if dimension == EvaluationDimension.EXCEL_SPECIFIC:
                    score = await self._evaluate_excel_specific(question, answer, question_type)
                else:
                    score = await self._evaluate_dimension(dimension, question, answer, context)
                
                dimension_scores[dimension] = score
                total_score += score.score
            
            # Generate overall feedback
            overall_feedback = await self._generate_overall_feedback(dimension_scores, question, answer)
            
            # Determine skill level and recommendations
            skill_level = self._determine_skill_level(total_score)
            recommended_topics = self._generate_recommendations(dimension_scores)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(dimension_scores)
            
            evaluation = ComprehensiveEvaluation(
                total_score=total_score,
                dimension_scores=dimension_scores,
                overall_feedback=overall_feedback,
                skill_level=skill_level,
                recommended_next_topics=recommended_topics,
                confidence_score=confidence_score,
                evaluation_timestamp=datetime.now(),
                question_context=question,
                answer_text=answer
            )
            
            logger.info(f"Evaluation completed. Total score: {total_score}/100")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise Exception(f"Failed to evaluate answer: {str(e)}")
    
    async def _evaluate_dimension(self, dimension: EvaluationDimension, question: str, answer: str, context: Optional[Dict] = None) -> DimensionScore:
        try:
            prompt = self.evaluation_prompts[dimension]
            
            messages = prompt.format_messages(
                question=question,
                answer=answer
            )
            
            response = await self.llm.ainvoke(messages)
            evaluation_text = response.content
            
            # Parse score and feedback from LLM response
            score, feedback, strengths, weaknesses, suggestions = self._parse_evaluation_response(evaluation_text)
            
            return DimensionScore(
                dimension=dimension,
                score=score,
                feedback=feedback,
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error evaluating dimension {dimension}: {str(e)}")
            return DimensionScore(
                dimension=dimension,
                score=0.0,
                feedback=f"Evaluation failed: {str(e)}",
                strengths=[],
                weaknesses=["Unable to evaluate this dimension"],
                suggestions=["Please provide more detailed response"]
            )
    
    async def _evaluate_excel_specific(self, question: str, answer: str, question_type: str) -> DimensionScore:
        try:
            # Extract Excel-specific elements from answer
            mentioned_formulas = self._extract_formulas(answer)
            mentioned_functions = self._extract_functions(answer)
            mentioned_features = self._extract_features(answer)
            
            # Calculate Excel-specific score
            score = self._calculate_excel_score(mentioned_formulas, mentioned_functions, mentioned_features, question_type)
            
            feedback = self._generate_excel_feedback(mentioned_formulas, mentioned_functions, mentioned_features)
            
            return DimensionScore(
                dimension=EvaluationDimension.EXCEL_SPECIFIC,
                score=score,
                feedback=feedback,
                strengths=self._identify_excel_strengths(mentioned_formulas, mentioned_functions, mentioned_features),
                weaknesses=self._identify_excel_weaknesses(mentioned_formulas, mentioned_functions, mentioned_features, question_type),
                suggestions=self._generate_excel_suggestions(mentioned_formulas, mentioned_functions, mentioned_features)
            )
            
        except Exception as e:
            logger.error(f"Error in Excel-specific evaluation: {str(e)}")
            return DimensionScore(
                dimension=EvaluationDimension.EXCEL_SPECIFIC,
                score=0.0,
                feedback="Excel-specific evaluation failed",
                strengths=[],
                weaknesses=["Unable to evaluate Excel knowledge"],
                suggestions=["Please demonstrate Excel knowledge in your response"]
            )
    
    def _extract_formulas(self, answer: str) -> List[str]:
        formulas = []
        answer_lower = answer.lower()
        
        for category, formula_list in self.excel_knowledge_base["formulas"].items():
            for formula in formula_list:
                if formula.lower() in answer_lower:
                    formulas.append(formula)
        
        return formulas
    
    def _extract_functions(self, answer: str) -> List[str]:
        functions = []
        answer_lower = answer.lower()
        
        for category, function_list in self.excel_knowledge_base["functions"].items():
            for function in function_list:
                if function.lower() in answer_lower:
                    functions.append(function)
        
        return functions
    
    def _extract_features(self, answer: str) -> List[str]:
        features = []
        answer_lower = answer.lower()
        
        for category, feature_list in self.excel_knowledge_base["features"].items():
            for feature in feature_list:
                if feature.lower().replace("_", " ") in answer_lower:
                    features.append(feature)
        
        return features
    
    def _calculate_excel_score(self, formulas: List[str], functions: List[str], features: List[str], question_type: str) -> float:
        base_score = 0.0
        
        # Score for formulas mentioned
        for formula in formulas:
            if formula in self.excel_knowledge_base["formulas"]["basic"]:
                base_score += 2.0
            elif formula in self.excel_knowledge_base["formulas"]["intermediate"]:
                base_score += 3.0
            elif formula in self.excel_knowledge_base["formulas"]["advanced"]:
                base_score += 5.0
        
        # Score for functions mentioned
        for function in functions:
            base_score += 2.5
        
        # Score for features mentioned
        for feature in features:
            base_score += 3.0
        
        # Adjust based on question type
        if question_type == "advanced":
            base_score *= 1.2
        elif question_type == "basic":
            base_score *= 0.8
        
        # Cap at 25
        return min(base_score, 25.0)
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, str, List[str], List[str], List[str]]:
        lines = response.strip().split('\n')
        score = 0.0
        feedback = ""
        strengths = []
        weaknesses = []
        suggestions = []
        
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if "score:" in line.lower() or line.replace('.', '').isdigit():
                try:
                    score = float(re.findall(r'\d+(?:\.\d+)?', line)[0])
                except:
                    score = 0.0
            elif "strength" in line.lower():
                current_section = "strengths"
            elif "weakness" in line.lower():
                current_section = "weaknesses"
            elif "suggestion" in line.lower():
                current_section = "suggestions"
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                if current_section == "strengths":
                    strengths.append(content)
                elif current_section == "weaknesses":
                    weaknesses.append(content)
                elif current_section == "suggestions":
                    suggestions.append(content)
                else:
                    feedback += content + " "
            elif line:
                feedback += line + " "
        
        return score, feedback.strip(), strengths, weaknesses, suggestions
    
    def _generate_overall_feedback(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], question: str, answer: str) -> str:
        total_score = sum(score.score for score in dimension_scores.values())
        
        if total_score >= 90:
            return "Excellent response! Demonstrated strong Excel knowledge with clear communication and practical understanding."
        elif total_score >= 75:
            return "Good response with solid Excel knowledge. Some areas for improvement but overall strong performance."
        elif total_score >= 60:
            return "Average response. Shows basic Excel understanding but needs improvement in several areas."
        elif total_score >= 40:
            return "Below average response. Limited Excel knowledge demonstrated, significant improvement needed."
        else:
            return "Poor response. Minimal Excel knowledge shown, requires substantial learning and practice."
    
    def _determine_skill_level(self, total_score: float) -> str:
        if total_score >= 90:
            return "Expert"
        elif total_score >= 75:
            return "Advanced"
        elif total_score >= 60:
            return "Intermediate"
        elif total_score >= 40:
            return "Basic"
        else:
            return "Novice"
    
    def _generate_recommendations(self, dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> List[str]:
        recommendations = []
        
        for dimension, score in dimension_scores.items():
            if score.score < 15:  # Below 60% of max
                if dimension == EvaluationDimension.TECHNICAL_ACCURACY:
                    recommendations.extend(["Review Excel formula syntax", "Practice basic functions", "Study Excel documentation"])
                elif dimension == EvaluationDimension.PROBLEM_SOLVING:
                    recommendations.extend(["Practice structured problem-solving", "Learn systematic approaches", "Work on analytical thinking"])
                elif dimension == EvaluationDimension.COMMUNICATION_CLARITY:
                    recommendations.extend(["Practice explaining technical concepts", "Improve presentation skills", "Work on clear documentation"])
                elif dimension == EvaluationDimension.PRACTICAL_APPLICATION:
                    recommendations.extend(["Gain real-world Excel experience", "Practice with business scenarios", "Study use cases"])
        
        return list(set(recommendations))[:5]  # Return unique recommendations, max 5
    
    def _calculate_confidence_score(self, dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> float:
        scores = [score.score for score in dimension_scores.values()]
        variance = np.var(scores)
        mean_score = np.mean(scores)
        
        # Lower variance and higher mean indicate higher confidence
        confidence = (1 - (variance / 625)) * (mean_score / 25)  # 625 = 25^2
        return max(0.0, min(1.0, confidence))
    
    def _generate_excel_feedback(self, formulas: List[str], functions: List[str], features: List[str]) -> str:
        feedback_parts = []
        
        if formulas:
            feedback_parts.append(f"Mentioned formulas: {', '.join(formulas)}")
        if functions:
            feedback_parts.append(f"Used functions: {', '.join(functions)}")
        if features:
            feedback_parts.append(f"Referenced features: {', '.join(features)}")
        
        if not feedback_parts:
            return "No specific Excel elements identified in the response."
        
        return "Excel elements detected: " + "; ".join(feedback_parts)
    
    def _identify_excel_strengths(self, formulas: List[str], functions: List[str], features: List[str]) -> List[str]:
        strengths = []
        
        if any(f in self.excel_knowledge_base["formulas"]["advanced"] for f in formulas):
            strengths.append("Knowledge of advanced Excel formulas")
        if len(functions) > 3:
            strengths.append("Good range of Excel functions")
        if any(f in ["PIVOT_TABLES", "MACROS", "POWER_QUERY"] for f in features):
            strengths.append("Understanding of advanced Excel features")
        
        return strengths if strengths else ["Basic Excel knowledge demonstrated"]
    
    def _identify_excel_weaknesses(self, formulas: List[str], functions: List[str], features: List[str], question_type: str) -> List[str]:
        weaknesses = []
        
        if not formulas and question_type != "general":
            weaknesses.append("No specific formulas mentioned")
        if not functions:
            weaknesses.append("No Excel functions identified")
        if question_type == "advanced" and not any(f in self.excel_knowledge_base["formulas"]["advanced"] for f in formulas):
            weaknesses.append("Limited advanced formula knowledge")
        
        return weaknesses if weaknesses else ["Could demonstrate more Excel-specific knowledge"]
    
    def _generate_excel_suggestions(self, formulas: List[str], functions: List[str], features: List[str]) -> List[str]:
        suggestions = []
        
        if not formulas:
            suggestions.append("Practice using specific Excel formulas")
        if not functions:
            suggestions.append("Learn common Excel functions")
        if not any(f in self.excel_knowledge_base["formulas"]["intermediate"] for f in formulas):
            suggestions.append("Study intermediate Excel formulas like VLOOKUP and IF")
        
        return suggestions if suggestions else ["Continue building Excel knowledge"]