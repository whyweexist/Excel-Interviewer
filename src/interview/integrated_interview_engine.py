import asyncio
import json
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import numpy as np

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from .conversation_state import ConversationStateMachine, InterviewState, Question, Answer
from .interview_engine import InterviewEngine
from ..agents.dynamic_pathing_engine import DynamicPathingEngine
from ..evaluation.answer_evaluator import AnswerEvaluator, ComprehensiveEvaluation
from ..simulation.excel_simulator import ExcelSimulator, ChallengeType, DifficultyLevel
from ..utils.config import settings
from ..utils.logger import get_logger, log_interaction, log_evaluation

logger = get_logger(__name__)

class IntegratedInterviewEngine:
    def __init__(self):
        self.conversation_state = ConversationStateMachine()
        self.interview_engine = InterviewEngine()
        self.dynamic_pathing = DynamicPathingEngine()
        self.answer_evaluator = AnswerEvaluator()
        self.excel_simulator = ExcelSimulator()
        
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )
        
        self.session_data = {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now(),
            "interactions": [],
            "evaluations": [],
            "challenges": [],
            "user_profile": {},
            "skill_assessment": {}
        }
        
        self.introduction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI Excel interviewer. Provide a professional, warm introduction that:
1. Welcomes the candidate and explains the interview process
2. Sets clear expectations about the format and duration
3. Emphasizes that this is a conversational assessment of Excel skills
4. Mentions that there will be both theoretical questions and practical challenges
5. Creates a comfortable, encouraging atmosphere

Keep it concise (2-3 paragraphs) and professional."""),
            HumanMessage(content="Generate a professional introduction for an Excel skills interview.")
        ])
        
        self.conclusion_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI Excel interviewer. Provide a professional conclusion that:
1. Thanks the candidate for their time and participation
2. Summarizes what was covered in the interview
3. Explains next steps in the process
4. Maintains a positive, encouraging tone

Keep it concise and professional."""),
            HumanMessage(content="Generate a professional conclusion for an Excel skills interview.")
        ])
    
    async def start_interview(self, candidate_name: str = "Candidate", experience_level: str = "intermediate") -> Dict[str, Any]:
        try:
            logger.info(f"Starting interview for {candidate_name} with experience level: {experience_level}")
            
            # Generate introduction
            introduction = await self._generate_introduction()
            
            # Initialize session
            self.session_data["candidate_name"] = candidate_name
            self.session_data["experience_level"] = experience_level
            self.session_data["interview_start"] = datetime.now()
            
            # Initialize conversation state
            self.conversation_state.start_interview()
            
            # Generate first question based on experience level
            first_question = await self._generate_first_question(experience_level)
            
            return {
                "introduction": introduction,
                "first_question": first_question,
                "session_id": self.session_data["session_id"],
                "state": self.conversation_state.current_state.value
            }
            
        except Exception as e:
            logger.error(f"Error starting interview: {str(e)}")
            raise Exception(f"Failed to start interview: {str(e)}")
    
    async def process_interaction(self, user_input: str, current_context: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            logger.info(f"Processing interaction: {user_input[:100]}...")
            
            # Get current state
            current_state = self.conversation_state.current_state
            
            # Process based on current state
            if current_state == InterviewState.INTRODUCTION:
                response = await self._handle_introduction(user_input)
            elif current_state == InterviewState.QUESTIONING:
                response = await self._handle_questioning(user_input, current_context)
            elif current_state == InterviewState.EVALUATION:
                response = await self._handle_evaluation(user_input, current_context)
            elif current_state == InterviewState.CHALLENGE:
                response = await self._handle_challenge(user_input, current_context)
            elif current_state == InterviewState.FEEDBACK:
                response = await self._handle_feedback(user_input)
            elif current_state == InterviewState.CONCLUSION:
                response = await self._handle_conclusion(user_input)
            else:
                response = await self._handle_unknown_state(user_input)
            
            # Log interaction
            log_interaction(
                interaction_type="user_input",
                user_input=user_input,
                ai_response=response.get("response", ""),
                session_id=self.session_data["session_id"],
                additional_data={
                    "state": current_state.value,
                    "next_state": self.conversation_state.current_state.value,
                    "question_type": response.get("question_type", "general")
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your response. Could you please rephrase?",
                "state": self.conversation_state.current_state.value,
                "error": str(e)
            }
    
    async def _generate_introduction(self) -> str:
        try:
            response = await self.llm.ainvoke(self.introduction_prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating introduction: {str(e)}")
            return "Welcome to your Excel skills interview! I'll be asking you questions about Excel and may give you some practical challenges to solve. Let's get started."
    
    async def _generate_first_question(self, experience_level: str) -> str:
        try:
            # Generate question based on experience level
            if experience_level == "beginner":
                return "Let's start with something basic. Can you tell me about your experience with Excel and what functions or features you're most comfortable using?"
            elif experience_level == "advanced":
                return "Given your advanced experience, let's dive deeper. Can you describe a complex Excel project you've worked on and the advanced techniques you used?"
            else:  # intermediate
                return "To get started, can you tell me about your Excel experience? What are some of the more complex functions or features you've used regularly?"
        except Exception as e:
            logger.error(f"Error generating first question: {str(e)}")
            return "Tell me about your experience with Excel."
    
    async def _handle_introduction(self, user_input: str) -> Dict[str, Any]:
        try:
            # Move to questioning state
            self.conversation_state.transition_to(InterviewState.QUESTIONING)
            
            # Analyze initial response for dynamic pathing
            pathing_decision = await self.dynamic_pathing.determine_next_action(
                user_input=user_input,
                current_context=self.session_data
            )
            
            # Generate next question based on pathing decision
            next_question = await self._generate_adaptive_question(pathing_decision)
            
            return {
                "response": f"Thank you for that introduction! {next_question}",
                "state": InterviewState.QUESTIONING.value,
                "question_type": pathing_decision.recommended_action,
                "next_topics": pathing_decision.recommended_topics
            }
            
        except Exception as e:
            logger.error(f"Error handling introduction: {str(e)}")
            self.conversation_state.transition_to(InterviewState.QUESTIONING)
            return {
                "response": "Thank you! Let's move on to some Excel questions.",
                "state": InterviewState.QUESTIONING.value
            }
    
    async def _handle_questioning(self, user_input: str, current_context: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            # Evaluate the answer
            evaluation = await self.answer_evaluator.evaluate_answer(
                question=current_context.get("current_question", "Tell me about Excel"),
                answer=user_input,
                question_type=current_context.get("question_type", "general")
            )
            
            # Store evaluation
            self.session_data["evaluations"].append({
                "question": current_context.get("current_question", ""),
                "answer": user_input,
                "evaluation": evaluation,
                "timestamp": datetime.now()
            })
            
            # Log evaluation
            log_evaluation(
                question=current_context.get("current_question", ""),
                answer=user_input,
                score=evaluation.total_score,
                evaluation_details={
                    "skill_level": evaluation.skill_level,
                    "dimensions": {dim.value: score.score for dim, score in evaluation.dimension_scores.items()}
                },
                session_id=self.session_data["session_id"]
            )
            
            # Use dynamic pathing to determine next action
            pathing_decision = await self.dynamic_pathing.determine_next_action(
                user_input=user_input,
                current_context=self.session_data,
                evaluation_score=evaluation.total_score
            )
            
            # Decide next step based on pathing and evaluation
            if evaluation.total_score < 40 and len(self.session_data["evaluations"]) < 3:
                # Low score, ask follow-up or easier question
                next_action = "follow_up_clarification"
            elif pathing_decision.confidence_score > 0.7:
                # High confidence pathing decision
                next_action = pathing_decision.recommended_action
            elif evaluation.total_score > 80 and len(self.session_data["evaluations"]) >= 3:
                # High score, offer challenge
                next_action = "challenge"
            else:
                # Continue with questioning
                next_action = "next_question"
            
            # Generate appropriate response
            if next_action == "challenge":
                return await self._transition_to_challenge(user_input, evaluation)
            elif next_action == "follow_up_clarification":
                return await self._generate_follow_up_question(user_input, evaluation)
            else:
                return await self._generate_next_question(user_input, evaluation, pathing_decision)
                
        except Exception as e:
            logger.error(f"Error handling questioning: {str(e)}")
            return {
                "response": "That's an interesting perspective. Let me ask you about another aspect of Excel.",
                "state": InterviewState.QUESTIONING.value
            }
    
    async def _handle_challenge(self, user_input: str, current_context: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if current_context and current_context.get("challenge_active"):
                # User is working on a challenge
                challenge_result = await self._evaluate_challenge_solution(
                    current_context["current_challenge"],
                    user_input
                )
                
                self.session_data["challenges"].append({
                    "challenge": current_context["current_challenge"],
                    "result": challenge_result,
                    "timestamp": datetime.now()
                })
                
                # Move back to questioning or to feedback
                if len(self.session_data["evaluations"]) >= 5:
                    self.conversation_state.transition_to(InterviewState.FEEDBACK)
                    return {
                        "response": f"Great work on that challenge! {challenge_result.feedback} Let me now provide you with overall feedback on your Excel skills.",
                        "state": InterviewState.FEEDBACK.value,
                        "challenge_result": challenge_result
                    }
                else:
                    self.conversation_state.transition_to(InterviewState.QUESTIONING)
                    return {
                        "response": f"Excellent work! {challenge_result.feedback} Let's continue with some more questions.",
                        "state": InterviewState.QUESTIONING.value,
                        "challenge_result": challenge_result
                    }
            else:
                # Generate a new challenge
                challenge = await self._generate_challenge()
                self.conversation_state.transition_to(InterviewState.CHALLENGE)
                
                return {
                    "response": f"I'd like to give you a practical challenge. {challenge.description}",
                    "state": InterviewState.CHALLENGE.value,
                    "challenge": challenge,
                    "challenge_data": self.excel_simulator.generate_dataset(challenge.dataset_config)
                }
                
        except Exception as e:
            logger.error(f"Error handling challenge: {str(e)}")
            self.conversation_state.transition_to(InterviewState.QUESTIONING)
            return {
                "response": "Let's continue with questions instead.",
                "state": InterviewState.QUESTIONING.value
            }
    
    async def _handle_evaluation(self, user_input: str, current_context: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            # This state is used for detailed evaluation discussions
            return await self._handle_questioning(user_input, current_context)
        except Exception as e:
            logger.error(f"Error handling evaluation: {str(e)}")
            return {
                "response": "Let's continue with the next part of our discussion.",
                "state": InterviewState.QUESTIONING.value
            }
    
    async def _handle_feedback(self, user_input: str) -> Dict[str, Any]:
        try:
            # Generate comprehensive feedback
            feedback = await self._generate_comprehensive_feedback()
            
            # Move to conclusion
            self.conversation_state.transition_to(InterviewState.CONCLUSION)
            
            return {
                "response": feedback,
                "state": InterviewState.FEEDBACK.value,
                "comprehensive_feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Error handling feedback: {str(e)}")
            self.conversation_state.transition_to(InterviewState.CONCLUSION)
            return {
                "response": "Thank you for participating in this interview. Let me conclude our session.",
                "state": InterviewState.CONCLUSION.value
            }
    
    async def _handle_conclusion(self, user_input: str) -> Dict[str, Any]:
        try:
            # Generate conclusion
            conclusion = await self._generate_conclusion()
            
            # Mark interview as complete
            self.conversation_state.transition_to(InterviewState.COMPLETED)
            self.session_data["interview_end"] = datetime.now()
            
            return {
                "response": conclusion,
                "state": InterviewState.COMPLETED.value,
                "interview_complete": True,
                "session_summary": self._generate_session_summary()
            }
            
        except Exception as e:
            logger.error(f"Error handling conclusion: {str(e)}")
            self.conversation_state.transition_to(InterviewState.COMPLETED)
            return {
                "response": "Thank you for your time. The interview is now complete.",
                "state": InterviewState.COMPLETED.value,
                "interview_complete": True
            }
    
    async def _handle_unknown_state(self, user_input: str) -> Dict[str, Any]:
        logger.warning(f"Unknown state encountered: {self.conversation_state.current_state}")
        self.conversation_state.transition_to(InterviewState.QUESTIONING)
        return {
            "response": "Let's continue with your Excel interview.",
            "state": InterviewState.QUESTIONING.value
        }
    
    async def _generate_adaptive_question(self, pathing_decision) -> str:
        try:
            # Generate question based on pathing decision
            if pathing_decision.recommended_action == "technical_deep_dive":
                return "That's interesting. Can you walk me through the technical details of how you would implement that solution in Excel?"
            elif pathing_decision.recommended_action == "practical_application":
                return "Great insight! How would you apply that knowledge in a real business scenario?"
            elif pathing_decision.recommended_action == "challenge":
                return "Excellent explanation! Let me give you a practical challenge to demonstrate that skill."
            else:
                topics = pathing_decision.recommended_topics
                if topics:
                    topic = topics[0]
                    return f"Let's explore {topic}. Can you tell me about your experience with that?"
                else:
                    return "Can you tell me more about your Excel experience with formulas and functions?"
                    
        except Exception as e:
            logger.error(f"Error generating adaptive question: {str(e)}")
            return "Tell me more about your Excel experience."
    
    async def _generate_follow_up_question(self, user_input: str, evaluation: ComprehensiveEvaluation) -> Dict[str, Any]:
        try:
            # Identify weak areas from evaluation
            weak_dimensions = [
                dim.value for dim, score in evaluation.dimension_scores.items() 
                if score.score < 15
            ]
            
            if weak_dimensions:
                weak_area = weak_dimensions[0]
                if weak_area == "technical_accuracy":
                    question = "I see. Let me ask you about something more specific. Can you explain how the VLOOKUP function works and when you would use it?"
                elif weak_area == "problem_solving":
                    question = "Let's try a different approach. If you had a dataset with missing values, how would you handle that in Excel?"
                else:
                    question = "Let me ask you something more specific. What's your experience with Excel pivot tables?"
            else:
                question = "Can you give me a specific example of an Excel project you've worked on?"
            
            return {
                "response": f"{evaluation.overall_feedback} {question}",
                "state": InterviewState.QUESTIONING.value,
                "evaluation": evaluation,
                "follow_up_type": "clarification"
            }
            
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            return {
                "response": f"{evaluation.overall_feedback} Let me ask you about something else.",
                "state": InterviewState.QUESTIONING.value,
                "evaluation": evaluation
            }
    
    async def _generate_next_question(self, user_input: str, evaluation: ComprehensiveEvaluation, pathing_decision) -> Dict[str, Any]:
        try:
            # Generate next question based on evaluation and pathing
            if evaluation.total_score > 80:
                # High performer, ask advanced question
                question = "Excellent! Let's go deeper. Have you ever used array formulas or advanced Excel functions like INDEX/MATCH combinations?"
            elif evaluation.total_score > 60:
                # Good performer, ask intermediate question
                question = "Good! Let's explore more. What's your experience with data analysis tools in Excel like pivot tables?"
            else:
                # Needs more basic questions
                question = "I see. Let's build on that. Can you tell me about some basic Excel functions you use regularly?"
            
            return {
                "response": f"{evaluation.overall_feedback} {question}",
                "state": InterviewState.QUESTIONING.value,
                "evaluation": evaluation,
                "next_topics": pathing_decision.recommended_topics
            }
            
        except Exception as e:
            logger.error(f"Error generating next question: {str(e)}")
            return {
                "response": f"{evaluation.overall_feedback} Tell me more about your Excel experience.",
                "state": InterviewState.QUESTIONING.value,
                "evaluation": evaluation
            }
    
    async def _transition_to_challenge(self, user_input: str, evaluation: ComprehensiveEvaluation) -> Dict[str, Any]:
        try:
            # Generate challenge based on evaluation
            if evaluation.total_score > 80:
                difficulty = DifficultyLevel.ADVANCED
            elif evaluation.total_score > 60:
                difficulty = DifficultyLevel.INTERMEDIATE
            else:
                difficulty = DifficultyLevel.BEGINNER
            
            # Select challenge type based on evaluation
            if any(dim == EvaluationDimension.TECHNICAL_ACCURACY and score.score > 15 
                   for dim, score in evaluation.dimension_scores.items()):
                challenge_type = ChallengeType.FORMULA_CREATION
            else:
                challenge_type = ChallengeType.DATA_ANALYSIS
            
            challenge = self.excel_simulator.generate_challenge(challenge_type, difficulty)
            
            self.conversation_state.transition_to(InterviewState.CHALLENGE)
            
            return {
                "response": f"{evaluation.overall_feedback} Let's put your skills to the test with a practical challenge. {challenge.description}",
                "state": InterviewState.CHALLENGE.value,
                "challenge": challenge,
                "challenge_data": self.excel_simulator.generate_dataset(challenge.dataset_config),
                "challenge_active": True
            }
            
        except Exception as e:
            logger.error(f"Error transitioning to challenge: {str(e)}")
            self.conversation_state.transition_to(InterviewState.QUESTIONING)
            return {
                "response": f"{evaluation.overall_feedback} Let's continue with more questions.",
                "state": InterviewState.QUESTIONING.value,
                "evaluation": evaluation
            }
    
    async def _generate_challenge(self) -> Any:
        try:
            # Generate challenge based on session progress
            if len(self.session_data["evaluations"]) >= 3:
                avg_score = sum(eval["evaluation"].total_score for eval in self.session_data["evaluations"]) / len(self.session_data["evaluations"])
                
                if avg_score > 75:
                    difficulty = DifficultyLevel.ADVANCED
                    challenge_type = random.choice([ChallengeType.FORMULA_CREATION, ChallengeType.AUTOMATION])
                elif avg_score > 50:
                    difficulty = DifficultyLevel.INTERMEDIATE
                    challenge_type = random.choice([ChallengeType.DATA_ANALYSIS, ChallengeType.DATA_VISUALIZATION])
                else:
                    difficulty = DifficultyLevel.BEGINNER
                    challenge_type = ChallengeType.DATA_ANALYSIS
            else:
                difficulty = DifficultyLevel.INTERMEDIATE
                challenge_type = ChallengeType.DATA_ANALYSIS
            
            return self.excel_simulator.generate_challenge(challenge_type, difficulty)
            
        except Exception as e:
            logger.error(f"Error generating challenge: {str(e)}")
            return self.excel_simulator.generate_challenge(ChallengeType.DATA_ANALYSIS, DifficultyLevel.INTERMEDIATE)
    
    async def _evaluate_challenge_solution(self, challenge: Any, user_solution: str) -> Any:
        try:
            # Parse user solution (this is simplified - in real implementation, would handle structured data)
            solution_data = {
                "solution_text": user_solution,
                "time_taken": 300  # Default 5 minutes
            }
            
            return self.excel_simulator.evaluate_challenge_solution(challenge, solution_data)
            
        except Exception as e:
            logger.error(f"Error evaluating challenge solution: {str(e)}")
            # Return a default result
            from ..simulation.excel_simulator import ChallengeResult
            return ChallengeResult(
                challenge_id=challenge.id,
                user_solution={"solution_text": user_solution},
                correctness_score=50.0,
                efficiency_score=50.0,
                approach_score=50.0,
                total_score=50.0,
                feedback="Solution evaluation encountered an issue, but good attempt!",
                completed_at=datetime.now(),
                time_taken=300
            )
    
    async def _generate_comprehensive_feedback(self) -> str:
        try:
            if not self.session_data["evaluations"]:
                return "Thank you for participating in the interview. I appreciate your time and effort."
            
            # Calculate averages
            avg_score = sum(eval["evaluation"].total_score for eval in self.session_data["evaluations"]) / len(self.session_data["evaluations"])
            
            # Calculate dimension averages
            dimension_averages = {}
            for eval in self.session_data["evaluations"]:
                for dim, score in eval["evaluation"].dimension_scores.items():
                    if dim.value not in dimension_averages:
                        dimension_averages[dim.value] = []
                    dimension_averages[dim.value].append(score.score)
            
            # Generate feedback based on performance
            if avg_score >= 90:
                performance_level = "Excellent"
                feedback = f"Outstanding performance! Your Excel skills are exceptional with an overall score of {avg_score:.1f}/100. "
                recommendations = ["Continue exploring advanced Excel features", "Consider mentoring others", "Explore Power BI for advanced analytics"]
            elif avg_score >= 75:
                performance_level = "Advanced"
                feedback = f"Very strong performance! You demonstrated solid Excel skills with a score of {avg_score:.1f}/100. "
                recommendations = ["Practice with more complex datasets", "Explore automation with macros", "Learn about Power Query"]
            elif avg_score >= 60:
                performance_level = "Intermediate"
                feedback = f"Good performance! You showed solid Excel understanding with a score of {avg_score:.1f}/100. "
                recommendations = ["Practice advanced functions like VLOOKUP", "Work with pivot tables", "Improve data visualization skills"]
            else:
                performance_level = "Developing"
                feedback = f"Keep working at it! You have basic Excel skills with a score of {avg_score:.1f}/100. "
                recommendations = ["Focus on basic Excel functions", "Practice with sample datasets", "Take online Excel courses"]
            
            # Add specific dimension feedback
            weak_areas = [dim for dim, scores in dimension_averages.items() if np.mean(scores) < 15]
            strong_areas = [dim for dim, scores in dimension_averages.items() if np.mean(scores) >= 20]
            
            if strong_areas:
                feedback += f"Your strongest areas include: {', '.join(strong_areas)}. "
            if weak_areas:
                feedback += f"Areas for improvement: {', '.join(weak_areas)}. "
            
            feedback += f"Recommendations: {', '.join(recommendations)}."
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating comprehensive feedback: {str(e)}")
            return "Thank you for participating in the interview. Your feedback will help you improve your Excel skills."
    
    async def _generate_conclusion(self) -> str:
        try:
            response = await self.llm.ainvoke(self.conclusion_prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            return "Thank you for your time and participation in this Excel skills interview. We appreciate your effort and wish you success in your Excel journey."
    
    def _generate_session_summary(self) -> Dict[str, Any]:
        try:
            duration = (self.session_data.get("interview_end", datetime.now()) - self.session_data["interview_start"]).total_seconds() / 60
            
            summary = {
                "session_id": self.session_data["session_id"],
                "candidate_name": self.session_data.get("candidate_name", "Unknown"),
                "duration_minutes": round(duration, 2),
                "total_questions": len(self.session_data["evaluations"]),
                "total_challenges": len(self.session_data["challenges"]),
                "average_score": 0.0,
                "skill_level": "Unknown",
                "strong_areas": [],
                "improvement_areas": []
            }
            
            if self.session_data["evaluations"]:
                avg_score = sum(eval["evaluation"].total_score for eval in self.session_data["evaluations"]) / len(self.session_data["evaluations"])
                summary["average_score"] = round(avg_score, 1)
                summary["skill_level"] = "Expert" if avg_score >= 90 else "Advanced" if avg_score >= 75 else "Intermediate" if avg_score >= 60 else "Basic"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return {"error": str(e)}
    
    def get_session_data(self) -> Dict[str, Any]:
        return self.session_data.copy()
    
    def reset_session(self):
        self.__init__()  # Reinitialize everything