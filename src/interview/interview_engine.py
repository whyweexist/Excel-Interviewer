import streamlit as st
import openai
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import random
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from .conversation_state import (
    ConversationStateMachine, InterviewState, Question, Answer, 
    EvaluationResult, QuestionType, DifficultyLevel, InterviewContext
)
from ..evaluation.answer_evaluator import AnswerEvaluator
from ..simulation.excel_simulator import ExcelSimulator
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class InterviewEngine:
    """Core engine that manages the entire interview process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state_machine = ConversationStateMachine()
        self.answer_evaluator = AnswerEvaluator(config)
        self.excel_simulator = ExcelSimulator()
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.openai_model,
            temperature=config.openai_temperature,
            max_tokens=config.openai_max_tokens
        )
        self.memory = ConversationBufferMemory(return_messages=True)
        self.question_bank = self._initialize_question_bank()
        self.current_question = None
        self.interview_start_time = None
        
    def _initialize_question_bank(self) -> Dict[str, List[Question]]:
        """Initialize the question bank with Excel-related questions."""
        return {
            "basic_formulas": [
                Question(
                    id="basic_1",
                    text="Can you explain the difference between relative and absolute cell references in Excel? Provide examples of when you would use each.",
                    question_type=QuestionType.TECHNICAL,
                    difficulty=DifficultyLevel.BEGINNER,
                    topic="basic_formulas",
                    expected_keywords=["relative", "absolute", "$", "reference", "copy", "formula"],
                    scoring_criteria={
                        "accuracy": 0.4,
                        "examples": 0.3,
                        "clarity": 0.3
                    },
                    max_time_seconds=180
                ),
                Question(
                    id="basic_2",
                    text="Walk me through how you would use VLOOKUP to find data in a large dataset. What are the limitations of VLOOKUP?",
                    question_type=QuestionType.TECHNICAL,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    topic="basic_formulas",
                    expected_keywords=["VLOOKUP", "lookup_value", "table_array", "col_index_num", "range_lookup", "leftmost", "exact_match"],
                    scoring_criteria={
                        "function_understanding": 0.4,
                        "limitations": 0.3,
                        "practical_application": 0.3
                    },
                    max_time_seconds=240
                )
            ],
            "advanced_functions": [
                Question(
                    id="advanced_1",
                    text="Explain how you would use INDEX and MATCH together as an alternative to VLOOKUP. What are the advantages of this approach?",
                    question_type=QuestionType.TECHNICAL,
                    difficulty=DifficultyLevel.ADVANCED,
                    topic="advanced_functions",
                    expected_keywords=["INDEX", "MATCH", "flexibility", "left", "right", "performance", "dynamic"],
                    scoring_criteria={
                        "function_combination": 0.4,
                        "advantages": 0.3,
                        "use_cases": 0.3
                    },
                    max_time_seconds=300
                ),
                Question(
                    id="advanced_2",
                    text="Describe a complex nested IF formula you've created. How do you handle multiple conditions efficiently?",
                    question_type=QuestionType.PRACTICAL,
                    difficulty=DifficultyLevel.ADVANCED,
                    topic="advanced_functions",
                    expected_keywords=["nested IF", "IFS", "SWITCH", "multiple conditions", "readability", "maintenance"],
                    scoring_criteria={
                        "complexity_handling": 0.4,
                        "best_practices": 0.3,
                        "efficiency": 0.3
                    },
                    max_time_seconds=300
                )
            ],
            "data_analysis": [
                Question(
                    id="analysis_1",
                    text="How would you approach cleaning and preparing a dataset with inconsistent formatting, missing values, and duplicates?",
                    question_type=QuestionType.SCENARIO_BASED,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    topic="data_analysis",
                    expected_keywords=["cleaning", "missing values", "duplicates", "formatting", "consistency", "validation"],
                    scoring_criteria={
                        "systematic_approach": 0.4,
                        "tools_techniques": 0.3,
                        "quality_focus": 0.3
                    },
                    max_time_seconds=300
                ),
                Question(
                    id="analysis_2",
                    text="Explain the difference between PivotTables and Power Query. When would you choose one over the other?",
                    question_type=QuestionType.TECHNICAL,
                    difficulty=DifficultyLevel.ADVANCED,
                    topic="data_analysis",
                    expected_keywords=["PivotTable", "Power Query", "summarization", "transformation", "automation", "data_source"],
                    scoring_criteria={
                        "tool_understanding": 0.4,
                        "use_case_selection": 0.3,
                        "integration": 0.3
                    },
                    max_time_seconds=240
                )
            ],
            "automation": [
                Question(
                    id="automation_1",
                    text="Describe your experience with Excel macros. What are the key considerations when deciding between VBA and modern alternatives?",
                    question_type=QuestionType.PRACTICAL,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    topic="automation",
                    expected_keywords=["macro", "VBA", "automation", "security", "maintenance", "Office Scripts", "Power Automate"],
                    scoring_criteria={
                        "experience_level": 0.3,
                        "decision_criteria": 0.4,
                        "modern_alternatives": 0.3
                    },
                    max_time_seconds=300
                )
            ]
        }
    
    def start_interview(self, candidate_name: str) -> str:
        """Start a new interview session."""
        interview_id = f"INT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state_machine.initialize_interview(candidate_name, interview_id)
        self.interview_start_time = datetime.now()
        self.memory.clear()
        
        logger.info(f"Started interview {interview_id} for candidate {candidate_name}")
        return interview_id
    
    def get_next_question(self) -> Optional[Question]:
        """Get the next question based on current state and context."""
        context = self.state_machine.get_context()
        if not context:
            return None
        
        current_state = self.state_machine.get_current_state()
        
        if current_state == InterviewState.INTRODUCTION:
            return self._generate_introduction_question()
        elif current_state == InterviewState.TOPIC_ASSESSMENT:
            return self._select_topic_question(context)
        elif current_state == InterviewState.PRACTICAL_CHALLENGE:
            return self._generate_practical_challenge(context)
        elif current_state == InterviewState.FOLLOW_UP_QUESTIONS:
            return self._generate_follow_up_question(context)
        else:
            return None
    
    def _generate_introduction_question(self) -> Question:
        """Generate the introduction question."""
        context = self.state_machine.get_context()
        return Question(
            id="intro_1",
            text=f"Hello {context.candidate_name}! I'm your AI Excel interviewer. I'll be assessing your Excel skills through a series of questions and practical challenges. To start, could you briefly tell me about your experience with Excel - how long have you been using it and what types of tasks do you typically perform?",
            question_type=QuestionType.TECHNICAL,
            difficulty=DifficultyLevel.BEGINNER,
            topic="introduction",
            max_time_seconds=180
        )
    
    def _select_topic_question(self, context: InterviewContext) -> Question:
        """Select the next topic-based question."""
        # Determine which topics haven't been covered yet
        covered_topics = set(result.question_id.split("_")[0] for result in context.evaluation_results)
        available_topics = [topic for topic in self.question_bank.keys() if topic not in covered_topics]
        
        if not available_topics:
            # All topics covered, move to summary
            self.state_machine.transition_to(InterviewState.SUMMARY, "All topics assessed")
            return None
        
        # Select topic based on difficulty progression and performance
        next_topic = self._select_next_topic(context, available_topics)
        questions = self.question_bank[next_topic]
        
        # Select question based on current difficulty level
        current_difficulty = self._determine_current_difficulty(context)
        suitable_questions = [q for q in questions if q.difficulty == current_difficulty]
        
        if not suitable_questions:
            suitable_questions = questions
        
        return random.choice(suitable_questions)
    
    def _select_next_topic(self, context: InterviewContext, available_topics: List[str]) -> str:
        """Intelligently select the next topic based on performance and progression."""
        # Simple logic: prioritize basic topics first, then move to advanced
        topic_priority = {
            "basic_formulas": 1,
            "data_analysis": 2,
            "advanced_functions": 3,
            "automation": 4
        }
        
        available_topics.sort(key=lambda x: topic_priority.get(x, 5))
        return available_topics[0]
    
    def _determine_current_difficulty(self, context: InterviewContext) -> DifficultyLevel:
        """Determine the appropriate difficulty level for the next question."""
        if not context.evaluation_results:
            return DifficultyLevel.BEGINNER
        
        # Simple adaptive difficulty: if last score was > 80, increase difficulty
        last_score = context.evaluation_results[-1].overall_score
        current_difficulty = context.difficulty_progression[-1] if context.difficulty_progression else DifficultyLevel.BEGINNER
        
        if last_score >= 80 and current_difficulty != DifficultyLevel.EXPERT:
            difficulty_order = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
            current_index = difficulty_order.index(current_difficulty)
            return difficulty_order[min(current_index + 1, len(difficulty_order) - 1)]
        elif last_score < 60 and current_difficulty != DifficultyLevel.BEGINNER:
            difficulty_order = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
            current_index = difficulty_order.index(current_difficulty)
            return difficulty_order[max(current_index - 1, 0)]
        
        return current_difficulty
    
    def _generate_practical_challenge(self, context: InterviewContext) -> Question:
        """Generate a practical Excel challenge."""
        # This will be implemented with the Excel simulator
        challenge_question = self.excel_simulator.generate_challenge(context)
        return challenge_question
    
    def _generate_follow_up_question(self, context: InterviewContext) -> Question:
        """Generate a follow-up question based on the last answer."""
        if not context.answers_given:
            return None
        
        last_answer = context.answers_given[-1]
        last_question = next((q for q in context.questions_asked if q.id == last_answer.question_id), None)
        
        if last_question and last_question.follow_up_questions:
            follow_up_text = random.choice(last_question.follow_up_questions)
            return Question(
                id=f"followup_{last_question.id}",
                text=follow_up_text,
                question_type=QuestionType.FOLLOW_UP,
                difficulty=last_question.difficulty,
                topic=last_question.topic,
                max_time_seconds=120
            )
        
        return None
    
    def process_answer(self, answer_text: str) -> EvaluationResult:
        """Process the candidate's answer and return evaluation."""
        if not self.current_question:
            return None
        
        # Create Answer object
        answer = Answer(
            question_id=self.current_question.id,
            text=answer_text,
            timestamp=datetime.now(),
            time_taken_seconds=180  # This should be calculated from actual time
        )
        
        # Record conversation turn
        self.state_machine.record_conversation_turn("candidate", answer_text)
        
        # Evaluate the answer
        evaluation = self.answer_evaluator.evaluate_answer(self.current_question, answer)
        
        # Update context
        context = self.state_machine.get_context()
        if context:
            context.answers_given.append(answer)
            context.evaluation_results.append(evaluation)
            context.questions_asked.append(self.current_question)
            
            # Update skill assessments
            topic = self.current_question.topic
            if topic not in context.skill_assessments:
                context.skill_assessments[topic] = []
            context.skill_assessments[topic].append(evaluation.overall_score)
            
            # Update difficulty progression
            context.difficulty_progression.append(self.current_question.difficulty)
        
        # Determine next state based on evaluation and interview progress
        self._determine_next_state(evaluation)
        
        return evaluation
    
    def _determine_next_state(self, evaluation: EvaluationResult):
        """Determine the next state based on evaluation and context."""
        context = self.state_machine.get_context()
        if not context:
            return
        
        # Simple state progression logic
        current_state = self.state_machine.get_current_state()
        questions_asked = len(context.questions_asked)
        
        if current_state == InterviewState.INTRODUCTION:
            self.state_machine.transition_to(InterviewState.TOPIC_ASSESSMENT, "Introduction complete")
        elif current_state == InterviewState.TOPIC_ASSESSMENT:
            # Alternate between topic assessment and practical challenges
            if questions_asked % 3 == 0:
                self.state_machine.transition_to(InterviewState.PRACTICAL_CHALLENGE, "Time for practical challenge")
            else:
                # Check if we should ask follow-up questions
                if evaluation.overall_score < 70:
                    self.state_machine.transition_to(InterviewState.FOLLOW_UP_QUESTIONS, "Low score, need follow-up")
                else:
                    self.state_machine.transition_to(InterviewState.TOPIC_ASSESSMENT, "Continue with next topic")
        elif current_state == InterviewState.PRACTICAL_CHALLENGE:
            self.state_machine.transition_to(InterviewState.FOLLOW_UP_QUESTIONS, "Challenge complete")
        elif current_state == InterviewState.FOLLOW_UP_QUESTIONS:
            self.state_machine.transition_to(InterviewState.TOPIC_ASSESSMENT, "Follow-up complete")
        
        # Check if interview should end (time limit or all topics covered)
        if self._should_end_interview():
            self.state_machine.transition_to(InterviewState.SUMMARY, "Interview time limit reached")
    
    def _should_end_interview(self) -> bool:
        """Determine if the interview should end based on time and coverage."""
        if not self.interview_start_time:
            return False
        
        # Check time limit
        duration = (datetime.now() - self.interview_start_time).total_seconds() / 60
        if duration >= self.config.max_interview_duration_minutes:
            return True
        
        # Check if all major topics have been covered
        context = self.state_machine.get_context()
        if context and len(context.questions_asked) >= 8:  # Maximum questions
            return True
        
        return False
    
    def get_interview_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the interview."""
        context = self.state_machine.get_context()
        if not context:
            return {}
        
        # Calculate overall statistics
        total_questions = len(context.questions_asked)
        total_answers = len(context.answers_given)
        avg_score = sum(result.overall_score for result in context.evaluation_results) / len(context.evaluation_results) if context.evaluation_results else 0
        
        # Calculate skill-specific scores
        skill_scores = {}
        for topic, scores in context.skill_assessments.items():
            skill_scores[topic] = sum(scores) / len(scores) if scores else 0
        
        return {
            "interview_id": context.interview_id,
            "candidate_name": context.candidate_name,
            "total_duration_minutes": self.state_machine._calculate_duration_minutes(),
            "total_questions": total_questions,
            "total_answers": total_answers,
            "overall_score": avg_score,
            "skill_scores": skill_scores,
            "state_history": [state.value for state in self.state_machine.get_state_history()],
            "conversation_summary": self.state_machine.get_conversation_summary()
        }
    
    def render_interview_interface(self):
        """Render the Streamlit interface for the interview."""
        if not st.session_state.get('interview_started'):
            self._render_interview_setup()
        else:
            self._render_active_interview()
    
    def _render_interview_setup(self):
        """Render the interview setup interface."""
        st.subheader("Interview Setup")
        
        with st.form("interview_setup"):
            candidate_name = st.text_input("Your Name:", placeholder="Enter your full name")
            experience_level = st.selectbox(
                "Experience Level:",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                help="This helps us tailor the interview difficulty"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("🚀 Start Interview", type="primary"):
                    if candidate_name:
                        interview_id = self.start_interview(candidate_name)
                        st.session_state.interview_started = True
                        st.session_state.candidate_name = candidate_name
                        st.session_state.interview_id = interview_id
                        st.session_state.experience_level = experience_level
                        st.rerun()
                    else:
                        st.error("Please enter your name to continue.")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.current_view = "home"
                    st.rerun()
    
    def _render_active_interview(self):
        """Render the active interview interface."""
        st.info(f"Interview in progress for: {st.session_state.candidate_name}")
        
        # Get current question
        if not self.current_question:
            self.current_question = self.get_next_question()
        
        if self.current_question:
            # Display question
            st.subheader(f"Question {len(self.state_machine.get_context().questions_asked) + 1}")
            st.markdown(f"**Topic:** {self.current_question.topic.replace('_', ' ').title()}")
            st.markdown(f"**Difficulty:** {self.current_question.difficulty.value.title()}")
            st.markdown("---")
            st.markdown(f"### {self.current_question.text}")
            
            # Answer input
            answer_text = st.text_area(
                "Your Answer:",
                height=150,
                placeholder="Please provide your detailed answer here...",
                help="Be as detailed and specific as possible in your response."
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer", type="primary"):
                    if answer_text:
                        with st.spinner("Evaluating your answer..."):
                            evaluation = self.process_answer(answer_text)
                            self._display_evaluation(evaluation)
                            self.current_question = None  # Reset for next question
                    else:
                        st.warning("Please provide an answer before submitting.")
            
            with col2:
                if st.button("Skip Question"):
                    self.current_question = None
                    st.rerun()
            
            with col3:
                if st.button("End Interview"):
                    self.state_machine.transition_to(InterviewState.SUMMARY, "User requested end")
                    self._render_interview_summary()
        else:
            # No more questions or interview complete
            if self.state_machine.is_interview_complete():
                self._render_interview_summary()
            else:
                st.info("Loading next question...")
                st.rerun()
    
    def _display_evaluation(self, evaluation: EvaluationResult):
        """Display the evaluation results."""
        st.success("Answer submitted successfully!")
        
        with st.expander("📊 Evaluation Results", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Technical Score", f"{evaluation.technical_score:.1f}%")
            with col2:
                st.metric("Problem Solving", f"{evaluation.problem_solving_score:.1f}%")
            with col3:
                st.metric("Communication", f"{evaluation.communication_score:.1f}%")
            with col4:
                st.metric("Overall Score", f"{evaluation.overall_score:.1f}%")
            
            st.markdown("### Detailed Feedback")
            st.markdown(evaluation.detailed_feedback)
            
            if evaluation.strengths:
                st.markdown("**Strengths:**")
                for strength in evaluation.strengths:
                    st.markdown(f"✅ {strength}")
            
            if evaluation.weaknesses:
                st.markdown("**Areas for Improvement:**")
                for weakness in evaluation.weaknesses:
                    st.markdown(f"⚠️ {weakness}")
            
            if evaluation.suggestions:
                st.markdown("**Suggestions:**")
                for suggestion in evaluation.suggestions:
                    st.markdown(f"💡 {suggestion}")
    
    def _render_interview_summary(self):
        """Render the interview summary."""
        st.header("🎯 Interview Summary")
        
        summary = self.get_interview_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", summary["total_questions"])
        with col2:
            st.metric("Overall Score", f"{summary['overall_score']:.1f}%")
        with col3:
            st.metric("Duration", f"{summary['total_duration_minutes']:.1f} minutes")
        
        st.markdown("### Skill Assessment Breakdown")
        skill_data = summary["skill_scores"]
        for skill, score in skill_data.items():
            skill_name = skill.replace("_", " ").title()
            st.progress(score / 100)
            st.text(f"{skill_name}: {score:.1f}%")
        
        st.markdown("### Recommendations")
        st.info("Your detailed feedback report is available in the Dashboard section.")
        
        if st.button("View Detailed Report"):
            st.session_state.current_view = "dashboard"
            st.rerun()
        
        if st.button("Start New Interview"):
            st.session_state.interview_started = False
            st.session_state.interview_engine = None
            st.rerun()