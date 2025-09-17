import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any

# Import our modules
from src.interview.integrated_interview_engine import IntegratedInterviewEngine
from src.utils.config import settings
from src.utils.logger import get_logger, log_interaction

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Excel Interviewer Prototype",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'interview_engine' not in st.session_state:
    st.session_state.interview_engine = IntegratedInterviewEngine()
    st.session_state.interview_started = False
    st.session_state.interview_completed = False
    st.session_state.messages = []
    st.session_state.current_context = {}
    st.session_state.session_summary = None

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .evaluation-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #4caf50;
    }
    .challenge-box {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 AI Excel Interviewer Prototype</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Interview Controls")
    
    # Candidate Information
    st.subheader("Candidate Information")
    candidate_name = st.text_input("Name", value="Candidate")
    experience_level = st.selectbox(
        "Experience Level",
        ["beginner", "intermediate", "advanced"],
        index=1
    )
    
    # Interview Controls
    st.subheader("Interview Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Interview", type="primary"):
            asyncio.run(start_interview(candidate_name, experience_level))
    
    with col2:
        if st.button("🔄 Reset Session"):
            reset_session()
    
    # Session Statistics
    if st.session_state.interview_started:
        st.subheader("📊 Session Stats")
        stats = get_session_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", stats.get("questions_asked", 0))
        with col2:
            st.metric("Avg Score", f"{stats.get('avg_score', 0):.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Challenges", stats.get("challenges_completed", 0))
        with col2:
            st.metric("Duration", stats.get("duration", "0m"))

# Main content area
if not st.session_state.interview_started:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🎯 Welcome to the AI Excel Interviewer Prototype
        
        This prototype demonstrates the core functionality of our AI-powered Excel interview system:
        
        **✨ Key Features:**
        - 🤖 AI-driven conversation flow
        - 📊 Multi-dimensional answer evaluation
        - 🎮 Dynamic challenge generation
        - 📈 Real-time performance tracking
        - 🧠 Adaptive questioning based on responses
        
        **🚀 How to Start:**
        1. Enter your name and select experience level
        2. Click "Start Interview" in the sidebar
        3. Engage in the conversation naturally
        4. Complete any challenges that appear
        5. Receive comprehensive feedback
        
        **💡 Tips:**
        - Be specific in your answers
        - Share real examples when possible
        - Ask for clarification if needed
        - Take your time with challenges
        """)
        
        st.info("📝 This is a prototype demonstration. Some features are simplified for demo purposes.")

else:
    # Interview interface
    st.markdown("### 💬 Interview Conversation")
    
    # Chat display
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # Input area
    if not st.session_state.interview_completed:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Your Response:",
                placeholder="Type your response here...",
                key="user_input",
                height=100
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")
            if st.button("📤 Submit Response", type="primary", use_container_width=True):
                if user_input.strip():
                    asyncio.run(process_user_response(user_input.strip()))
                    st.rerun()
            
            if st.button("⏭️ Skip Question", use_container_width=True):
                asyncio.run(skip_question())
                st.rerun()

# Results section
if st.session_state.interview_completed and st.session_state.session_summary:
    st.markdown("---")
    st.markdown("### 📊 Interview Results")
    
    display_results()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🛠️ <strong>AI Excel Interviewer Prototype</strong> | Built with Streamlit & LangChain</p>
    <p>📝 This is a demonstration of the core interview system architecture</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
async def start_interview(candidate_name: str, experience_level: str):
    """Start the interview session"""
    try:
        with st.spinner("🤖 Initializing interview..."):
            result = await st.session_state.interview_engine.start_interview(candidate_name, experience_level)
            
            # Add introduction to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["introduction"],
                "type": "introduction"
            })
            
            # Add first question
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["first_question"],
                "type": "question"
            })
            
            st.session_state.interview_started = True
            st.session_state.current_context = {
                "current_question": result["first_question"],
                "question_type": "initial"
            }
            
            logger.info(f"Interview started for {candidate_name} ({experience_level})")
            
    except Exception as e:
        st.error(f"❌ Error starting interview: {str(e)}")
        logger.error(f"Failed to start interview: {str(e)}")

async def process_user_response(user_input: str):
    """Process user response and generate next interaction"""
    try:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Process interaction
        with st.spinner("🤖 Processing your response..."):
            result = await st.session_state.interview_engine.process_interaction(
                user_input, 
                st.session_state.current_context
            )
            
            # Add AI response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "type": result.get("type", "response"),
                "evaluation": result.get("evaluation"),
                "challenge": result.get("challenge"),
                "timestamp": datetime.now()
            })
            
            # Update context for next interaction
            if result.get("challenge"):
                st.session_state.current_context = {
                    "challenge_active": True,
                    "current_challenge": result["challenge"],
                    "challenge_data": result.get("challenge_data")
                }
            elif result.get("next_topics"):
                st.session_state.current_context = {
                    "current_question": result["response"],
                    "question_type": result.get("question_type", "adaptive"),
                    "next_topics": result["next_topics"]
                }
            else:
                st.session_state.current_context = {
                    "current_question": result["response"],
                    "question_type": result.get("question_type", "adaptive")
                }
            
            # Check if interview is complete
            if result.get("interview_complete"):
                st.session_state.interview_completed = True
                st.session_state.session_summary = result.get("session_summary", {})
                
            logger.info(f"Processed interaction, state: {result.get('state', 'unknown')}")
            
    except Exception as e:
        st.error(f"❌ Error processing response: {str(e)}")
        logger.error(f"Failed to process user response: {str(e)}")

async def skip_question():
    """Skip current question and move to next"""
    try:
        # Add skip notice
        st.session_state.messages.append({
            "role": "user",
            "content": "[Question skipped by user]",
            "timestamp": datetime.now(),
            "type": "skip"
        })
        
        # Generate next question
        result = await st.session_state.interview_engine.process_interaction(
            "skip question",
            {"skip_question": True}
        )
        
        # Add AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "type": "question",
            "timestamp": datetime.now()
        })
        
        # Update context
        st.session_state.current_context = {
            "current_question": result["response"],
            "question_type": "adaptive"
        }
        
    except Exception as e:
        st.error(f"❌ Error skipping question: {str(e)}")

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # AI message with optional evaluation
            content = message["content"]
            
            if message.get("evaluation"):
                eval_obj = message["evaluation"]
                st.markdown(f'<div class="ai-message">🤖 **AI:** {content}</div>', unsafe_allow_html=True)
                
                # Display evaluation score
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{eval_obj.total_score:.1f}%")
                with col2:
                    st.metric("Skill Level", eval_obj.skill_level.title())
                with col3:
                    st.metric("Confidence", f"{eval_obj.confidence_score:.1f}%")
                
                # Display dimension scores
                with st.expander("📊 Detailed Evaluation"):
                    for dim, score in eval_obj.dimension_scores.items():
                        st.write(f"**{dim.value.replace('_', ' ').title()}:** {score.score}/25")
                        st.write(f"*Feedback:* {score.feedback}")
                        st.write("")
                    
                    st.write(f"**Overall Feedback:** {eval_obj.overall_feedback}")
                    if eval_obj.recommendations:
                        st.write(f"**Recommendations:** {', '.join(eval_obj.recommendations)}")
            
            elif message.get("challenge"):
                st.markdown(f'<div class="ai-message">🤖 **AI:** {content}</div>', unsafe_allow_html=True)
                
                # Display challenge
                challenge = message["challenge"]
                with st.container():
                    st.markdown('<div class="challenge-box">', unsafe_allow_html=True)
                    st.write(f"**Challenge Type:** {challenge.challenge_type.value.replace('_', ' ').title()}")
                    st.write(f"**Difficulty:** {challenge.difficulty.value.title()}")
                    st.write(f"**Dataset:** {challenge.dataset_config.description}")
                    st.write(f"**Expected Time:** {challenge.expected_time_minutes} minutes")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown(f'<div class="ai-message">🤖 **AI:** {content}</div>', unsafe_allow_html=True)

def get_session_stats() -> Dict[str, Any]:
    """Get current session statistics"""
    try:
        session_data = st.session_state.interview_engine.get_session_data()
        
        stats = {
            "questions_asked": len(session_data.get("evaluations", [])),
            "challenges_completed": len(session_data.get("challenges", [])),
            "avg_score": 0.0,
            "duration": "0m"
        }
        
        if session_data.get("evaluations"):
            total_score = sum(eval["evaluation"].total_score for eval in session_data["evaluations"])
            stats["avg_score"] = total_score / len(session_data["evaluations"])
        
        if session_data.get("interview_start") and session_data.get("interview_end"):
            duration = (session_data["interview_end"] - session_data["interview_start"]).total_seconds() / 60
            stats["duration"] = f"{duration:.0f}m"
        elif session_data.get("interview_start"):
            duration = (datetime.now() - session_data["interview_start"]).total_seconds() / 60
            stats["duration"] = f"{duration:.0f}m"
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}")
        return {"questions_asked": 0, "challenges_completed": 0, "avg_score": 0.0, "duration": "0m"}

def display_results():
    """Display comprehensive interview results"""
    try:
        summary = st.session_state.session_summary
        session_data = st.session_state.interview_engine.get_session_data()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Performance Summary")
            
            # Overall metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Overall Score", f"{summary.get('average_score', 0):.1f}%")
            with col_b:
                st.metric("Skill Level", summary.get('skill_level', 'Unknown'))
            with col_c:
                st.metric("Duration", f"{summary.get('duration_minutes', 0):.0f} minutes")
            
            # Detailed evaluations
            if session_data.get("evaluations"):
                st.subheader("📊 Question Performance")
                
                # Create evaluation dataframe
                eval_data = []
                for i, eval_item in enumerate(session_data["evaluations"]):
                    eval_obj = eval_item["evaluation"]
                    eval_data.append({
                        "Question #": i + 1,
                        "Score": eval_obj.total_score,
                        "Skill Level": eval_obj.skill_level,
                        "Technical": eval_obj.dimension_scores.get("technical_accuracy", {}).get("score", 0),
                        "Problem Solving": eval_obj.dimension_scores.get("problem_solving", {}).get("score", 0),
                        "Communication": eval_obj.dimension_scores.get("communication_clarity", {}).get("score", 0),
                        "Practical": eval_obj.dimension_scores.get("practical_application", {}).get("score", 0)
                    })
                
                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True)
                
                # Performance chart
                if len(eval_data) > 1:
                    st.line_chart(eval_df.set_index("Question #")["Score"])
            
            # Challenges
            if session_data.get("challenges"):
                st.subheader("🎮 Challenge Results")
                
                for i, challenge_item in enumerate(session_data["challenges"]):
                    result = challenge_item["result"]
                    with st.expander(f"Challenge {i+1}: {challenge_item['challenge'].challenge_type.value}"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Correctness", f"{result.correctness_score:.1f}%")
                        with col_b:
                            st.metric("Efficiency", f"{result.efficiency_score:.1f}%")
                        with col_c:
                            st.metric("Approach", f"{result.approach_score:.1f}%")
                        
                        st.write(f"**Feedback:** {result.feedback}")
        
        with col2:
            st.subheader("🎯 Key Insights")
            
            # Strengths and weaknesses
            if session_data.get("evaluations"):
                # Calculate dimension averages
                dimensions = {}
                for eval_item in session_data["evaluations"]:
                    eval_obj = eval_item["evaluation"]
                    for dim, score in eval_obj.dimension_scores.items():
                        if dim.value not in dimensions:
                            dimensions[dim.value] = []
                        dimensions[dim.value].append(score.score)
                
                # Display strengths
                strengths = [dim.replace("_", " ").title() for dim, scores in dimensions.items() 
                           if sum(scores)/len(scores) >= 20]
                if strengths:
                    st.write("**✅ Strengths:**")
                    for strength in strengths:
                        st.write(f"• {strength}")
                
                # Display improvement areas
                improvements = [dim.replace("_", " ").title() for dim, scores in dimensions.items() 
                              if sum(scores)/len(scores) < 15]
                if improvements:
                    st.write("**📚 Areas for Improvement:**")
                    for improvement in improvements:
                        st.write(f"• {improvement}")
            
            # Recommendations
            st.write("**💡 Recommendations:**")
            if summary.get('average_score', 0) >= 90:
                recommendations = [
                    "Explore advanced Excel features",
                    "Consider Excel certification",
                    "Mentor junior colleagues",
                    "Explore Power BI integration"
                ]
            elif summary.get('average_score', 0) >= 75:
                recommendations = [
                    "Practice complex formulas",
                    "Learn advanced functions",
                    "Work with larger datasets",
                    "Improve data visualization"
                ]
            elif summary.get('average_score', 0) >= 60:
                recommendations = [
                    "Master basic functions",
                    "Practice pivot tables",
                    "Learn VLOOKUP/INDEX-MATCH",
                    "Improve chart creation"
                ]
            else:
                recommendations = [
                    "Start with Excel basics",
                    "Practice with sample data",
                    "Take online courses",
                    "Use Excel daily for practice"
                ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
            
            # Export options
            st.write("**📤 Export Options:**")
            if st.button("📄 Generate Report"):
                st.success("Report generation would be implemented here!")
            
            if st.button("📊 Export Data"):
                st.success("Data export would be implemented here!")
    
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")
        logger.error(f"Failed to display results: {str(e)}")

def reset_session():
    """Reset the interview session"""
    st.session_state.interview_engine = IntegratedInterviewEngine()
    st.session_state.interview_started = False
    st.session_state.interview_completed = False
    st.session_state.messages = []
    st.session_state.current_context = {}
    st.session_state.session_summary = None
    logger.info("Session reset")

# Add numpy import for the integrated engine
import numpy as np