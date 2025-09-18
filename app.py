import streamlit as st
import os
from dotenv import load_dotenv
from src.interview.interview_engine import InterviewEngine
from src.dashboard import dashboard_ui  # ✅ updated import
from src.utils.config import Config

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Excel Interviewer AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point."""
    st.title("🎯 AI-Powered Excel Interviewer")
    st.markdown("### Professional Excel Skills Assessment Platform")
    
    # Initialize session state
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_engine' not in st.session_state:
        st.session_state.interview_engine = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "home"
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = "home"
            st.rerun()
        
        if st.button("🎤 Start Interview", use_container_width=True):
            st.session_state.current_view = "interview"
            st.rerun()
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_view = "dashboard"
            st.rerun()
        
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.current_view = "analytics"
            st.rerun()
        
        st.divider()
        st.info("This AI-powered system conducts comprehensive Excel interviews with real-time evaluation and detailed feedback.")
    
    # Main content area
    if st.session_state.current_view == "home":
        show_home_page()
    elif st.session_state.current_view == "interview":
        show_interview_page()
    elif st.session_state.current_view == "dashboard":
        show_dashboard_page()
    elif st.session_state.current_view == "analytics":
        show_analytics_page()

def show_home_page():
    """Display the home page with system overview."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to AI-Powered Excel Interviewer")
        st.markdown("""
        ### 🎯 What We Offer
        
        **Comprehensive Assessment**: Our AI system evaluates candidates across multiple dimensions:
        - Technical Excel proficiency
        - Problem-solving methodology  
        - Communication clarity
        - Practical application knowledge
        
        **Advanced Features**:
        - 🧠 Adaptive difficulty adjustment
        - 📊 Real-time performance evaluation
        - 🎮 Interactive Excel simulations
        - 📈 Detailed skill competency heatmaps
        - 🏆 Gamification and benchmarking
        
        **Professional Experience**:
        - Structured multi-turn conversations
        - Context-aware follow-up questions
        - Professional tone maintenance
        - Comprehensive feedback reports
        """)
        
        if st.button("🚀 Start Your Interview", type="primary", use_container_width=True):
            st.session_state.current_view = "interview"
            st.rerun()
    
    with col2:
        st.header("Quick Stats")
        st.metric("Total Interviews Conducted", "1,247")
        st.metric("Average Interview Duration", "32 minutes")
        st.metric("Candidate Satisfaction Rate", "94%")
        st.metric("Skills Assessed", "15+")
        
        st.divider()
        st.header("🎯 Interview Process")
        st.markdown("""
        1. **Introduction** (2 min)
           - Professional greeting
           - Process explanation
        
        2. **Skill Assessment** (25-35 min)
           - Sequential questions
           - Adaptive difficulty
           - Real-time evaluation
        
        3. **Practical Challenges** (10-15 min)
           - Excel simulations
           - Problem-solving tasks
        
        4. **Feedback & Summary** (5 min)
           - Performance analysis
           - Improvement recommendations
        """)

def show_interview_page():
    """Display the interview interface."""
    if not st.session_state.interview_engine:
        config = Config()
        st.session_state.interview_engine = InterviewEngine(config)
    
    st.header("🎤 Excel Skills Interview")
    
    interview_engine = st.session_state.interview_engine
    interview_engine.render_interview_interface()

def show_dashboard_page():
    """Display the performance dashboard."""
    st.header("📊 Performance Dashboard")
    dashboard_ui.render_dashboard()  # ✅ updated

def show_analytics_page():
    """Display advanced analytics and insights."""
    st.header("📈 Advanced Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Skill Distribution")
        dashboard_ui.render_skill_distribution()
    
    with col2:
        st.subheader("Performance Trends")
        dashboard_ui.render_trend_analysis()
    
    with col3:
        st.subheader("Radar Analysis")
        dashboard_ui.render_performance_radar()

if __name__ == "__main__":
    main()

# import streamlit as st
# import os
# from dotenv import load_dotenv
# from src.interview.interview_engine import InterviewEngine
# # from src.dashboard.interview_dashboard import InterviewDashboard
# from src.dashboard import dashboard_ui
# from src.utils.config import Config

# # Load environment variables
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Excel Interviewer AI",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def main():
#     """Main application entry point."""
#     st.title("🎯 AI-Powered Excel Interviewer")
#     st.markdown("### Professional Excel Skills Assessment Platform")
    
#     # Initialize session state
#     if 'interview_started' not in st.session_state:
#         st.session_state.interview_started = False
#     if 'interview_engine' not in st.session_state:
#         st.session_state.interview_engine = None
#     if 'current_view' not in st.session_state:
#         st.session_state.current_view = "home"
    
#     # Sidebar navigation
#     with st.sidebar:
#         st.header("Navigation")
        
#         if st.button("🏠 Home", use_container_width=True):
#             st.session_state.current_view = "home"
#             st.rerun()
        
#         if st.button("🎤 Start Interview", use_container_width=True):
#             st.session_state.current_view = "interview"
#             st.rerun()
        
#         if st.button("📊 Dashboard", use_container_width=True):
#             st.session_state.current_view = "dashboard"
#             st.rerun()
        
#         if st.button("📈 Analytics", use_container_width=True):
#             st.session_state.current_view = "analytics"
#             st.rerun()
        
#         st.divider()
#         st.info("This AI-powered system conducts comprehensive Excel interviews with real-time evaluation and detailed feedback.")
    
#     # Main content area
#     if st.session_state.current_view == "home":
#         show_home_page()
#     elif st.session_state.current_view == "interview":
#         show_interview_page()
#     elif st.session_state.current_view == "dashboard":
#         show_dashboard_page()
#     elif st.session_state.current_view == "analytics":
#         show_analytics_page()

# def show_home_page():
#     """Display the home page with system overview."""
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.header("Welcome to AI-Powered Excel Interviewer")
#         st.markdown("""
#         ### 🎯 What We Offer
        
#         **Comprehensive Assessment**: Our AI system evaluates candidates across multiple dimensions:
#         - Technical Excel proficiency
#         - Problem-solving methodology  
#         - Communication clarity
#         - Practical application knowledge
        
#         **Advanced Features**:
#         - 🧠 Adaptive difficulty adjustment
#         - 📊 Real-time performance evaluation
#         - 🎮 Interactive Excel simulations
#         - 📈 Detailed skill competency heatmaps
#         - 🏆 Gamification and benchmarking
        
#         **Professional Experience**:
#         - Structured multi-turn conversations
#         - Context-aware follow-up questions
#         - Professional tone maintenance
#         - Comprehensive feedback reports
#         """)
        
#         if st.button("🚀 Start Your Interview", type="primary", use_container_width=True):
#             st.session_state.current_view = "interview"
#             st.rerun()
    
#     with col2:
#         st.header("Quick Stats")
#         st.metric("Total Interviews Conducted", "1,247")
#         st.metric("Average Interview Duration", "32 minutes")
#         st.metric("Candidate Satisfaction Rate", "94%")
#         st.metric("Skills Assessed", "15+")
        
#         st.divider()
#         st.header("🎯 Interview Process")
#         st.markdown("""
#         1. **Introduction** (2 min)
#            - Professional greeting
#            - Process explanation
        
#         2. **Skill Assessment** (25-35 min)
#            - Sequential questions
#            - Adaptive difficulty
#            - Real-time evaluation
        
#         3. **Practical Challenges** (10-15 min)
#            - Excel simulations
#            - Problem-solving tasks
        
#         4. **Feedback & Summary** (5 min)
#            - Performance analysis
#            - Improvement recommendations
#         """)

# def show_interview_page():
#     """Display the interview interface."""
#     if not st.session_state.interview_engine:
#         config = Config()
#         st.session_state.interview_engine = InterviewEngine(config)
    
#     st.header("🎤 Excel Skills Interview")
    
#     # Interview interface will be implemented in the interview engine
#     interview_engine = st.session_state.interview_engine
#     interview_engine.render_interview_interface()

# def show_dashboard_page():
#     """Display the performance dashboard."""
#     st.header("📊 Performance Dashboard")
    
#     dashboard = InterviewDashboard()
#     dashboard.render_dashboard()

# def show_analytics_page():
#     """Display advanced analytics and insights."""
#     st.header("📈 Advanced Analytics")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.subheader("Skill Distribution")
#         st.markdown("Coming soon...")
    
#     with col2:
#         st.subheader("Performance Trends")
#         st.markdown("Coming soon...")
    
#     with col3:
#         st.subheader("Benchmarking Analysis")
#         st.markdown("Coming soon...")

# if __name__ == "__main__":
#     main()