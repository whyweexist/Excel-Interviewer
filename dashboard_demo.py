"""
Interactive Dashboard Demo for AI Excel Interviewer
Demonstrates the skill competency heatmaps and performance analytics
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dashboard.dashboard_ui import render_dashboard
from dashboard.dashboard_engine import InteractiveDashboard, SkillCompetencyData
from evaluation.answer_evaluator import ComprehensiveEvaluation, EvaluationDimension, ScoreLevel
from interview.conversation_state import DifficultyLevel
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Main dashboard demo function"""
    
    # Configure page
    st.set_page_config(
        page_title="AI Excel Interviewer - Interactive Dashboard Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h1>🎯 AI Excel Interviewer Dashboard</h1>
        <p>Interactive Skill Competency Heatmaps & Performance Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different demo modes
    tabs = st.tabs([
        "🎯 Interactive Dashboard",
        "📊 Sample Data Demo",
        "🔧 Dashboard Configuration",
        "📈 Analytics Overview"
    ])
    
    with tabs[0]:
        render_interactive_dashboard()
    
    with tabs[1]:
        render_sample_data_demo()
    
    with tabs[2]:
        render_configuration_demo()
    
    with tabs[3]:
        render_analytics_overview()

def render_interactive_dashboard():
    """Render the main interactive dashboard"""
    st.markdown("### 🎯 Interactive Skill Competency Dashboard")
    st.markdown("Experience the full interactive dashboard with real-time skill competency heatmaps and performance analytics.")
    
    # Initialize dashboard
    dashboard = InteractiveDashboard()
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Launch Full Dashboard", type="primary", use_container_width=True):
            st.info("Launching full dashboard in a new tab...")
            # This would typically open in a new tab or modal
            render_dashboard()
    
    with col2:
        if st.button("📊 Load Sample Session", use_container_width=True):
            # Load sample competency data
            sample_data = create_sample_competency_data()
            st.session_state['competency_data'] = sample_data
            st.success("Sample session data loaded!")
    
    with col3:
        if st.button("🔄 Reset Dashboard", use_container_width=True):
            if 'competency_data' in st.session_state:
                del st.session_state['competency_data']
            st.success("Dashboard reset!")
    
    # Display dashboard if data is available
    if 'competency_data' in st.session_state:
        competency_data = st.session_state['competency_data']
        
        st.markdown("---")
        
        # Create dashboard visualizations
        try:
            figures = dashboard.generate_comprehensive_dashboard(competency_data)
            
            # Display visualizations in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Skill Competency Heatmap")
                st.plotly_chart(figures["skill_heatmap"], use_container_width=True, key="demo_heatmap")
                
                st.markdown("#### 📈 Performance Trend")
                st.plotly_chart(figures["performance_trend"], use_container_width=True, key="demo_trend")
            
            with col2:
                st.markdown("#### 🎯 Performance Radar")
                st.plotly_chart(figures["performance_radar"], use_container_width=True, key="demo_radar")
                
                st.markdown("#### 📊 Skill Distribution")
                st.plotly_chart(figures["skill_distribution"], use_container_width=True, key="demo_distribution")
            
        except Exception as e:
            st.error(f"Error generating dashboard: {str(e)}")
            logger.error(f"Dashboard generation error: {str(e)}")
    else:
        st.info("👆 Load sample session data to see the interactive dashboard in action!")

def render_sample_data_demo():
    """Render sample data demonstration"""
    st.markdown("### 📊 Sample Data Demonstration")
    st.markdown("Explore different sample datasets and see how they visualize in the dashboard.")
    
    # Sample data options
    sample_options = [
        "High Performer (Expert Level)",
        "Average Performer (Intermediate Level)", 
        "Developing Candidate (Basic Level)",
        "Mixed Profile (Variable Performance)",
        "Custom Scenario"
    ]
    
    selected_sample = st.selectbox("Choose a sample profile:", sample_options)
    
    if st.button("Generate Sample Data", type="primary"):
        sample_data = create_custom_sample_data(selected_sample)
        
        # Display sample data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{sample_data.overall_score:.1f}%")
        
        with col2:
            st.metric("Skill Level", sample_data.skill_level)
        
        with col3:
            st.metric("Categories Assessed", len(sample_data.skill_categories))
        
        # Display detailed breakdown
        st.markdown("#### 📋 Skill Category Breakdown")
        
        for category, score in sample_data.skill_categories.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{category}**")
                st.progress(int(score / 100))
            
            with col2:
                st.markdown(f"{score:.1f}%")
        
        # Store in session for dashboard view
        if st.button("View in Full Dashboard"):
            st.session_state['competency_data'] = sample_data
            st.success("Sample data loaded into dashboard!")

def render_configuration_demo():
    """Render dashboard configuration demonstration"""
    st.markdown("### 🔧 Dashboard Configuration")
    st.markdown("Explore different configuration options for the dashboard.")
    
    # Configuration options
    with st.expander("🎨 Visualization Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Theme", ["professional", "modern", "classic"])
            color_scheme = st.selectbox("Color Scheme", ["viridis", "blues", "rainbow", "plasma", "cividis"])
            heatmap_granularity = st.selectbox("Heatmap Detail", ["basic", "detailed", "comprehensive"])
        
        with col2:
            comparison_mode = st.selectbox("Comparison Mode", ["individual", "peer", "benchmark"])
            show_annotations = st.checkbox("Show Annotations", value=True)
            enable_interactions = st.checkbox("Enable Interactions", value=True)
        
        if st.button("Apply Configuration", type="primary"):
            st.success("Configuration applied! (Settings would be applied to dashboard)")
    
    with st.expander("📊 Data Display Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_sub_skills = st.checkbox("Show Sub-skills", value=True)
            show_confidence_scores = st.checkbox("Show Confidence Scores", value=True)
            show_time_analysis = st.checkbox("Show Time Analysis", value=True)
        
        with col2:
            show_trend_lines = st.checkbox("Show Trend Lines", value=True)
            show_benchmarks = st.checkbox("Show Benchmarks", value=False)
            show_percentiles = st.checkbox("Show Percentiles", value=False)
        
        if st.button("Update Display Settings"):
            st.success("Display settings updated!")
    
    with st.expander("🎯 Advanced Features"):
        st.markdown("#### Advanced Dashboard Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            real_time_updates = st.checkbox("Real-time Updates", value=False)
            predictive_analytics = st.checkbox("Predictive Analytics", value=False)
            skill_gap_analysis = st.checkbox("Skill Gap Analysis", value=True)
        
        with col2:
            peer_comparison = st.checkbox("Peer Comparison", value=False)
            historical_tracking = st.checkbox("Historical Tracking", value=False)
            export_enabled = st.checkbox("Enable Export", value=True)
        
        if st.button("Configure Advanced Features"):
            st.success("Advanced features configured!")

def render_analytics_overview():
    """Render analytics overview and insights"""
    st.markdown("### 📈 Analytics Overview")
    st.markdown("Comprehensive analytics and insights from the dashboard system.")
    
    # Analytics metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions Analyzed", "1,247", delta="+12%")
        st.caption("vs. last month")
    
    with col2:
        st.metric("Average Overall Score", "73.2%", delta="+3.1%")
        st.caption("vs. last month")
    
    with col3:
        st.metric("Expert Level Candidates", "156", delta="+8%")
        st.caption("vs. last month")
    
    with col4:
        st.metric("Skill Improvement Rate", "68%", delta="+5%")
        st.caption("vs. last month")
    
    # Analytics charts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Skill Distribution Analysis")
        
        # Sample skill distribution data
        skill_categories = ["Technical Accuracy", "Problem Solving", "Communication", "Practical Application", "Excel Expertise"]
        avg_scores = [72.5, 68.8, 79.2, 71.1, 75.6]
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=skill_categories,
                y=avg_scores,
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
                text=[f"{score:.1f}%" for score in avg_scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Average Scores by Skill Category",
            xaxis_title="Skill Categories",
            yaxis_title="Average Score (%)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="analytics_skill_dist")
    
    with col2:
        st.markdown("#### 📈 Performance Trends")
        
        # Sample trend data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        avg_performance = [69.2, 70.8, 72.1, 71.5, 73.2, 74.1]
        expert_percentage = [8.5, 9.2, 10.1, 9.8, 11.2, 12.1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=avg_performance,
            mode='lines+markers',
            name='Average Performance',
            line=dict(color='rgb(55, 128, 191)', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=expert_percentage,
            mode='lines+markers',
            name='Expert Level %',
            line=dict(color='rgb(255, 127, 14)', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Month",
            yaxis=dict(title="Average Score (%)", side="left"),
            yaxis2=dict(title="Expert Level %", side="right", overlaying="y"),
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="analytics_trends")
    
    # Key insights
    st.markdown("---")
    st.markdown("#### 💡 Key Analytics Insights")
    
    insights = [
        "📈 **Performance Improvement**: 68% of candidates show improvement in their second interview session",
        "🎯 **Skill Focus**: Technical Accuracy and Problem Solving are the most challenging areas for candidates",
        "⚡ **Efficiency Gain**: Candidates with prior Excel training complete interviews 23% faster",
        "🌟 **Top Performers**: Expert-level candidates consistently score above 85% in Communication Clarity",
        "📊 **Learning Curve**: Most improvement occurs between sessions 2-4, with diminishing returns after session 6"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")

def create_sample_competency_data() -> SkillCompetencyData:
    """Create sample competency data for demonstration"""
    return SkillCompetencyData(
        candidate_id="demo_001",
        candidate_name="Demo Candidate",
        session_date=datetime.now(),
        skill_categories={
            "Technical Accuracy": 75.5,
            "Problem Solving": 68.2,
            "Communication Clarity": 82.1,
            "Practical Application": 71.8,
            "Excel Expertise": 79.3
        },
        sub_skills={
            "Technical Accuracy": {
                "Formula Knowledge": 78.0,
                "Function Understanding": 72.5,
                "Data Structure Awareness": 76.0
            },
            "Problem Solving": {
                "Analytical Thinking": 65.0,
                "Solution Approach": 70.0,
                "Error Handling": 69.5
            },
            "Communication Clarity": {
                "Explanation Quality": 85.0,
                "Technical Terminology": 80.0,
                "Response Structure": 81.3
            },
            "Practical Application": {
                "Real-world Relevance": 73.0,
                "Use Case Understanding": 70.5,
                "Implementation Strategy": 72.0
            },
            "Excel Expertise": {
                "Advanced Features": 76.0,
                "Best Practices": 82.0,
                "Efficiency Techniques": 80.0
            }
        },
        performance_trend=[65, 68, 72, 75, 78, 76, 79, 82, 78, 75],
        difficulty_progression=[],
        time_spent_per_skill={
            "Technical Accuracy": 120.5,
            "Problem Solving": 98.2,
            "Communication Clarity": 85.1,
            "Practical Application": 110.8,
            "Excel Expertise": 95.3
        },
        confidence_scores={
            "Technical Accuracy": 0.78,
            "Problem Solving": 0.65,
            "Communication Clarity": 0.85,
            "Practical Application": 0.72,
            "Excel Expertise": 0.80
        },
        improvement_areas=[
            "Problem Solving - Analytical Thinking",
            "Practical Application - Use Case Understanding"
        ],
        strengths=[
            "Communication Clarity - Explanation Quality",
            "Excel Expertise - Best Practices",
            "Technical Accuracy - Formula Knowledge"
        ],
        overall_score=75.4,
        skill_level="Advanced"
    )

def create_custom_sample_data(profile_type: str) -> SkillCompetencyData:
    """Create custom sample data based on profile type"""
    
    profiles = {
        "High Performer (Expert Level)": {
            "overall": 88.5,
            "skills": {"Technical Accuracy": 92, "Problem Solving": 89, "Communication": 91, "Practical Application": 85, "Excel Expertise": 94},
            "level": "Expert"
        },
        "Average Performer (Intermediate Level)": {
            "overall": 72.3,
            "skills": {"Technical Accuracy": 75, "Problem Solving": 68, "Communication": 78, "Practical Application": 71, "Excel Expertise": 73},
            "level": "Intermediate"
        },
        "Developing Candidate (Basic Level)": {
            "overall": 45.7,
            "skills": {"Technical Accuracy": 48, "Problem Solving": 42, "Communication": 52, "Practical Application": 45, "Excel Expertise": 47},
            "level": "Basic"
        },
        "Mixed Profile (Variable Performance)": {
            "overall": 65.8,
            "skills": {"Technical Accuracy": 82, "Problem Solving": 58, "Communication": 75, "Practical Application": 52, "Excel Expertise": 68},
            "level": "Intermediate"
        }
    }
    
    base_profile = profiles.get(selected_sample, profiles["Average Performer (Intermediate Level)"])
    
    # Add some variation to make it more realistic
    import random
    variation = lambda x: max(0, min(100, x + random.uniform(-5, 5)))
    
    return SkillCompetencyData(
        candidate_id=f"sample_{selected_sample.lower().replace(' ', '_')}",
        candidate_name=f"Sample {selected_sample}",
        session_date=datetime.now(),
        skill_categories={k: variation(v) for k, v in base_profile["skills"].items()},
        sub_skills={},  # Would be populated with detailed breakdown
        performance_trend=[variation(base_profile["overall"] - 10 + i * 2) for i in range(10)],
        difficulty_progression=[],
        time_spent_per_skill={k: random.uniform(80, 150) for k in base_profile["skills"].keys()},
        confidence_scores={k: random.uniform(0.6, 0.9) for k in base_profile["skills"].keys()},
        improvement_areas=[],
        strengths=[],
        overall_score=variation(base_profile["overall"]),
        skill_level=base_profile["level"]
    )

if __name__ == "__main__":
    main()