"""
Interactive Dashboard UI for AI Excel Interviewer
Streamlit-based interface for skill competency visualization and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
import json

from .dashboard_engine import (
    InteractiveDashboard, SkillCompetencyData, DashboardConfig, 
    VisualizationType, SkillCategory
)
from ..evaluation.answer_evaluator import ComprehensiveEvaluation
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class DashboardUI:
    """Streamlit-based dashboard user interface"""
    
    def __init__(self):
        self.dashboard_engine = InteractiveDashboard()
        self.current_competency_data: Optional[SkillCompetencyData] = None
        self.session_evaluations: List[ComprehensiveEvaluation] = []
        
    def render_dashboard(self):
        """Render the complete dashboard interface"""
        
        # Configure page
        st.set_page_config(
            page_title="AI Excel Interviewer - Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_styles()
        
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
        
        # Footer
        self._render_footer()
    
    def _apply_custom_styles(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
        }
        
        /* Skill level badges */
        .skill-badge-expert {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .skill-badge-advanced {
            background: linear-gradient(135deg, #feca57, #ff9ff3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .skill-badge-intermediate {
            background: linear-gradient(135deg, #48dbfb, #0abde3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .skill-badge-basic {
            background: linear-gradient(135deg, #26de81, #20bf6b);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        /* Heatmap styling */
        .heatmap-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Progress indicators */
        .progress-indicator {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Insight cards */
        .insight-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .strength-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .improvement-card {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        /* Custom button styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background: transparent;
            border-bottom: 2px solid #e9ecef;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: transparent;
            border: none;
            color: #495057;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px 10px 0 0;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
        }
        
        /* Loading animation */
        .loading-animation {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="dashboard-header">
            <h1>🎯 AI Excel Interviewer Dashboard</h1>
            <p>Comprehensive Performance Analytics & Skill Competency Visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar with controls and filters"""
        with st.sidebar:
            st.markdown("### 🎛️ Dashboard Controls")
            
            # Dashboard configuration
            with st.expander("⚙️ Configuration", expanded=True):
                self._render_configuration_controls()
            
            # Data management
            with st.expander("📊 Data Management", expanded=False):
                self._render_data_controls()
            
            # Export options
            with st.expander("📤 Export Options", expanded=False):
                self._render_export_controls()
            
            # Session info
            if self.current_competency_data:
                with st.expander("ℹ️ Current Session", expanded=False):
                    self._render_session_info()
            
            # Quick stats
            self._render_quick_stats()
    
    def _render_configuration_controls(self):
        """Render dashboard configuration controls"""
        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["professional", "modern", "classic"],
            help="Select dashboard visual theme"
        )
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme",
            ["viridis", "blues", "rainbow", "plasma"],
            help="Select color palette for visualizations"
        )
        
        # Heatmap granularity
        heatmap_granularity = st.selectbox(
            "Heatmap Detail",
            ["basic", "detailed", "comprehensive"],
            help="Level of detail in skill heatmaps"
        )
        
        # Comparison mode
        comparison_mode = st.selectbox(
            "Comparison Mode",
            ["individual", "peer", "benchmark"],
            help="Select comparison baseline"
        )
        
        # Update configuration
        if st.button("Apply Configuration"):
            self.dashboard_engine.config.theme = theme
            self.dashboard_engine.config.color_scheme = color_scheme
            self.dashboard_engine.config.heatmap_granularity = heatmap_granularity
            self.dashboard_engine.config.comparison_mode = comparison_mode
            st.success("Configuration updated!")
    
    def _render_data_controls(self):
        """Render data management controls"""
        # Load sample data
        if st.button("Load Sample Data"):
            self._load_sample_data()
            st.success("Sample data loaded!")
        
        # Clear current data
        if st.button("Clear Current Data"):
            self.current_competency_data = None
            self.session_evaluations = []
            st.success("Data cleared!")
        
        # Refresh visualizations
        if st.button("Refresh Visualizations"):
            st.rerun()
    
    def _render_export_controls(self):
        """Render export controls"""
        if self.current_competency_data:
            # Export format selection
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "PDF Report"]
            )
            
            if st.button("Export Data"):
                exported_data = self.dashboard_engine.export_dashboard_data(
                    self.current_competency_data, 
                    export_format.lower()
                )
                
                # Create download button
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"interview_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                    mime="text/plain"
                )
    
    def _render_session_info(self):
        """Render current session information"""
        if self.current_competency_data:
            st.markdown(f"**Candidate:** {self.current_competency_data.candidate_name}")
            st.markdown(f"**Session Date:** {self.current_competency_data.session_date.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Overall Score:** {self.current_competency_data.overall_score:.1f}%")
            st.markdown(f"**Skill Level:** {self.current_competency_data.skill_level}")
    
    def _render_quick_stats(self):
        """Render quick statistics"""
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        
        if self.current_competency_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Overall Score",
                    f"{self.current_competency_data.overall_score:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Skills Assessed",
                    len(self.current_competency_data.skill_categories),
                    delta=None
                )
        else:
            st.info("No session data available")
    
    def _render_main_content(self):
        """Render main dashboard content"""
        
        # Check if we have data to display
        if not self.current_competency_data:
            self._render_empty_state()
            return
        
        # Create tabs for different views
        tabs = st.tabs([
            "🎯 Skill Competency Heatmap",
            "📊 Performance Analytics",
            "📈 Trend Analysis",
            "🔍 Detailed Insights",
            "📋 Executive Summary"
        ])
        
        # Skill Competency Heatmap Tab
        with tabs[0]:
            self._render_skill_heatmap_tab()
        
        # Performance Analytics Tab
        with tabs[1]:
            self._render_performance_analytics_tab()
        
        # Trend Analysis Tab
        with tabs[2]:
            self._render_trend_analysis_tab()
        
        # Detailed Insights Tab
        with tabs[3]:
            self._render_detailed_insights_tab()
        
        # Executive Summary Tab
        with tabs[4]:
            self._render_executive_summary_tab()
    
    def _render_empty_state(self):
        """Render empty state when no data is available"""
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Welcome to the AI Excel Interviewer Dashboard</h3>
            <p>This dashboard provides comprehensive performance analytics and skill competency visualization for interview sessions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Real-time Analytics</h4>
                <p>Live performance tracking and skill assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Interactive Heatmaps</h4>
                <p>Visual skill competency representation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Trend Analysis</h4>
                <p>Performance progression insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Use the sidebar to load sample data or process interview results to see the dashboard in action.")
    
    def _render_skill_heatmap_tab(self):
        """Render skill competency heatmap visualization"""
        st.markdown("### 🔥 Skill Competency Heatmap")
        st.markdown("Interactive visualization of candidate performance across all skill dimensions.")
        
        try:
            # Generate heatmap
            figures = self.dashboard_engine.generate_comprehensive_dashboard(self.current_competency_data)
            heatmap_fig = figures["skill_heatmap"]
            
            # Display heatmap
            st.plotly_chart(heatmap_fig, use_container_width=True, key="skill_heatmap")
            
            # Add interactive features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 Zoom to Details"):
                    self._show_detailed_heatmap()
            
            with col2:
                if st.button("📊 Compare with Benchmark"):
                    self._show_benchmark_comparison()
            
            with col3:
                if st.button("🔄 Refresh Heatmap"):
                    st.rerun()
            
            # Skill breakdown
            self._render_skill_breakdown()
            
        except Exception as e:
            st.error(f"Error rendering heatmap: {str(e)}")
            logger.error(f"Heatmap rendering error: {str(e)}")
    
    def _render_performance_analytics_tab(self):
        """Render performance analytics visualizations"""
        st.markdown("### 📊 Performance Analytics")
        st.markdown("Comprehensive performance analysis across multiple dimensions.")
        
        try:
            # Generate performance visualizations
            figures = self.dashboard_engine.generate_comprehensive_dashboard(self.current_competency_data)
            
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Performance Radar Chart")
                radar_fig = figures["performance_radar"]
                st.plotly_chart(radar_fig, use_container_width=True, key="performance_radar")
            
            with col2:
                st.markdown("#### 📈 Skill Distribution")
                distribution_fig = figures["skill_distribution"]
                st.plotly_chart(distribution_fig, use_container_width=True, key="skill_distribution")
            
            # Performance metrics table
            self._render_performance_metrics_table()
            
            # Key insights
            self._render_key_insights()
            
        except Exception as e:
            st.error(f"Error rendering performance analytics: {str(e)}")
            logger.error(f"Performance analytics error: {str(e)}")
    
    def _render_trend_analysis_tab(self):
        """Render trend analysis visualizations"""
        st.markdown("### 📈 Performance Trend Analysis")
        st.markdown("Analysis of performance progression throughout the interview session.")
        
        try:
            # Generate trend visualization
            figures = self.dashboard_engine.generate_comprehensive_dashboard(self.current_competency_data)
            trend_fig = figures["performance_trend"]
            
            st.plotly_chart(trend_fig, use_container_width=True, key="performance_trend")
            
            # Trend analysis insights
            self._render_trend_insights()
            
            # Progress indicators
            self._render_progress_indicators()
            
        except Exception as e:
            st.error(f"Error rendering trend analysis: {str(e)}")
            logger.error(f"Trend analysis error: {str(e)}")
    
    def _render_detailed_insights_tab(self):
        """Render detailed insights and recommendations"""
        st.markdown("### 🔍 Detailed Skill Insights")
        st.markdown("In-depth analysis of candidate strengths and improvement areas.")
        
        # Strengths section
        self._render_strengths_section()
        
        # Improvement areas section
        self._render_improvement_areas_section()
        
        # Detailed skill breakdown
        self._render_detailed_skill_breakdown()
        
        # Recommendations
        self._render_recommendations()
    
    def _render_executive_summary_tab(self):
        """Render executive summary view"""
        st.markdown("### 📋 Executive Summary")
        st.markdown("High-level overview of candidate performance and recommendations.")
        
        # Executive summary card
        self._render_executive_summary_card()
        
        # Key performance indicators
        self._render_kpi_summary()
        
        # Hiring recommendation
        self._render_hiring_recommendation()
        
        # Action items
        self._render_action_items()
    
    def _render_skill_breakdown(self):
        """Render detailed skill breakdown"""
        st.markdown("#### 📋 Skill Category Breakdown")
        
        if not self.current_competency_data:
            return
        
        # Create expandable sections for each skill category
        for category, score in self.current_competency_data.skill_categories.items():
            with st.expander(f"**{category}** - Score: {score:.1f}%"):
                # Sub-skills
                if category in self.current_competency_data.sub_skills:
                    sub_skills = self.current_competency_data.sub_skills[category]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Sub-skills:**")
                        for sub_skill, sub_score in sub_skills.items():
                            st.markdown(f"- {sub_skill}: {sub_score:.1f}%")
                    
                    with col2:
                        # Mini progress bars for sub-skills
                        st.markdown("**Performance:**")
                        for sub_skill, sub_score in sub_skills.items():
                            st.progress(int(sub_score / 100))
                
                # Confidence score
                confidence = self.current_competency_data.confidence_scores.get(category, 0)
                st.markdown(f"**Confidence Level:** {confidence:.1%}")
                
                # Time spent
                time_spent = self.current_competency_data.time_spent_per_skill.get(category, 0)
                st.markdown(f"**Time Spent:** {time_spent:.1f} seconds")
    
    def _render_performance_metrics_table(self):
        """Render performance metrics in table format"""
        if not self.current_competency_data:
            return
        
        st.markdown("#### 📊 Performance Metrics Table")
        
        # Create metrics dataframe
        metrics_data = []
        for category, score in self.current_competency_data.skill_categories.items():
            metrics_data.append({
                "Skill Category": category,
                "Score (%)": f"{score:.1f}",
                "Confidence": f"{self.current_competency_data.confidence_scores.get(category, 0):.1%}",
                "Time Spent (s)": f"{self.current_competency_data.time_spent_per_skill.get(category, 0):.1f}",
                "Performance Level": self._get_performance_level(score)
            })
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = np.mean(list(self.current_competency_data.skill_categories.values()))
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col2:
            max_score = max(self.current_competency_data.skill_categories.values())
            st.metric("Highest Score", f"{max_score:.1f}%")
        
        with col3:
            min_score = min(self.current_competency_data.skill_categories.values())
            st.metric("Lowest Score", f"{min_score:.1f}%")
        
        with col4:
            std_score = np.std(list(self.current_competency_data.skill_categories.values()))
            st.metric("Score Variability", f"{std_score:.1f}%")
    
    def _render_key_insights(self):
        """Render key performance insights"""
        st.markdown("#### 💡 Key Performance Insights")
        
        if not self.current_competency_data:
            return
        
        # Generate insights based on data
        insights = self._generate_insights()
        
        for insight in insights:
            st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
    
    def _render_strengths_section(self):
        """Render candidate strengths section"""
        st.markdown("#### 🏆 Key Strengths")
        
        if not self.current_competency_data or not self.current_competency_data.strengths:
            st.info("No specific strengths identified yet.")
            return
        
        for strength in self.current_competency_data.strengths[:5]:  # Top 5 strengths
            st.markdown(f"<div class='strength-card'>✅ {strength}</div>", unsafe_allow_html=True)
    
    def _render_improvement_areas_section(self):
        """Render improvement areas section"""
        st.markdown("#### 🎯 Areas for Improvement")
        
        if not self.current_competency_data or not self.current_competency_data.improvement_areas:
            st.info("No specific improvement areas identified yet.")
            return
        
        for area in self.current_competency_data.improvement_areas[:5]:  # Top 5 areas
            st.markdown(f"<div class='improvement-card'>⚠️ {area}</div>", unsafe_allow_html=True)
    
    def _render_detailed_skill_breakdown(self):
        """Render detailed skill breakdown with visualizations"""
        st.markdown("#### 🔍 Detailed Skill Analysis")
        
        if not self.current_competency_data:
            return
        
        # Create skill comparison chart
        categories = list(self.current_competency_data.skill_categories.keys())
        scores = list(self.current_competency_data.skill_categories.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
                text=[f"{score:.1f}%" for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Skill Category Comparison",
            xaxis_title="Skill Categories",
            yaxis_title="Score (%)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="skill_comparison")
    
    def _render_recommendations(self):
        """Render personalized recommendations"""
        st.markdown("#### 📋 Personalized Recommendations")
        
        recommendations = self._generate_recommendations()
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {recommendation['title']}**")
            st.markdown(f"{recommendation['description']}")
            st.markdown(f"*Priority: {recommendation['priority']}*")
            st.markdown("---")
    
    def _render_executive_summary_card(self):
        """Render executive summary card"""
        if not self.current_competency_data:
            return
        
        # Skill level badge
        skill_level = self.current_competency_data.skill_level.lower()
        badge_class = f"skill-badge-{skill_level}"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Executive Summary</h3>
            <p><strong>Candidate:</strong> {self.current_competency_data.candidate_name}</p>
            <p><strong>Overall Score:</strong> {self.current_competency_data.overall_score:.1f}%</p>
            <p><strong>Skill Level:</strong> <span class="{badge_class}">{self.current_competency_data.skill_level}</span></p>
            <p><strong>Assessment Date:</strong> {self.current_competency_data.session_date.strftime('%B %d, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_kpi_summary(self):
        """Render key performance indicators summary"""
        if not self.current_competency_data:
            return
        
        st.markdown("#### 📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = np.mean(list(self.current_competency_data.skill_categories.values()))
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col2:
            top_skill = max(self.current_competency_data.skill_categories.items(), key=lambda x: x[1])
            st.metric("Top Skill", f"{top_skill[0]}: {top_skill[1]:.1f}%")
        
        with col3:
            improvement_needed = len(self.current_competency_data.improvement_areas)
            st.metric("Improvement Areas", improvement_needed)
        
        with col4:
            strengths_count = len(self.current_competency_data.strengths)
            st.metric("Key Strengths", strengths_count)
    
    def _render_hiring_recommendation(self):
        """Render hiring recommendation"""
        if not self.current_competency_data:
            return
        
        st.markdown("#### 🤔 Hiring Recommendation")
        
        recommendation = self._generate_hiring_recommendation()
        
        # Color code based on recommendation
        if recommendation["decision"] == "Strongly Recommend":
            color = "🟢"
        elif recommendation["decision"] == "Recommend":
            color = "🟡"
        elif recommendation["decision"] == "Neutral":
            color = "🟠"
        else:
            color = "🔴"
        
        st.markdown(f"""
        <div class="info-box">
            <h4>{color} {recommendation['decision']}</h4>
            <p><strong>Score:</strong> {recommendation['score']:.1f}/10</p>
            <p><strong>Justification:</strong> {recommendation['justification']}</p>
            <p><strong>Key Factors:</strong> {', '.join(recommendation['key_factors'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_action_items(self):
        """Render action items for next steps"""
        st.markdown("#### ✅ Action Items")
        
        action_items = self._generate_action_items()
        
        for i, item in enumerate(action_items, 1):
            st.markdown(f"{i}. **{item['action']}**")
            st.markdown(f"   *Priority: {item['priority']}* | *Timeline: {item['timeline']}*")
            if item['notes']:
                st.markdown(f"   *Notes: {item['notes']}*")
            st.markdown("")
    
    def _render_footer(self):
        """Render dashboard footer"""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("*Powered by AI Excel Interviewer*")
        
        with col2:
            st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        with col3:
            st.markdown("*© 2024 AI Interviewer System*")
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 55:
            return "Average"
        elif score >= 40:
            return "Below Average"
        else:
            return "Needs Improvement"
    
    def _generate_insights(self) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if not self.current_competency_data:
            return insights
        
        # Analyze performance patterns
        scores = list(self.current_competency_data.skill_categories.values())
        avg_score = np.mean(scores)
        
        if avg_score >= 80:
            insights.append("🌟 **Outstanding Performance**: Candidate demonstrates exceptional Excel skills across all dimensions.")
        elif avg_score >= 65:
            insights.append("✅ **Strong Performance**: Candidate shows solid Excel competency with room for growth.")
        else:
            insights.append("⚠️ **Development Needed**: Candidate requires additional training in Excel fundamentals.")
        
        # Identify top and bottom skills
        top_skill = max(self.current_competency_data.skill_categories.items(), key=lambda x: x[1])
        bottom_skill = min(self.current_competency_data.skill_categories.items(), key=lambda x: x[1])
        
        insights.append(f"🏆 **Top Skill**: {top_skill[0]} ({top_skill[1]:.1f}%)")
        insights.append(f"📈 **Focus Area**: {bottom_skill[0]} ({bottom_skill[1]:.1f}%)")
        
        # Consistency analysis
        score_std = np.std(scores)
        if score_std < 10:
            insights.append("📊 **Consistent Performance**: Skills are well-balanced across all categories.")
        else:
            insights.append("⚖️ **Variable Performance**: Significant differences between skill categories.")
        
        return insights
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if not self.current_competency_data:
            return recommendations
        
        # Analyze improvement areas
        for area in self.current_competency_data.improvement_areas[:3]:
            recommendations.append({
                "title": f"Improve {area}",
                "description": f"Focus on developing {area} through targeted practice and training.",
                "priority": "High"
            })
        
        # Add general recommendations based on skill level
        if self.current_competency_data.skill_level == "Novice":
            recommendations.append({
                "title": "Excel Fundamentals Training",
                "description": "Complete basic Excel training covering formulas, functions, and data manipulation.",
                "priority": "Critical"
            })
        elif self.current_competency_data.skill_level == "Expert":
            recommendations.append({
                "title": "Advanced Specialization",
                "description": "Consider specializing in advanced Excel features like Power Query, VBA, or data modeling.",
                "priority": "Medium"
            })
        
        return recommendations
    
    def _generate_hiring_recommendation(self) -> Dict[str, Any]:
        """Generate hiring recommendation"""
        if not self.current_competency_data:
            return {"decision": "Insufficient Data", "score": 0, "justification": "", "key_factors": []}
        
        overall_score = self.current_competency_data.overall_score
        
        if overall_score >= 80:
            decision = "Strongly Recommend"
            score = 9.0
            justification = "Candidate demonstrates exceptional Excel skills suitable for advanced roles."
            key_factors = ["High technical accuracy", "Strong problem-solving", "Excellent communication"]
        elif overall_score >= 65:
            decision = "Recommend"
            score = 7.5
            justification = "Candidate shows solid Excel competency with potential for growth."
            key_factors = ["Good technical foundation", "Adequate problem-solving", "Clear communication"]
        elif overall_score >= 50:
            decision = "Neutral"
            score = 6.0
            justification = "Candidate meets basic requirements but may need additional training."
            key_factors = ["Basic technical skills", "Limited problem-solving", "Adequate communication"]
        else:
            decision = "Do Not Recommend"
            score = 4.0
            justification = "Candidate requires significant development in Excel skills."
            key_factors = ["Low technical accuracy", "Weak problem-solving", "Communication gaps"]
        
        return {
            "decision": decision,
            "score": score,
            "justification": justification,
            "key_factors": key_factors
        }
    
    def _generate_action_items(self) -> List[Dict[str, str]]:
        """Generate action items for next steps"""
        action_items = []
        
        if not self.current_competency_data:
            return action_items
        
        # Immediate actions
        action_items.append({
            "action": "Review detailed assessment results",
            "priority": "High",
            "timeline": "Today",
            "notes": "Share comprehensive feedback with hiring team"
        })
        
        # Follow-up actions based on performance
        if self.current_competency_data.overall_score < 60:
            action_items.append({
                "action": "Schedule follow-up technical interview",
                "priority": "Medium",
                "timeline": "Within 1 week",
                "notes": "Focus on areas needing improvement"
            })
        
        if len(self.current_competency_data.improvement_areas) > 3:
            action_items.append({
                "action": "Provide skill development resources",
                "priority": "Low",
                "timeline": "Within 2 weeks",
                "notes": "Share training materials for identified gaps"
            })
        
        return action_items
    
    def _load_sample_data(self):
        """Load sample data for demonstration"""
        # Create sample competency data
        sample_data = SkillCompetencyData(
            candidate_id="sample_001",
            candidate_name="John Doe",
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
        
        self.current_competency_data = sample_data
        logger.info("Sample data loaded successfully")
    
    def _show_detailed_heatmap(self):
        """Show detailed heatmap view"""
        st.info("Detailed heatmap view would show granular skill breakdowns and sub-competencies.")
    
    def _show_benchmark_comparison(self):
        """Show benchmark comparison"""
        st.info("Benchmark comparison would show candidate performance against peer averages and industry standards.")
    
    def _render_trend_insights(self):
        """Render trend analysis insights"""
        if not self.current_competency_data or not self.current_competency_data.performance_trend:
            return
        
        trend_data = self.current_competency_data.performance_trend
        
        # Calculate trend metrics
        if len(trend_data) > 1:
            overall_trend = trend_data[-1] - trend_data[0]
            max_improvement = max(trend_data) - min(trend_data)
            consistency = np.std(trend_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if overall_trend > 0:
                    st.markdown(f"<div class='strength-card'>📈 Overall Improvement: +{overall_trend:.1f}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='improvement-card'>📉 Overall Decline: {overall_trend:.1f}%</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Maximum Variation", f"{max_improvement:.1f}%")
            
            with col3:
                st.metric("Consistency Score", f"{100 - consistency:.1f}%")
    
    def _render_progress_indicators(self):
        """Render progress indicators"""
        if not self.current_competency_data:
            return
        
        st.markdown("#### 📊 Progress Indicators")
        
        # Skill progression indicators
        for category, score in self.current_competency_data.skill_categories.items():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{category}**")
                
                with col2:
                    st.progress(int(score / 100))
                
                with col3:
                    st.markdown(f"{score:.1f}%")

# Global dashboard UI instance
dashboard_ui = DashboardUI()

def render_dashboard():
    """Main function to render the dashboard"""
    dashboard_ui.render_dashboard()

if __name__ == "__main__":
    render_dashboard()