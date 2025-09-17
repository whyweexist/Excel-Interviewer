"""
Active Learning and Continuous Improvement Demo

This script demonstrates the active learning system and continuous improvement pipeline
with interactive visualizations and performance analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from learning.continuous_improvement_engine import (
    ImprovementType, LearningSession, ModelPerformance, 
    ImprovementRecommendation, ContinuousImprovementEngine
)
from learning.active_learning_system import (
    ActiveLearningStrategy, QueryType, ActiveLearningSample,
    StrategicQuery, LearningAdaptation, ActiveLearningSystem
)

# Page configuration
st.set_page_config(
    page_title="AI Learning System Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .learning-insight {
        background-color: #f0f8ff;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .adaptation-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'learning_system' not in st.session_state:
    st.session_state.learning_system = ActiveLearningSystem()

if 'improvement_engine' not in st.session_state:
    st.session_state.improvement_engine = ContinuousImprovementEngine()

if 'demo_data' not in st.session_state:
    st.session_state.demo_data = None

def generate_sample_learning_sessions(count: int = 50) -> list:
    """Generate sample learning sessions for demonstration"""
    sessions = []
    
    for i in range(count):
        # Generate realistic evaluation scores with some variation
        base_scores = {
            'technical_accuracy': np.random.uniform(0.4, 0.95),
            'problem_solving': np.random.uniform(0.3, 0.9),
            'communication_clarity': np.random.uniform(0.5, 0.95),
            'practical_application': np.random.uniform(0.3, 0.85),
            'excel_expertise': np.random.uniform(0.4, 0.9)
        }
        
        # Add some correlation between scores
        base_scores['problem_solving'] = base_scores['technical_accuracy'] * 0.8 + np.random.uniform(-0.1, 0.1)
        base_scores['practical_application'] = base_scores['excel_expertise'] * 0.9 + np.random.uniform(-0.1, 0.1)
        
        # Ensure scores are within bounds
        for key in base_scores:
            base_scores[key] = max(0.1, min(1.0, base_scores[key]))
        
        session = LearningSession(
            session_id=f"session_{i+1:03d}",
            candidate_id=f"candidate_{np.random.randint(1, 20):03d}",
            start_time=datetime.now() - timedelta(days=np.random.randint(0, 30)),
            end_time=datetime.now() - timedelta(days=np.random.randint(0, 30)) + timedelta(minutes=np.random.randint(15, 45)),
            conversation_transcript=[
                {"role": "interviewer", "content": f"Question about {['formulas', 'data analysis', 'visualization', 'automation'][i % 4]}"},
                {"role": "candidate", "content": f"Response with {['basic', 'intermediate', 'advanced', 'expert'][i % 4]} level detail"}
            ],
            evaluation_scores=base_scores,
            difficulty_level=['easy', 'medium', 'hard'][i % 3],
            final_score=np.mean(list(base_scores.values())),
            areas_for_improvement=np.random.choice([
                'formula complexity', 'data interpretation', 'visualization clarity',
                'automation logic', 'error handling', 'efficiency optimization'
            ], size=np.random.randint(1, 3), replace=False).tolist(),
            candidate_responses=[
                {
                    'question': f"Sample question {j+1}",
                    'response': f"Sample response {j+1}",
                    'evaluation_score': np.random.uniform(0.3, 0.9),
                    'response_time': np.random.uniform(30, 120)
                }
                for j in range(np.random.randint(3, 8))
            ]
        )
        sessions.append(session)
    
    return sessions

def create_learning_metrics_visualization(sessions: list) -> go.Figure:
    """Create visualization for learning metrics"""
    # Extract scores
    scores_data = []
    for session in sessions:
        scores_data.append(session.evaluation_scores)
    
    df = pd.DataFrame(scores_data)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add average scores
    avg_scores = df.mean()
    fig.add_trace(go.Scatterpolar(
        r=avg_scores.values,
        theta=avg_scores.index,
        fill='toself',
        name='Average Performance',
        line_color='blue',
        opacity=0.7
    ))
    
    # Add best performance
    best_scores = df.max()
    fig.add_trace(go.Scatterpolar(
        r=best_scores.values,
        theta=best_scores.index,
        fill='toself',
        name='Best Performance',
        line_color='green',
        opacity=0.5
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Learning Performance Metrics"
    )
    
    return fig

def create_active_learning_visualization(active_samples: list) -> go.Figure:
    """Create visualization for active learning samples"""
    if not active_samples:
        return go.Figure()
    
    # Extract data
    strategic_values = [sample.strategic_value for sample in active_samples]
    uncertainties = [sample.prediction_uncertainty for sample in active_samples]
    diversity_scores = [sample.sample_diversity_score for sample in active_samples]
    query_types = [sample.query_type.value for sample in active_samples]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter plot for each query type
    for query_type in set(query_types):
        mask = [qt == query_type for qt in query_types]
        x_vals = np.array(strategic_values)[mask]
        y_vals = np.array(uncertainties)[mask]
        sizes = np.array(diversity_scores)[mask] * 100
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=query_type,
            marker=dict(
                size=sizes,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=[f"Diversity: {div:.3f}" for div in np.array(diversity_scores)[mask]]
        ))
    
    fig.update_layout(
        title="Active Learning Sample Selection",
        xaxis_title="Strategic Value",
        yaxis_title="Prediction Uncertainty",
        hovermode='closest',
        height=500
    )
    
    return fig

def main():
    """Main demo function"""
    st.markdown('<div class="main-header">🧠 AI Learning System Demo</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # System activation
        system_active = st.toggle("Active Learning System", value=True)
        st.session_state.learning_system.is_active = system_active
        
        # Learning strategy
        strategy = st.selectbox(
            "Active Learning Strategy",
            ["UNCERTAINTY_SAMPLING", "DIVERSITY_SAMPLING", "HYBRID_STRATEGY"]
        )
        
        # Sample generation
        num_samples = st.slider("Number of Samples", 10, 100, 50)
        
        if st.button("Generate New Data"):
            with st.spinner("Generating learning sessions..."):
                st.session_state.demo_data = generate_sample_learning_sessions(num_samples)
            st.success(f"Generated {num_samples} learning sessions!")
        
        if st.button("Run Active Learning"):
            if st.session_state.demo_data:
                with st.spinner("Running active learning..."):
                    # Process sessions through active learning
                    active_samples = st.session_state.learning_system.select_active_learning_samples(
                        st.session_state.demo_data,
                        current_models={},
                        strategy=ActiveLearningStrategy[strategy],
                        max_samples=20
                    )
                    st.session_state.active_samples = active_samples
                st.success(f"Selected {len(active_samples)} active learning samples!")
            else:
                st.warning("Please generate demo data first!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Active Learning", "📈 Performance Analytics", "🔧 System Configuration"])
    
    with tab1:
        st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = st.session_state.learning_system.get_active_learning_status()
            st.metric("Learning Cycles", status['learning_cycle'])
        
        with col2:
            st.metric("Active Samples", status['active_samples_count'])
        
        with col3:
            st.metric("Strategic Queries", status['total_queries_generated'])
        
        with col4:
            st.metric("Adaptations Applied", status['adaptations_applied'])
        
        if st.session_state.demo_data:
            # Learning metrics visualization
            st.plotly_chart(create_learning_metrics_visualization(st.session_state.demo_data), use_container_width=True)
            
            # Recent learning insights
            st.markdown('<div class="section-header">Recent Learning Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance trends
                sessions = st.session_state.demo_data[-10:]  # Last 10 sessions
                scores = [session.final_score for session in sessions]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(scores))),
                    y=scores,
                    mode='lines+markers',
                    name='Session Scores',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title="Recent Performance Trend",
                    xaxis_title="Session",
                    yaxis_title="Final Score",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Skill distribution
                all_scores = []
                for session in st.session_state.demo_data:
                    all_scores.extend(session.evaluation_scores.values())
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=all_scores,
                    nbinsx=20,
                    name='Score Distribution',
                    marker_color='green'
                ))
                fig.update_layout(
                    title="Score Distribution Across All Sessions",
                    xaxis_title="Score",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate demo data to see system overview visualizations")
    
    with tab2:
        st.markdown('<div class="section-header">Active Learning Analysis</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'active_samples') and st.session_state.active_samples:
            # Active learning visualization
            st.plotly_chart(create_active_learning_visualization(st.session_state.active_samples), use_container_width=True)
            
            # Sample details
            st.markdown('<div class="section-header">Selected Active Learning Samples</div>', unsafe_allow_html=True)
            
            for i, sample in enumerate(st.session_state.active_samples[:5]):  # Show top 5
                with st.expander(f"Sample {i+1}: {sample.query_type.value}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Strategic Value:** {sample.strategic_value:.3f}")
                        st.write(f"**Uncertainty:** {sample.prediction_uncertainty:.3f}")
                        st.write(f"**Diversity Score:** {sample.sample_diversity_score:.3f}")
                    
                    with col2:
                        st.write(f"**Strategy:** {sample.learning_strategy.value}")
                        st.write(f"**Selection Reason:** {sample.selection_reasoning}")
                        st.write(f"**Session ID:** {sample.session_id}")
                    
                    if hasattr(sample, 'strategic_query'):
                        st.write(f"**Strategic Query:** {sample.strategic_query.query_text}")
                        st.write(f"**Target Skill:** {sample.strategic_query.target_skill_area}")
                        st.write(f"**Expected Difficulty:** {sample.strategic_query.expected_difficulty}")
            
            # Strategic queries
            if st.button("Generate Strategic Queries"):
                skill_areas = ['formulas', 'data_analysis', 'visualization', 'automation']
                query_types = [QueryType.KNOWLEDGE_PROBE, QueryType.SKILL_ASSESSMENT, QueryType.EDGE_CASE_EXPLORATION]
                
                strategic_queries = st.session_state.learning_system.generate_strategic_queries(
                    skill_areas, query_types
                )
                
                st.write(f"Generated {len(strategic_queries)} strategic queries")
                
                for i, query in enumerate(strategic_queries[:3]):  # Show first 3
                    with st.expander(f"Query {i+1}: {query.query_type.value}"):
                        st.write(f"**Text:** {query.query_text}")
                        st.write(f"**Target Skill:** {query.target_skill_area}")
                        st.write(f"**Expected Difficulty:** {query.expected_difficulty}")
                        st.write(f"**Purpose:** {query.strategic_purpose}")
        else:
            st.info("Run active learning to see analysis results")
    
    with tab3:
        st.markdown('<div class="section-header">Performance Analytics</div>', unsafe_allow_html=True)
        
        if st.session_state.demo_data:
            # Model performance simulation
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy over time
                sessions = st.session_state.demo_data
                accuracies = [0.65 + (i * 0.01) + np.random.uniform(-0.05, 0.05) for i in range(len(sessions))]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(accuracies))),
                    y=accuracies,
                    mode='lines+markers',
                    name='Model Accuracy',
                    line=dict(color='green', width=2)
                ))
                fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Target Accuracy")
                fig.update_layout(
                    title="Model Accuracy Over Time",
                    xaxis_title="Session",
                    yaxis_title="Accuracy",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Learning rate analysis
                learning_rates = [0.001 * (0.95 ** i) for i in range(len(sessions))]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(learning_rates))),
                    y=learning_rates,
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='orange', width=2)
                ))
                fig.update_layout(
                    title="Learning Rate Decay",
                    xaxis_title="Epoch",
                    yaxis_title="Learning Rate",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Adaptation triggers
            st.markdown('<div class="section-header">Learning Adaptations</div>', unsafe_allow_html=True)
            
            # Simulate performance data
            recent_performance = {
                'model_accuracy': 0.78,
                'avg_uncertainty': 0.25,
                'diversity_score': 0.65,
                'learning_rate': 0.0005
            }
            
            active_learning_metrics = {
                'uncertainty_threshold': 0.3,
                'diversity_threshold': 0.7,
                'sample_selection_efficiency': 0.82
            }
            
            adaptations = st.session_state.learning_system.adapt_learning_strategy(
                recent_performance, active_learning_metrics
            )
            
            if adaptations:
                for adaptation in adaptations:
                    st.markdown(f'<div class="adaptation-alert">', unsafe_allow_html=True)
                    st.write(f"**{adaptation.adaptation_type}:** {adaptation.description}")
                    st.write(f"**Expected Outcome:** {adaptation.expected_outcome}")
                    st.write(f"**Success Metrics:** {', '.join(adaptation.success_metrics)}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("No adaptations needed - system performing optimally!")
        else:
            st.info("Generate demo data to see performance analytics")
    
    with tab4:
        st.markdown('<div class="section-header">System Configuration</div>', unsafe_allow_html=True)
        
        # System status
        status = st.session_state.learning_system.get_active_learning_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Configuration:**")
            st.json({
                'uncertainty_threshold': status['sampler_status']['uncertainty_threshold'],
                'diversity_threshold': status['sampler_status']['diversity_threshold'],
                'system_active': status['is_active'],
                'learning_cycle': status['learning_cycle']
            })
        
        with col2:
            st.write("**System Metrics:**")
            st.metric("Total Learning Sessions", len(st.session_state.demo_data) if st.session_state.demo_data else 0)
            st.metric("Active Samples", status['active_samples_count'])
            st.metric("Adaptations Applied", status['adaptations_applied'])
        
        # Export insights
        if st.button("Export Learning Insights"):
            insights = st.session_state.learning_system.export_learning_insights()
            
            st.download_button(
                label="Download Learning Insights JSON",
                data=json.dumps(insights, indent=2, default=str),
                file_name=f"learning_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # System reset
        if st.button("Reset Learning System", type="secondary"):
            st.session_state.learning_system = ActiveLearningSystem()
            st.session_state.demo_data = None
            if hasattr(st.session_state, 'active_samples'):
                del st.session_state.active_samples
            st.success("Learning system reset successfully!")
            st.rerun()

if __name__ == "__main__":
    main()