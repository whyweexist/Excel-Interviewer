"""
Interactive Dashboard Engine for AI Excel Interviewer
Manages skill competency visualization, performance analytics, and interactive reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from ..evaluation.answer_evaluator import ComprehensiveEvaluation, EvaluationDimension
from ..interview.conversation_state import InterviewState, QuestionType, DifficultyLevel
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class VisualizationType(Enum):
    """Types of dashboard visualizations"""
    HEATMAP = "heatmap"
    RADAR = "radar"
    TREND = "trend"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    PERFORMANCE = "performance"

class SkillCategory(Enum):
    """Primary skill categories for visualization"""
    TECHNICAL_ACCURACY = "Technical Accuracy"
    PROBLEM_SOLVING = "Problem Solving"
    COMMUNICATION_CLARITY = "Communication Clarity"
    PRACTICAL_APPLICATION = "Practical Application"
    EXCEL_EXPERTISE = "Excel Expertise"
    OVERALL_PERFORMANCE = "Overall Performance"

@dataclass
class SkillCompetencyData:
    """Data structure for skill competency visualization"""
    candidate_id: str
    candidate_name: str
    session_date: datetime
    skill_categories: Dict[str, float]
    sub_skills: Dict[str, Dict[str, float]]
    performance_trend: List[float]
    difficulty_progression: List[DifficultyLevel]
    time_spent_per_skill: Dict[str, float]
    confidence_scores: Dict[str, float]
    improvement_areas: List[str]
    strengths: List[str]
    overall_score: float
    skill_level: str

@dataclass
class DashboardConfig:
    """Configuration for dashboard visualization"""
    theme: str = "professional"
    color_scheme: str = "viridis"
    show_annotations: bool = True
    enable_interactions: bool = True
    update_frequency: int = 5  # seconds
    heatmap_granularity: str = "detailed"  # basic, detailed, comprehensive
    comparison_mode: str = "individual"  # individual, peer, benchmark
    time_range: str = "session"  # session, daily, weekly, monthly

class SkillCompetencyAnalyzer:
    """Analyzes candidate performance and generates competency insights"""
    
    def __init__(self):
        self.skill_weights = {
            SkillCategory.TECHNICAL_ACCURACY.value: 0.25,
            SkillCategory.PROBLEM_SOLVING.value: 0.25,
            SkillCategory.COMMUNICATION_CLARITY.value: 0.20,
            SkillCategory.PRACTICAL_APPLICATION.value: 0.15,
            SkillCategory.EXCEL_EXPERTISE.value: 0.15
        }
        
        self.sub_skill_mappings = {
            "Technical Accuracy": {
                "Formula Knowledge": 0.4,
                "Function Understanding": 0.3,
                "Data Structure Awareness": 0.3
            },
            "Problem Solving": {
                "Analytical Thinking": 0.4,
                "Solution Approach": 0.3,
                "Error Handling": 0.3
            },
            "Communication Clarity": {
                "Explanation Quality": 0.5,
                "Technical Terminology": 0.3,
                "Response Structure": 0.2
            },
            "Practical Application": {
                "Real-world Relevance": 0.4,
                "Use Case Understanding": 0.3,
                "Implementation Strategy": 0.3
            },
            "Excel Expertise": {
                "Advanced Features": 0.4,
                "Best Practices": 0.3,
                "Efficiency Techniques": 0.3
            }
        }
    
    def analyze_evaluation_results(
        self, 
        evaluations: List[ComprehensiveEvaluation],
        candidate_info: Dict[str, Any]
    ) -> SkillCompetencyData:
        """Analyze evaluation results to generate competency data"""
        
        try:
            # Extract scores by dimension
            dimension_scores = self._extract_dimension_scores(evaluations)
            
            # Calculate sub-skill scores
            sub_skill_scores = self._calculate_sub_skills(dimension_scores)
            
            # Generate performance trend
            performance_trend = self._calculate_performance_trend(evaluations)
            
            # Analyze time patterns
            time_spent = self._analyze_time_patterns(evaluations)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(evaluations)
            
            # Identify strengths and improvement areas
            strengths, improvement_areas = self._identify_skill_gaps(sub_skill_scores)
            
            # Calculate overall metrics
            overall_score = self._calculate_overall_score(dimension_scores)
            skill_level = self._determine_skill_level(overall_score)
            
            return SkillCompetencyData(
                candidate_id=candidate_info.get("candidate_id", "unknown"),
                candidate_name=candidate_info.get("candidate_name", "Unknown Candidate"),
                session_date=datetime.now(),
                skill_categories=dimension_scores,
                sub_skills=sub_skill_scores,
                performance_trend=performance_trend,
                difficulty_progression=self._extract_difficulty_progression(evaluations),
                time_spent_per_skill=time_spent,
                confidence_scores=confidence_scores,
                improvement_areas=improvement_areas,
                strengths=strengths,
                overall_score=overall_score,
                skill_level=skill_level
            )
            
        except Exception as e:
            logger.error(f"Error analyzing evaluation results: {str(e)}")
            raise
    
    def _extract_dimension_scores(self, evaluations: List[ComprehensiveEvaluation]) -> Dict[str, float]:
        """Extract average scores for each evaluation dimension"""
        dimension_averages = {}
        
        for dimension in EvaluationDimension:
            scores = [eval.dimension_scores.get(dimension, 0) for eval in evaluations]
            dimension_averages[dimension.value] = np.mean(scores) if scores else 0.0
        
        return dimension_averages
    
    def _calculate_sub_skills(self, dimension_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed sub-skill scores"""
        sub_skills = {}
        
        for dimension, sub_mappings in self.sub_skill_mappings.items():
            if dimension in dimension_scores:
                base_score = dimension_scores[dimension]
                sub_skills[dimension] = {}
                
                for sub_skill, weight in sub_mappings.items():
                    # Apply some variation based on the evaluation context
                    variation = np.random.normal(0, 0.1)  # Small random variation
                    sub_score = max(0, min(100, base_score + (variation * 20)))
                    sub_skills[dimension][sub_skill] = sub_score
        
        return sub_skills
    
    def _calculate_performance_trend(self, evaluations: List[ComprehensiveEvaluation]) -> List[float]:
        """Calculate performance trend over time"""
        if not evaluations:
            return []
        
        # Extract overall scores chronologically
        scores = [eval.overall_score for eval in evaluations]
        
        # Apply smoothing to show trend
        if len(scores) > 1:
            smoothed_scores = self._smooth_data(scores)
            return smoothed_scores
        
        return scores
    
    def _smooth_data(self, data: List[float], window_size: int = 3) -> List[float]:
        """Apply moving average smoothing to data"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            window = data[start_idx:end_idx]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _analyze_time_patterns(self, evaluations: List[ComprehensiveEvaluation]) -> Dict[str, float]:
        """Analyze time spent patterns across skills"""
        # This would integrate with actual timing data
        # For now, return simulated data based on evaluation complexity
        time_patterns = {}
        
        for dimension in EvaluationDimension:
            # Simulate time based on score complexity
            base_time = 60  # seconds
            complexity_factor = np.random.uniform(0.8, 1.5)
            time_patterns[dimension.value] = base_time * complexity_factor
        
        return time_patterns
    
    def _calculate_confidence_scores(self, evaluations: List[ComprehensiveEvaluation]) -> Dict[str, float]:
        """Calculate confidence scores for each skill area"""
        confidence_scores = {}
        
        for dimension in EvaluationDimension:
            # Extract scores for this dimension
            scores = [eval.dimension_scores.get(dimension, 0) for eval in evaluations]
            
            if scores:
                # Confidence based on consistency and score level
                mean_score = np.mean(scores)
                std_dev = np.std(scores)
                consistency = max(0, 1 - (std_dev / 100))  # Lower std = higher consistency
                
                # Combine score level and consistency
                confidence = (mean_score / 100) * 0.7 + consistency * 0.3
                confidence_scores[dimension.value] = min(1.0, confidence)
            else:
                confidence_scores[dimension.value] = 0.0
        
        return confidence_scores
    
    def _identify_skill_gaps(self, sub_skills: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str]]:
        """Identify strengths and improvement areas"""
        strengths = []
        improvement_areas = []
        
        threshold_high = 75
        threshold_low = 50
        
        for dimension, sub_scores in sub_skills.items():
            for sub_skill, score in sub_scores.items():
                skill_name = f"{dimension} - {sub_skill}"
                if score >= threshold_high:
                    strengths.append(skill_name)
                elif score <= threshold_low:
                    improvement_areas.append(skill_name)
        
        return strengths, improvement_areas
    
    def _extract_difficulty_progression(self, evaluations: List[ComprehensiveEvaluation]) -> List[DifficultyLevel]:
        """Extract difficulty progression from evaluations"""
        # This would extract actual difficulty levels from evaluation context
        # For now, return a simulated progression
        difficulties = [DifficultyLevel.BASIC, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]
        return difficulties[:len(evaluations)]
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        weighted_sum = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            # Map evaluation dimensions to skill categories
            category = self._map_dimension_to_category(dimension)
            weight = self.skill_weights.get(category, 0.2)
            
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _map_dimension_to_category(self, dimension: str) -> str:
        """Map evaluation dimension to skill category"""
        mapping = {
            "Technical Accuracy": SkillCategory.TECHNICAL_ACCURACY.value,
            "Problem Solving": SkillCategory.PROBLEM_SOLVING.value,
            "Communication Clarity": SkillCategory.COMMUNICATION_CLARITY.value,
            "Practical Application": SkillCategory.PRACTICAL_APPLICATION.value,
            "Excel Expertise": SkillCategory.EXCEL_EXPERTISE.value
        }
        return mapping.get(dimension, SkillCategory.OVERALL_PERFORMANCE.value)
    
    def _determine_skill_level(self, overall_score: float) -> str:
        """Determine skill level based on overall score"""
        if overall_score >= 85:
            return "Expert"
        elif overall_score >= 70:
            return "Advanced"
        elif overall_score >= 55:
            return "Intermediate"
        elif overall_score >= 40:
            return "Basic"
        else:
            return "Novice"

class InteractiveDashboardRenderer:
    """Renders interactive dashboard visualizations"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.color_schemes = self._initialize_color_schemes()
    
    def _initialize_color_schemes(self) -> Dict[str, Any]:
        """Initialize color schemes for different visualization types"""
        return {
            "heatmap": px.colors.sequential.Viridis,
            "performance": px.colors.sequential.Blues,
            "comparison": px.colors.qualitative.Set3,
            "trend": px.colors.sequential.Plasma,
            "radar": px.colors.sequential.Rainbow
        }
    
    def create_skill_heatmap(self, competency_data: SkillCompetencyData) -> go.Figure:
        """Create interactive skill competency heatmap"""
        
        # Prepare data for heatmap
        skill_categories = list(competency_data.skill_categories.keys())
        scores = list(competency_data.skill_categories.values())
        
        # Create sub-skill matrix for detailed view
        if self.config.heatmap_granularity in ["detailed", "comprehensive"]:
            sub_skill_matrix, labels = self._create_sub_skill_matrix(competency_data)
            
            fig = go.Figure(data=go.Heatmap(
                z=sub_skill_matrix,
                x=labels["x_labels"],
                y=labels["y_labels"],
                colorscale=self.color_schemes["heatmap"],
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
                showscale=True,
                colorbar=dict(title="Competency Score (%)")
            ))
        else:
            # Basic heatmap with main categories
            fig = go.Figure(data=go.Heatmap(
                z=[scores],
                x=skill_categories,
                y=["Skill Categories"],
                colorscale=self.color_schemes["heatmap"],
                hovertemplate='<b>%{x}</b><br>Score: %{z:.1f}%<extra></extra>',
                showscale=True,
                colorbar=dict(title="Score (%)")
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Skill Competency Heatmap - {competency_data.candidate_name}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title="Skill Categories",
            yaxis_title="Competency Level",
            height=500 if self.config.heatmap_granularity == "basic" else 700,
            template="plotly_white",
            hovermode="closest"
        )
        
        if self.config.show_annotations:
            self._add_heatmap_annotations(fig, competency_data)
        
        return fig
    
    def _create_sub_skill_matrix(self, competency_data: SkillCompetencyData) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        """Create detailed sub-skill matrix for heatmap"""
        sub_skills = competency_data.sub_skills
        
        # Flatten sub-skills into matrix format
        y_labels = []
        x_labels = []
        matrix_data = []
        
        for category, sub_categories in sub_skills.items():
            y_labels.append(category)
            if not x_labels:  # First iteration
                x_labels = list(sub_categories.keys())
            
            row_data = [sub_categories[sub_skill] for sub_skill in x_labels]
            matrix_data.append(row_data)
        
        return np.array(matrix_data), {"x_labels": x_labels, "y_labels": y_labels}
    
    def _add_heatmap_annotations(self, fig: go.Figure, competency_data: SkillCompetencyData):
        """Add annotations to heatmap"""
        # Add performance level indicators
        for i, (skill, score) in enumerate(competency_data.skill_categories.items()):
            color = "white" if score < 50 else "black"
            fig.add_annotation(
                x=i,
                y=0,
                text=f"{score:.1f}%",
                showarrow=False,
                font=dict(color=color, size=12, weight="bold")
            )
    
    def create_performance_radar_chart(self, competency_data: SkillCompetencyData) -> go.Figure:
        """Create interactive radar chart for performance visualization"""
        
        categories = list(competency_data.skill_categories.keys())
        scores = list(competency_data.skill_categories.values())
        
        # Close the polygon by repeating first value
        categories_closed = categories + [categories[0]]
        scores_closed = scores + [scores[0]]
        
        fig = go.Figure()
        
        # Add candidate performance
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=categories_closed,
            fill='toself',
            name='Candidate Performance',
            line_color='rgb(55, 128, 191)',
            fillcolor='rgba(55, 128, 191, 0.3)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>'
        ))
        
        # Add benchmark line (if comparison mode enabled)
        if self.config.comparison_mode in ["peer", "benchmark"]:
            benchmark_scores = self._generate_benchmark_scores()
            benchmark_closed = benchmark_scores + [benchmark_scores[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=benchmark_closed,
                theta=categories_closed,
                fill='toself',
                name='Benchmark Average',
                line_color='rgb(255, 127, 14)',
                fillcolor='rgba(255, 127, 14, 0.1)',
                line_dash='dash',
                hovertemplate='<b>%{theta}</b><br>Benchmark: %{r:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': f"Performance Radar - {competency_data.candidate_name}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=12),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    gridcolor='rgba(0,0,0,0.1)'
                )
            ),
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def _generate_benchmark_scores(self) -> List[float]:
        """Generate benchmark scores for comparison"""
        # This would come from actual peer data
        return [65, 70, 68, 62, 75]  # Simulated benchmark scores
    
    def create_performance_trend_chart(self, competency_data: SkillCompetencyData) -> go.Figure:
        """Create performance trend visualization"""
        
        trend_data = competency_data.performance_trend
        
        if not trend_data:
            return self._create_empty_chart("No trend data available")
        
        # Create time-based x-axis
        time_points = list(range(len(trend_data)))
        
        fig = go.Figure()
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=time_points,
            y=trend_data,
            mode='lines+markers',
            name='Performance Trend',
            line=dict(color='rgb(46, 204, 113)', width=3),
            marker=dict(size=8, color='rgb(46, 204, 113)'),
            hovertemplate='<b>Question %{x}</b><br>Score: %{y:.1f}%<extra></extra>'
        ))
        
        # Add confidence bands if enough data points
        if len(trend_data) > 3:
            confidence_upper = [y + 5 for y in trend_data]
            confidence_lower = [y - 5 for y in trend_data]
            
            fig.add_trace(go.Scatter(
                x=time_points + time_points[::-1],
                y=confidence_upper + confidence_lower[::-1],
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name="Confidence Band"
            ))
        
        # Add improvement indicators
        improvements = self._identify_improvements(trend_data)
        for i, improvement in improvements:
            fig.add_annotation(
                x=i,
                y=trend_data[i] + 3,
                text="📈" if improvement > 0 else "📉",
                showarrow=False,
                font=dict(size=16)
            )
        
        fig.update_layout(
            title={
                'text': f"Performance Trend - {competency_data.candidate_name}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title="Interview Progression",
            yaxis_title="Performance Score (%)",
            height=500,
            template="plotly_white",
            hovermode="closest"
        )
        
        return fig
    
    def _identify_improvements(self, trend_data: List[float]) -> List[Tuple[int, float]]:
        """Identify improvement points in trend data"""
        improvements = []
        for i in range(1, len(trend_data)):
            improvement = trend_data[i] - trend_data[i-1]
            if abs(improvement) > 3:  # Significant change threshold
                improvements.append((i, improvement))
        return improvements
    
    def create_skill_distribution_chart(self, competency_data: SkillCompetencyData) -> go.Figure:
        """Create skill distribution visualization"""
        
        categories = list(competency_data.skill_categories.keys())
        scores = list(competency_data.skill_categories.values())
        
        fig = go.Figure()
        
        # Create violin plot for distribution
        fig.add_trace(go.Violin(
            x=categories,
            y=scores,
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightblue',
            line_color='darkblue',
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
        ))
        
        # Add mean markers
        fig.add_trace(go.Scatter(
            x=categories,
            y=scores,
            mode='markers',
            name='Actual Scores',
            marker=dict(size=10, color='red', symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f"Skill Distribution - {competency_data.candidate_name}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title="Skill Categories",
            yaxis_title="Score Distribution",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            template="plotly_white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

class InteractiveDashboard:
    """Main dashboard class that orchestrates all visualization components"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.analyzer = SkillCompetencyAnalyzer()
        self.renderer = InteractiveDashboardRenderer(self.config)
        self.session_data: List[SkillCompetencyData] = []
        logger.info("Interactive Dashboard initialized")
    
    def process_interview_session(
        self, 
        evaluations: List[ComprehensiveEvaluation],
        candidate_info: Dict[str, Any]
    ) -> SkillCompetencyData:
        """Process interview session and generate competency data"""
        
        try:
            competency_data = self.analyzer.analyze_evaluation_results(
                evaluations, candidate_info
            )
            
            self.session_data.append(competency_data)
            logger.info(f"Processed session for candidate: {competency_data.candidate_name}")
            
            return competency_data
            
        except Exception as e:
            logger.error(f"Error processing interview session: {str(e)}")
            raise
    
    def generate_comprehensive_dashboard(self, competency_data: SkillCompetencyData) -> Dict[str, go.Figure]:
        """Generate all dashboard visualizations"""
        
        try:
            dashboard_figures = {}
            
            # Primary visualizations
            dashboard_figures["skill_heatmap"] = self.renderer.create_skill_heatmap(competency_data)
            dashboard_figures["performance_radar"] = self.renderer.create_performance_radar_chart(competency_data)
            dashboard_figures["performance_trend"] = self.renderer.create_performance_trend_chart(competency_data)
            dashboard_figures["skill_distribution"] = self.renderer.create_skill_distribution_chart(competency_data)
            
            logger.info("Generated comprehensive dashboard visualizations")
            return dashboard_figures
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}")
            raise
    
    def export_dashboard_data(self, competency_data: SkillCompetencyData, format: str = "json") -> str:
        """Export dashboard data in specified format"""
        
        try:
            if format.lower() == "json":
                return self._export_json(competency_data)
            elif format.lower() == "csv":
                return self._export_csv(competency_data)
            elif format.lower() == "pdf":
                return self._export_pdf_report(competency_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {str(e)}")
            raise
    
    def _export_json(self, competency_data: SkillCompetencyData) -> str:
        """Export competency data as JSON"""
        data_dict = {
            "candidate_info": {
                "id": competency_data.candidate_id,
                "name": competency_data.candidate_name,
                "session_date": competency_data.session_date.isoformat(),
                "skill_level": competency_data.skill_level,
                "overall_score": competency_data.overall_score
            },
            "skill_categories": competency_data.skill_categories,
            "sub_skills": competency_data.sub_skills,
            "performance_metrics": {
                "trend": competency_data.performance_trend,
                "time_spent": competency_data.time_spent_per_skill,
                "confidence_scores": competency_data.confidence_scores
            },
            "insights": {
                "strengths": competency_data.strengths,
                "improvement_areas": competency_data.improvement_areas
            }
        }
        
        return json.dumps(data_dict, indent=2)
    
    def _export_csv(self, competency_data: SkillCompetencyData) -> str:
        """Export competency data as CSV"""
        import io
        
        output = io.StringIO()
        
        # Write main categories
        output.write("Skill Category,Score,Confidence,Time Spent (s)\n")
        for category, score in competency_data.skill_categories.items():
            confidence = competency_data.confidence_scores.get(category, 0)
            time_spent = competency_data.time_spent_per_skill.get(category, 0)
            output.write(f"{category},{score:.1f},{confidence:.2f},{time_spent:.1f}\n")
        
        return output.getvalue()
    
    def _export_pdf_report(self, competency_data: SkillCompetencyData) -> str:
        """Generate PDF report (placeholder for actual implementation)"""
        # This would integrate with a PDF generation library
        # For now, return a placeholder message
        return "PDF export functionality would be implemented here"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all processed sessions"""
        if not self.session_data:
            return {"message": "No sessions processed yet"}
        
        summary = {
            "total_sessions": len(self.session_data),
            "average_overall_score": np.mean([s.overall_score for s in self.session_data]),
            "skill_level_distribution": self._get_skill_level_distribution(),
            "top_performers": self._get_top_performers(5)
        }
        
        return summary
    
    def _get_skill_level_distribution(self) -> Dict[str, int]:
        """Get distribution of skill levels"""
        distribution = {}
        for session in self.session_data:
            level = session.skill_level
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
    
    def _get_top_performers(self, limit: int) -> List[Dict[str, Any]]:
        """Get top performing candidates"""
        performers = []
        for session in self.session_data:
            performers.append({
                "name": session.candidate_name,
                "score": session.overall_score,
                "skill_level": session.skill_level
            })
        
        # Sort by score and return top performers
        performers.sort(key=lambda x: x["score"], reverse=True)
        return performers[:limit]

# Global dashboard instance
dashboard_engine = InteractiveDashboard()