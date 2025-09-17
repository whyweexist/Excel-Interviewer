"""
Interactive Dashboard Package for AI Excel Interviewer

This package provides comprehensive skill competency visualization and analytics
capabilities for the AI-powered Excel interview system.
"""

from .dashboard_engine import (
    InteractiveDashboard,
    SkillCompetencyAnalyzer,
    InteractiveDashboardRenderer,
    SkillCompetencyData,
    DashboardConfig,
    VisualizationType,
    SkillCategory
)

from .dashboard_ui import (
    DashboardUI,
    render_dashboard,
    render_skill_heatmap,
    render_performance_radar,
    render_trend_analysis,
    render_skill_distribution
)

__all__ = [
    # Core Engine Components
    'InteractiveDashboard',
    'SkillCompetencyAnalyzer', 
    'InteractiveDashboardRenderer',
    'SkillCompetencyData',
    'DashboardConfig',
    'VisualizationType',
    'SkillCategory',
    
    # UI Components
    'DashboardUI',
    'render_dashboard',
    'render_skill_heatmap',
    'render_performance_radar',
    'render_trend_analysis',
    'render_skill_distribution'
]