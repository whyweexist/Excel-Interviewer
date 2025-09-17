import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import json
from datetime import datetime, timedelta
import uuid

from ..utils.config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ChallengeType(Enum):
    DATA_ANALYSIS = "data_analysis"
    FORMULA_CREATION = "formula_creation"
    DATA_VISUALIZATION = "data_visualization"
    PROBLEM_SOLVING = "problem_solving"
    AUTOMATION = "automation"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class DatasetConfig:
    name: str
    rows: int
    columns: int
    data_types: Dict[str, str]
    business_context: str
    complexity_level: DifficultyLevel

@dataclass
class Challenge:
    id: str
    type: ChallengeType
    difficulty: DifficultyLevel
    title: str
    description: str
    dataset_config: DatasetConfig
    expected_solution: Dict[str, Any]
    evaluation_criteria: List[str]
    hints: List[str]
    time_limit: int  # in minutes
    scoring_weights: Dict[str, float]

@dataclass
class ChallengeResult:
    challenge_id: str
    user_solution: Dict[str, Any]
    correctness_score: float
    efficiency_score: float
    approach_score: float
    total_score: float
    feedback: str
    completed_at: datetime
    time_taken: int  # in seconds

class ExcelSimulator:
    def __init__(self):
        self.challenge_templates = self._load_challenge_templates()
        self.dataset_generators = self._initialize_dataset_generators()
        self.evaluation_engine = ChallengeEvaluationEngine()
        
    def _load_challenge_templates(self) -> Dict[ChallengeType, List[Dict]]:
        return {
            ChallengeType.DATA_ANALYSIS: [
                {
                    "title": "Sales Performance Analysis",
                    "description": "Analyze quarterly sales data to identify top performers and trends.",
                    "expected_solution": {
                        "pivot_table": True,
                        "charts": ["bar_chart", "line_chart"],
                        "formulas": ["SUMIF", "AVERAGE", "GROWTH_RATE"],
                        "insights": ["top_performers", "growth_trends", "seasonal_patterns"]
                    },
                    "evaluation_criteria": ["data_organization", "formula_accuracy", "chart_quality", "insight_depth"],
                    "hints": ["Use pivot tables to summarize data", "Create charts to visualize trends", "Calculate growth rates"],
                    "time_limit": 15,
                    "scoring_weights": {"correctness": 0.4, "efficiency": 0.3, "approach": 0.3}
                },
                {
                    "title": "Customer Segmentation Analysis",
                    "description": "Segment customers based on purchase behavior and demographics.",
                    "expected_solution": {
                        "clustering": True,
                        "formulas": ["IF", "VLOOKUP", "COUNTIF", "AVERAGEIF"],
                        "segments": ["high_value", "regular", "new", "at_risk"],
                        "metrics": ["avg_order_value", "purchase_frequency", "recency"]
                    },
                    "evaluation_criteria": ["segmentation_logic", "formula_complexity", "business_relevance", "data_quality"],
                    "hints": ["Use IF statements for segmentation", "Calculate customer metrics", "Create segment profiles"],
                    "time_limit": 20,
                    "scoring_weights": {"correctness": 0.5, "efficiency": 0.25, "approach": 0.25}
                }
            ],
            ChallengeType.FORMULA_CREATION: [
                {
                    "title": "Dynamic Commission Calculator",
                    "description": "Create a commission calculator with tiered rates and performance bonuses.",
                    "expected_solution": {
                        "formulas": ["IF", "VLOOKUP", "SUMPRODUCT", "MIN", "MAX"],
                        "features": ["tiered_rates", "bonus_calculation", "performance_tracking"],
                        "validation": ["input_validation", "error_handling"]
                    },
                    "evaluation_criteria": ["formula_accuracy", "logic_complexity", "error_handling", "user_friendliness"],
                    "hints": ["Use nested IF statements", "Implement tiered logic", "Add input validation"],
                    "time_limit": 12,
                    "scoring_weights": {"correctness": 0.6, "efficiency": 0.2, "approach": 0.2}
                },
                {
                    "title": "Inventory Management System",
                    "description": "Build formulas to track inventory levels, reorder points, and costs.",
                    "expected_solution": {
                        "formulas": ["SUMIF", "COUNTIF", "VLOOKUP", "IF", "DATEDIF"],
                        "tracking": ["stock_levels", "reorder_alerts", "cost_analysis"],
                        "automation": ["conditional_formatting", "data_validation"]
                    },
                    "evaluation_criteria": ["formula_integration", "business_logic", "automation_features", "data_consistency"],
                    "hints": ["Use SUMIF for conditional sums", "Implement reorder point logic", "Add conditional formatting"],
                    "time_limit": 18,
                    "scoring_weights": {"correctness": 0.5, "efficiency": 0.3, "approach": 0.2}
                }
            ],
            ChallengeType.DATA_VISUALIZATION: [
                {
                    "title": "Executive Dashboard Creation",
                    "description": "Create an interactive dashboard for executive reporting with key metrics.",
                    "expected_solution": {
                        "charts": ["dashboard", "kpi_cards", "trend_charts", "comparison_charts"],
                        "interactivity": ["slicers", "timelines", "drill_down"],
                        "formatting": ["professional_design", "color_scheme", "layout"]
                    },
                    "evaluation_criteria": ["visual_clarity", "interactivity", "professional_design", "insight_communication"],
                    "hints": ["Use consistent color schemes", "Create interactive elements", "Focus on key metrics"],
                    "time_limit": 25,
                    "scoring_weights": {"correctness": 0.3, "efficiency": 0.3, "approach": 0.4}
                }
            ],
            ChallengeType.PROBLEM_SOLVING: [
                {
                    "title": "Data Quality Issues Resolution",
                    "description": "Identify and fix data quality issues in a messy dataset.",
                    "expected_solution": {
                        "data_cleaning": ["remove_duplicates", "handle_missing_values", "standardize_formats"],
                        "validation": ["data_validation", "error_identification", "quality_checks"],
                        "reporting": ["quality_metrics", "issue_summary", "resolution_tracking"]
                    },
                    "evaluation_criteria": ["problem_identification", "solution_effectiveness", "thoroughness", "documentation"],
                    "hints": ["Look for inconsistencies", "Use data validation tools", "Document your process"],
                    "time_limit": 20,
                    "scoring_weights": {"correctness": 0.4, "efficiency": 0.3, "approach": 0.3}
                }
            ],
            ChallengeType.AUTOMATION: [
                {
                    "title": "Monthly Report Automation",
                    "description": "Automate the generation of monthly performance reports.",
                    "expected_solution": {
                        "automation": ["macros", "power_query", "scheduled_refresh"],
                        "templates": ["report_template", "formatting_automation"],
                        "distribution": ["export_automation", "email_integration"]
                    },
                    "evaluation_criteria": ["automation_efficiency", "reliability", "scalability", "error_handling"],
                    "hints": ["Use macros for repetitive tasks", "Create reusable templates", "Implement error handling"],
                    "time_limit": 30,
                    "scoring_weights": {"correctness": 0.4, "efficiency": 0.4, "approach": 0.2}
                }
            ]
        }
    
    def _initialize_dataset_generators(self) -> Dict[str, Any]:
        return {
            "sales_data": self._generate_sales_data,
            "customer_data": self._generate_customer_data,
            "inventory_data": self._generate_inventory_data,
            "financial_data": self._generate_financial_data,
            "employee_data": self._generate_employee_data
        }
    
    def generate_challenge(self, challenge_type: ChallengeType, difficulty: DifficultyLevel, 
                          context: Optional[Dict] = None) -> Challenge:
        try:
            # Select appropriate template
            templates = self.challenge_templates.get(challenge_type, [])
            if not templates:
                raise ValueError(f"No templates available for challenge type: {challenge_type}")
            
            template = random.choice(templates)
            
            # Generate dataset configuration
            dataset_config = self._generate_dataset_config(difficulty, template["title"])
            
            # Create challenge
            challenge = Challenge(
                id=str(uuid.uuid4()),
                type=challenge_type,
                difficulty=difficulty,
                title=template["title"],
                description=template["description"],
                dataset_config=dataset_config,
                expected_solution=template["expected_solution"],
                evaluation_criteria=template["evaluation_criteria"],
                hints=template["hints"],
                time_limit=template["time_limit"],
                scoring_weights=template["scoring_weights"]
            )
            
            logger.info(f"Generated challenge: {challenge.title} (ID: {challenge.id})")
            return challenge
            
        except Exception as e:
            logger.error(f"Error generating challenge: {str(e)}")
            raise Exception(f"Failed to generate challenge: {str(e)}")
    
    def generate_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        try:
            generator_func = self.dataset_generators.get(dataset_config.name.lower().replace(" ", "_"))
            if generator_func:
                return generator_func(dataset_config)
            else:
                return self._generate_generic_dataset(dataset_config)
                
        except Exception as e:
            logger.error(f"Error generating dataset: {str(e)}")
            raise Exception(f"Failed to generate dataset: {str(e)}")
    
    def evaluate_challenge_solution(self, challenge: Challenge, user_solution: Dict[str, Any]) -> ChallengeResult:
        try:
            return self.evaluation_engine.evaluate_solution(challenge, user_solution)
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            raise Exception(f"Failed to evaluate solution: {str(e)}")
    
    def _generate_dataset_config(self, difficulty: DifficultyLevel, context: str) -> DatasetConfig:
        base_configs = {
            DifficultyLevel.BEGINNER: {
                "rows": (50, 200),
                "columns": (3, 6),
                "complexity": "simple"
            },
            DifficultyLevel.INTERMEDIATE: {
                "rows": (200, 1000),
                "columns": (5, 12),
                "complexity": "moderate"
            },
            DifficultyLevel.ADVANCED: {
                "rows": (1000, 5000),
                "columns": (8, 20),
                "complexity": "complex"
            }
        }
        
        config = base_configs[difficulty]
        
        # Select appropriate dataset type based on context
        if "sales" in context.lower():
            dataset_name = "sales_data"
            data_types = {
                "date": "datetime",
                "product": "string",
                "quantity": "integer",
                "price": "float",
                "region": "string",
                "salesperson": "string"
            }
            business_context = "Sales performance tracking and analysis"
            
        elif "customer" in context.lower():
            dataset_name = "customer_data"
            data_types = {
                "customer_id": "string",
                "name": "string",
                "email": "string",
                "signup_date": "datetime",
                "total_spent": "float",
                "purchase_count": "integer"
            }
            business_context = "Customer behavior and segmentation analysis"
            
        else:
            dataset_name = "generic_business_data"
            data_types = {
                "id": "integer",
                "category": "string",
                "value": "float",
                "date": "datetime",
                "status": "string"
            }
            business_context = "General business data analysis"
        
        # Adjust data types based on difficulty
        if difficulty == DifficultyLevel.ADVANCED:
            data_types.update({
                "calculated_field": "formula",
                "reference_data": "lookup"
            })
        
        return DatasetConfig(
            name=dataset_name,
            rows=random.randint(*config["rows"]),
            columns=len(data_types),
            data_types=data_types,
            business_context=business_context,
            complexity_level=difficulty
        )
    
    def _generate_sales_data(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime.now() - timedelta(days=365)
        dates = pd.date_range(start=start_date, periods=config.rows, freq='D')
        
        data = {}
        
        for col, dtype in config.data_types.items():
            if dtype == "datetime":
                data[col] = np.random.choice(dates, config.rows)
            elif dtype == "string" and col == "product":
                products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
                data[col] = np.random.choice(products, config.rows)
            elif dtype == "string" and col == "region":
                regions = ["North", "South", "East", "West", "Central"]
                data[col] = np.random.choice(regions, config.rows)
            elif dtype == "string" and col == "salesperson":
                names = ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Wilson", "David Brown"]
                data[col] = np.random.choice(names, config.rows)
            elif dtype == "integer":
                data[col] = np.random.randint(1, 100, config.rows)
            elif dtype == "float":
                data[col] = np.round(np.random.uniform(10, 1000, config.rows), 2)
        
        return pd.DataFrame(data)
    
    def _generate_customer_data(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        data = {}
        
        for col, dtype in config.data_types.items():
            if col == "customer_id":
                data[col] = [f"CUST_{i:06d}" for i in range(config.rows)]
            elif col == "name":
                first_names = ["John", "Jane", "Mike", "Sarah", "David", "Lisa", "Tom", "Emily", "Chris", "Anna"]
                last_names = ["Smith", "Doe", "Johnson", "Wilson", "Brown", "Davis", "Miller", "Garcia", "Martinez", "Anderson"]
                data[col] = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(config.rows)]
            elif col == "email":
                domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
                data[col] = [f"user{i}@{np.random.choice(domains)}" for i in range(config.rows)]
            elif col == "signup_date":
                start_date = datetime.now() - timedelta(days=730)
                dates = pd.date_range(start=start_date, periods=config.rows, freq='D')
                data[col] = np.random.choice(dates, config.rows)
            elif col == "total_spent":
                data[col] = np.round(np.random.exponential(500, config.rows), 2)
            elif col == "purchase_count":
                data[col] = np.random.poisson(5, config.rows)
        
        return pd.DataFrame(data)
    
    def _generate_inventory_data(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        data = {
            "product_id": [f"PROD_{i:05d}" for i in range(config.rows)],
            "product_name": [f"Product {i}" for i in range(config.rows)],
            "current_stock": np.random.randint(0, 1000, config.rows),
            "reorder_point": np.random.randint(10, 100, config.rows),
            "unit_cost": np.round(np.random.uniform(5, 500, config.rows), 2),
            "supplier": np.random.choice(["Supplier A", "Supplier B", "Supplier C"], config.rows),
            "last_order_date": pd.date_range(start=datetime.now() - timedelta(days=90), periods=config.rows, freq='D')
        }
        
        return pd.DataFrame(data)
    
    def _generate_financial_data(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        data = {
            "date": pd.date_range(start=datetime.now() - timedelta(days=365), periods=config.rows, freq='M'),
            "revenue": np.round(np.random.normal(100000, 20000, config.rows), 2),
            "expenses": np.round(np.random.normal(80000, 15000, config.rows), 2),
            "profit_margin": np.round(np.random.uniform(0.1, 0.3, config.rows), 3),
            "cash_flow": np.round(np.random.normal(50000, 10000, config.rows), 2)
        }
        
        return pd.DataFrame(data)
    
    def _generate_employee_data(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        departments = ["Sales", "Marketing", "IT", "HR", "Finance", "Operations"]
        positions = ["Manager", "Senior", "Junior", "Intern"]
        
        data = {
            "employee_id": [f"EMP_{i:05d}" for i in range(config.rows)],
            "name": [f"Employee {i}" for i in range(config.rows)],
            "department": np.random.choice(departments, config.rows),
            "position": np.random.choice(positions, config.rows),
            "salary": np.round(np.random.normal(60000, 15000, config.rows), 2),
            "hire_date": pd.date_range(start=datetime.now() - timedelta(days=1825), periods=config.rows, freq='M'),
            "performance_score": np.round(np.random.uniform(2.5, 5.0, config.rows), 1)
        }
        
        return pd.DataFrame(data)
    
    def _generate_generic_dataset(self, config: DatasetConfig) -> pd.DataFrame:
        np.random.seed(42)
        
        data = {}
        
        for col, dtype in config.data_types.items():
            if dtype == "datetime":
                data[col] = pd.date_range(start=datetime.now() - timedelta(days=365), periods=config.rows, freq='D')
            elif dtype == "string":
                if col == "category":
                    categories = ["A", "B", "C", "D", "E"]
                    data[col] = np.random.choice(categories, config.rows)
                elif col == "status":
                    statuses = ["Active", "Inactive", "Pending", "Complete"]
                    data[col] = np.random.choice(statuses, config.rows)
                else:
                    data[col] = [f"{col}_{i}" for i in range(config.rows)]
            elif dtype == "integer":
                data[col] = np.random.randint(1, 1000, config.rows)
            elif dtype == "float":
                data[col] = np.round(np.random.uniform(1, 1000, config.rows), 2)
        
        return pd.DataFrame(data)

class ChallengeEvaluationEngine:
    def __init__(self):
        self.evaluation_criteria = {
            "correctness": self._evaluate_correctness,
            "efficiency": self._evaluate_efficiency,
            "approach": self._evaluate_approach
        }
    
    def evaluate_solution(self, challenge: Challenge, user_solution: Dict[str, Any]) -> ChallengeResult:
        try:
            correctness_score = self._evaluate_correctness(challenge, user_solution)
            efficiency_score = self._evaluate_efficiency(challenge, user_solution)
            approach_score = self._evaluate_approach(challenge, user_solution)
            
            # Calculate weighted total score
            weights = challenge.scoring_weights
            total_score = (
                correctness_score * weights["correctness"] +
                efficiency_score * weights["efficiency"] +
                approach_score * weights["approach"]
            )
            
            feedback = self._generate_feedback(correctness_score, efficiency_score, approach_score, total_score)
            
            return ChallengeResult(
                challenge_id=challenge.id,
                user_solution=user_solution,
                correctness_score=correctness_score,
                efficiency_score=efficiency_score,
                approach_score=approach_score,
                total_score=total_score,
                feedback=feedback,
                completed_at=datetime.now(),
                time_taken=user_solution.get("time_taken", 0)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            raise Exception(f"Failed to evaluate solution: {str(e)}")
    
    def _evaluate_correctness(self, challenge: Challenge, user_solution: Dict[str, Any]) -> float:
        expected = challenge.expected_solution
        user = user_solution
        
        score = 0.0
        total_checks = 0
        
        # Check formulas
        if "formulas" in expected:
            total_checks += len(expected["formulas"])
            user_formulas = user.get("formulas", [])
            for formula in expected["formulas"]:
                if any(formula.lower() in uf.lower() for uf in user_formulas):
                    score += 1.0
        
        # Check features
        if "features" in expected:
            total_checks += len(expected["features"])
            user_features = user.get("features", [])
            for feature in expected["features"]:
                if feature in user_features:
                    score += 1.0
        
        # Check charts
        if "charts" in expected:
            total_checks += len(expected["charts"])
            user_charts = user.get("charts", [])
            for chart in expected["charts"]:
                if chart in user_charts:
                    score += 1.0
        
        return (score / total_checks) * 100 if total_checks > 0 else 0.0
    
    def _evaluate_efficiency(self, challenge: Challenge, user_solution: Dict[str, Any]) -> float:
        # Evaluate efficiency based on solution approach and time taken
        time_taken = user_solution.get("time_taken", challenge.time_limit * 60)  # Convert to seconds
        time_limit_seconds = challenge.time_limit * 60
        
        # Time efficiency (faster is better, but with diminishing returns)
        if time_taken <= time_limit_seconds:
            time_score = 100 - (time_taken / time_limit_seconds) * 50  # 50-100 range
        else:
            time_score = max(0, 100 - ((time_taken - time_limit_seconds) / time_limit_seconds) * 100)
        
        # Solution efficiency (based on approach complexity)
        approach_complexity = len(user_solution.get("formulas", [])) + len(user_solution.get("features", []))
        expected_complexity = len(challenge.expected_solution.get("formulas", [])) + len(challenge.expected_solution.get("features", []))
        
        if expected_complexity > 0:
            complexity_score = min(100, (approach_complexity / expected_complexity) * 100)
        else:
            complexity_score = 50  # Neutral score
        
        # Combine scores (60% time, 40% complexity)
        return (time_score * 0.6) + (complexity_score * 0.4)
    
    def _evaluate_approach(self, challenge: Challenge, user_solution: Dict[str, Any]) -> float:
        # Evaluate problem-solving approach
        score = 0.0
        
        # Check if user provided explanation of their approach
        if "approach_explanation" in user_solution:
            score += 20.0
        
        # Check for systematic approach (step-by-step)
        if "steps" in user_solution and len(user_solution["steps"]) > 0:
            score += 30.0
        
        # Check for error handling
        if "error_handling" in user_solution:
            score += 25.0
        
        # Check for documentation/commenting
        if "documentation" in user_solution:
            score += 25.0
        
        return min(100.0, score)
    
    def _generate_feedback(self, correctness: float, efficiency: float, approach: float, total: float) -> str:
        feedback_parts = []
        
        if correctness >= 80:
            feedback_parts.append("Excellent technical accuracy!")
        elif correctness >= 60:
            feedback_parts.append("Good technical understanding with room for improvement.")
        else:
            feedback_parts.append("Technical accuracy needs significant improvement.")
        
        if efficiency >= 80:
            feedback_parts.append("Very efficient solution approach.")
        elif efficiency >= 60:
            feedback_parts.append("Reasonably efficient with some optimization opportunities.")
        else:
            feedback_parts.append("Solution efficiency could be improved.")
        
        if approach >= 80:
            feedback_parts.append("Strong problem-solving methodology.")
        elif approach >= 60:
            feedback_parts.append("Good problem-solving approach with minor gaps.")
        else:
            feedback_parts.append("Problem-solving approach needs development.")
        
        overall_grade = "Excellent" if total >= 90 else "Good" if total >= 75 else "Average" if total >= 60 else "Needs Improvement"
        feedback_parts.append(f"Overall: {overall_grade} ({total:.1f}/100)")
        
        return " ".join(feedback_parts)