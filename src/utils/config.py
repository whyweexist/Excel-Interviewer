import os
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    """Configuration management for the Excel Interviewer system."""
    
    # Application settings
    app_name: str = Field(default="Excel Interviewer AI", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")  # ✅ added
    
    # OpenAI Configuration
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    
    # Interview Configuration
    max_interview_duration_minutes: int = Field(default=45, env="MAX_INTERVIEW_DURATION_MINUTES")
    min_questions_per_topic: int = Field(default=3, env="MIN_QUESTIONS_PER_TOPIC")
    max_questions_per_topic: int = Field(default=8, env="MAX_QUESTIONS_PER_TOPIC")
    adaptive_difficulty_enabled: bool = Field(default=True, env="ADAPTIVE_DIFFICULTY_ENABLED")
    
    # Evaluation Configuration
    technical_weight: float = Field(default=0.4, env="TECHNICAL_WEIGHT")
    problem_solving_weight: float = Field(default=0.3, env="PROBLEM_SOLVING_WEIGHT")
    communication_weight: float = Field(default=0.2, env="COMMUNICATION_WEIGHT")
    practical_weight: float = Field(default=0.1, env="PRACTICAL_WEIGHT")
    min_passing_score: float = Field(default=70.0, env="MIN_PASSING_SCORE")
    
    # # Database Configuration (optional)
    # database_url: str = Field(default="sqlite:///excel_interviewer.db", env="DATABASE_URL")
    # redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Dashboard Configuration
    dashboard_theme: str = Field(default="light", env="DASHBOARD_THEME")
    enable_real_time_updates: bool = Field(default=True, env="ENABLE_REAL_TIME_UPDATES")
    chart_animation_duration: int = Field(default=1000, env="CHART_ANIMATION_DURATION")
    
    # Gamification Configuration
    enable_achievements: bool = Field(default=True, env="ENABLE_ACHIEVEMENTS")
    enable_leaderboard: bool = Field(default=True, env="ENABLE_LEADERBOARD")
    points_per_correct_answer: int = Field(default=10, env="POINTS_PER_CORRECT_ANSWER")
    points_per_bonus_question: int = Field(default=20, env="POINTS_PER_BONUS_QUESTION")
    
    # Security Configuration
    session_timeout_minutes: int = Field(default=30, env="SESSION_TIMEOUT_MINUTES")
    max_login_attempts: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
    encrypt_sensitive_data: bool = Field(default=True, env="ENCRYPT_SENSITIVE_DATA")
    
    # Performance Configuration
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    max_concurrent_interviews: int = Field(default=10, env="MAX_CONCURRENT_INTERVIEWS")
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_evaluation_weights(self) -> Dict[str, float]:
        return {
            "technical": self.technical_weight,
            "problem_solving": self.problem_solving_weight,
            "communication": self.communication_weight,
            "practical": self.practical_weight,
        }
    
    def get_interview_config(self) -> Dict[str, Any]:
        return {
            "max_duration_minutes": self.max_interview_duration_minutes,
            "min_questions_per_topic": self.min_questions_per_topic,
            "max_questions_per_topic": self.max_questions_per_topic,
            "adaptive_difficulty_enabled": self.adaptive_difficulty_enabled,
        }
    
    def validate_openai_config(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
    
    def get_scoring_thresholds(self) -> Dict[str, float]:
        return {
            "excellent": 90.0,
            "good": 80.0,
            "satisfactory": 70.0,
            "needs_improvement": 60.0,
            "poor": 0.0,
        }

# ✅ singleton instance
settings = Config()

# import os
# from typing import Dict, Any, Optional
# from pydantic_settings import BaseSettings
# from pydantic import Field
# from dotenv import load_dotenv

# load_dotenv()

# class Config(BaseSettings):
#     """Configuration management for the Excel Interviewer system."""
    
#     # Application settings
#     app_name: str = Field(default="Excel Interviewer AI", env="APP_NAME")
#     app_version: str = Field(default="1.0.0", env="APP_VERSION")
#     debug_mode: bool = Field(default=False, env="DEBUG_MODE")
#     log_level: str = Field(default="INFO", env="LOG_LEVEL")
#     LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
#     # OpenAI Configuration
#     openai_api_key: str = Field(env="OPENAI_API_KEY")
#     openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
#     openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
#     openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    
#     # Interview Configuration
#     max_interview_duration_minutes: int = Field(default=45, env="MAX_INTERVIEW_DURATION_MINUTES")
#     min_questions_per_topic: int = Field(default=3, env="MIN_QUESTIONS_PER_TOPIC")
#     max_questions_per_topic: int = Field(default=8, env="MAX_QUESTIONS_PER_TOPIC")
#     adaptive_difficulty_enabled: bool = Field(default=True, env="ADAPTIVE_DIFFICULTY_ENABLED")
    
#     # Evaluation Configuration
#     technical_weight: float = Field(default=0.4, env="TECHNICAL_WEIGHT")
#     problem_solving_weight: float = Field(default=0.3, env="PROBLEM_SOLVING_WEIGHT")
#     communication_weight: float = Field(default=0.2, env="COMMUNICATION_WEIGHT")
#     practical_weight: float = Field(default=0.1, env="PRACTICAL_WEIGHT")
#     min_passing_score: float = Field(default=70.0, env="MIN_PASSING_SCORE")
    
#     # # Database Configuration
#     # database_url: str = Field(default="sqlite:///excel_interviewer.db", env="DATABASE_URL")
#     # redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
#     # Dashboard Configuration
#     dashboard_theme: str = Field(default="light", env="DASHBOARD_THEME")
#     enable_real_time_updates: bool = Field(default=True, env="ENABLE_REAL_TIME_UPDATES")
#     chart_animation_duration: int = Field(default=1000, env="CHART_ANIMATION_DURATION")
    
#     # Gamification Configuration
#     enable_achievements: bool = Field(default=True, env="ENABLE_ACHIEVEMENTS")
#     enable_leaderboard: bool = Field(default=True, env="ENABLE_LEADERBOARD")
#     points_per_correct_answer: int = Field(default=10, env="POINTS_PER_CORRECT_ANSWER")
#     points_per_bonus_question: int = Field(default=20, env="POINTS_PER_BONUS_QUESTION")
    
#     # Security Configuration
#     session_timeout_minutes: int = Field(default=30, env="SESSION_TIMEOUT_MINUTES")
#     max_login_attempts: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
#     encrypt_sensitive_data: bool = Field(default=True, env="ENCRYPT_SENSITIVE_DATA")
    
#     # Performance Configuration
#     cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
#     max_concurrent_interviews: int = Field(default=10, env="MAX_CONCURRENT_INTERVIEWS")
#     rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
#     class Config:
#         env_file = ".env"
#         case_sensitive = False
    
#     def get_evaluation_weights(self) -> Dict[str, float]:
#         """Get evaluation weights as a dictionary."""
#         return {
#             "technical": self.technical_weight,
#             "problem_solving": self.problem_solving_weight,
#             "communication": self.communication_weight,
#             "practical": self.practical_weight
#         }
    
#     def get_interview_config(self) -> Dict[str, Any]:
#         """Get interview configuration as a dictionary."""
#         return {
#             "max_duration_minutes": self.max_interview_duration_minutes,
#             "min_questions_per_topic": self.min_questions_per_topic,
#             "max_questions_per_topic": self.max_questions_per_topic,
#             "adaptive_difficulty_enabled": self.adaptive_difficulty_enabled
#         }
    
#     # def LOG_FILE(self) -> str:
#     #     return self.log_file
    
#     def validate_openai_config(self) -> bool:
#         """Validate OpenAI configuration."""
#         if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
#             return False
#         return True
    
#     def get_scoring_thresholds(self) -> Dict[str, float]:
#         """Get scoring thresholds for different performance levels."""
#         return {
#             "excellent": 90.0,
#             "good": 80.0,
#             "satisfactory": 70.0,
#             "needs_improvement": 60.0,
#             "poor": 0.0
#         }
# settings = Config()