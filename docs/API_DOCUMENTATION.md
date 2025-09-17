# 📋 API Documentation
## AI-Powered Excel Interviewer System

### Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Interview API](#interview-api)
4. [Evaluation API](#evaluation-api)
5. [Dashboard API](#dashboard-api)
6. [Learning System API](#learning-system-api)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [WebSocket API](#websocket-api)

---

## Overview

The AI Interviewer API provides RESTful endpoints for managing interview sessions, evaluating candidates, and accessing analytics. All API responses follow a consistent JSON format with appropriate HTTP status codes.

### Base URL
```
https://api.ai-interviewer.com/v1
```

### Response Format
All API responses follow this structure:
```json
{
  "success": boolean,
  "data": object | array,
  "message": string,
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-v4"
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Too Many Requests
- `500` - Internal Server Error

---

## Authentication

### API Key Authentication
Include your API key in the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```

### JWT Token Authentication
For user-specific operations, use JWT tokens:
```
Authorization: Bearer JWT_TOKEN
```

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

---

## Interview API

### Start Interview Session

**POST** `/interviews/start`

Start a new interview session for a candidate.

**Request Body:**
```json
{
  "candidate_id": "string",
  "position_level": "junior|mid|senior|expert",
  "focus_areas": ["formulas", "data_analysis", "visualization", "automation"],
  "difficulty_preference": "adaptive|easy|medium|hard",
  "session_config": {
    "max_questions": 10,
    "time_limit_minutes": 45,
    "allow_hints": true,
    "show_progress": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "sess_123456789",
    "conversation_state": {
      "current_phase": "introduction",
      "difficulty_level": "medium",
      "questions_asked": 0,
      "time_remaining": 2700
    },
    "first_question": {
      "question_id": "q_001",
      "text": "Can you describe your experience with Excel formulas?",
      "type": "open_ended",
      "difficulty": "medium"
    }
  },
  "message": "Interview session started successfully"
}
```

### Submit Response

**POST** `/interviews/{session_id}/respond`

Submit a candidate's response to the current question.

**Request Body:**
```json
{
  "response_text": "string",
  "response_type": "text|code|file_upload",
  "metadata": {
    "response_time_seconds": 45,
    "confidence_level": 8,
    "additional_notes": "string"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "evaluation": {
      "overall_score": 7.5,
      "dimension_scores": {
        "technical_accuracy": 8.0,
        "problem_solving": 7.0,
        "communication_clarity": 8.5,
        "practical_application": 7.0,
        "excel_expertise": 7.5
      },
      "strengths": ["Clear explanation", "Good technical knowledge"],
      "improvement_areas": ["Could provide more examples"]
    },
    "next_question": {
      "question_id": "q_002",
      "text": "Follow-up: Can you give a specific example of a complex formula you've created?",
      "type": "follow_up",
      "difficulty": "medium"
    },
    "conversation_update": {
      "questions_asked": 1,
      "session_progress": 10,
      "difficulty_adjustment": "maintained"
    }
  },
  "message": "Response evaluated successfully"
}
```

### Get Session Status

**GET** `/interviews/{session_id}/status`

Get the current status of an interview session.

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "sess_123456789",
    "status": "active|paused|completed|expired",
    "progress": {
      "questions_asked": 5,
      "total_questions": 10,
      "time_elapsed": 1200,
      "time_remaining": 1500
    },
    "current_state": {
      "phase": "technical_assessment",
      "difficulty": "medium",
      "focus_area": "formulas"
    },
    "performance_summary": {
      "average_score": 7.2,
      "consistency": 0.85,
      "improvement_trend": "positive"
    }
  }
}
```

### End Interview Session

**POST** `/interviews/{session_id}/end`

End an interview session and generate final report.

**Request Body:**
```json
{
  "end_reason": "completed|timeout|interrupted|abandoned",
  "final_notes": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "final_report": {
      "session_summary": {
        "total_questions": 8,
        "average_score": 7.3,
        "time_taken": 1800,
        "completion_rate": 1.0
      },
      "skill_assessment": {
        "formulas": 7.5,
        "data_analysis": 7.0,
        "visualization": 6.5,
        "automation": 8.0
      },
      "recommendations": [
        "Strong in automation and formulas",
        "Could improve data visualization skills"
      ]
    },
    "candidate_profile": {
      "overall_rating": "proficient",
      "skill_level": "intermediate",
      "suitable_roles": ["Data Analyst", "Business Analyst"]
    }
  },
  "message": "Interview session ended successfully"
}
```

---

## Evaluation API

### Get Evaluation Details

**GET** `/evaluations/{evaluation_id}`

Get detailed evaluation results for a specific response.

**Response:**
```json
{
  "success": true,
  "data": {
    "evaluation_id": "eval_123456789",
    "response_id": "resp_987654321",
    "scores": {
      "technical_accuracy": {
        "score": 8.0,
        "max_score": 10,
        "criteria": ["Correct formula usage", "Proper syntax", "Efficient approach"]
      },
      "problem_solving": {
        "score": 7.0,
        "max_score": 10,
        "criteria": ["Logical thinking", "Systematic approach", "Solution completeness"]
      },
      "communication_clarity": {
        "score": 8.5,
        "max_score": 10,
        "criteria": ["Clear explanation", "Structured response", "Appropriate detail"]
      }
    },
    "detailed_feedback": {
      "strengths": [
        "Demonstrated good understanding of VLOOKUP function",
        "Provided clear step-by-step explanation"
      ],
      "improvements": [
        "Consider mentioning error handling",
        "Could discuss alternative approaches"
      ],
      "suggestions": [
        "Practice with more complex scenarios",
        "Explore advanced Excel features"
      ]
    },
    "confidence_metrics": {
      "overall_confidence": 0.82,
      "score_reliability": 0.89,
      "evaluation_completeness": 0.95
    }
  }
}
```

### Batch Evaluation

**POST** `/evaluations/batch`

Evaluate multiple responses in a single request.

**Request Body:**
```json
{
  "responses": [
    {
      "response_id": "resp_001",
      "response_text": "Sample response 1",
      "question_context": "Context for question 1"
    },
    {
      "response_id": "resp_002",
      "response_text": "Sample response 2",
      "question_context": "Context for question 2"
    }
  ],
  "evaluation_config": {
    "include_feedback": true,
    "detailed_analysis": true,
    "consistency_check": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_123456789",
    "evaluations": [
      {
        "response_id": "resp_001",
        "overall_score": 7.5,
        "evaluation_time_ms": 250,
        "status": "completed"
      },
      {
        "response_id": "resp_002",
        "overall_score": 8.0,
        "evaluation_time_ms": 275,
        "status": "completed"
      }
    ],
    "batch_summary": {
      "total_evaluated": 2,
      "average_score": 7.75,
      "processing_time_ms": 525
    }
  }
}
```

---

## Dashboard API

### Get Dashboard Data

**GET** `/dashboard/summary`

Get comprehensive dashboard data for analytics.

**Query Parameters:**
- `time_range`: 7d|30d|90d|1y (default: 30d)
- `candidate_level`: all|junior|mid|senior|expert
- `focus_area`: all|formulas|data_analysis|visualization|automation

**Response:**
```json
{
  "success": true,
  "data": {
    "overview_metrics": {
      "total_interviews": 156,
      "average_score": 7.2,
      "completion_rate": 0.89,
      "average_duration_minutes": 32
    },
    "skill_distribution": {
      "formulas": {
        "average": 7.5,
        "trend": "improving",
        "distribution": {
          "excellent": 25,
          "good": 45,
          "average": 30,
          "needs_improvement": 15
        }
      },
      "data_analysis": {
        "average": 6.8,
        "trend": "stable",
        "distribution": {
          "excellent": 20,
          "good": 35,
          "average": 40,
          "needs_improvement": 25
        }
      }
    },
    "performance_trends": [
      {
        "date": "2024-01-01",
        "average_score": 7.0,
        "total_interviews": 12
      },
      {
        "date": "2024-01-02",
        "average_score": 7.3,
        "total_interviews": 15
      }
    ],
    "competency_heatmap": {
      "junior": {
        "formulas": 6.5,
        "data_analysis": 5.8,
        "visualization": 6.2,
        "automation": 4.5
      },
      "mid": {
        "formulas": 7.2,
        "data_analysis": 6.9,
        "visualization": 7.1,
        "automation": 6.8
      },
      "senior": {
        "formulas": 8.1,
        "data_analysis": 8.3,
        "visualization": 7.8,
        "automation": 8.5
      }
    }
  }
}
```

### Get Candidate Analytics

**GET** `/dashboard/candidates/{candidate_id}`

Get detailed analytics for a specific candidate.

**Response:**
```json
{
  "success": true,
  "data": {
    "candidate_profile": {
      "candidate_id": "cand_123456789",
      "name": "John Doe",
      "email": "john.doe@example.com",
      "total_interviews": 3,
      "average_score": 7.8,
      "skill_level": "intermediate"
    },
    "interview_history": [
      {
        "session_id": "sess_001",
        "date": "2024-01-01",
        "score": 7.5,
        "duration_minutes": 35,
        "focus_areas": ["formulas", "data_analysis"]
      },
      {
        "session_id": "sess_002",
        "date": "2024-01-15",
        "score": 8.1,
        "duration_minutes": 28,
        "focus_areas": ["visualization", "automation"]
      }
    ],
    "skill_progression": {
      "formulas": {
        "initial_score": 6.5,
        "current_score": 8.0,
        "improvement": 1.5,
        "trend": "improving"
      },
      "data_analysis": {
        "initial_score": 7.0,
        "current_score": 7.8,
        "improvement": 0.8,
        "trend": "improving"
      }
    },
    "competency_radar": {
      "formulas": 8.0,
      "data_analysis": 7.8,
      "visualization": 7.2,
      "automation": 6.5,
      "problem_solving": 8.1
    }
  }
}
```

### Export Dashboard Data

**POST** `/dashboard/export`

Export dashboard data in various formats.

**Request Body:**
```json
{
  "export_format": "csv|json|pdf|excel",
  "data_types": ["interviews", "evaluations", "analytics", "reports"],
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "filters": {
    "candidate_level": "all",
    "focus_area": "all",
    "min_score": 0,
    "max_score": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "export_id": "export_123456789",
    "download_url": "https://api.ai-interviewer.com/downloads/export_123456789.csv",
    "expires_at": "2024-01-02T00:00:00Z",
    "file_info": {
      "format": "csv",
      "size_bytes": 1048576,
      "rows_exported": 156
    }
  }
}
```

---

## Learning System API

### Trigger Active Learning

**POST** `/learning/active-learning/trigger`

Trigger the active learning system to select new samples.

**Request Body:**
```json
{
  "strategy": "uncertainty_sampling|diversity_sampling|hybrid_strategy",
  "max_samples": 20,
  "selection_criteria": {
    "min_uncertainty": 0.3,
    "min_diversity_score": 0.6,
    "focus_areas": ["formulas", "automation"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "selection_id": "sel_123456789",
    "selected_samples": [
      {
        "sample_id": "sample_001",
        "session_id": "sess_001",
        "strategic_value": 0.85,
        "uncertainty": 0.42,
        "diversity_score": 0.73,
        "query_type": "knowledge_probe",
        "selection_reason": "High uncertainty in formula evaluation"
      }
    ],
    "summary": {
      "total_selected": 15,
      "average_strategic_value": 0.78,
      "strategy_used": "hybrid_strategy"
    }
  }
}
```

### Get Learning Insights

**GET** `/learning/insights`

Get insights from the learning system.

**Query Parameters:**
- `insight_type`: performance|adaptation|recommendations|all
- `time_period`: 7d|30d|90d|1y

**Response:**
```json
{
  "success": true,
  "data": {
    "performance_insights": {
      "model_accuracy_improvement": 0.08,
      "evaluation_consistency": 0.89,
      "learning_efficiency": 0.76,
      "adaptation_success_rate": 0.82
    },
    "adaptation_history": [
      {
        "adaptation_type": "threshold_optimization",
        "description": "Adjusted uncertainty threshold from 0.35 to 0.28",
        "trigger_condition": "High uncertainty samples detected",
        "implementation_date": "2024-01-15",
        "impact_assessment": "Improved sample selection by 12%"
      }
    ],
    "recommendations": [
      {
        "type": "model_training",
        "priority": "high",
        "description": "Retrain evaluation models with recent high-quality samples",
        "expected_impact": "5-8% improvement in accuracy",
        "implementation_effort": "medium"
      },
      {
        "type": "data_collection",
        "priority": "medium",
        "description": "Increase collection of edge case responses",
        "expected_impact": "Better handling of complex scenarios",
        "implementation_effort": "low"
      }
    ]
  }
}
```

### Generate Strategic Queries

**POST** `/learning/strategic-queries`

Generate strategic queries for active learning.

**Request Body:**
```json
{
  "skill_areas": ["formulas", "data_analysis", "visualization"],
  "query_types": ["knowledge_probe", "skill_assessment", "edge_case_exploration"],
  "difficulty_levels": ["medium", "hard"],
  "context": {
    "previous_topics": ["VLOOKUP", "pivot_tables"],
    "candidate_level": "intermediate"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "queries": [
      {
        "query_id": "query_001",
        "query_text": "Can you explain the difference between VLOOKUP and INDEX-MATCH in Excel?",
        "query_type": "knowledge_probe",
        "target_skill_area": "formulas",
        "expected_difficulty": "medium",
        "strategic_purpose": "Assess understanding of lookup functions",
        "alternative_formulations": [
          "What are the advantages of INDEX-MATCH over VLOOKUP?",
          "When would you choose INDEX-MATCH instead of VLOOKUP?"
        ]
      },
      {
        "query_id": "query_002",
        "query_text": "How would you handle a dataset with missing values when creating a pivot table?",
        "query_type": "skill_assessment",
        "target_skill_area": "data_analysis",
        "expected_difficulty": "hard",
        "strategic_purpose": "Evaluate practical data handling skills",
        "follow_up_questions": [
          "What impact do missing values have on analysis?",
          "How would you communicate this to stakeholders?"
        ]
      }
    ],
    "generation_summary": {
      "total_queries": 8,
      "skill_coverage": ["formulas", "data_analysis", "visualization"],
      "difficulty_distribution": {"medium": 5, "hard": 3}
    }
  }
}
```

---

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "specific_field",
      "reason": "validation_failed"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-v4"
}
```

### Common Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Invalid or expired authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server internal error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `EVALUATION_FAILED` | 422 | Unable to evaluate response |
| `SESSION_EXPIRED` | 410 | Interview session has expired |
| `INVALID_STATE` | 409 | Invalid session state for operation |

### Validation Errors
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": {
      "errors": [
        {
          "field": "candidate_id",
          "message": "Candidate ID is required",
          "code": "required"
        },
        {
          "field": "position_level",
          "message": "Invalid position level. Must be one of: junior, mid, senior, expert",
          "code": "invalid_value"
        }
      ]
    }
  }
}
```

---

## Rate Limiting

### Rate Limit Tiers

| Tier | Requests per Minute | Requests per Hour | Monthly Quota |
|------|-------------------|------------------|---------------|
| **Free** | 60 | 1,000 | 10,000 |
| **Basic** | 300 | 10,000 | 100,000 |
| **Professional** | 1,000 | 50,000 | 500,000 |
| **Enterprise** | 5,000 | 250,000 | 2,500,000 |

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 856
X-RateLimit-Reset: 1640995200
X-RateLimit-Reset-After: 3600
```

### Handling Rate Limits
When rate limit is exceeded:
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds.",
    "details": {
      "limit": 1000,
      "reset_time": "2024-01-01T01:00:00Z",
      "retry_after": 3600
    }
  }
}
```

---

## WebSocket API

### Connection
Connect to the WebSocket API for real-time updates:
```
wss://api.ai-interviewer.com/v1/ws
```

### Authentication
Include API key in connection URL:
```
wss://api.ai-interviewer.com/v1/ws?api_key=YOUR_API_KEY
```

### Events

#### Interview Progress Updates
```json
{
  "event": "interview_progress",
  "data": {
    "session_id": "sess_123456789",
    "progress": {
      "questions_asked": 5,
      "total_questions": 10,
      "current_score": 7.5
    },
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

#### Evaluation Complete
```json
{
  "event": "evaluation_complete",
  "data": {
    "session_id": "sess_123456789",
    "evaluation": {
      "overall_score": 8.0,
      "dimension_scores": {
        "technical_accuracy": 8.5,
        "problem_solving": 7.5
      }
    },
    "next_question": {
      "question_id": "q_006",
      "text": "Next question text..."
    }
  }
}
```

#### System Status
```json
{
  "event": "system_status",
  "data": {
    "status": "operational",
    "load": 0.65,
    "active_sessions": 42,
    "queue_length": 3
  }
}
```

### Error Events
```json
{
  "event": "error",
  "data": {
    "error_code": "SESSION_ERROR",
    "message": "Session expired",
    "session_id": "sess_123456789"
  }
}
```

---

## SDK Examples

### Python SDK
```python
import ai_interviewer

# Initialize client
client = ai_interviewer.Client(api_key="your_api_key")

# Start interview session
session = client.interviews.start(
    candidate_id="cand_123",
    position_level="mid",
    focus_areas=["formulas", "data_analysis"]
)

# Submit response
evaluation = client.interviews.respond(
    session_id=session.session_id,
    response_text="My experience with Excel formulas..."
)

# Get dashboard data
dashboard = client.dashboard.get_summary(time_range="30d")
```

### JavaScript SDK
```javascript
const AIInterviewer = require('ai-interviewer-sdk');

// Initialize client
const client = new AIInterviewer.Client({
  apiKey: 'your_api_key'
});

// Start interview session
const session = await client.interviews.start({
  candidate_id: 'cand_123',
  position_level: 'mid',
  focus_areas: ['formulas', 'data_analysis']
});

// Submit response
const evaluation = await client.interviews.respond({
  session_id: session.session_id,
  response_text: 'My experience with Excel formulas...'
});
```

---

## Changelog

### Version 1.0.0 (Current)
- Initial API release
- Core interview functionality
- Evaluation system
- Dashboard analytics
- Active learning integration

### Planned Features
- GraphQL API support
- Bulk operations
- Advanced filtering
- Custom evaluation models
- Real-time collaboration features

---

For support and questions, contact: api-support@ai-interviewer.com