#!/usr/bin/env python3
"""
Test script for the AI Excel Interviewer System
This script tests the core functionality without the Streamlit UI
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.interview.integrated_interview_engine import IntegratedInterviewEngine
from src.utils.config import settings
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

async def test_basic_interview():
    """Test basic interview functionality"""
    print("🚀 Starting AI Excel Interviewer System Test...")
    print("=" * 60)
    
    try:
        # Initialize the interview engine
        print("🔄 Initializing interview engine...")
        engine = IntegratedInterviewEngine()
        print("✅ Interview engine initialized successfully")
        
        # Start interview
        print("\n🎯 Starting interview session...")
        result = await engine.start_interview("Test Candidate", "intermediate")
        
        print(f"✅ Interview started successfully")
        print(f"📋 Session ID: {result['session_id']}")
        print(f"🤖 Introduction: {result['introduction'][:100]}...")
        print(f"❓ First Question: {result['first_question']}")
        
        # Test a few interactions
        test_responses = [
            "I have about 3 years of experience with Excel. I use it regularly for data analysis, creating reports, and building dashboards. I'm comfortable with formulas like VLOOKUP, pivot tables, and basic charts.",
            "I recently worked on a sales analysis project where I had to combine data from multiple sources, create pivot tables to summarize the information, and build interactive dashboards for management review.",
            "I would first clean the data by removing duplicates and handling missing values. Then I'd use pivot tables to analyze trends, create charts to visualize key metrics, and build a summary dashboard with key performance indicators."
        ]
        
        print(f"\n🧪 Testing {len(test_responses)} sample interactions...")
        
        for i, response in enumerate(test_responses, 1):
            print(f"\n--- Interaction {i} ---")
            print(f"👤 User: {response[:50]}...")
            
            # Process interaction
            interaction_result = await engine.process_interaction(response, {})
            
            print(f"🤖 AI Response: {interaction_result['response'][:100]}...")
            print(f"📊 State: {interaction_result['state']}")
            
            if 'evaluation' in interaction_result:
                eval_obj = interaction_result['evaluation']
                print(f"📈 Score: {eval_obj.total_score:.1f}% | Level: {eval_obj.skill_level}")
            
            if 'challenge' in interaction_result:
                challenge = interaction_result['challenge']
                print(f"🎮 Challenge Generated: {challenge.challenge_type.value} (Difficulty: {challenge.difficulty.value})")
        
        # Test challenge completion
        print(f"\n🎮 Testing challenge completion...")
        challenge_response = "I would use a combination of INDEX and MATCH functions to create a dynamic lookup that can handle the data analysis requirements."
        
        challenge_result = await engine.process_interaction(challenge_response, {
            "challenge_active": True,
            "current_challenge": interaction_result.get("challenge")
        })
        
        print(f"✅ Challenge completed")
        print(f"📝 Result: {challenge_result['response'][:100]}...")
        
        # Generate final feedback
        print(f"\n📊 Generating comprehensive feedback...")
        feedback_result = await engine.process_interaction("I'm ready for feedback", {})
        
        print(f"✅ Feedback generated")
        print(f"💬 Feedback: {feedback_result['response'][:150]}...")
        
        # Get session summary
        session_data = engine.get_session_data()
        summary = engine._generate_session_summary()
        
        print(f"\n📋 Session Summary:")
        print(f"   • Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        print(f"   • Questions Asked: {summary.get('total_questions', 0)}")
        print(f"   • Challenges Completed: {summary.get('total_challenges', 0)}")
        print(f"   • Average Score: {summary.get('average_score', 0):.1f}%")
        print(f"   • Skill Level: {summary.get('skill_level', 'Unknown')}")
        
        print(f"\n✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        logger.error(f"System test failed: {str(e)}", exc_info=True)
        return False

async def test_evaluation_system():
    """Test the evaluation system independently"""
    print("\n🔍 Testing Evaluation System...")
    
    try:
        from src.evaluation.answer_evaluator import AnswerEvaluator
        
        evaluator = AnswerEvaluator()
        
        test_cases = [
            {
                "question": "How would you use VLOOKUP in Excel?",
                "answer": "VLOOKUP is used to search for a value in the first column of a range and return a value in the same row from another column. For example, =VLOOKUP(A2, Sheet2!A:B, 2, FALSE) would look up the value in A2 in the first column of Sheet2's A:B range and return the corresponding value from the second column.",
                "expected_score_range": (80, 100)
            },
            {
                "question": "What's the difference between COUNT and COUNTA?",
                "answer": "COUNT counts only numeric values, while COUNTA counts all non-empty cells including text and numbers.",
                "expected_score_range": (70, 90)
            },
            {
                "question": "How do you create a pivot table?",
                "answer": "I don't know, I've never used pivot tables.",
                "expected_score_range": (0, 30)
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"❓ Question: {test_case['question']}")
            print(f"👤 Answer: {test_case['answer'][:50]}...")
            
            evaluation = await evaluator.evaluate_answer(
                test_case["question"],
                test_case["answer"],
                "technical"
            )
            
            print(f"📊 Score: {evaluation.total_score:.1f}%")
            print(f"🎯 Skill Level: {evaluation.skill_level}")
            
            # Check if score is in expected range
            min_score, max_score = test_case["expected_score_range"]
            if min_score <= evaluation.total_score <= max_score:
                print(f"✅ Score within expected range ({min_score}-{max_score})")
            else:
                print(f"⚠️ Score outside expected range ({min_score}-{max_score})")
            
            # Show dimension breakdown
            print(f"📈 Dimension Scores:")
            for dim, score in evaluation.dimension_scores.items():
                print(f"   • {dim.value.replace('_', ' ').title()}: {score.score}/25")
        
        print(f"\n✅ Evaluation system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {str(e)}")
        logger.error(f"Evaluation test failed: {str(e)}", exc_info=True)
        return False

async def test_excel_simulator():
    """Test the Excel simulator"""
    print("\n🔢 Testing Excel Simulator...")
    
    try:
        from src.simulation.excel_simulator import ExcelSimulator, ChallengeType, DifficultyLevel
        
        simulator = ExcelSimulator()
        
        # Test challenge generation
        print("🎯 Testing challenge generation...")
        challenge = simulator.generate_challenge(ChallengeType.DATA_ANALYSIS, DifficultyLevel.INTERMEDIATE)
        
        print(f"✅ Challenge generated:")
        print(f"   • Type: {challenge.challenge_type.value}")
        print(f"   • Difficulty: {challenge.difficulty.value}")
        print(f"   • Description: {challenge.description[:50]}...")
        print(f"   • Expected Time: {challenge.expected_time_minutes} minutes")
        
        # Test dataset generation
        print("\n📊 Testing dataset generation...")
        dataset = simulator.generate_dataset(challenge.dataset_config)
        
        print(f"✅ Dataset generated:")
        print(f"   • Size: {len(dataset.data)} rows")
        print(f"   • Columns: {len(dataset.data[0]) if dataset.data else 0}")
        print(f"   • Data Preview: {str(dataset.data[0])[:50]}..." if dataset.data else "No data")
        
        # Test solution evaluation
        print("\n🔍 Testing solution evaluation...")
        test_solution = {
            "solution_text": "I would use pivot tables to analyze the data and create charts to visualize trends.",
            "time_taken": 180  # 3 minutes
        }
        
        result = simulator.evaluate_challenge_solution(challenge, test_solution)
        
        print(f"✅ Solution evaluated:")
        print(f"   • Total Score: {result.total_score:.1f}%")
        print(f"   • Correctness: {result.correctness_score:.1f}%")
        print(f"   • Efficiency: {result.efficiency_score:.1f}%")
        print(f"   • Approach: {result.approach_score:.1f}%")
        print(f"   • Feedback: {result.feedback}")
        
        print(f"\n✅ Excel simulator test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Excel simulator test failed: {str(e)}")
        logger.error(f"Excel simulator test failed: {str(e)}", exc_info=True)
        return False

async def main():
    """Main test function"""
    print("🚀 AI Excel Interviewer System - Comprehensive Test Suite")
    print("=" * 70)
    print("This test will verify all core components of the system:")
    print("• Interview Engine Integration")
    print("• Answer Evaluation System")
    print("• Excel Simulator")
    print("• Conversation Flow")
    print("• Challenge Generation")
    print("• Feedback Generation")
    print("=" * 70)
    
    # Run all tests
    test_results = []
    
    # Test 1: Evaluation System
    print("\n" + "="*50)
    test_results.append(await test_evaluation_system())
    
    # Test 2: Excel Simulator
    print("\n" + "="*50)
    test_results.append(await test_excel_simulator())
    
    # Test 3: Full Integration
    print("\n" + "="*50)
    test_results.append(await test_basic_interview())
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    
    test_names = ["Evaluation System", "Excel Simulator", "Full Integration"]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name:<20} {status}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for deployment.")
        print("\n🚀 To run the Streamlit prototype:")
        print("   streamlit run prototype_demo.py")
    else:
        print("⚠️ Some tests failed. Please review the logs and fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)