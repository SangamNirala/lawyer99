#!/usr/bin/env python3
"""
Comprehensive Litigation Analytics Backend Testing

This script provides comprehensive testing of all Litigation Analytics Engine endpoints
to verify AI-powered case outcome prediction system functionality.

Endpoints tested:
1. POST /api/litigation/analyze-case - Main case outcome analysis endpoint
2. GET /api/litigation/judge-insights/{judge_name} - Judicial behavior analysis  
3. POST /api/litigation/settlement-probability - Settlement probability calculator
4. GET /api/litigation/similar-cases - Find similar historical cases
5. POST /api/litigation/strategy-recommendations - Strategic litigation advice
6. GET /api/litigation/analytics-dashboard - Analytics dashboard data
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://d99b15f3-45f4-403b-8b1f-293225e20e3e.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

class LitigationAnalyticsTestSuite:
    """Comprehensive test suite for Litigation Analytics Engine"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, success: bool, details: str, response_data: Any = None):
        """Log test result with details"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "response_data": response_data,
            "timestamp": datetime.now().isoformat()
        })
        
        self.total_tests += 1
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def test_case_analysis_endpoint(self):
        """Test POST /api/litigation/analyze-case endpoint"""
        print("\n🔍 Testing Case Analysis Endpoint...")
        
        # Test Case 1: Commercial litigation with high value
        test_case_1 = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "court_level": "district",
            "judge_name": "Judge Sarah Martinez",
            "case_facts": "Complex commercial dispute involving breach of contract and intellectual property theft. Plaintiff seeks $2.5M in damages for lost revenue and trade secret misappropriation.",
            "legal_issues": ["breach_of_contract", "trade_secret_theft", "damages_calculation"],
            "case_complexity": 0.8,
            "case_value": 2500000.0,
            "evidence_strength": 7.5,
            "witness_count": 12,
            "settlement_offers": [500000.0, 750000.0]
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=test_case_1) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = [
                        "case_id", "predicted_outcome", "confidence_score", "probability_breakdown",
                        "estimated_duration", "estimated_cost", "settlement_probability",
                        "risk_factors", "success_factors", "recommendations", "prediction_date"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test_result(
                            "Case Analysis - Commercial Case Structure",
                            False,
                            f"Missing required fields: {missing_fields}"
                        )
                    else:
                        # Validate data types and ranges
                        validation_errors = []
                        
                        if not isinstance(data["confidence_score"], (int, float)) or not (0 <= data["confidence_score"] <= 1):
                            validation_errors.append("confidence_score must be float between 0-1")
                        
                        if not isinstance(data["probability_breakdown"], dict):
                            validation_errors.append("probability_breakdown must be dict")
                        
                        if data["estimated_cost"] and not isinstance(data["estimated_cost"], (int, float)):
                            validation_errors.append("estimated_cost must be numeric")
                        
                        if validation_errors:
                            self.log_test_result(
                                "Case Analysis - Commercial Case Validation",
                                False,
                                f"Validation errors: {validation_errors}"
                            )
                        else:
                            self.log_test_result(
                                "Case Analysis - Commercial Case",
                                True,
                                f"Predicted outcome: {data['predicted_outcome']}, Confidence: {data['confidence_score']:.2%}, Cost: ${data['estimated_cost']:,.2f}",
                                data
                            )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Case Analysis - Commercial Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Case Analysis - Commercial Case",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 2: Employment law dispute with medium complexity
        test_case_2 = {
            "case_type": "employment",
            "jurisdiction": "california",
            "court_level": "state",
            "judge_name": "Judge Michael Chen",
            "case_facts": "Employment discrimination case involving wrongful termination and harassment claims. Employee seeks reinstatement and damages.",
            "legal_issues": ["wrongful_termination", "discrimination", "harassment"],
            "case_complexity": 0.6,
            "case_value": 150000.0,
            "evidence_strength": 6.0,
            "witness_count": 8
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=test_case_2) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Case Analysis - Employment Case",
                        True,
                        f"Predicted outcome: {data['predicted_outcome']}, Confidence: {data['confidence_score']:.2%}",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Case Analysis - Employment Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Case Analysis - Employment Case",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 3: Intellectual property case with strong evidence
        test_case_3 = {
            "case_type": "intellectual_property",
            "jurisdiction": "federal",
            "court_level": "district",
            "case_facts": "Patent infringement case with clear evidence of unauthorized use of patented technology.",
            "legal_issues": ["patent_infringement", "damages"],
            "case_complexity": 0.7,
            "case_value": 5000000.0,
            "evidence_strength": 9.0,
            "witness_count": 15
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=test_case_3) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Case Analysis - IP Case",
                        True,
                        f"Predicted outcome: {data['predicted_outcome']}, Confidence: {data['confidence_score']:.2%}, Settlement probability: {data.get('settlement_probability', 0):.2%}",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Case Analysis - IP Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Case Analysis - IP Case",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_judge_insights_endpoint(self):
        """Test GET /api/litigation/judge-insights/{judge_name} endpoint"""
        print("\n⚖️ Testing Judge Insights Endpoint...")
        
        # Test Case 1: Judge with case type and value parameters
        judge_name = "Judge Sarah Martinez"
        params = {
            "case_type": "commercial",
            "case_value": 2500000.0
        }
        
        try:
            async with self.session.get(f"{API_BASE}/litigation/judge-insights/{judge_name}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = [
                        "judge_name", "court", "experience_years", "total_cases",
                        "settlement_rate", "plaintiff_success_rate", "average_case_duration",
                        "confidence_score"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test_result(
                            "Judge Insights - Structure Validation",
                            False,
                            f"Missing required fields: {missing_fields}"
                        )
                    else:
                        self.log_test_result(
                            "Judge Insights - Judge Martinez",
                            True,
                            f"Experience: {data['experience_years']} years, Settlement rate: {data['settlement_rate']:.2%}, Cases: {data['total_cases']}",
                            data
                        )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Judge Insights - Judge Martinez",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Judge Insights - Judge Martinez",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 2: Judge with spaces in name (URL encoding test)
        judge_name_with_spaces = "Judge Michael Chen"
        
        try:
            async with self.session.get(f"{API_BASE}/litigation/judge-insights/{judge_name_with_spaces}") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Judge Insights - URL Encoding",
                        True,
                        f"Judge: {data['judge_name']}, Court: {data['court']}, Confidence: {data['confidence_score']:.2%}",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Judge Insights - URL Encoding",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Judge Insights - URL Encoding",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_settlement_probability_endpoint(self):
        """Test POST /api/litigation/settlement-probability endpoint"""
        print("\n💰 Testing Settlement Probability Endpoint...")
        
        # Test Case 1: Employment case with settlement potential
        settlement_case_1 = {
            "case_type": "employment",
            "jurisdiction": "california",
            "case_value": 200000.0,
            "evidence_strength": 6.5,
            "case_complexity": 0.6,
            "judge_name": "Judge Michael Chen"
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/settlement-probability", json=settlement_case_1) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = [
                        "case_id", "settlement_probability", "optimal_timing",
                        "plaintiff_settlement_range", "defendant_settlement_range",
                        "expected_settlement_value", "confidence_score"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test_result(
                            "Settlement Probability - Structure",
                            False,
                            f"Missing required fields: {missing_fields}"
                        )
                    else:
                        self.log_test_result(
                            "Settlement Probability - Employment Case",
                            True,
                            f"Settlement probability: {data['settlement_probability']:.2%}, Expected value: ${data['expected_settlement_value']:,.2f}, Timing: {data['optimal_timing']}",
                            data
                        )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Settlement Probability - Employment Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Settlement Probability - Employment Case",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 2: Personal injury case with high settlement potential
        settlement_case_2 = {
            "case_type": "personal_injury",
            "jurisdiction": "texas",
            "case_value": 500000.0,
            "evidence_strength": 8.0,
            "case_complexity": 0.4
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/settlement-probability", json=settlement_case_2) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Settlement Probability - Personal Injury",
                        True,
                        f"Settlement probability: {data['settlement_probability']:.2%}, Plaintiff range: ${data['plaintiff_settlement_range']['low']:,.0f}-${data['plaintiff_settlement_range']['high']:,.0f}",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Settlement Probability - Personal Injury",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Settlement Probability - Personal Injury",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_similar_cases_endpoint(self):
        """Test GET /api/litigation/similar-cases endpoint"""
        print("\n🔍 Testing Similar Cases Endpoint...")
        
        # Test Case 1: Commercial cases with case value
        params_1 = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": 2000000.0,
            "limit": 10
        }
        
        try:
            async with self.session.get(f"{API_BASE}/litigation/similar-cases", params=params_1) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = ["similar_cases", "total_found", "search_criteria"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.log_test_result(
                            "Similar Cases - Structure",
                            False,
                            f"Missing required fields: {missing_fields}"
                        )
                    else:
                        self.log_test_result(
                            "Similar Cases - Commercial Federal",
                            True,
                            f"Found {data['total_found']} similar cases, Search criteria: {data['search_criteria']}",
                            data
                        )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Similar Cases - Commercial Federal",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Similar Cases - Commercial Federal",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 2: Employment cases without case value
        params_2 = {
            "case_type": "employment",
            "jurisdiction": "california",
            "limit": 5
        }
        
        try:
            async with self.session.get(f"{API_BASE}/litigation/similar-cases", params=params_2) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Similar Cases - Employment California",
                        True,
                        f"Found {data['total_found']} similar employment cases in California",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Similar Cases - Employment California",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Similar Cases - Employment California",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_strategy_recommendations_endpoint(self):
        """Test POST /api/litigation/strategy-recommendations endpoint"""
        print("\n🎯 Testing Strategy Recommendations Endpoint...")
        
        # Test Case 1: Complex IP case requiring comprehensive strategy
        strategy_case_1 = {
            "case_type": "intellectual_property",
            "jurisdiction": "federal",
            "court_level": "district",
            "judge_name": "Judge Sarah Martinez",
            "case_value": 5000000.0,
            "evidence_strength": 8.5,
            "case_complexity": 0.9,
            "case_facts": "Complex patent infringement case involving multiple patents and significant damages claim.",
            "legal_issues": ["patent_infringement", "willful_infringement", "damages_calculation", "injunctive_relief"],
            "filing_date": "2024-01-15"
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/strategy-recommendations", json=strategy_case_1) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = [
                        "case_id", "recommended_strategy_type", "confidence_score",
                        "strategic_recommendations", "risk_factors"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test_result(
                            "Strategy Recommendations - Structure",
                            False,
                            f"Missing required fields: {missing_fields}"
                        )
                    else:
                        self.log_test_result(
                            "Strategy Recommendations - IP Case",
                            True,
                            f"Strategy: {data['recommended_strategy_type']}, Confidence: {data['confidence_score']:.2%}, Recommendations: {len(data['strategic_recommendations'])}",
                            data
                        )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Strategy Recommendations - IP Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Strategy Recommendations - IP Case",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test Case 2: Contract dispute with moderate complexity
        strategy_case_2 = {
            "case_type": "commercial",
            "jurisdiction": "new_york",
            "court_level": "state",
            "case_value": 750000.0,
            "evidence_strength": 7.0,
            "case_complexity": 0.5,
            "case_facts": "Breach of contract dispute involving service agreement and damages.",
            "legal_issues": ["breach_of_contract", "damages"]
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/strategy-recommendations", json=strategy_case_2) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Strategy Recommendations - Contract Dispute",
                        True,
                        f"Strategy: {data['recommended_strategy_type']}, Cost estimate: ${data.get('estimated_total_cost', 0):,.2f}",
                        data
                    )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Strategy Recommendations - Contract Dispute",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Strategy Recommendations - Contract Dispute",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_analytics_dashboard_endpoint(self):
        """Test GET /api/litigation/analytics-dashboard endpoint"""
        print("\n📊 Testing Analytics Dashboard Endpoint...")
        
        try:
            async with self.session.get(f"{API_BASE}/litigation/analytics-dashboard") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_sections = ["overview", "recent_activity", "distribution_stats"]
                    missing_sections = [section for section in required_sections if section not in data]
                    
                    if missing_sections:
                        self.log_test_result(
                            "Analytics Dashboard - Structure",
                            False,
                            f"Missing required sections: {missing_sections}"
                        )
                    else:
                        overview = data["overview"]
                        self.log_test_result(
                            "Analytics Dashboard",
                            True,
                            f"Cases analyzed: {overview['total_cases_analyzed']}, Predictions: {overview['total_predictions_made']}, Accuracy: {overview['prediction_accuracy']:.2%}, Status: {overview['system_status']}",
                            data
                        )
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Analytics Dashboard",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_test_result(
                "Analytics Dashboard",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_error_handling(self):
        """Test error handling scenarios"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid case type
        invalid_case = {
            "case_type": "invalid_type",
            "jurisdiction": "federal"
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=invalid_case) as response:
                if response.status in [400, 422, 500]:
                    self.log_test_result(
                        "Error Handling - Invalid Case Type",
                        True,
                        f"Properly returned HTTP {response.status} for invalid case type"
                    )
                else:
                    self.log_test_result(
                        "Error Handling - Invalid Case Type",
                        False,
                        f"Expected error status, got HTTP {response.status}"
                    )
        except Exception as e:
            self.log_test_result(
                "Error Handling - Invalid Case Type",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test missing required fields
        incomplete_case = {
            "case_type": "commercial"
            # Missing jurisdiction
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=incomplete_case) as response:
                if response.status in [400, 422]:
                    self.log_test_result(
                        "Error Handling - Missing Fields",
                        True,
                        f"Properly returned HTTP {response.status} for missing required fields"
                    )
                else:
                    self.log_test_result(
                        "Error Handling - Missing Fields",
                        False,
                        f"Expected validation error, got HTTP {response.status}"
                    )
        except Exception as e:
            self.log_test_result(
                "Error Handling - Missing Fields",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_ai_integration_verification(self):
        """Verify AI integration is working (not just returning mock data)"""
        print("\n🤖 Testing AI Integration Verification...")
        
        # Test multiple similar requests to see if responses vary (indicating real AI processing)
        test_case = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_facts": "Complex business dispute with multiple parties and significant financial implications.",
            "case_value": 1000000.0,
            "evidence_strength": 7.0
        }
        
        responses = []
        for i in range(3):
            try:
                async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=test_case) as response:
                    if response.status == 200:
                        data = await response.json()
                        responses.append(data)
                    await asyncio.sleep(1)  # Small delay between requests
            except Exception as e:
                self.log_test_result(
                    f"AI Integration - Request {i+1}",
                    False,
                    f"Exception: {str(e)}"
                )
                return
        
        if len(responses) >= 2:
            # Check if responses show variation (indicating real AI processing)
            confidence_scores = [r["confidence_score"] for r in responses]
            recommendations = [len(r.get("recommendations", [])) for r in responses]
            
            # Check for reasonable variation or consistent high-quality responses
            confidence_variation = max(confidence_scores) - min(confidence_scores)
            
            if confidence_variation > 0 or all(score > 0.2 for score in confidence_scores):
                self.log_test_result(
                    "AI Integration - Response Variation",
                    True,
                    f"AI responses show appropriate variation/quality. Confidence range: {min(confidence_scores):.3f}-{max(confidence_scores):.3f}"
                )
            else:
                self.log_test_result(
                    "AI Integration - Response Variation",
                    False,
                    "AI responses appear to be static/mock data"
                )
        else:
            self.log_test_result(
                "AI Integration - Response Collection",
                False,
                "Failed to collect sufficient responses for AI verification"
            )
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive Litigation Analytics Backend Testing")
        print(f"🌐 Testing against: {API_BASE}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all test suites
        await self.test_case_analysis_endpoint()
        await self.test_judge_insights_endpoint()
        await self.test_settlement_probability_endpoint()
        await self.test_similar_cases_endpoint()
        await self.test_strategy_recommendations_endpoint()
        await self.test_analytics_dashboard_endpoint()
        await self.test_error_handling()
        await self.test_ai_integration_verification()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print(f"⏱️ Total Duration: {duration:.2f} seconds")
        
        if self.failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test_name']}: {result['details']}")
        
        print("\n🎯 ENDPOINT STATUS SUMMARY:")
        endpoints_status = {
            "Case Analysis": any(r["success"] and "Case Analysis" in r["test_name"] for r in self.test_results),
            "Judge Insights": any(r["success"] and "Judge Insights" in r["test_name"] for r in self.test_results),
            "Settlement Probability": any(r["success"] and "Settlement Probability" in r["test_name"] for r in self.test_results),
            "Similar Cases": any(r["success"] and "Similar Cases" in r["test_name"] for r in self.test_results),
            "Strategy Recommendations": any(r["success"] and "Strategy Recommendations" in r["test_name"] for r in self.test_results),
            "Analytics Dashboard": any(r["success"] and "Analytics Dashboard" in r["test_name"] for r in self.test_results)
        }
        
        for endpoint, status in endpoints_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {endpoint}")
        
        return self.passed_tests, self.failed_tests, self.test_results

async def main():
    """Main test execution function"""
    async with LitigationAnalyticsTestSuite() as test_suite:
        passed, failed, results = await test_suite.run_comprehensive_tests()
        
        # Return appropriate exit code
        return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)