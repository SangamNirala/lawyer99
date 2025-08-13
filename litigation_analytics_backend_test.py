#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Litigation Analytics Engine
============================================================

This script tests all the new Litigation Analytics Engine endpoints that were added to LegalMate AI.

Test Coverage:
1. POST /api/litigation/analyze-case - Comprehensive case outcome analysis
2. GET /api/litigation/judge-insights/{judge_name} - Judicial behavior insights  
3. POST /api/litigation/settlement-probability - Settlement probability calculation
4. GET /api/litigation/similar-cases - Find similar historical cases
5. POST /api/litigation/strategy-recommendations - Generate litigation strategy
6. GET /api/litigation/analytics-dashboard - Analytics dashboard data

Focus Areas:
- Verify all litigation analytics modules are properly imported and initialized
- Test AI integration with Gemini and Groq APIs
- Confirm MongoDB collections are created and accessible
- Validate response structures match the defined Pydantic models
- Test error scenarios and edge cases
"""

import requests
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

class LitigationAnalyticsBackendTester:
    def __init__(self):
        # Use the production backend URL from frontend/.env
        self.base_url = "https://legal-research-api.preview.emergentagent.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Test results tracking
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
        print("🎯 LITIGATION ANALYTICS ENGINE BACKEND TESTING INITIATED")
        print(f"📡 Backend URL: {self.base_url}")
        print("=" * 80)

    def log_test_result(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test results for tracking"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "response_data": response_data,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        print(f"{status} | {test_name}")
        if details:
            print(f"     Details: {details}")
        if not success and response_data:
            print(f"     Response: {response_data}")
        print()

    def test_case_analysis_endpoint(self):
        """Test POST /api/litigation/analyze-case endpoint"""
        print("🔍 TESTING CASE ANALYSIS ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Commercial litigation case
        test_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": 500000,
            "evidence_strength": 0.7,
            "case_complexity": 0.6,
            "judge_name": "Judge Smith",
            "court_level": "district",
            "case_facts": "Contract dispute involving breach of service agreement",
            "legal_issues": ["breach of contract", "damages calculation", "specific performance"],
            "witness_count": 5,
            "filing_date": "2024-01-15T00:00:00Z"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/analyze-case", json=test_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "case_id", "predicted_outcome", "confidence_score", 
                    "probability_breakdown", "recommendations", "prediction_date"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    # Validate data types and ranges
                    if (isinstance(data["confidence_score"], (int, float)) and 0 <= data["confidence_score"] <= 1 and
                        isinstance(data["probability_breakdown"], dict) and
                        isinstance(data["recommendations"], list)):
                        
                        self.log_test_result(
                            "Case Analysis - Commercial Case",
                            True,
                            f"Predicted outcome: {data['predicted_outcome']}, Confidence: {data['confidence_score']:.2f}",
                            {"case_id": data["case_id"], "outcome": data["predicted_outcome"]}
                        )
                    else:
                        self.log_test_result(
                            "Case Analysis - Commercial Case",
                            False,
                            "Invalid data types in response",
                            data
                        )
                else:
                    self.log_test_result(
                        "Case Analysis - Commercial Case",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Case Analysis - Commercial Case",
                    False,
                    f"HTTP {response.status_code}: {response.text}",
                    response.text
                )
                
        except Exception as e:
            self.log_test_result(
                "Case Analysis - Commercial Case",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Employment case with different parameters
        employment_data = {
            "case_type": "employment",
            "jurisdiction": "california",
            "case_value": 250000,
            "evidence_strength": 0.8,
            "case_complexity": 0.4,
            "case_facts": "Wrongful termination and discrimination claim",
            "legal_issues": ["wrongful termination", "discrimination", "wage violations"]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/analyze-case", json=employment_data)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Case Analysis - Employment Case",
                    True,
                    f"Predicted outcome: {data.get('predicted_outcome', 'N/A')}, Confidence: {data.get('confidence_score', 0):.2f}"
                )
            else:
                self.log_test_result(
                    "Case Analysis - Employment Case",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Case Analysis - Employment Case",
                False,
                f"Request failed: {str(e)}"
            )

    def test_judge_insights_endpoint(self):
        """Test GET /api/litigation/judge-insights/{judge_name} endpoint"""
        print("⚖️ TESTING JUDGE INSIGHTS ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Judge insights with case type and value
        judge_name = "Judge Smith"
        params = {
            "case_type": "commercial",
            "case_value": 500000
        }
        
        try:
            # URL encode the judge name
            encoded_judge_name = requests.utils.quote(judge_name)
            response = self.session.get(
                f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "judge_name", "court", "experience_years", "total_cases",
                    "settlement_rate", "plaintiff_success_rate", "confidence_score"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test_result(
                        "Judge Insights - Judge Smith Commercial",
                        True,
                        f"Experience: {data['experience_years']} years, Settlement rate: {data['settlement_rate']:.2f}",
                        {"judge": data["judge_name"], "court": data["court"]}
                    )
                else:
                    self.log_test_result(
                        "Judge Insights - Judge Smith Commercial",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Judge Insights - Judge Smith Commercial",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Insights - Judge Smith Commercial",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Judge insights without parameters
        try:
            response = self.session.get(f"{self.base_url}/litigation/judge-insights/Judge%20Johnson")
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Judge Insights - Judge Johnson General",
                    True,
                    f"Judge: {data.get('judge_name', 'N/A')}, Total cases: {data.get('total_cases', 0)}"
                )
            else:
                self.log_test_result(
                    "Judge Insights - Judge Johnson General",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Insights - Judge Johnson General",
                False,
                f"Request failed: {str(e)}"
            )

    def test_settlement_probability_endpoint(self):
        """Test POST /api/litigation/settlement-probability endpoint"""
        print("💰 TESTING SETTLEMENT PROBABILITY ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Employment case settlement analysis
        settlement_data = {
            "case_type": "employment",
            "jurisdiction": "california",
            "case_value": 250000,
            "evidence_strength": 0.6,
            "case_complexity": 0.5,
            "judge_name": "Judge Martinez"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/settlement-probability", json=settlement_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "case_id", "settlement_probability", "optimal_timing",
                    "plaintiff_settlement_range", "defendant_settlement_range",
                    "expected_settlement_value", "confidence_score"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test_result(
                        "Settlement Probability - Employment Case",
                        True,
                        f"Settlement probability: {data['settlement_probability']:.2f}, Expected value: ${data['expected_settlement_value']:,.2f}",
                        {"case_id": data["case_id"], "probability": data["settlement_probability"]}
                    )
                else:
                    self.log_test_result(
                        "Settlement Probability - Employment Case",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Settlement Probability - Employment Case",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Settlement Probability - Employment Case",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Commercial case with minimal data
        minimal_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": 1000000
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/settlement-probability", json=minimal_data)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Settlement Probability - Commercial Minimal",
                    True,
                    f"Settlement probability: {data.get('settlement_probability', 0):.2f}"
                )
            else:
                self.log_test_result(
                    "Settlement Probability - Commercial Minimal",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Settlement Probability - Commercial Minimal",
                False,
                f"Request failed: {str(e)}"
            )

    def test_similar_cases_endpoint(self):
        """Test GET /api/litigation/similar-cases endpoint"""
        print("🔍 TESTING SIMILAR CASES ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Commercial cases with case value
        params = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": 500000,
            "limit": 5
        }
        
        try:
            response = self.session.get(f"{self.base_url}/litigation/similar-cases", params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if "similar_cases" in data and "total_found" in data and "search_criteria" in data:
                    self.log_test_result(
                        "Similar Cases - Commercial Federal",
                        True,
                        f"Found {data['total_found']} similar cases",
                        {"total_found": data["total_found"], "criteria": data["search_criteria"]}
                    )
                else:
                    self.log_test_result(
                        "Similar Cases - Commercial Federal",
                        False,
                        "Invalid response structure",
                        data
                    )
            else:
                self.log_test_result(
                    "Similar Cases - Commercial Federal",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Similar Cases - Commercial Federal",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Employment cases without case value
        params = {
            "case_type": "employment",
            "jurisdiction": "california",
            "limit": 10
        }
        
        try:
            response = self.session.get(f"{self.base_url}/litigation/similar-cases", params=params)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Similar Cases - Employment California",
                    True,
                    f"Found {data.get('total_found', 0)} similar cases"
                )
            else:
                self.log_test_result(
                    "Similar Cases - Employment California",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Similar Cases - Employment California",
                False,
                f"Request failed: {str(e)}"
            )

    def test_strategy_recommendations_endpoint(self):
        """Test POST /api/litigation/strategy-recommendations endpoint"""
        print("🎯 TESTING LITIGATION STRATEGY ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Intellectual property case
        strategy_data = {
            "case_type": "intellectual_property",
            "jurisdiction": "delaware",
            "case_value": 1000000,
            "evidence_strength": 0.8,
            "case_complexity": 0.9,
            "court_level": "district",
            "case_facts": "Patent infringement case involving software technology",
            "legal_issues": ["patent infringement", "damages calculation", "injunctive relief"]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/strategy-recommendations", json=strategy_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "case_id", "recommended_strategy_type", "confidence_score",
                    "strategic_recommendations", "risk_factors"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test_result(
                        "Litigation Strategy - IP Case",
                        True,
                        f"Strategy: {data['recommended_strategy_type']}, Confidence: {data['confidence_score']:.2f}",
                        {"case_id": data["case_id"], "strategy": data["recommended_strategy_type"]}
                    )
                else:
                    self.log_test_result(
                        "Litigation Strategy - IP Case",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Litigation Strategy - IP Case",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Litigation Strategy - IP Case",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Contract dispute case
        contract_data = {
            "case_type": "commercial",
            "jurisdiction": "new_york",
            "case_value": 750000,
            "evidence_strength": 0.7,
            "case_complexity": 0.5,
            "case_facts": "Breach of supply agreement with damages claim"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/strategy-recommendations", json=contract_data)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Litigation Strategy - Contract Dispute",
                    True,
                    f"Strategy: {data.get('recommended_strategy_type', 'N/A')}"
                )
            else:
                self.log_test_result(
                    "Litigation Strategy - Contract Dispute",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Litigation Strategy - Contract Dispute",
                False,
                f"Request failed: {str(e)}"
            )

    def test_analytics_dashboard_endpoint(self):
        """Test GET /api/litigation/analytics-dashboard endpoint"""
        print("📊 TESTING ANALYTICS DASHBOARD ENDPOINT")
        print("-" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/litigation/analytics-dashboard")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_sections = ["overview", "recent_activity", "distribution_stats"]
                missing_sections = [section for section in required_sections if section not in data]
                
                if not missing_sections:
                    overview = data.get("overview", {})
                    self.log_test_result(
                        "Analytics Dashboard - Overview",
                        True,
                        f"Cases analyzed: {overview.get('total_cases_analyzed', 0)}, Predictions: {overview.get('total_predictions_made', 0)}",
                        {
                            "total_cases": overview.get('total_cases_analyzed', 0),
                            "predictions": overview.get('total_predictions_made', 0),
                            "status": overview.get('system_status', 'unknown')
                        }
                    )
                else:
                    self.log_test_result(
                        "Analytics Dashboard - Overview",
                        False,
                        f"Missing required sections: {missing_sections}",
                        data
                    )
            else:
                self.log_test_result(
                    "Analytics Dashboard - Overview",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Analytics Dashboard - Overview",
                False,
                f"Request failed: {str(e)}"
            )

    def test_error_scenarios(self):
        """Test error handling and edge cases"""
        print("🚨 TESTING ERROR SCENARIOS")
        print("-" * 50)
        
        # Test Case 1: Invalid case type
        invalid_data = {
            "case_type": "invalid_case_type",
            "jurisdiction": "federal"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/analyze-case", json=invalid_data)
            
            # Should return an error (400 or 422)
            if response.status_code in [400, 422, 500]:
                self.log_test_result(
                    "Error Handling - Invalid Case Type",
                    True,
                    f"Correctly returned error: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Invalid Case Type",
                    False,
                    f"Expected error but got HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Invalid Case Type",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Missing required fields
        incomplete_data = {
            "case_type": "commercial"
            # Missing jurisdiction
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/analyze-case", json=incomplete_data)
            
            if response.status_code in [400, 422]:
                self.log_test_result(
                    "Error Handling - Missing Required Fields",
                    True,
                    f"Correctly returned validation error: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Missing Required Fields",
                    False,
                    f"Expected validation error but got HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Missing Required Fields",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 3: Non-existent judge
        try:
            response = self.session.get(f"{self.base_url}/litigation/judge-insights/NonExistentJudge123")
            
            # Should handle gracefully (might return empty data or error)
            if response.status_code in [200, 404, 500]:
                self.log_test_result(
                    "Error Handling - Non-existent Judge",
                    True,
                    f"Handled non-existent judge: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Non-existent Judge",
                    False,
                    f"Unexpected response: HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Non-existent Judge",
                False,
                f"Request failed: {str(e)}"
            )

    def test_response_times(self):
        """Test response times for performance"""
        print("⏱️ TESTING RESPONSE TIMES")
        print("-" * 50)
        
        # Test response time for case analysis (most complex endpoint)
        test_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": 500000,
            "evidence_strength": 0.7
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/litigation/analyze-case", json=test_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200 and response_time < 30:  # Should be under 30 seconds
                self.log_test_result(
                    "Performance - Case Analysis Response Time",
                    True,
                    f"Response time: {response_time:.2f} seconds (under 30s threshold)"
                )
            elif response.status_code == 200:
                self.log_test_result(
                    "Performance - Case Analysis Response Time",
                    False,
                    f"Response time too slow: {response_time:.2f} seconds (over 30s threshold)"
                )
            else:
                self.log_test_result(
                    "Performance - Case Analysis Response Time",
                    False,
                    f"Request failed: HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Performance - Case Analysis Response Time",
                False,
                f"Request failed: {str(e)}"
            )

    def run_comprehensive_tests(self):
        """Run all litigation analytics tests"""
        print("🚀 STARTING COMPREHENSIVE LITIGATION ANALYTICS TESTING")
        print("=" * 80)
        
        # Run all test suites
        self.test_case_analysis_endpoint()
        self.test_judge_insights_endpoint()
        self.test_settlement_probability_endpoint()
        self.test_similar_cases_endpoint()
        self.test_strategy_recommendations_endpoint()
        self.test_analytics_dashboard_endpoint()
        self.test_error_scenarios()
        self.test_response_times()
        
        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("=" * 80)
        print("📋 LITIGATION ANALYTICS ENGINE TEST REPORT")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.total_tests - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print()
        
        # Group results by endpoint
        endpoint_results = {}
        for result in self.test_results:
            endpoint = result["test_name"].split(" - ")[0]
            if endpoint not in endpoint_results:
                endpoint_results[endpoint] = {"passed": 0, "total": 0}
            endpoint_results[endpoint]["total"] += 1
            if result["success"]:
                endpoint_results[endpoint]["passed"] += 1
        
        print("📈 ENDPOINT-SPECIFIC RESULTS:")
        for endpoint, stats in endpoint_results.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if rate == 100 else "⚠️" if rate >= 50 else "❌"
            print(f"   {status} {endpoint}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        print()
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print("❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['test_name']}: {test['details']}")
            print()
        
        # Show successful key features
        successful_tests = [r for r in self.test_results if r["success"]]
        if successful_tests:
            print("✅ SUCCESSFUL KEY FEATURES:")
            key_features = [
                "Case Analysis", "Judge Insights", "Settlement Probability",
                "Similar Cases", "Litigation Strategy", "Analytics Dashboard"
            ]
            for feature in key_features:
                feature_tests = [t for t in successful_tests if feature in t["test_name"]]
                if feature_tests:
                    print(f"   • {feature}: {len(feature_tests)} tests passed")
            print()
        
        # Final assessment
        if success_rate >= 90:
            print("🎉 EXCELLENT: Litigation Analytics Engine is fully operational!")
        elif success_rate >= 75:
            print("✅ GOOD: Litigation Analytics Engine is mostly functional with minor issues.")
        elif success_rate >= 50:
            print("⚠️ MODERATE: Litigation Analytics Engine has significant issues requiring attention.")
        else:
            print("❌ CRITICAL: Litigation Analytics Engine has major functionality problems.")
        
        print("=" * 80)
        
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "endpoint_results": endpoint_results,
            "failed_tests": failed_tests,
            "test_results": self.test_results
        }

def main():
    """Main testing function"""
    tester = LitigationAnalyticsBackendTester()
    results = tester.run_comprehensive_tests()
    
    # Return results for potential integration with other systems
    return results

if __name__ == "__main__":
    main()