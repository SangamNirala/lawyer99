#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Judge Comparison Functionality
==============================================================

This script tests the new judge comparison functionality as requested:

Test Coverage:
1. POST /api/litigation/judge-comparison - Judge comparison endpoint with sample data
2. GET /api/litigation/judge-insights/{judge_name} - Existing judge insights endpoint
3. Integration Test - Complete workflow testing
4. Error Scenarios - Edge cases and error handling

Focus Areas:
- Verify judge comparison API endpoint with exact sample data from request
- Test response structure includes: judges_compared, comparative_metrics, recommendations, analysis_date, confidence_score
- Test error handling for less than 2 judges
- Test with different case types
- Verify existing judge insights endpoint still works (no regressions)
- Test complete workflow: Get insights for Judge 1 → Get insights for Judge 2 → Compare both judges
"""

import requests
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

class JudgeComparisonBackendTester:
    def __init__(self):
        # Use the production backend URL from frontend/.env
        self.base_url = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Test results tracking
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
        print("🎯 JUDGE COMPARISON FUNCTIONALITY BACKEND TESTING INITIATED")
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

    def test_judge_comparison_endpoint(self):
        """Test POST /api/litigation/judge-comparison endpoint with exact sample data"""
        print("⚖️ TESTING JUDGE COMPARISON ENDPOINT")
        print("-" * 50)
        
        # Test Case 1: Exact sample data from review request
        sample_data = {
            "judge_names": ["Judge Sarah Martinez", "Judge John Smith"],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=sample_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure includes required fields from review request
                required_fields = [
                    "judges_compared", "comparative_metrics", "recommendations", 
                    "analysis_date", "confidence_score"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    # Validate specific data types and values
                    if (isinstance(data["judges_compared"], int) and data["judges_compared"] == 2 and
                        isinstance(data["comparative_metrics"], dict) and
                        isinstance(data["recommendations"], dict) and
                        isinstance(data["confidence_score"], (int, float)) and 0 <= data["confidence_score"] <= 1):
                        
                        self.log_test_result(
                            "Judge Comparison - Sample Data (Sarah Martinez vs John Smith)",
                            True,
                            f"Compared {data['judges_compared']} judges, Confidence: {data['confidence_score']:.2f}",
                            {
                                "judges_compared": data["judges_compared"],
                                "case_type": data.get("case_type_focus"),
                                "analysis_date": data["analysis_date"]
                            }
                        )
                        
                        # Verify comparative metrics structure
                        metrics = data["comparative_metrics"]
                        if isinstance(metrics, dict) and len(metrics) > 0:
                            self.log_test_result(
                                "Judge Comparison - Comparative Metrics Structure",
                                True,
                                f"Metrics include: {list(metrics.keys())[:3]}..." if len(metrics) > 3 else f"Metrics: {list(metrics.keys())}"
                            )
                        else:
                            self.log_test_result(
                                "Judge Comparison - Comparative Metrics Structure",
                                False,
                                "Comparative metrics is empty or invalid"
                            )
                        
                        # Verify recommendations structure
                        recommendations = data["recommendations"]
                        if isinstance(recommendations, dict) and len(recommendations) > 0:
                            self.log_test_result(
                                "Judge Comparison - Recommendations Structure",
                                True,
                                f"Recommendations provided for: {list(recommendations.keys())}"
                            )
                        else:
                            self.log_test_result(
                                "Judge Comparison - Recommendations Structure",
                                False,
                                "Recommendations is empty or invalid"
                            )
                    else:
                        self.log_test_result(
                            "Judge Comparison - Sample Data (Sarah Martinez vs John Smith)",
                            False,
                            "Invalid data types or values in response",
                            data
                        )
                else:
                    self.log_test_result(
                        "Judge Comparison - Sample Data (Sarah Martinez vs John Smith)",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Judge Comparison - Sample Data (Sarah Martinez vs John Smith)",
                    False,
                    f"HTTP {response.status_code}: {response.text}",
                    response.text
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Comparison - Sample Data (Sarah Martinez vs John Smith)",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Different case type
        commercial_data = {
            "judge_names": ["Judge Sarah Martinez", "Judge John Smith"],
            "case_type": "commercial"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=commercial_data)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Judge Comparison - Commercial Case Type",
                    True,
                    f"Commercial case comparison successful, Case type focus: {data.get('case_type_focus', 'N/A')}"
                )
            else:
                self.log_test_result(
                    "Judge Comparison - Commercial Case Type",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Comparison - Commercial Case Type",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 3: Multiple judges (more than 2)
        multiple_judges_data = {
            "judge_names": ["Judge Sarah Martinez", "Judge John Smith", "Judge Emily Johnson"],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=multiple_judges_data)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Judge Comparison - Multiple Judges (3 judges)",
                    True,
                    f"Successfully compared {data.get('judges_compared', 0)} judges"
                )
            else:
                self.log_test_result(
                    "Judge Comparison - Multiple Judges (3 judges)",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Comparison - Multiple Judges (3 judges)",
                False,
                f"Request failed: {str(e)}"
            )

    def test_existing_judge_insights_endpoint(self):
        """Test GET /api/litigation/judge-insights/{judge_name} to ensure no regressions"""
        print("🔍 TESTING EXISTING JUDGE INSIGHTS ENDPOINT (REGRESSION TEST)")
        print("-" * 50)
        
        # Test Case 1: Judge Sarah Martinez (from sample data)
        judge_name = "Judge Sarah Martinez"
        
        try:
            # URL encode the judge name
            encoded_judge_name = requests.utils.quote(judge_name)
            response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
            
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
                        "Judge Insights - Judge Sarah Martinez",
                        True,
                        f"Judge: {data['judge_name']}, Experience: {data['experience_years']} years, Settlement rate: {data['settlement_rate']:.2f}",
                        {"judge": data["judge_name"], "court": data["court"]}
                    )
                else:
                    self.log_test_result(
                        "Judge Insights - Judge Sarah Martinez",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Judge Insights - Judge Sarah Martinez",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Insights - Judge Sarah Martinez",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Judge John Smith (from sample data)
        judge_name = "Judge John Smith"
        
        try:
            encoded_judge_name = requests.utils.quote(judge_name)
            response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
            
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(
                    "Judge Insights - Judge John Smith",
                    True,
                    f"Judge: {data.get('judge_name', 'N/A')}, Total cases: {data.get('total_cases', 0)}"
                )
            else:
                self.log_test_result(
                    "Judge Insights - Judge John Smith",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Insights - Judge John Smith",
                False,
                f"Request failed: {str(e)}"
            )

    def test_integration_workflow(self):
        """Test complete workflow: Get insights for Judge 1 → Get insights for Judge 2 → Compare both judges"""
        print("🔄 TESTING INTEGRATION WORKFLOW")
        print("-" * 50)
        
        judge1_name = "Judge Sarah Martinez"
        judge2_name = "Judge John Smith"
        judge1_data = None
        judge2_data = None
        
        # Step 1: Get insights for Judge 1
        try:
            encoded_judge1 = requests.utils.quote(judge1_name)
            response1 = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge1}")
            
            if response1.status_code == 200:
                judge1_data = response1.json()
                self.log_test_result(
                    "Integration Workflow - Step 1 (Judge 1 Insights)",
                    True,
                    f"Retrieved insights for {judge1_data.get('judge_name', 'N/A')}"
                )
            else:
                self.log_test_result(
                    "Integration Workflow - Step 1 (Judge 1 Insights)",
                    False,
                    f"Failed to get Judge 1 insights: HTTP {response1.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Integration Workflow - Step 1 (Judge 1 Insights)",
                False,
                f"Request failed: {str(e)}"
            )

        # Step 2: Get insights for Judge 2
        try:
            encoded_judge2 = requests.utils.quote(judge2_name)
            response2 = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge2}")
            
            if response2.status_code == 200:
                judge2_data = response2.json()
                self.log_test_result(
                    "Integration Workflow - Step 2 (Judge 2 Insights)",
                    True,
                    f"Retrieved insights for {judge2_data.get('judge_name', 'N/A')}"
                )
            else:
                self.log_test_result(
                    "Integration Workflow - Step 2 (Judge 2 Insights)",
                    False,
                    f"Failed to get Judge 2 insights: HTTP {response2.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Integration Workflow - Step 2 (Judge 2 Insights)",
                False,
                f"Request failed: {str(e)}"
            )

        # Step 3: Compare both judges
        if judge1_data and judge2_data:
            comparison_data = {
                "judge_names": [judge1_name, judge2_name],
                "case_type": "civil"
            }
            
            try:
                response3 = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=comparison_data)
                
                if response3.status_code == 200:
                    comparison_result = response3.json()
                    
                    # Verify comparison shows differences in settlement rates, plaintiff success rates, etc.
                    metrics = comparison_result.get("comparative_metrics", {})
                    
                    if isinstance(metrics, dict) and len(metrics) > 0:
                        # Look for settlement rate and plaintiff success rate differences
                        has_settlement_data = any("settlement" in str(key).lower() for key in metrics.keys())
                        has_success_rate_data = any("success" in str(key).lower() or "plaintiff" in str(key).lower() for key in metrics.keys())
                        
                        self.log_test_result(
                            "Integration Workflow - Step 3 (Judge Comparison)",
                            True,
                            f"Comparison successful. Settlement data: {has_settlement_data}, Success rate data: {has_success_rate_data}",
                            {
                                "judges_compared": comparison_result.get("judges_compared"),
                                "metrics_keys": list(metrics.keys())[:5]  # First 5 keys
                            }
                        )
                        
                        # Verify differences are shown
                        if has_settlement_data or has_success_rate_data:
                            self.log_test_result(
                                "Integration Workflow - Differences Verification",
                                True,
                                "Comparison shows differences in settlement rates and/or plaintiff success rates"
                            )
                        else:
                            self.log_test_result(
                                "Integration Workflow - Differences Verification",
                                False,
                                "Comparison does not show expected settlement/success rate differences"
                            )
                    else:
                        self.log_test_result(
                            "Integration Workflow - Step 3 (Judge Comparison)",
                            False,
                            "Comparison returned empty or invalid metrics"
                        )
                else:
                    self.log_test_result(
                        "Integration Workflow - Step 3 (Judge Comparison)",
                        False,
                        f"Comparison failed: HTTP {response3.status_code}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    "Integration Workflow - Step 3 (Judge Comparison)",
                    False,
                    f"Request failed: {str(e)}"
                )
        else:
            self.log_test_result(
                "Integration Workflow - Step 3 (Judge Comparison)",
                False,
                "Cannot perform comparison - missing judge insights data"
            )

    def test_error_scenarios(self):
        """Test edge cases and error handling"""
        print("🚨 TESTING ERROR SCENARIOS")
        print("-" * 50)
        
        # Test Case 1: Less than 2 judges (should return error)
        single_judge_data = {
            "judge_names": ["Judge Sarah Martinez"],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=single_judge_data)
            
            # Should return an error (400 or 422)
            if response.status_code in [400, 422]:
                self.log_test_result(
                    "Error Handling - Less than 2 judges",
                    True,
                    f"Correctly returned error for single judge: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Less than 2 judges",
                    False,
                    f"Expected error but got HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Less than 2 judges",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Empty judge names list
        empty_judges_data = {
            "judge_names": [],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=empty_judges_data)
            
            if response.status_code in [400, 422]:
                self.log_test_result(
                    "Error Handling - Empty judge names",
                    True,
                    f"Correctly returned error for empty judge list: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Empty judge names",
                    False,
                    f"Expected error but got HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Empty judge names",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 3: Non-existent judges
        nonexistent_judges_data = {
            "judge_names": ["Judge NonExistent One", "Judge NonExistent Two"],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=nonexistent_judges_data)
            
            # Should handle gracefully (might return data with low confidence or error)
            if response.status_code in [200, 404, 500]:
                if response.status_code == 200:
                    data = response.json()
                    confidence = data.get("confidence_score", 1.0)
                    self.log_test_result(
                        "Error Handling - Non-existent judges",
                        True,
                        f"Handled non-existent judges gracefully: HTTP {response.status_code}, Confidence: {confidence:.2f}"
                    )
                else:
                    self.log_test_result(
                        "Error Handling - Non-existent judges",
                        True,
                        f"Handled non-existent judges with appropriate error: HTTP {response.status_code}"
                    )
            else:
                self.log_test_result(
                    "Error Handling - Non-existent judges",
                    False,
                    f"Unexpected response: HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Non-existent judges",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 4: Invalid request format (missing judge_names field)
        invalid_format_data = {
            "case_type": "civil"
            # Missing judge_names field
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=invalid_format_data)
            
            if response.status_code in [400, 422]:
                self.log_test_result(
                    "Error Handling - Invalid request format",
                    True,
                    f"Correctly returned validation error: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Invalid request format",
                    False,
                    f"Expected validation error but got HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Invalid request format",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 5: Non-existent judge in insights endpoint
        try:
            response = self.session.get(f"{self.base_url}/litigation/judge-insights/NonExistentJudge123")
            
            # Should handle gracefully
            if response.status_code in [200, 404, 500]:
                self.log_test_result(
                    "Error Handling - Non-existent judge insights",
                    True,
                    f"Handled non-existent judge insights: HTTP {response.status_code}"
                )
            else:
                self.log_test_result(
                    "Error Handling - Non-existent judge insights",
                    False,
                    f"Unexpected response: HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling - Non-existent judge insights",
                False,
                f"Request failed: {str(e)}"
            )

    def run_comprehensive_tests(self):
        """Run all judge comparison tests"""
        print("🚀 STARTING COMPREHENSIVE JUDGE COMPARISON TESTING")
        print("=" * 80)
        
        # Run all test suites
        self.test_judge_comparison_endpoint()
        self.test_existing_judge_insights_endpoint()
        self.test_integration_workflow()
        self.test_error_scenarios()
        
        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("=" * 80)
        print("📋 JUDGE COMPARISON FUNCTIONALITY TEST REPORT")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.total_tests - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print()
        
        # Group results by test category
        category_results = {}
        for result in self.test_results:
            category = result["test_name"].split(" - ")[0]
            if category not in category_results:
                category_results[category] = {"passed": 0, "total": 0}
            category_results[category]["total"] += 1
            if result["success"]:
                category_results[category]["passed"] += 1
        
        print("📈 CATEGORY-SPECIFIC RESULTS:")
        for category, stats in category_results.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if rate == 100 else "⚠️" if rate >= 50 else "❌"
            print(f"   {status} {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
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
                "Judge Comparison", "Judge Insights", "Integration Workflow", "Error Handling"
            ]
            for feature in key_features:
                feature_tests = [t for t in successful_tests if feature in t["test_name"]]
                if feature_tests:
                    print(f"   • {feature}: {len(feature_tests)} tests passed")
            print()
        
        # Final assessment
        if success_rate >= 90:
            print("🎉 EXCELLENT: Judge Comparison functionality is fully operational!")
        elif success_rate >= 75:
            print("✅ GOOD: Judge Comparison functionality is mostly functional with minor issues.")
        elif success_rate >= 50:
            print("⚠️ MODERATE: Judge Comparison functionality has significant issues requiring attention.")
        else:
            print("❌ CRITICAL: Judge Comparison functionality has major problems.")
        
        print("=" * 80)
        
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "category_results": category_results,
            "failed_tests": failed_tests,
            "test_results": self.test_results
        }

def main():
    """Main testing function"""
    tester = JudgeComparisonBackendTester()
    results = tester.run_comprehensive_tests()
    
    # Return results for potential integration with other systems
    return results

if __name__ == "__main__":
    main()