#!/usr/bin/env python3
"""
Judicial Behavior Analyzer Fix Testing
=====================================

This script tests the specific fixes for the judicial behavior analyzer,
focusing on the judge comparison issue where different judges were getting
the same values instead of realistic different values.

SPECIFIC TESTING REQUIREMENTS:
1. Test individual judge analysis endpoint to verify different judges get different default profiles
2. Test judge comparison endpoint (/api/litigation/judge-comparison) with different judge names
3. Test with judges like "John Smith" and "Mary Johnson" to ensure seed-based randomization
4. Verify same judge queried multiple times gets same profile (consistent based on name hash)
5. Test confidence scores are appropriately low (0.15-0.45) indicating default profiles
6. Verify enhanced analysis summary mentions "Limited historical data available"

FIXES IMPLEMENTED:
- Updated _create_default_profile method to use judge name as seed for generating consistent but varied data
- Updated compare_judges method to accept force_refresh parameter
- Updated API endpoint to use force_refresh=True for fresh data
- Updated _get_default_insights method to also generate varied data
"""

import requests
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List

class JudicialBehaviorFixTester:
    def __init__(self):
        # Use the production backend URL from frontend/.env
        self.base_url = "https://legalengine.preview.emergentagent.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Test results tracking
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
        print("🎯 JUDICIAL BEHAVIOR ANALYZER FIX TESTING INITIATED")
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

    def test_individual_judge_different_profiles(self):
        """Test that different judges get different default profiles with varied settlement rates"""
        print("⚖️ TESTING INDIVIDUAL JUDGE DIFFERENT PROFILES")
        print("-" * 50)
        
        # Test judges as specified in the review request
        test_judges = ["John Smith", "Mary Johnson", "Robert Davis", "Sarah Wilson"]
        judge_profiles = {}
        
        for judge_name in test_judges:
            try:
                # URL encode the judge name
                encoded_judge_name = requests.utils.quote(judge_name)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    judge_profiles[judge_name] = data
                    
                    # Verify required fields exist
                    required_fields = [
                        "judge_name", "settlement_rate", "plaintiff_success_rate", 
                        "confidence_score", "overall_metrics"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        # Check confidence score is appropriately low (0.15-0.45)
                        confidence = data.get("confidence_score", 1.0)
                        confidence_ok = 0.15 <= confidence <= 0.45
                        
                        self.log_test_result(
                            f"Individual Judge Profile - {judge_name}",
                            True,
                            f"Settlement rate: {data['settlement_rate']:.2f}, Plaintiff success: {data['plaintiff_success_rate']:.2f}, Confidence: {confidence:.2f} ({'✓' if confidence_ok else '✗ should be 0.15-0.45'})"
                        )
                    else:
                        self.log_test_result(
                            f"Individual Judge Profile - {judge_name}",
                            False,
                            f"Missing required fields: {missing_fields}",
                            data
                        )
                else:
                    self.log_test_result(
                        f"Individual Judge Profile - {judge_name}",
                        False,
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Individual Judge Profile - {judge_name}",
                    False,
                    f"Request failed: {str(e)}"
                )
        
        # Now verify that different judges have different values
        if len(judge_profiles) >= 2:
            settlement_rates = [profile.get("settlement_rate", 0) for profile in judge_profiles.values()]
            plaintiff_rates = [profile.get("plaintiff_success_rate", 0) for profile in judge_profiles.values()]
            
            # Check if all settlement rates are different
            settlement_unique = len(set(settlement_rates)) == len(settlement_rates)
            plaintiff_unique = len(set(plaintiff_rates)) == len(plaintiff_rates)
            
            if settlement_unique and plaintiff_unique:
                self.log_test_result(
                    "Judge Profile Variation - Different Values",
                    True,
                    f"All judges have different settlement rates: {settlement_rates} and plaintiff success rates: {plaintiff_rates}"
                )
            else:
                self.log_test_result(
                    "Judge Profile Variation - Different Values",
                    False,
                    f"Some judges have identical values. Settlement rates: {settlement_rates}, Plaintiff rates: {plaintiff_rates}"
                )
        else:
            self.log_test_result(
                "Judge Profile Variation - Different Values",
                False,
                "Insufficient judge profiles to compare variation"
            )

    def test_judge_comparison_different_values(self):
        """Test judge comparison endpoint returns different values for different judges"""
        print("⚖️ TESTING JUDGE COMPARISON DIFFERENT VALUES")
        print("-" * 50)
        
        # Test Case 1: Compare John Smith vs Mary Johnson
        comparison_data = {
            "judge_names": ["John Smith", "Mary Johnson"],
            "case_type": "civil"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=comparison_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "judges_compared", "comparative_metrics", "recommendations", 
                    "analysis_date", "confidence_score"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    # Check comparative metrics for different values
                    metrics = data.get("comparative_metrics", {})
                    
                    # Look for settlement rate differences
                    settlement_differences = []
                    plaintiff_differences = []
                    
                    for key, value in metrics.items():
                        if "settlement" in key.lower() and isinstance(value, dict):
                            settlement_differences.extend(value.values())
                        elif "plaintiff" in key.lower() and isinstance(value, dict):
                            plaintiff_differences.extend(value.values())
                    
                    # Check if we have different values
                    has_different_settlements = len(set(settlement_differences)) > 1 if settlement_differences else False
                    has_different_plaintiff = len(set(plaintiff_differences)) > 1 if plaintiff_differences else False
                    
                    if has_different_settlements or has_different_plaintiff:
                        self.log_test_result(
                            "Judge Comparison - Different Values (John Smith vs Mary Johnson)",
                            True,
                            f"Comparison shows different values. Settlement differences: {has_different_settlements}, Plaintiff differences: {has_different_plaintiff}"
                        )
                    else:
                        self.log_test_result(
                            "Judge Comparison - Different Values (John Smith vs Mary Johnson)",
                            False,
                            f"Comparison shows identical values. Metrics: {metrics}"
                        )
                else:
                    self.log_test_result(
                        "Judge Comparison - Different Values (John Smith vs Mary Johnson)",
                        False,
                        f"Missing required fields: {missing_fields}",
                        data
                    )
            else:
                self.log_test_result(
                    "Judge Comparison - Different Values (John Smith vs Mary Johnson)",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Comparison - Different Values (John Smith vs Mary Johnson)",
                False,
                f"Request failed: {str(e)}"
            )

        # Test Case 2: Compare different set of judges
        comparison_data2 = {
            "judge_names": ["Robert Davis", "Sarah Wilson"],
            "case_type": "commercial"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/litigation/judge-comparison", json=comparison_data2)
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get("comparative_metrics", {})
                
                # Check if metrics contain varied data
                has_varied_data = False
                for key, value in metrics.items():
                    if isinstance(value, dict) and len(set(value.values())) > 1:
                        has_varied_data = True
                        break
                
                self.log_test_result(
                    "Judge Comparison - Different Values (Robert Davis vs Sarah Wilson)",
                    has_varied_data,
                    f"Comparison shows {'varied' if has_varied_data else 'identical'} values in metrics"
                )
            else:
                self.log_test_result(
                    "Judge Comparison - Different Values (Robert Davis vs Sarah Wilson)",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "Judge Comparison - Different Values (Robert Davis vs Sarah Wilson)",
                False,
                f"Request failed: {str(e)}"
            )

    def test_consistent_same_judge_queries(self):
        """Test that the same judge queried multiple times gets the same profile (consistent based on name hash)"""
        print("🔄 TESTING CONSISTENT SAME JUDGE QUERIES")
        print("-" * 50)
        
        judge_name = "John Smith"
        profiles = []
        
        # Query the same judge 3 times
        for i in range(3):
            try:
                encoded_judge_name = requests.utils.quote(judge_name)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    profiles.append(data)
                    
                    self.log_test_result(
                        f"Consistency Test - Query {i+1} for {judge_name}",
                        True,
                        f"Settlement rate: {data.get('settlement_rate', 0):.2f}, Plaintiff success: {data.get('plaintiff_success_rate', 0):.2f}"
                    )
                else:
                    self.log_test_result(
                        f"Consistency Test - Query {i+1} for {judge_name}",
                        False,
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Consistency Test - Query {i+1} for {judge_name}",
                    False,
                    f"Request failed: {str(e)}"
                )
            
            # Small delay between requests
            time.sleep(0.5)
        
        # Verify consistency
        if len(profiles) >= 2:
            # Check if settlement rates are consistent
            settlement_rates = [profile.get("settlement_rate", 0) for profile in profiles]
            plaintiff_rates = [profile.get("plaintiff_success_rate", 0) for profile in profiles]
            
            settlement_consistent = len(set(settlement_rates)) == 1
            plaintiff_consistent = len(set(plaintiff_rates)) == 1
            
            if settlement_consistent and plaintiff_consistent:
                self.log_test_result(
                    "Consistency Verification - Same Judge Multiple Queries",
                    True,
                    f"All queries returned identical values. Settlement: {settlement_rates[0]:.2f}, Plaintiff: {plaintiff_rates[0]:.2f}"
                )
            else:
                self.log_test_result(
                    "Consistency Verification - Same Judge Multiple Queries",
                    False,
                    f"Queries returned different values. Settlement rates: {settlement_rates}, Plaintiff rates: {plaintiff_rates}"
                )
        else:
            self.log_test_result(
                "Consistency Verification - Same Judge Multiple Queries",
                False,
                "Insufficient successful queries to verify consistency"
            )

    def test_confidence_scores_appropriate(self):
        """Test that confidence scores are appropriately low (0.15-0.45) indicating default profiles"""
        print("📊 TESTING CONFIDENCE SCORES APPROPRIATE RANGE")
        print("-" * 50)
        
        test_judges = ["John Smith", "Mary Johnson", "Robert Davis"]
        
        for judge_name in test_judges:
            try:
                encoded_judge_name = requests.utils.quote(judge_name)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    confidence = data.get("confidence_score", 1.0)
                    
                    # Check if confidence is in appropriate range (0.15-0.45)
                    confidence_appropriate = 0.15 <= confidence <= 0.45
                    
                    self.log_test_result(
                        f"Confidence Score Range - {judge_name}",
                        confidence_appropriate,
                        f"Confidence score: {confidence:.2f} ({'✓ appropriate' if confidence_appropriate else '✗ should be 0.15-0.45'})"
                    )
                else:
                    self.log_test_result(
                        f"Confidence Score Range - {judge_name}",
                        False,
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Confidence Score Range - {judge_name}",
                    False,
                    f"Request failed: {str(e)}"
                )

    def test_enhanced_analysis_summary(self):
        """Test that enhanced analysis summary mentions 'Limited historical data available'"""
        print("📝 TESTING ENHANCED ANALYSIS SUMMARY")
        print("-" * 50)
        
        test_judges = ["John Smith", "Mary Johnson"]
        
        for judge_name in test_judges:
            try:
                encoded_judge_name = requests.utils.quote(judge_name)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for analysis summary or similar field
                    summary_fields = ["ai_analysis_summary", "analysis_summary", "summary", "insights"]
                    summary_found = False
                    limited_data_mentioned = False
                    
                    for field in summary_fields:
                        if field in data and isinstance(data[field], str):
                            summary_found = True
                            summary_text = data[field].lower()
                            if "limited" in summary_text and ("historical" in summary_text or "data" in summary_text):
                                limited_data_mentioned = True
                                break
                    
                    if summary_found and limited_data_mentioned:
                        self.log_test_result(
                            f"Enhanced Analysis Summary - {judge_name}",
                            True,
                            "Analysis summary mentions limited historical data availability"
                        )
                    elif summary_found:
                        self.log_test_result(
                            f"Enhanced Analysis Summary - {judge_name}",
                            False,
                            "Analysis summary found but doesn't mention limited data availability"
                        )
                    else:
                        self.log_test_result(
                            f"Enhanced Analysis Summary - {judge_name}",
                            False,
                            "No analysis summary field found in response"
                        )
                else:
                    self.log_test_result(
                        f"Enhanced Analysis Summary - {judge_name}",
                        False,
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Enhanced Analysis Summary - {judge_name}",
                    False,
                    f"Request failed: {str(e)}"
                )

    def test_seed_based_randomization(self):
        """Test that seed-based randomization produces consistent but different values for each judge"""
        print("🎲 TESTING SEED-BASED RANDOMIZATION")
        print("-" * 50)
        
        # Test that the same judge name produces the same hash/seed
        judge_name = "John Smith"
        
        # Calculate expected seed (same logic as in the backend)
        expected_seed = int(hashlib.md5(judge_name.encode()).hexdigest()[:8], 16)
        
        # Query the judge multiple times and verify consistency
        profiles = []
        for i in range(2):
            try:
                encoded_judge_name = requests.utils.quote(judge_name)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    profiles.append(data)
                    
            except Exception as e:
                self.log_test_result(
                    f"Seed-based Randomization - Query {i+1}",
                    False,
                    f"Request failed: {str(e)}"
                )
        
        if len(profiles) >= 2:
            # Verify that values are identical (same seed produces same results)
            settlement_rates = [profile.get("settlement_rate", 0) for profile in profiles]
            consistent = len(set(settlement_rates)) == 1
            
            self.log_test_result(
                "Seed-based Randomization - Consistency",
                consistent,
                f"Expected seed: {expected_seed}, Settlement rates: {settlement_rates} ({'consistent' if consistent else 'inconsistent'})"
            )
        
        # Test that different judges produce different values
        different_judges = ["John Smith", "Mary Johnson"]
        judge_values = {}
        
        for judge in different_judges:
            try:
                encoded_judge_name = requests.utils.quote(judge)
                response = self.session.get(f"{self.base_url}/litigation/judge-insights/{encoded_judge_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    judge_values[judge] = {
                        'settlement_rate': data.get('settlement_rate', 0),
                        'plaintiff_success_rate': data.get('plaintiff_success_rate', 0)
                    }
                    
            except Exception as e:
                self.log_test_result(
                    f"Seed-based Randomization - {judge}",
                    False,
                    f"Request failed: {str(e)}"
                )
        
        if len(judge_values) >= 2:
            # Check if different judges have different values
            settlement_values = [values['settlement_rate'] for values in judge_values.values()]
            plaintiff_values = [values['plaintiff_success_rate'] for values in judge_values.values()]
            
            settlement_different = len(set(settlement_values)) > 1
            plaintiff_different = len(set(plaintiff_values)) > 1
            
            if settlement_different and plaintiff_different:
                self.log_test_result(
                    "Seed-based Randomization - Different Judges Different Values",
                    True,
                    f"Different judges produce different values: {judge_values}"
                )
            else:
                self.log_test_result(
                    "Seed-based Randomization - Different Judges Different Values",
                    False,
                    f"Different judges produce similar values: {judge_values}"
                )

    def run_comprehensive_tests(self):
        """Run all judicial behavior analyzer fix tests"""
        print("🚀 STARTING COMPREHENSIVE JUDICIAL BEHAVIOR ANALYZER FIX TESTING")
        print("=" * 80)
        
        # Run all test suites
        self.test_individual_judge_different_profiles()
        self.test_judge_comparison_different_values()
        self.test_consistent_same_judge_queries()
        self.test_confidence_scores_appropriate()
        self.test_enhanced_analysis_summary()
        self.test_seed_based_randomization()
        
        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("=" * 80)
        print("📋 JUDICIAL BEHAVIOR ANALYZER FIX TEST REPORT")
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
                "Individual Judge Profile", "Judge Comparison", "Consistency", 
                "Confidence Score", "Enhanced Analysis", "Seed-based Randomization"
            ]
            for feature in key_features:
                feature_tests = [t for t in successful_tests if feature in t["test_name"]]
                if feature_tests:
                    print(f"   • {feature}: {len(feature_tests)} tests passed")
            print()
        
        # Final assessment
        if success_rate >= 90:
            print("🎉 EXCELLENT: Judicial behavior analyzer fixes are working perfectly!")
            print("   ✅ Different judges now get different realistic values")
            print("   ✅ Same judge queries are consistent")
            print("   ✅ Confidence scores indicate default profiles appropriately")
        elif success_rate >= 75:
            print("✅ GOOD: Judicial behavior analyzer fixes are mostly working with minor issues.")
        elif success_rate >= 50:
            print("⚠️ MODERATE: Judicial behavior analyzer fixes have significant issues requiring attention.")
        else:
            print("❌ CRITICAL: Judicial behavior analyzer fixes are not working properly.")
        
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
    tester = JudicialBehaviorFixTester()
    results = tester.run_comprehensive_tests()
    
    # Return results for potential integration with other systems
    return results

if __name__ == "__main__":
    main()