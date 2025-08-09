#!/usr/bin/env python3
"""
Backend Testing Suite for Appeal Probability Analysis Metrics Clarification Fix
Testing Agent - Comprehensive Backend API Testing

Focus: Appeal Probability Analysis endpoint testing with metrics clarification verification
"""

import requests
import json
import sys
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/backend/.env')

# Get backend URL from frontend environment
frontend_env_path = '/app/frontend/.env'
backend_url = None

try:
    with open(frontend_env_path, 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                backend_url = line.split('=', 1)[1].strip()
                break
except Exception as e:
    print(f"❌ Error reading frontend .env: {e}")
    sys.exit(1)

if not backend_url:
    print("❌ REACT_APP_BACKEND_URL not found in frontend/.env")
    sys.exit(1)

# Construct API base URL
API_BASE_URL = f"{backend_url}/api"
print(f"🔗 Testing backend at: {API_BASE_URL}")

class AppealAnalysisTestSuite:
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = f"{status}: {test_name}"
        if details:
            result += f" - {details}"
        
        print(result)
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        
    def test_appeal_analysis_endpoint_basic(self):
        """Test 1: Basic Appeal Analysis Endpoint Availability"""
        try:
            # Test with minimal required data
            test_data = {
                "case_type": "civil",
                "jurisdiction": "federal"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/litigation/appeal-analysis",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    'appeal_probability', 'appeal_confidence', 'appeal_factors',
                    'appeal_timeline', 'appeal_cost_estimate', 'appeal_success_probability',
                    'preventive_measures', 'jurisdictional_appeal_rate'
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("Appeal Analysis Endpoint Basic", True, 
                                f"All required fields present: {required_fields}")
                else:
                    self.log_test("Appeal Analysis Endpoint Basic", False, 
                                f"Missing fields: {missing_fields}")
            else:
                self.log_test("Appeal Analysis Endpoint Basic", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Appeal Analysis Endpoint Basic", False, f"Exception: {str(e)}")
    
    def test_user_specific_scenario(self):
        """Test 2: User's Exact Scenario from Review Request"""
        try:
            # Exact user scenario from review request
            test_data = {
                "case_type": "civil",
                "jurisdiction": "federal",
                "case_value": 250000,
                "judge_name": "Judge Rebecca Morgan",
                "evidence_strength": 7.0,
                "case_complexity": 0.65,
                "case_facts": "Plaintiff alleges breach of contract by federal contractor involving delayed delivery of critical medical equipment. Agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
            }
            
            response = requests.post(
                f"{API_BASE_URL}/litigation/appeal-analysis",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify both metrics are present and different
                appeal_probability = data.get('appeal_probability', 0)
                appeal_success_probability = data.get('appeal_success_probability', 0)
                
                # Check that both metrics are reasonable values
                if 0 <= appeal_probability <= 1 and 0 <= appeal_success_probability <= 1:
                    # Verify they represent different aspects (should be different values)
                    if appeal_probability != appeal_success_probability:
                        self.log_test("User Specific Scenario", True, 
                                    f"Appeal filing probability: {appeal_probability:.1%}, " +
                                    f"Appeal success probability: {appeal_success_probability:.1%}")
                    else:
                        self.log_test("User Specific Scenario", False, 
                                    f"Both metrics identical: {appeal_probability:.1%} - should be different")
                else:
                    self.log_test("User Specific Scenario", False, 
                                f"Invalid probability values: filing={appeal_probability}, success={appeal_success_probability}")
            else:
                self.log_test("User Specific Scenario", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("User Specific Scenario", False, f"Exception: {str(e)}")
    
    def test_metrics_separation_verification(self):
        """Test 3: Verify Metric Separation - Appeal Filing vs Success Probability"""
        try:
            # Test with strong evidence (should have low appeal filing probability)
            strong_evidence_data = {
                "case_type": "civil",
                "jurisdiction": "federal",
                "case_value": 250000,
                "evidence_strength": 9.0,
                "case_complexity": 0.3,
                "case_facts": "Clear-cut breach of contract with overwhelming evidence including signed contracts, witness testimony, and documented damages."
            }
            
            response = requests.post(
                f"{API_BASE_URL}/litigation/appeal-analysis",
                json=strong_evidence_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                appeal_filing_prob = data.get('appeal_probability', 0)
                appeal_success_prob = data.get('appeal_success_probability', 0)
                
                # With strong evidence, appeal filing probability should be low
                # but success probability (if filed) could be moderate
                if appeal_filing_prob < 0.2:  # Low filing probability with strong evidence
                    self.log_test("Metrics Separation - Strong Evidence", True, 
                                f"Low appeal filing probability ({appeal_filing_prob:.1%}) with strong evidence")
                else:
                    self.log_test("Metrics Separation - Strong Evidence", False, 
                                f"Expected low filing probability with strong evidence, got {appeal_filing_prob:.1%}")
                
                # Test with weak evidence (should have higher appeal filing probability)
                weak_evidence_data = {
                    "case_type": "civil",
                    "jurisdiction": "federal", 
                    "case_value": 250000,
                    "evidence_strength": 3.0,
                    "case_complexity": 0.8,
                    "case_facts": "Disputed contract terms with limited documentation and conflicting witness accounts."
                }
                
                response2 = requests.post(
                    f"{API_BASE_URL}/litigation/appeal-analysis",
                    json=weak_evidence_data,
                    timeout=30
                )
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    weak_appeal_filing_prob = data2.get('appeal_probability', 0)
                    
                    # Weak evidence should result in higher appeal filing probability
                    if weak_appeal_filing_prob > appeal_filing_prob:
                        self.log_test("Metrics Separation - Evidence Correlation", True, 
                                    f"Weak evidence higher filing prob ({weak_appeal_filing_prob:.1%}) vs strong ({appeal_filing_prob:.1%})")
                    else:
                        self.log_test("Metrics Separation - Evidence Correlation", False, 
                                    f"Expected higher filing prob with weak evidence")
                else:
                    self.log_test("Metrics Separation - Evidence Correlation", False, 
                                f"Weak evidence test failed: HTTP {response2.status_code}")
            else:
                self.log_test("Metrics Separation - Strong Evidence", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Metrics Separation Verification", False, f"Exception: {str(e)}")
    
    def test_ai_analysis_mode_verification(self):
        """Test 4: Verify AI Analysis Mode (90% confidence vs fallback)"""
        try:
            # Test with detailed case facts to trigger full AI analysis
            detailed_case_data = {
                "case_type": "civil",
                "jurisdiction": "federal",
                "case_value": 250000,
                "judge_name": "Judge Rebecca Morgan",
                "evidence_strength": 7.0,
                "case_complexity": 0.65,
                "case_facts": "Plaintiff alleges breach of contract by federal contractor involving delayed delivery of critical medical equipment. Agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
            }
            
            response = requests.post(
                f"{API_BASE_URL}/litigation/appeal-analysis",
                json=detailed_case_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                confidence = data.get('appeal_confidence', 0)
                appeal_factors = data.get('appeal_factors', [])
                
                # Full AI analysis should have high confidence (90%+) and detailed factors
                if confidence >= 0.85:  # 85%+ indicates full AI analysis
                    if len(appeal_factors) >= 3:  # Should have multiple detailed factors
                        self.log_test("AI Analysis Mode", True, 
                                    f"High confidence ({confidence:.1%}) with {len(appeal_factors)} factors")
                    else:
                        self.log_test("AI Analysis Mode", False, 
                                    f"High confidence but few factors: {len(appeal_factors)}")
                else:
                    self.log_test("AI Analysis Mode", False, 
                                f"Low confidence ({confidence:.1%}) suggests fallback mode")
            else:
                self.log_test("AI Analysis Mode", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("AI Analysis Mode Verification", False, f"Exception: {str(e)}")
    
    def test_response_structure_validation(self):
        """Test 5: Validate Complete Response Structure"""
        try:
            test_data = {
                "case_type": "civil",
                "jurisdiction": "federal",
                "case_value": 250000,
                "judge_name": "Judge Rebecca Morgan",
                "evidence_strength": 7.0,
                "case_complexity": 0.65,
                "case_facts": "Contract breach case with medical equipment delivery delays."
            }
            
            response = requests.post(
                f"{API_BASE_URL}/litigation/appeal-analysis",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate all required fields with proper types
                validations = [
                    ('appeal_probability', float, lambda x: 0 <= x <= 1),
                    ('appeal_confidence', float, lambda x: 0 <= x <= 1),
                    ('appeal_factors', list, lambda x: len(x) > 0),
                    ('appeal_timeline', (int, type(None)), lambda x: x is None or x > 0),
                    ('appeal_cost_estimate', (float, type(None)), lambda x: x is None or x > 0),
                    ('appeal_success_probability', (float, type(None)), lambda x: x is None or 0 <= x <= 1),
                    ('preventive_measures', list, lambda x: True),
                    ('jurisdictional_appeal_rate', (float, type(None)), lambda x: x is None or 0 <= x <= 1)
                ]
                
                validation_results = []
                for field, expected_type, validator in validations:
                    if field in data:
                        value = data[field]
                        if isinstance(value, expected_type) and validator(value):
                            validation_results.append(f"✓ {field}")
                        else:
                            validation_results.append(f"✗ {field} (invalid value/type)")
                    else:
                        validation_results.append(f"✗ {field} (missing)")
                
                passed_validations = len([r for r in validation_results if r.startswith('✓')])
                total_validations = len(validation_results)
                
                if passed_validations == total_validations:
                    self.log_test("Response Structure Validation", True, 
                                f"All {total_validations} fields valid")
                else:
                    self.log_test("Response Structure Validation", False, 
                                f"{passed_validations}/{total_validations} fields valid: {validation_results}")
            else:
                self.log_test("Response Structure Validation", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Response Structure Validation", False, f"Exception: {str(e)}")
    
    def test_edge_cases(self):
        """Test 6: Edge Cases - Different Case Values and Evidence Strengths"""
        try:
            edge_cases = [
                {
                    "name": "High Value Case",
                    "data": {
                        "case_type": "civil",
                        "jurisdiction": "federal",
                        "case_value": 5000000,  # $5M case
                        "evidence_strength": 8.0,
                        "case_complexity": 0.4
                    }
                },
                {
                    "name": "Low Value Case", 
                    "data": {
                        "case_type": "civil",
                        "jurisdiction": "federal",
                        "case_value": 50000,  # $50K case
                        "evidence_strength": 6.0,
                        "case_complexity": 0.3
                    }
                },
                {
                    "name": "Complex Case",
                    "data": {
                        "case_type": "civil",
                        "jurisdiction": "federal",
                        "case_value": 250000,
                        "evidence_strength": 5.0,
                        "case_complexity": 0.9  # Very complex
                    }
                }
            ]
            
            edge_case_results = []
            
            for case in edge_cases:
                response = requests.post(
                    f"{API_BASE_URL}/litigation/appeal-analysis",
                    json=case["data"],
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    appeal_prob = data.get('appeal_probability', 0)
                    success_prob = data.get('appeal_success_probability', 0)
                    
                    edge_case_results.append(f"{case['name']}: filing={appeal_prob:.1%}, success={success_prob:.1%}")
                else:
                    edge_case_results.append(f"{case['name']}: FAILED ({response.status_code})")
            
            # Check if we got reasonable results for all edge cases
            successful_cases = len([r for r in edge_case_results if "FAILED" not in r])
            
            if successful_cases == len(edge_cases):
                self.log_test("Edge Cases", True, f"All {successful_cases} cases processed: {edge_case_results}")
            else:
                self.log_test("Edge Cases", False, f"Only {successful_cases}/{len(edge_cases)} cases successful")
                
        except Exception as e:
            self.log_test("Edge Cases", False, f"Exception: {str(e)}")
    
    def run_all_tests(self):
        """Run all appeal analysis tests"""
        print("🎯 APPEAL PROBABILITY ANALYSIS METRICS CLARIFICATION FIX TESTING")
        print("=" * 80)
        print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all tests
        self.test_appeal_analysis_endpoint_basic()
        self.test_user_specific_scenario()
        self.test_metrics_separation_verification()
        self.test_ai_analysis_mode_verification()
        self.test_response_structure_validation()
        self.test_edge_cases()
        
        # Print summary
        print()
        print("=" * 80)
        print("🎯 APPEAL ANALYSIS TESTING SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"📊 Overall Success Rate: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests} tests passed)")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT: Appeal Analysis functionality is working well!")
        elif success_rate >= 60:
            print("⚠️  GOOD: Appeal Analysis mostly working with some issues")
        else:
            print("❌ NEEDS ATTENTION: Appeal Analysis has significant issues")
        
        print()
        print("🔍 KEY FINDINGS:")
        
        # Analyze results for key findings
        if any("Appeal filing probability" in r['details'] for r in self.test_results if r['passed']):
            print("✅ Metrics separation working - appeal filing vs success probability correctly calculated")
        
        if any("High confidence" in r['details'] for r in self.test_results if r['passed']):
            print("✅ AI analysis mode operational - high confidence scores indicate full AI processing")
        
        if any("All" in r['details'] and "fields valid" in r['details'] for r in self.test_results if r['passed']):
            print("✅ Response structure complete - all required fields present and valid")
        
        failed_tests = [r for r in self.test_results if not r['passed']]
        if failed_tests:
            print()
            print("❌ ISSUES FOUND:")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        return success_rate >= 80

if __name__ == "__main__":
    test_suite = AppealAnalysisTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 TESTING COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n⚠️  TESTING COMPLETED WITH ISSUES")
        sys.exit(1)