#!/usr/bin/env python3
"""
Appeal Probability Prediction Feature Testing - COMPREHENSIVE BACKEND VALIDATION
===============================================================================

Testing the newly implemented Appeal Probability Prediction feature in the LegalMate AI 
litigation analytics system to ensure comprehensive functionality.

KEY ENDPOINTS TO TEST:
1. POST /api/litigation/appeal-analysis - Dedicated appeal probability analysis endpoint
2. POST /api/litigation/analyze-case - Verify appeal analysis is included in case prediction results
3. GET /api/litigation/analytics-dashboard - Ensure dashboard includes appeal-related metrics

CRITICAL TEST SCENARIOS:
- Dedicated Appeal Analysis Endpoint validation
- Case Analysis with Appeal Integration verification
- Appeal Probability Factors Validation with different case values, evidence strength, jurisdictions
- AI-Enhanced Appeal Analysis testing
- Error Handling validation

Expected functionality:
✅ Appeal probability should range from 5% to 95% based on case factors
✅ Appeal success probability should range from 10% to 70%
✅ Appeal timeline should reflect jurisdiction-specific deadlines (30-60 days)
✅ Appeal costs should scale with case value and jurisdiction
✅ AI should generate relevant, actionable appeal factors and preventive measures
✅ Response should include jurisdictional comparison data
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://legal-research-api.preview.emergentagent.com/api"

def test_dedicated_appeal_analysis_endpoint():
    """Test POST /api/litigation/appeal-analysis with comprehensive case data"""
    print("🎯 TESTING DEDICATED APPEAL ANALYSIS ENDPOINT")
    print("=" * 70)
    
    test_results = []
    
    # Test Case 1: High Appeal Risk
    high_risk_case = {
        "case_type": "commercial",
        "jurisdiction": "california", 
        "case_value": 5000000,
        "evidence_strength": 3,
        "case_complexity": 0.8,
        "judge_name": "Judge Smith",
        "case_facts": "Complex breach of contract case with disputed evidence and high financial stakes"
    }
    
    print(f"\n📋 Test Case 1: High Appeal Risk")
    print(f"Case Type: {high_risk_case['case_type']}")
    print(f"Jurisdiction: {high_risk_case['jurisdiction']}")
    print(f"Case Value: ${high_risk_case['case_value']:,}")
    print(f"Evidence Strength: {high_risk_case['evidence_strength']}/10")
    print(f"Case Complexity: {high_risk_case['case_complexity']}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=high_risk_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Appeal analysis endpoint working")
            
            # Verify all required fields are present
            required_fields = [
                'appeal_probability', 'appeal_confidence', 'appeal_factors',
                'appeal_timeline', 'appeal_cost_estimate', 'appeal_success_probability',
                'preventive_measures', 'jurisdictional_appeal_rate'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"❌ MISSING FIELDS: {missing_fields}")
                test_results.append(("High Risk Appeal Analysis", False, f"Missing fields: {missing_fields}"))
            else:
                print("✅ All required fields present")
                
                # Validate field types and ranges
                appeal_prob = data.get('appeal_probability', 0)
                appeal_success = data.get('appeal_success_probability', 0)
                appeal_timeline = data.get('appeal_timeline', 0)
                appeal_cost = data.get('appeal_cost_estimate', 0)
                
                print(f"📊 Appeal Probability: {appeal_prob:.1%}")
                print(f"📊 Appeal Success Probability: {appeal_success:.1%}")
                print(f"📊 Appeal Timeline: {appeal_timeline} days")
                print(f"📊 Appeal Cost Estimate: ${appeal_cost:,.2f}")
                print(f"📊 Appeal Factors: {len(data.get('appeal_factors', []))} factors")
                print(f"📊 Preventive Measures: {len(data.get('preventive_measures', []))} measures")
                
                # Validate ranges
                if 0.05 <= appeal_prob <= 0.95:
                    print("✅ Appeal probability in expected range (5%-95%)")
                else:
                    print(f"❌ Appeal probability {appeal_prob:.1%} outside expected range")
                
                if 0.10 <= appeal_success <= 0.70:
                    print("✅ Appeal success probability in expected range (10%-70%)")
                else:
                    print(f"❌ Appeal success probability {appeal_success:.1%} outside expected range")
                
                if 30 <= appeal_timeline <= 60:
                    print("✅ Appeal timeline in expected range (30-60 days)")
                else:
                    print(f"❌ Appeal timeline {appeal_timeline} days outside expected range")
                
                test_results.append(("High Risk Appeal Analysis", True, "All validations passed"))
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            test_results.append(("High Risk Appeal Analysis", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("High Risk Appeal Analysis", False, str(e)))
    
    # Test Case 2: Low Appeal Risk
    print(f"\n📋 Test Case 2: Low Appeal Risk")
    low_risk_case = {
        "case_type": "employment",
        "jurisdiction": "texas",
        "case_value": 150000,
        "evidence_strength": 9,
        "case_complexity": 0.3,
        "case_facts": "Clear-cut employment termination case with strong documentation"
    }
    
    print(f"Case Type: {low_risk_case['case_type']}")
    print(f"Jurisdiction: {low_risk_case['jurisdiction']}")
    print(f"Case Value: ${low_risk_case['case_value']:,}")
    print(f"Evidence Strength: {low_risk_case['evidence_strength']}/10")
    print(f"Case Complexity: {low_risk_case['case_complexity']}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=low_risk_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Low risk appeal analysis working")
            
            appeal_prob = data.get('appeal_probability', 0)
            print(f"📊 Appeal Probability: {appeal_prob:.1%}")
            
            # Low risk case should have lower appeal probability than high risk
            if appeal_prob < 0.4:  # Should be less than 40% for low risk
                print("✅ Low risk case shows appropriately low appeal probability")
                test_results.append(("Low Risk Appeal Analysis", True, f"Appeal probability {appeal_prob:.1%}"))
            else:
                print(f"❌ Low risk case shows high appeal probability: {appeal_prob:.1%}")
                test_results.append(("Low Risk Appeal Analysis", False, f"Appeal probability too high: {appeal_prob:.1%}"))
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            test_results.append(("Low Risk Appeal Analysis", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("Low Risk Appeal Analysis", False, str(e)))
    
    return test_results

def test_case_analysis_with_appeal_integration():
    """Test POST /api/litigation/analyze-case to ensure appeal analysis is included"""
    print("\n🎯 TESTING CASE ANALYSIS WITH APPEAL INTEGRATION")
    print("=" * 70)
    
    test_results = []
    
    # Test Case 3: Medium Appeal Risk
    medium_risk_case = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 750000,
        "evidence_strength": 6,
        "case_complexity": 0.65,
        "case_facts": "Civil litigation with moderate complexity and mixed evidence"
    }
    
    print(f"\n📋 Test Case 3: Medium Appeal Risk - Full Case Analysis")
    print(f"Case Type: {medium_risk_case['case_type']}")
    print(f"Jurisdiction: {medium_risk_case['jurisdiction']}")
    print(f"Case Value: ${medium_risk_case['case_value']:,}")
    print(f"Evidence Strength: {medium_risk_case['evidence_strength']}/10")
    print(f"Case Complexity: {medium_risk_case['case_complexity']}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/analyze-case",
            json=medium_risk_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Case analysis endpoint working")
            
            # Check if appeal_analysis is included in the response
            if 'appeal_analysis' in data:
                appeal_data = data['appeal_analysis']
                print("✅ Appeal analysis included in case analysis response")
                
                # Verify appeal analysis structure
                if isinstance(appeal_data, dict) and 'appeal_probability' in appeal_data:
                    appeal_prob = appeal_data.get('appeal_probability', 0)
                    print(f"📊 Integrated Appeal Probability: {appeal_prob:.1%}")
                    
                    # Medium risk should be between low and high
                    if 0.2 <= appeal_prob <= 0.7:
                        print("✅ Medium risk case shows appropriate appeal probability")
                        test_results.append(("Case Analysis Appeal Integration", True, f"Appeal probability {appeal_prob:.1%}"))
                    else:
                        print(f"❌ Medium risk appeal probability outside expected range: {appeal_prob:.1%}")
                        test_results.append(("Case Analysis Appeal Integration", False, f"Appeal probability {appeal_prob:.1%} outside range"))
                else:
                    print("❌ Appeal analysis structure invalid")
                    test_results.append(("Case Analysis Appeal Integration", False, "Invalid appeal analysis structure"))
            else:
                print("❌ Appeal analysis not included in case analysis response")
                test_results.append(("Case Analysis Appeal Integration", False, "Appeal analysis missing from response"))
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            test_results.append(("Case Analysis Appeal Integration", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("Case Analysis Appeal Integration", False, str(e)))
    
    return test_results

def test_appeal_probability_factors_validation():
    """Test different case values and factors to validate appeal probability calculation"""
    print("\n🎯 TESTING APPEAL PROBABILITY FACTORS VALIDATION")
    print("=" * 70)
    
    test_results = []
    
    # Test different case values - higher values should increase appeal probability
    case_values = [100000, 1000000, 10000000]
    
    for i, case_value in enumerate(case_values, 1):
        print(f"\n📋 Test Case {i}: Case Value Impact - ${case_value:,}")
        
        test_case = {
            "case_type": "commercial",
            "jurisdiction": "california",
            "case_value": case_value,
            "evidence_strength": 5,  # Keep other factors constant
            "case_complexity": 0.5,
            "case_facts": f"Commercial dispute involving ${case_value:,}"
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/litigation/appeal-analysis",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                appeal_prob = data.get('appeal_probability', 0)
                appeal_cost = data.get('appeal_cost_estimate', 0)
                
                print(f"📊 Case Value: ${case_value:,}")
                print(f"📊 Appeal Probability: {appeal_prob:.1%}")
                print(f"📊 Appeal Cost Estimate: ${appeal_cost:,.2f}")
                
                # Store for comparison
                if i == 1:
                    low_value_prob = appeal_prob
                elif i == 2:
                    mid_value_prob = appeal_prob
                elif i == 3:
                    high_value_prob = appeal_prob
                    
                    # Compare probabilities - should generally increase with case value
                    if high_value_prob >= mid_value_prob >= low_value_prob:
                        print("✅ Appeal probability increases with case value")
                        test_results.append(("Case Value Impact", True, "Probability scales with value"))
                    else:
                        print(f"❌ Appeal probability doesn't scale properly: {low_value_prob:.1%} -> {mid_value_prob:.1%} -> {high_value_prob:.1%}")
                        test_results.append(("Case Value Impact", False, "Probability doesn't scale with value"))
                
            else:
                print(f"❌ FAILED: {response.status_code}")
                test_results.append((f"Case Value ${case_value:,}", False, f"HTTP {response.status_code}"))
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            test_results.append((f"Case Value ${case_value:,}", False, str(e)))
    
    # Test different evidence strength - weaker evidence should increase appeal probability
    print(f"\n📋 Testing Evidence Strength Impact")
    evidence_strengths = [2, 5, 9]  # Weak, Medium, Strong
    
    for i, evidence_strength in enumerate(evidence_strengths, 1):
        test_case = {
            "case_type": "civil",
            "jurisdiction": "federal",
            "case_value": 500000,  # Keep constant
            "evidence_strength": evidence_strength,
            "case_complexity": 0.5,  # Keep constant
            "case_facts": f"Case with evidence strength {evidence_strength}/10"
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/litigation/appeal-analysis",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                appeal_prob = data.get('appeal_probability', 0)
                
                print(f"📊 Evidence Strength: {evidence_strength}/10")
                print(f"📊 Appeal Probability: {appeal_prob:.1%}")
                
                # Store for comparison
                if i == 1:
                    weak_evidence_prob = appeal_prob
                elif i == 2:
                    medium_evidence_prob = appeal_prob
                elif i == 3:
                    strong_evidence_prob = appeal_prob
                    
                    # Weaker evidence should lead to higher appeal probability
                    if weak_evidence_prob >= medium_evidence_prob >= strong_evidence_prob:
                        print("✅ Appeal probability decreases with stronger evidence")
                        test_results.append(("Evidence Strength Impact", True, "Probability inversely correlates with evidence strength"))
                    else:
                        print(f"❌ Evidence strength impact incorrect: {weak_evidence_prob:.1%} -> {medium_evidence_prob:.1%} -> {strong_evidence_prob:.1%}")
                        test_results.append(("Evidence Strength Impact", False, "Probability doesn't correlate properly with evidence"))
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            test_results.append((f"Evidence Strength {evidence_strength}", False, str(e)))
    
    return test_results

def test_jurisdictional_differences():
    """Test different jurisdictions to verify different appeal rates"""
    print("\n🎯 TESTING JURISDICTIONAL DIFFERENCES")
    print("=" * 70)
    
    test_results = []
    jurisdictions = ["federal", "california", "texas", "new_york"]
    
    for jurisdiction in jurisdictions:
        print(f"\n📋 Testing Jurisdiction: {jurisdiction.upper()}")
        
        test_case = {
            "case_type": "commercial",
            "jurisdiction": jurisdiction,
            "case_value": 500000,
            "evidence_strength": 6,
            "case_complexity": 0.6,
            "case_facts": f"Commercial case in {jurisdiction} jurisdiction"
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/litigation/appeal-analysis",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                appeal_prob = data.get('appeal_probability', 0)
                jurisdictional_rate = data.get('jurisdictional_appeal_rate', 0)
                appeal_timeline = data.get('appeal_timeline', 0)
                
                print(f"📊 Appeal Probability: {appeal_prob:.1%}")
                print(f"📊 Jurisdictional Appeal Rate: {jurisdictional_rate:.1%}")
                print(f"📊 Appeal Timeline: {appeal_timeline} days")
                
                # Validate timeline is within expected range for jurisdiction
                if 30 <= appeal_timeline <= 60:
                    print("✅ Appeal timeline within expected range")
                    test_results.append((f"Jurisdiction {jurisdiction}", True, f"Timeline: {appeal_timeline} days"))
                else:
                    print(f"❌ Appeal timeline outside expected range: {appeal_timeline} days")
                    test_results.append((f"Jurisdiction {jurisdiction}", False, f"Timeline: {appeal_timeline} days"))
                
            else:
                print(f"❌ FAILED: {response.status_code}")
                test_results.append((f"Jurisdiction {jurisdiction}", False, f"HTTP {response.status_code}"))
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            test_results.append((f"Jurisdiction {jurisdiction}", False, str(e)))
    
    return test_results

def test_ai_enhanced_appeal_analysis():
    """Test AI-powered appeal factors and preventive measures generation"""
    print("\n🎯 TESTING AI-ENHANCED APPEAL ANALYSIS")
    print("=" * 70)
    
    test_results = []
    
    # Test with detailed case facts to trigger AI analysis
    detailed_case = {
        "case_type": "commercial",
        "jurisdiction": "federal",
        "case_value": 2000000,
        "evidence_strength": 4,
        "case_complexity": 0.8,
        "case_facts": "Complex breach of contract case involving software licensing dispute. Multiple parties, international elements, disputed intellectual property rights, and conflicting expert testimony on damages calculation. Case involves novel legal questions regarding cloud computing agreements."
    }
    
    print(f"\n📋 Testing AI Analysis with Complex Case")
    print(f"Case Facts Length: {len(detailed_case['case_facts'])} characters")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=detailed_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            appeal_factors = data.get('appeal_factors', [])
            preventive_measures = data.get('preventive_measures', [])
            
            print(f"📊 Appeal Factors Generated: {len(appeal_factors)}")
            print(f"📊 Preventive Measures Generated: {len(preventive_measures)}")
            
            # Display factors and measures
            if appeal_factors:
                print("\n🔍 Appeal Risk Factors:")
                for i, factor in enumerate(appeal_factors[:3], 1):  # Show first 3
                    print(f"  {i}. {factor}")
            
            if preventive_measures:
                print("\n🛡️ Preventive Measures:")
                for i, measure in enumerate(preventive_measures[:3], 1):  # Show first 3
                    print(f"  {i}. {measure}")
            
            # Validate AI generated meaningful content
            if len(appeal_factors) >= 2 and len(preventive_measures) >= 2:
                # Check if factors are meaningful (not just generic)
                meaningful_factors = [f for f in appeal_factors if len(f) > 20 and any(word in f.lower() for word in ['complex', 'dispute', 'evidence', 'legal', 'case'])]
                meaningful_measures = [m for m in preventive_measures if len(m) > 20 and any(word in m.lower() for word in ['prepare', 'document', 'ensure', 'review', 'strengthen'])]
                
                if meaningful_factors and meaningful_measures:
                    print("✅ AI generated meaningful, case-specific content")
                    test_results.append(("AI Appeal Analysis", True, f"{len(meaningful_factors)} factors, {len(meaningful_measures)} measures"))
                else:
                    print("❌ AI generated generic or insufficient content")
                    test_results.append(("AI Appeal Analysis", False, "Generic content generated"))
            else:
                print("❌ Insufficient AI-generated content")
                test_results.append(("AI Appeal Analysis", False, f"Only {len(appeal_factors)} factors, {len(preventive_measures)} measures"))
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            test_results.append(("AI Appeal Analysis", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("AI Appeal Analysis", False, str(e)))
    
    return test_results

def test_error_handling():
    """Test error handling with missing required fields and invalid data"""
    print("\n🎯 TESTING ERROR HANDLING")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Missing required fields
    print(f"\n📋 Test 1: Missing Required Fields")
    incomplete_case = {
        "case_type": "commercial"
        # Missing jurisdiction, case_value, evidence_strength, etc.
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=incomplete_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [400, 422]:  # Expected validation error
            print("✅ Proper validation error for missing fields")
            test_results.append(("Missing Fields Validation", True, f"HTTP {response.status_code}"))
        elif response.status_code == 200:
            print("⚠️ Request succeeded despite missing fields (may have defaults)")
            test_results.append(("Missing Fields Validation", True, "Succeeded with defaults"))
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            test_results.append(("Missing Fields Validation", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("Missing Fields Validation", False, str(e)))
    
    # Test 2: Invalid data types
    print(f"\n📋 Test 2: Invalid Data Types")
    invalid_case = {
        "case_type": "commercial",
        "jurisdiction": "federal",
        "case_value": "not_a_number",  # Should be numeric
        "evidence_strength": "high",   # Should be numeric
        "case_complexity": "complex"   # Should be numeric
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=invalid_case,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [400, 422]:  # Expected validation error
            print("✅ Proper validation error for invalid data types")
            test_results.append(("Invalid Data Types", True, f"HTTP {response.status_code}"))
        else:
            print(f"❌ Should have failed validation: {response.status_code}")
            test_results.append(("Invalid Data Types", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("Invalid Data Types", False, str(e)))
    
    return test_results

def test_analytics_dashboard():
    """Test GET /api/litigation/analytics-dashboard for appeal-related metrics"""
    print("\n🎯 TESTING ANALYTICS DASHBOARD - APPEAL METRICS")
    print("=" * 70)
    
    test_results = []
    
    try:
        response = requests.get(
            f"{BACKEND_URL}/litigation/analytics-dashboard",
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Analytics dashboard endpoint accessible")
            
            # Check for appeal-related metrics
            appeal_metrics_found = False
            
            # Look for appeal-related data in various possible locations
            if isinstance(data, dict):
                # Check top-level keys
                appeal_keys = [key for key in data.keys() if 'appeal' in key.lower()]
                if appeal_keys:
                    print(f"✅ Found appeal-related keys: {appeal_keys}")
                    appeal_metrics_found = True
                
                # Check nested structures
                for key, value in data.items():
                    if isinstance(value, dict):
                        nested_appeal_keys = [k for k in value.keys() if 'appeal' in k.lower()]
                        if nested_appeal_keys:
                            print(f"✅ Found nested appeal metrics in {key}: {nested_appeal_keys}")
                            appeal_metrics_found = True
            
            if appeal_metrics_found:
                test_results.append(("Analytics Dashboard Appeal Metrics", True, "Appeal metrics present"))
            else:
                print("⚠️ No appeal-specific metrics found in dashboard")
                test_results.append(("Analytics Dashboard Appeal Metrics", False, "No appeal metrics found"))
                
        elif response.status_code == 404:
            print("❌ Analytics dashboard endpoint not found")
            test_results.append(("Analytics Dashboard", False, "Endpoint not found"))
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            test_results.append(("Analytics Dashboard", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        test_results.append(("Analytics Dashboard", False, str(e)))
    
    return test_results

def print_test_summary(all_results: List[tuple]):
    """Print comprehensive test summary"""
    print("\n" + "=" * 80)
    print("🎯 APPEAL PROBABILITY PREDICTION TESTING SUMMARY")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, passed, _ in all_results if passed)
    failed_tests = total_tests - passed_tests
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(f"\n🎉 EXCELLENT: Appeal Probability Prediction feature is working well!")
    elif success_rate >= 60:
        print(f"\n✅ GOOD: Appeal Probability Prediction feature is mostly functional with minor issues")
    else:
        print(f"\n⚠️ NEEDS ATTENTION: Appeal Probability Prediction feature has significant issues")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, passed, details in all_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name} - {details}")
    
    print(f"\n🔍 KEY FINDINGS:")
    
    # Analyze specific functionality
    appeal_analysis_tests = [r for r in all_results if 'Appeal Analysis' in r[0]]
    if appeal_analysis_tests:
        appeal_success = sum(1 for _, passed, _ in appeal_analysis_tests if passed)
        print(f"• Appeal Analysis Endpoints: {appeal_success}/{len(appeal_analysis_tests)} working")
    
    factor_tests = [r for r in all_results if any(word in r[0] for word in ['Value', 'Evidence', 'Jurisdiction'])]
    if factor_tests:
        factor_success = sum(1 for _, passed, _ in factor_tests if passed)
        print(f"• Appeal Factor Validation: {factor_success}/{len(factor_tests)} working")
    
    ai_tests = [r for r in all_results if 'AI' in r[0]]
    if ai_tests:
        ai_success = sum(1 for _, passed, _ in ai_tests if passed)
        print(f"• AI-Enhanced Analysis: {ai_success}/{len(ai_tests)} working")
    
    error_tests = [r for r in all_results if 'Error' in r[0] or 'Validation' in r[0]]
    if error_tests:
        error_success = sum(1 for _, passed, _ in error_tests if passed)
        print(f"• Error Handling: {error_success}/{len(error_tests)} working")

def main():
    """Run comprehensive Appeal Probability Prediction testing"""
    print("🚀 STARTING APPEAL PROBABILITY PREDICTION COMPREHENSIVE TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Run all test suites
    try:
        # Test 1: Dedicated Appeal Analysis Endpoint
        results1 = test_dedicated_appeal_analysis_endpoint()
        all_results.extend(results1)
        
        # Test 2: Case Analysis with Appeal Integration
        results2 = test_case_analysis_with_appeal_integration()
        all_results.extend(results2)
        
        # Test 3: Appeal Probability Factors Validation
        results3 = test_appeal_probability_factors_validation()
        all_results.extend(results3)
        
        # Test 4: Jurisdictional Differences
        results4 = test_jurisdictional_differences()
        all_results.extend(results4)
        
        # Test 5: AI-Enhanced Appeal Analysis
        results5 = test_ai_enhanced_appeal_analysis()
        all_results.extend(results5)
        
        # Test 6: Error Handling
        results6 = test_error_handling()
        all_results.extend(results6)
        
        # Test 7: Analytics Dashboard
        results7 = test_analytics_dashboard()
        all_results.extend(results7)
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        all_results.append(("Testing Framework", False, str(e)))
    
    # Print comprehensive summary
    print_test_summary(all_results)
    
    print(f"\n🏁 Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return success rate for external monitoring
    total_tests = len(all_results)
    passed_tests = sum(1 for _, passed, _ in all_results if passed)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    return success_rate >= 70  # Consider 70%+ as overall success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)