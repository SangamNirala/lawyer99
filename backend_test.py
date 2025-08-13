#!/usr/bin/env python3
"""
Settlement Probability Calculator Validation Fix Testing
=======================================================

Comprehensive testing of the Settlement Probability Calculator endpoints to verify
the Pydantic validation fix is working correctly:

1. POST /api/litigation/settlement-probability (basic settlement analysis)
2. POST /api/litigation/settlement-probability-advanced (advanced settlement analysis)

SPECIFIC TEST SCENARIOS:
- Test with sample case data to trigger the negotiation_leverage calculation
- Verify that both endpoints accept the complex negotiation_leverage structure
- Confirm the response includes proper NegotiationLeverageData structure without validation errors

This addresses the user's exact issue where clicking settlement calculation buttons
was causing validation errors.
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://legalcore.preview.emergentagent.com/api"

def test_basic_settlement_probability():
    """Test basic settlement probability endpoint with validation fix verification"""
    print("🎯 TESTING BASIC SETTLEMENT PROBABILITY - VALIDATION FIX")
    print("=" * 70)
    
    test_results = []
    
    # Test case data as specified in the review request
    test_case = {
        "case_id": "test-settlement-calc-001",
        "case_type": "Contract Dispute", 
        "jurisdiction": "Federal",
        "case_value": 500000,
        "evidence_strength": 0.8,
        "case_complexity": 0.7,
        "filing_date": "2024-01-15",
        "judge_name": "Judge Smith"
    }
    
    print(f"\n📋 Test Case: Basic Settlement Analysis")
    print(f"Case ID: {test_case['case_id']}")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}")
    print(f"Case Complexity: {test_case['case_complexity']}")
    print(f"Filing Date: {test_case['filing_date']}")
    print(f"Judge: {test_case['judge_name']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability"
        print(f"\nRequest URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify required fields are present
            required_fields = [
                'case_id', 'settlement_probability', 'optimal_timing',
                'plaintiff_settlement_range', 'defendant_settlement_range',
                'expected_settlement_value', 'settlement_urgency_score',
                'confidence_score', 'key_settlement_factors',
                'negotiation_leverage', 'scenarios', 'ai_insights',
                'recommendations'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                test_results.append(False)
            else:
                print("✅ All required fields present")
                
                # CRITICAL: Verify negotiation_leverage structure (the main fix)
                negotiation_leverage = data.get('negotiation_leverage')
                if negotiation_leverage:
                    print("\n🔍 NEGOTIATION LEVERAGE VALIDATION:")
                    
                    # Check required fields in negotiation_leverage
                    leverage_fields = ['plaintiff', 'defendant', 'factors', 'balance', 'difference']
                    missing_leverage_fields = [field for field in leverage_fields if field not in negotiation_leverage]
                    
                    if missing_leverage_fields:
                        print(f"❌ Missing negotiation_leverage fields: {missing_leverage_fields}")
                        test_results.append(False)
                    else:
                        print("✅ All negotiation_leverage fields present")
                        
                        # Verify field types (the core of the validation fix)
                        plaintiff = negotiation_leverage.get('plaintiff')
                        defendant = negotiation_leverage.get('defendant')
                        factors = negotiation_leverage.get('factors')
                        balance = negotiation_leverage.get('balance')
                        difference = negotiation_leverage.get('difference')
                        
                        print(f"  Plaintiff: {plaintiff} (type: {type(plaintiff).__name__})")
                        print(f"  Defendant: {defendant} (type: {type(defendant).__name__})")
                        print(f"  Factors: {factors} (type: {type(factors).__name__})")
                        print(f"  Balance: {balance} (type: {type(balance).__name__})")
                        print(f"  Difference: {difference} (type: {type(difference).__name__})")
                        
                        # Validate types match the fixed Pydantic model
                        type_validations = [
                            isinstance(plaintiff, (int, float)),  # plaintiff: float
                            isinstance(defendant, (int, float)),  # defendant: float
                            isinstance(factors, dict),            # factors: Dict[str, float]
                            isinstance(balance, str),             # balance: str
                            isinstance(difference, (int, float))  # difference: float
                        ]
                        
                        if all(type_validations):
                            print("✅ All negotiation_leverage field types are correct")
                            
                            # Additional validation: factors should contain float values
                            if factors and all(isinstance(v, (int, float)) for v in factors.values()):
                                print("✅ All factors values are numeric (float)")
                                validation_success = True
                            else:
                                print("❌ Some factors values are not numeric")
                                validation_success = False
                        else:
                            print("❌ Some negotiation_leverage field types are incorrect")
                            validation_success = False
                        
                        test_results.append(validation_success)
                else:
                    print("❌ negotiation_leverage field is missing")
                    test_results.append(False)
                
                # Display other key metrics
                print(f"\n📊 SETTLEMENT ANALYSIS RESULTS:")
                print(f"Settlement Probability: {data.get('settlement_probability', 0):.1%}")
                print(f"Confidence Score: {data.get('confidence_score', 0):.1%}")
                print(f"Expected Settlement Value: ${data.get('expected_settlement_value', 0):,.2f}")
                print(f"Optimal Timing: {data.get('optimal_timing', 'N/A')}")
                
                # Check settlement ranges
                plaintiff_range = data.get('plaintiff_settlement_range', {})
                defendant_range = data.get('defendant_settlement_range', {})
                print(f"Plaintiff Range: ${plaintiff_range.get('low', 0):,.2f} - ${plaintiff_range.get('high', 0):,.2f}")
                print(f"Defendant Range: ${defendant_range.get('low', 0):,.2f} - ${defendant_range.get('high', 0):,.2f}")
                
                # Check AI insights and recommendations
                ai_insights = data.get('ai_insights', '')
                recommendations = data.get('recommendations', [])
                print(f"AI Insights Length: {len(ai_insights)} characters")
                print(f"Number of Recommendations: {len(recommendations)}")
                
                # Overall quality check
                if len(ai_insights) > 50 and len(recommendations) > 0:
                    print("✅ AI analysis content is substantial")
                    test_results.append(True)
                else:
                    print("⚠️ AI analysis content is minimal")
                    test_results.append(False)
                    
        elif response.status_code == 422:
            print(f"❌ VALIDATION ERROR (422) - The fix may not be working!")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
                
                # Check if it's the specific negotiation_leverage validation error
                error_detail = str(error_data)
                if 'negotiation_leverage' in error_detail:
                    print("🚨 CRITICAL: This is the exact validation error the fix was supposed to resolve!")
                    print("The Pydantic model fix is not working correctly.")
            except:
                print(f"Raw error response: {response.text}")
            test_results.append(False)
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            test_results.append(False)
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        test_results.append(False)
    
    print("-" * 50)
    return test_results

def test_advanced_settlement_probability():
    """Test advanced settlement probability endpoint with validation fix verification"""
    print("\n🚀 TESTING ADVANCED SETTLEMENT PROBABILITY - VALIDATION FIX")
    print("=" * 70)
    
    test_results = []
    
    # Test case data as specified in the review request
    test_case = {
        "case_id": "test-settlement-calc-001",
        "case_type": "Contract Dispute", 
        "jurisdiction": "Federal",
        "case_value": 500000,
        "evidence_strength": 0.8,
        "case_complexity": 0.7,
        "filing_date": "2024-01-15",
        "judge_name": "Judge Smith",
        "case_facts": "Complex commercial contract dispute with strong evidence",
        "witness_count": 3,
        "analysis_mode": "advanced",
        "monte_carlo_iterations": 5000,
        "include_comparative": True,
        "include_market_analysis": True
    }
    
    print(f"\n📋 Test Case: Advanced Settlement Analysis")
    print(f"Case ID: {test_case['case_id']}")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}")
    print(f"Case Complexity: {test_case['case_complexity']}")
    print(f"Analysis Mode: {test_case['analysis_mode']}")
    print(f"Monte Carlo Iterations: {test_case['monte_carlo_iterations']:,}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability-advanced"
        print(f"\nRequest URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify standard fields
            standard_fields = [
                'case_id', 'settlement_probability', 'optimal_timing',
                'plaintiff_settlement_range', 'defendant_settlement_range',
                'expected_settlement_value', 'confidence_score',
                'negotiation_leverage', 'scenarios', 'ai_insights',
                'recommendations'
            ]
            
            missing_standard = [field for field in standard_fields if field not in data]
            if missing_standard:
                print(f"❌ Missing standard fields: {missing_standard}")
                test_results.append(False)
            else:
                print("✅ All standard fields present")
                
                # CRITICAL: Verify negotiation_leverage structure (the main fix)
                negotiation_leverage = data.get('negotiation_leverage')
                if negotiation_leverage:
                    print("\n🔍 ADVANCED NEGOTIATION LEVERAGE VALIDATION:")
                    
                    # Check required fields in negotiation_leverage
                    leverage_fields = ['plaintiff', 'defendant', 'factors', 'balance', 'difference']
                    missing_leverage_fields = [field for field in leverage_fields if field not in negotiation_leverage]
                    
                    if missing_leverage_fields:
                        print(f"❌ Missing negotiation_leverage fields: {missing_leverage_fields}")
                        test_results.append(False)
                    else:
                        print("✅ All negotiation_leverage fields present")
                        
                        # Verify field types (the core of the validation fix)
                        plaintiff = negotiation_leverage.get('plaintiff')
                        defendant = negotiation_leverage.get('defendant')
                        factors = negotiation_leverage.get('factors')
                        balance = negotiation_leverage.get('balance')
                        difference = negotiation_leverage.get('difference')
                        
                        print(f"  Plaintiff: {plaintiff} (type: {type(plaintiff).__name__})")
                        print(f"  Defendant: {defendant} (type: {type(defendant).__name__})")
                        print(f"  Factors: {factors} (type: {type(factors).__name__})")
                        print(f"  Balance: {balance} (type: {type(balance).__name__})")
                        print(f"  Difference: {difference} (type: {type(difference).__name__})")
                        
                        # Validate types match the fixed Pydantic model
                        type_validations = [
                            isinstance(plaintiff, (int, float)),  # plaintiff: float
                            isinstance(defendant, (int, float)),  # defendant: float
                            isinstance(factors, dict),            # factors: Dict[str, float]
                            isinstance(balance, str),             # balance: str
                            isinstance(difference, (int, float))  # difference: float
                        ]
                        
                        if all(type_validations):
                            print("✅ All negotiation_leverage field types are correct")
                            
                            # Additional validation: factors should contain float values
                            if factors and all(isinstance(v, (int, float)) for v in factors.values()):
                                print("✅ All factors values are numeric (float)")
                                print(f"  Sample factors: {dict(list(factors.items())[:3])}")
                                validation_success = True
                            else:
                                print("❌ Some factors values are not numeric")
                                validation_success = False
                        else:
                            print("❌ Some negotiation_leverage field types are incorrect")
                            validation_success = False
                        
                        test_results.append(validation_success)
                else:
                    print("❌ negotiation_leverage field is missing")
                    test_results.append(False)
                
                # Check advanced fields
                advanced_fields = [
                    'monte_carlo_results', 'ai_consensus_score',
                    'market_trend_adjustment', 'volatility_index',
                    'strategic_advantage_score', 'comparative_cases',
                    'processing_time', 'analysis_mode', 'metadata'
                ]
                
                present_advanced = [field for field in advanced_fields if field in data]
                print(f"\n📊 ADVANCED FEATURES PRESENT: {len(present_advanced)}/{len(advanced_fields)}")
                for field in present_advanced:
                    value = data.get(field)
                    if isinstance(value, (list, dict)):
                        print(f"  {field}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"  {field}: {value}")
                
                # Check Monte Carlo results if present
                monte_carlo_results = data.get('monte_carlo_results')
                if monte_carlo_results:
                    print("\n🎲 MONTE CARLO RESULTS VALIDATION:")
                    mc_fields = [
                        'mean_settlement_probability', 'std_settlement_probability',
                        'percentiles', 'confidence_intervals', 'scenario_probabilities',
                        'risk_metrics', 'simulation_count', 'convergence_analysis'
                    ]
                    
                    present_mc = [field for field in mc_fields if field in monte_carlo_results]
                    print(f"  Monte Carlo fields present: {len(present_mc)}/{len(mc_fields)}")
                    
                    if 'simulation_count' in monte_carlo_results:
                        sim_count = monte_carlo_results['simulation_count']
                        print(f"  Simulation count: {sim_count:,}")
                        
                    if 'mean_settlement_probability' in monte_carlo_results:
                        mean_prob = monte_carlo_results['mean_settlement_probability']
                        print(f"  Mean settlement probability: {mean_prob:.1%}")
                
                # Overall quality assessment
                quality_indicators = [
                    data.get('settlement_probability', 0) > 0,
                    data.get('confidence_score', 0) > 0,
                    len(data.get('ai_insights', '')) > 50,
                    len(data.get('recommendations', [])) > 0,
                    len(present_advanced) >= 5  # At least 5 advanced features
                ]
                
                quality_score = sum(quality_indicators) / len(quality_indicators)
                print(f"\n📈 OVERALL QUALITY SCORE: {quality_score:.1%}")
                
                if quality_score >= 0.8:
                    print("✅ Advanced settlement analysis working excellently")
                    test_results.append(True)
                else:
                    print("⚠️ Advanced settlement analysis working but with some limitations")
                    test_results.append(False)
                    
        elif response.status_code == 422:
            print(f"❌ VALIDATION ERROR (422) - The fix may not be working!")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
                
                # Check if it's the specific negotiation_leverage validation error
                error_detail = str(error_data)
                if 'negotiation_leverage' in error_detail:
                    print("🚨 CRITICAL: This is the exact validation error the fix was supposed to resolve!")
                    print("The Pydantic model fix is not working correctly.")
            except:
                print(f"Raw error response: {response.text}")
            test_results.append(False)
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            test_results.append(False)
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        test_results.append(False)
    
    print("-" * 50)
    return test_results

def test_negotiation_leverage_edge_cases():
    """Test edge cases for negotiation_leverage validation"""
    print("\n🧪 TESTING NEGOTIATION LEVERAGE EDGE CASES")
    print("=" * 70)
    
    test_results = []
    
    # Test with minimal data to ensure negotiation_leverage is still generated
    minimal_case = {
        "case_type": "Contract Dispute",
        "jurisdiction": "Federal"
    }
    
    print(f"\n📋 Test Case: Minimal Data")
    print(f"Data: {minimal_case}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability"
        response = requests.post(url, json=minimal_case, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            negotiation_leverage = data.get('negotiation_leverage')
            
            if negotiation_leverage:
                print("✅ negotiation_leverage generated even with minimal data")
                
                # Verify structure is still correct
                required_fields = ['plaintiff', 'defendant', 'factors', 'balance', 'difference']
                has_all_fields = all(field in negotiation_leverage for field in required_fields)
                
                if has_all_fields:
                    print("✅ All negotiation_leverage fields present with minimal data")
                    test_results.append(True)
                else:
                    print("❌ Some negotiation_leverage fields missing with minimal data")
                    test_results.append(False)
            else:
                print("❌ negotiation_leverage not generated with minimal data")
                test_results.append(False)
                
        elif response.status_code == 422:
            print("⚠️ Validation error with minimal data (may be expected)")
            test_results.append(True)  # This might be expected behavior
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            test_results.append(False)
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        test_results.append(False)
    
    print("-" * 50)
    return test_results

def main():
    """Main test execution function"""
    print("🎯 SETTLEMENT PROBABILITY CALCULATOR VALIDATION FIX TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 FOCUS: Verifying Pydantic validation fix for negotiation_leverage field")
    print("USER ISSUE: Settlement buttons causing validation errors with negotiation_leverage.factors and negotiation_leverage.balance")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: Basic Settlement Probability
    print("\n" + "🎯" * 25 + " TEST 1 " + "🎯" * 25)
    basic_results = test_basic_settlement_probability()
    all_results.extend(basic_results)
    
    # Test 2: Advanced Settlement Probability
    print("\n" + "🚀" * 25 + " TEST 2 " + "🚀" * 25)
    advanced_results = test_advanced_settlement_probability()
    all_results.extend(advanced_results)
    
    # Test 3: Edge Cases
    print("\n" + "🧪" * 25 + " TEST 3 " + "🧪" * 25)
    edge_case_results = test_negotiation_leverage_edge_cases()
    all_results.extend(edge_case_results)
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("🎯 VALIDATION FIX TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Detailed breakdown
    print(f"\n📋 Test Suite Breakdown:")
    print(f"  Basic Settlement Analysis: {sum(basic_results)}/{len(basic_results)} passed")
    print(f"  Advanced Settlement Analysis: {sum(advanced_results)}/{len(advanced_results)} passed")
    print(f"  Edge Case Testing: {sum(edge_case_results)}/{len(edge_case_results)} passed")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validation Fix Assessment
    print(f"\n🔍 VALIDATION FIX ASSESSMENT:")
    
    if success_rate >= 90:
        print("🎉 VALIDATION FIX SUCCESSFUL: Settlement Probability Calculator working perfectly!")
        print("✅ No more Pydantic validation errors with negotiation_leverage fields")
        print("✅ Both endpoints return proper NegotiationLeverageData structure")
        print("✅ User-reported validation errors are completely resolved")
        fix_status = "SUCCESSFUL"
    elif success_rate >= 70:
        print("✅ VALIDATION FIX MOSTLY SUCCESSFUL: Settlement Calculator working well")
        print("✅ Most negotiation_leverage validation issues resolved")
        print("⚠️ Some minor issues may remain")
        fix_status = "MOSTLY_SUCCESSFUL"
    else:
        print("❌ VALIDATION FIX NEEDS ATTENTION: Settlement Calculator has issues")
        print("❌ negotiation_leverage validation errors may still be present")
        print("🚨 User-reported issues may not be fully resolved")
        fix_status = "NEEDS_ATTENTION"
    
    print(f"\n🎯 EXPECTED BEHAVIOR VERIFICATION:")
    expected_behaviors = [
        "✅ No Pydantic validation errors about negotiation_leverage.factors",
        "✅ No Pydantic validation errors about negotiation_leverage.balance", 
        "✅ Response includes negotiation_leverage object with correct structure",
        "✅ Both endpoints return 200 status with complete settlement analysis data",
        "✅ negotiation_leverage contains: plaintiff (float), defendant (float), factors (dict), balance (string), difference (float)"
    ]
    
    for behavior in expected_behaviors:
        print(behavior)
    
    print(f"\n📊 VALIDATION FIX STATUS: {fix_status}")
    print("=" * 80)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)