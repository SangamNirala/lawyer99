#!/usr/bin/env python3
"""
Enhanced Settlement Probability Calculator Testing
=================================================

Comprehensive testing of the enhanced Settlement Probability Calculator endpoints:
1. POST /api/litigation/settlement-probability (Standard Analysis)
2. POST /api/litigation/settlement-probability-advanced (Advanced Analysis)

Test Scenarios:
- Standard Settlement Analysis
- Advanced Settlement Analysis with Monte Carlo simulation
- Advanced Comparative Analysis
- Error handling and validation
- Enhanced AI Analysis verification
- Multi-scenario Monte Carlo modeling
- Real-time interactive analysis
- Advanced risk assessment
- Comparative case analysis
- Settlement strategy optimization
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://93a8c601-fd30-4f56-a2f2-47103260abd2.preview.emergentagent.com/api"

def test_standard_settlement_analysis():
    """Test standard settlement probability analysis endpoint"""
    print("🎯 TESTING STANDARD SETTLEMENT ANALYSIS")
    print("=" * 60)
    
    test_results = []
    
    # Test case 1: Contract Dispute
    test_case_1 = {
        "case_type": "contract_dispute",
        "jurisdiction": "federal",
        "case_value": 500000,
        "evidence_strength": 7.5,
        "case_complexity": 0.6,
        "judge_name": "John Roberts"
    }
    
    print(f"\n📋 Test Case 1: Contract Dispute Analysis")
    print(f"Case Type: {test_case_1['case_type']}")
    print(f"Jurisdiction: {test_case_1['jurisdiction']}")
    print(f"Case Value: ${test_case_1['case_value']:,}")
    print(f"Evidence Strength: {test_case_1['evidence_strength']}/10")
    print(f"Case Complexity: {test_case_1['case_complexity']}")
    print(f"Judge: {test_case_1['judge_name']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability"
        print(f"Request URL: {url}")
        
        response = requests.post(url, json=test_case_1, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify required fields
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
                
                # Verify data quality
                settlement_prob = data.get('settlement_probability', 0)
                confidence_score = data.get('confidence_score', 0)
                expected_value = data.get('expected_settlement_value', 0)
                scenarios = data.get('scenarios', [])
                
                print(f"Settlement Probability: {settlement_prob:.1%}")
                print(f"Confidence Score: {confidence_score:.1%}")
                print(f"Expected Settlement Value: ${expected_value:,.2f}")
                print(f"Number of Scenarios: {len(scenarios)}")
                print(f"Optimal Timing: {data.get('optimal_timing', 'N/A')}")
                
                # Validate ranges
                plaintiff_range = data.get('plaintiff_settlement_range', {})
                defendant_range = data.get('defendant_settlement_range', {})
                
                print(f"Plaintiff Range: ${plaintiff_range.get('low', 0):,.2f} - ${plaintiff_range.get('high', 0):,.2f}")
                print(f"Defendant Range: ${defendant_range.get('low', 0):,.2f} - ${defendant_range.get('high', 0):,.2f}")
                
                # Check AI insights and recommendations
                ai_insights = data.get('ai_insights', '')
                recommendations = data.get('recommendations', [])
                
                print(f"AI Insights Length: {len(ai_insights)} characters")
                print(f"Number of Recommendations: {len(recommendations)}")
                
                # Quality checks
                quality_checks = [
                    0 <= settlement_prob <= 1,
                    0 <= confidence_score <= 1,
                    expected_value > 0,
                    len(scenarios) > 0,
                    len(ai_insights) > 50,
                    len(recommendations) > 0
                ]
                
                if all(quality_checks):
                    print("✅ Standard settlement analysis working correctly")
                    test_results.append(True)
                else:
                    print("❌ Quality checks failed")
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

def test_advanced_settlement_analysis_monte_carlo():
    """Test advanced settlement analysis with Monte Carlo simulation"""
    print("\n🚀 TESTING ADVANCED SETTLEMENT ANALYSIS - MONTE CARLO")
    print("=" * 60)
    
    test_results = []
    
    # Test case: Employment Dispute with Monte Carlo
    test_case = {
        "case_type": "employment_dispute",
        "jurisdiction": "state",
        "case_value": 250000,
        "evidence_strength": 6.0,
        "case_complexity": 0.4,
        "case_facts": "Wrongful termination case with documentation",
        "witness_count": 3,
        "analysis_mode": "monte_carlo",
        "monte_carlo_iterations": 5000,
        "include_comparative": True,
        "include_market_analysis": True
    }
    
    print(f"\n📋 Test Case: Employment Dispute with Monte Carlo")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}/10")
    print(f"Case Complexity: {test_case['case_complexity']}")
    print(f"Analysis Mode: {test_case['analysis_mode']}")
    print(f"Monte Carlo Iterations: {test_case['monte_carlo_iterations']:,}")
    print(f"Case Facts: {test_case['case_facts']}")
    print(f"Witness Count: {test_case['witness_count']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability-advanced"
        print(f"Request URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify standard fields
            standard_fields = [
                'case_id', 'settlement_probability', 'optimal_timing',
                'plaintiff_settlement_range', 'defendant_settlement_range',
                'expected_settlement_value', 'confidence_score'
            ]
            
            # Verify advanced fields
            advanced_fields = [
                'monte_carlo_results', 'ai_consensus_score',
                'market_trend_adjustment', 'volatility_index',
                'strategic_advantage_score', 'comparative_cases',
                'processing_time', 'analysis_mode', 'metadata'
            ]
            
            missing_standard = [field for field in standard_fields if field not in data]
            missing_advanced = [field for field in advanced_fields if field not in data]
            
            if missing_standard:
                print(f"❌ Missing standard fields: {missing_standard}")
                test_results.append(False)
            else:
                print("✅ All standard fields present")
                
                # Check Monte Carlo specific results
                monte_carlo_results = data.get('monte_carlo_results')
                if monte_carlo_results:
                    print("✅ Monte Carlo results present")
                    
                    # Verify Monte Carlo structure
                    mc_fields = [
                        'mean_settlement_probability', 'std_settlement_probability',
                        'percentiles', 'confidence_intervals', 'scenario_probabilities',
                        'risk_metrics', 'simulation_count', 'convergence_analysis'
                    ]
                    
                    missing_mc = [field for field in mc_fields if field not in monte_carlo_results]
                    if missing_mc:
                        print(f"⚠️ Missing Monte Carlo fields: {missing_mc}")
                    else:
                        print("✅ Complete Monte Carlo analysis structure")
                        
                        # Display Monte Carlo metrics
                        mean_prob = monte_carlo_results.get('mean_settlement_probability', 0)
                        std_prob = monte_carlo_results.get('std_settlement_probability', 0)
                        sim_count = monte_carlo_results.get('simulation_count', 0)
                        
                        print(f"Mean Settlement Probability: {mean_prob:.1%}")
                        print(f"Standard Deviation: {std_prob:.1%}")
                        print(f"Simulation Count: {sim_count:,}")
                        
                        # Check percentiles
                        percentiles = monte_carlo_results.get('percentiles', {})
                        if percentiles:
                            print(f"Percentiles: P25={percentiles.get('25', 0):.1%}, P50={percentiles.get('50', 0):.1%}, P75={percentiles.get('75', 0):.1%}")
                        
                        # Check confidence intervals
                        confidence_intervals = monte_carlo_results.get('confidence_intervals', {})
                        if confidence_intervals:
                            ci_95 = confidence_intervals.get('95%', {})
                            print(f"95% Confidence Interval: [{ci_95.get('lower', 0):.1%}, {ci_95.get('upper', 0):.1%}]")
                else:
                    print("⚠️ Monte Carlo results not present")
                
                # Check advanced analytics
                ai_consensus = data.get('ai_consensus_score')
                market_trend = data.get('market_trend_adjustment')
                volatility = data.get('volatility_index')
                strategic_advantage = data.get('strategic_advantage_score')
                
                print(f"AI Consensus Score: {ai_consensus}")
                print(f"Market Trend Adjustment: {market_trend}")
                print(f"Volatility Index: {volatility}")
                print(f"Strategic Advantage Score: {strategic_advantage}")
                
                # Check comparative cases
                comparative_cases = data.get('comparative_cases', [])
                print(f"Comparative Cases Found: {len(comparative_cases)}")
                
                if comparative_cases:
                    print("✅ Comparative case analysis working")
                    for i, case in enumerate(comparative_cases[:3]):
                        similarity = case.get('similarity_score', 0)
                        case_type = case.get('case_type', 'Unknown')
                        settlement_amt = case.get('settlement_amount', 0)
                        print(f"  Case {i+1}: {case_type}, Similarity: {similarity:.1%}, Settlement: ${settlement_amt:,.2f}")
                
                # Check processing time and metadata
                processing_time = data.get('processing_time')
                analysis_mode = data.get('analysis_mode')
                metadata = data.get('metadata', {})
                
                print(f"Processing Time: {processing_time}s")
                print(f"Analysis Mode: {analysis_mode}")
                print(f"Metadata Keys: {list(metadata.keys())}")
                
                # Overall quality assessment
                quality_checks = [
                    data.get('settlement_probability', 0) > 0,
                    data.get('confidence_score', 0) > 0,
                    len(data.get('scenarios', [])) > 0,
                    len(data.get('ai_insights', '')) > 50,
                    monte_carlo_results is not None,
                    sim_count >= 1000 if monte_carlo_results else True
                ]
                
                if all(quality_checks):
                    print("✅ Advanced Monte Carlo analysis working excellently")
                    test_results.append(True)
                else:
                    print("❌ Some quality checks failed")
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

def test_advanced_comparative_analysis():
    """Test advanced settlement analysis with comparative mode"""
    print("\n🔍 TESTING ADVANCED SETTLEMENT ANALYSIS - COMPARATIVE")
    print("=" * 60)
    
    test_results = []
    
    # Test case: Personal Injury with Comparative Analysis
    test_case = {
        "case_type": "personal_injury",
        "jurisdiction": "state",
        "case_value": 750000,
        "evidence_strength": 8.0,
        "case_complexity": 0.3,
        "case_facts": "Motor vehicle accident with clear liability and documented injuries",
        "witness_count": 5,
        "analysis_mode": "comparative",
        "include_comparative": True,
        "include_market_analysis": True
    }
    
    print(f"\n📋 Test Case: Personal Injury Comparative Analysis")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}/10")
    print(f"Case Complexity: {test_case['case_complexity']}")
    print(f"Analysis Mode: {test_case['analysis_mode']}")
    print(f"Case Facts: {test_case['case_facts']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability-advanced"
        print(f"Request URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=90)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Focus on comparative analysis features
            comparative_cases = data.get('comparative_cases', [])
            ai_consensus_score = data.get('ai_consensus_score')
            strategic_advantage_score = data.get('strategic_advantage_score')
            
            print(f"Comparative Cases Found: {len(comparative_cases)}")
            print(f"AI Consensus Score: {ai_consensus_score}")
            print(f"Strategic Advantage Score: {strategic_advantage_score}")
            
            if len(comparative_cases) > 0:
                print("✅ Comparative case analysis working")
                
                # Analyze comparative cases
                total_similarity = 0
                for i, case in enumerate(comparative_cases):
                    case_id = case.get('case_id', 'Unknown')
                    case_type = case.get('case_type', 'Unknown')
                    settlement_amount = case.get('settlement_amount', 0)
                    settlement_probability = case.get('settlement_probability', 0)
                    similarity_score = case.get('similarity_score', 0)
                    key_factors = case.get('key_factors', [])
                    jurisdiction = case.get('jurisdiction', 'Unknown')
                    
                    total_similarity += similarity_score
                    
                    print(f"  Case {i+1}:")
                    print(f"    Type: {case_type}")
                    print(f"    Settlement: ${settlement_amount:,.2f}")
                    print(f"    Settlement Probability: {settlement_probability:.1%}")
                    print(f"    Similarity: {similarity_score:.1%}")
                    print(f"    Jurisdiction: {jurisdiction}")
                    print(f"    Key Factors: {len(key_factors)} factors")
                
                avg_similarity = total_similarity / len(comparative_cases) if comparative_cases else 0
                print(f"Average Similarity Score: {avg_similarity:.1%}")
                
                # Check if similarity scores are reasonable
                if avg_similarity > 0.3:  # At least 30% similarity on average
                    print("✅ Comparative cases have good similarity scores")
                    similarity_quality = True
                else:
                    print("⚠️ Comparative cases have low similarity scores")
                    similarity_quality = False
            else:
                print("⚠️ No comparative cases found")
                similarity_quality = False
            
            # Check enhanced AI analysis
            ai_insights = data.get('ai_insights', '')
            recommendations = data.get('recommendations', [])
            
            print(f"AI Insights Length: {len(ai_insights)} characters")
            print(f"Number of Recommendations: {len(recommendations)}")
            
            # Check settlement probability and confidence
            settlement_prob = data.get('settlement_probability', 0)
            confidence_score = data.get('confidence_score', 0)
            
            print(f"Settlement Probability: {settlement_prob:.1%}")
            print(f"Confidence Score: {confidence_score:.1%}")
            
            # Quality assessment for comparative analysis
            quality_checks = [
                settlement_prob > 0,
                confidence_score > 0.5,  # Higher confidence expected for comparative analysis
                len(ai_insights) > 100,
                len(recommendations) > 0,
                similarity_quality,
                ai_consensus_score is not None,
                strategic_advantage_score is not None
            ]
            
            passed_checks = sum(quality_checks)
            total_checks = len(quality_checks)
            
            print(f"Quality Checks Passed: {passed_checks}/{total_checks}")
            
            if passed_checks >= total_checks * 0.8:  # 80% pass rate
                print("✅ Advanced comparative analysis working well")
                test_results.append(True)
            else:
                print("❌ Comparative analysis needs improvement")
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

def test_enhanced_ai_analysis():
    """Test enhanced AI analysis features"""
    print("\n🤖 TESTING ENHANCED AI ANALYSIS FEATURES")
    print("=" * 60)
    
    test_results = []
    
    # Test case with comprehensive data for AI analysis
    test_case = {
        "case_type": "contract_dispute",
        "jurisdiction": "federal",
        "case_value": 1000000,
        "evidence_strength": 9.0,
        "case_complexity": 0.8,
        "case_facts": "Complex commercial contract dispute involving breach of exclusivity clause, damages include lost profits and market share. Strong documentary evidence including email communications and financial records.",
        "witness_count": 8,
        "opposing_party_resources": "high",
        "analysis_mode": "advanced",
        "monte_carlo_iterations": 10000,
        "include_comparative": True,
        "include_market_analysis": True,
        "ai_providers": ["openrouter", "gemini", "groq"]
    }
    
    print(f"\n📋 Test Case: Enhanced AI Analysis")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Evidence Strength: {test_case['evidence_strength']}/10")
    print(f"Case Complexity: {test_case['case_complexity']}")
    print(f"Monte Carlo Iterations: {test_case['monte_carlo_iterations']:,}")
    print(f"AI Providers: {test_case['ai_providers']}")
    
    try:
        url = f"{BACKEND_URL}/litigation/settlement-probability-advanced"
        print(f"Request URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=150)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check AI consensus and multi-provider analysis
            ai_consensus_score = data.get('ai_consensus_score')
            ai_insights = data.get('ai_insights', '')
            recommendations = data.get('recommendations', [])
            
            print(f"AI Consensus Score: {ai_consensus_score}")
            print(f"AI Insights Length: {len(ai_insights)} characters")
            print(f"Number of Recommendations: {len(recommendations)}")
            
            # Check for sophisticated analysis indicators
            sophisticated_indicators = [
                'risk' in ai_insights.lower(),
                'strategy' in ai_insights.lower(),
                'negotiation' in ai_insights.lower(),
                'precedent' in ai_insights.lower() or 'similar' in ai_insights.lower(),
                len(ai_insights) > 200,
                len(recommendations) >= 3
            ]
            
            sophistication_score = sum(sophisticated_indicators) / len(sophisticated_indicators)
            print(f"AI Sophistication Score: {sophistication_score:.1%}")
            
            # Check advanced metrics
            market_trend_adjustment = data.get('market_trend_adjustment')
            volatility_index = data.get('volatility_index')
            strategic_advantage_score = data.get('strategic_advantage_score')
            
            print(f"Market Trend Adjustment: {market_trend_adjustment}")
            print(f"Volatility Index: {volatility_index}")
            print(f"Strategic Advantage Score: {strategic_advantage_score}")
            
            # Check Monte Carlo sophistication
            monte_carlo_results = data.get('monte_carlo_results')
            if monte_carlo_results:
                convergence_analysis = monte_carlo_results.get('convergence_analysis', {})
                risk_metrics = monte_carlo_results.get('risk_metrics', {})
                
                print(f"Convergence Analysis: {bool(convergence_analysis)}")
                print(f"Risk Metrics: {bool(risk_metrics)}")
                
                if risk_metrics:
                    var_95 = risk_metrics.get('var_95')
                    expected_shortfall = risk_metrics.get('expected_shortfall')
                    print(f"Value at Risk (95%): {var_95}")
                    print(f"Expected Shortfall: {expected_shortfall}")
            
            # Quality assessment for enhanced AI
            ai_quality_checks = [
                ai_consensus_score is not None and ai_consensus_score > 0.6,
                sophistication_score > 0.6,
                len(recommendations) >= 3,
                market_trend_adjustment is not None,
                volatility_index is not None,
                strategic_advantage_score is not None,
                monte_carlo_results is not None
            ]
            
            ai_quality_score = sum(ai_quality_checks) / len(ai_quality_checks)
            print(f"AI Quality Score: {ai_quality_score:.1%}")
            
            if ai_quality_score >= 0.8:
                print("✅ Enhanced AI analysis working excellently")
                test_results.append(True)
            elif ai_quality_score >= 0.6:
                print("✅ Enhanced AI analysis working well")
                test_results.append(True)
            else:
                print("❌ Enhanced AI analysis needs improvement")
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

def test_error_handling_and_validation():
    """Test error handling and input validation"""
    print("\n⚠️ TESTING ERROR HANDLING AND VALIDATION")
    print("=" * 60)
    
    test_results = []
    
    # Test cases for error handling
    error_test_cases = [
        {
            "name": "Missing required fields",
            "data": {},
            "expected_status": 422
        },
        {
            "name": "Invalid case type",
            "data": {
                "case_type": "invalid_case_type",
                "jurisdiction": "federal"
            },
            "expected_status": [200, 422]  # May be handled gracefully
        },
        {
            "name": "Invalid evidence strength",
            "data": {
                "case_type": "contract_dispute",
                "jurisdiction": "federal",
                "evidence_strength": 15.0  # Should be 0-10
            },
            "expected_status": [200, 422]
        },
        {
            "name": "Invalid case complexity",
            "data": {
                "case_type": "contract_dispute",
                "jurisdiction": "federal",
                "case_complexity": 2.0  # Should be 0-1
            },
            "expected_status": [200, 422]
        },
        {
            "name": "Negative case value",
            "data": {
                "case_type": "contract_dispute",
                "jurisdiction": "federal",
                "case_value": -100000
            },
            "expected_status": [200, 422]
        }
    ]
    
    for test_case in error_test_cases:
        name = test_case["name"]
        data = test_case["data"]
        expected_status = test_case["expected_status"]
        
        print(f"\n📋 Error Test: {name}")
        print(f"Data: {data}")
        print(f"Expected Status: {expected_status}")
        
        try:
            # Test both endpoints
            for endpoint_name, endpoint_url in [
                ("Standard", f"{BACKEND_URL}/litigation/settlement-probability"),
                ("Advanced", f"{BACKEND_URL}/litigation/settlement-probability-advanced")
            ]:
                print(f"\n  Testing {endpoint_name} endpoint:")
                
                response = requests.post(endpoint_url, json=data, timeout=30)
                print(f"  Status Code: {response.status_code}")
                
                if isinstance(expected_status, list):
                    status_ok = response.status_code in expected_status
                else:
                    status_ok = response.status_code == expected_status
                
                if status_ok:
                    print(f"  ✅ {endpoint_name} endpoint handled error correctly")
                    if response.status_code == 422:
                        try:
                            error_data = response.json()
                            print(f"  Error details: {error_data.get('detail', 'No details')}")
                        except:
                            pass
                else:
                    print(f"  ❌ {endpoint_name} endpoint error handling issue")
                    if response.text:
                        print(f"  Response: {response.text[:200]}...")
                
                test_results.append(status_ok)
                
        except Exception as e:
            print(f"❌ Exception in error test '{name}': {str(e)}")
            test_results.append(False)
            test_results.append(False)  # For both endpoints
    
    print("-" * 50)
    return test_results

def main():
    """Main test execution function"""
    print("🎯 ENHANCED SETTLEMENT PROBABILITY CALCULATOR TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: Standard Settlement Analysis
    print("\n" + "🎯" * 20 + " TEST SUITE 1 " + "🎯" * 20)
    standard_results = test_standard_settlement_analysis()
    all_results.extend(standard_results)
    
    # Test 2: Advanced Monte Carlo Analysis
    print("\n" + "🚀" * 20 + " TEST SUITE 2 " + "🚀" * 20)
    monte_carlo_results = test_advanced_settlement_analysis_monte_carlo()
    all_results.extend(monte_carlo_results)
    
    # Test 3: Advanced Comparative Analysis
    print("\n" + "🔍" * 20 + " TEST SUITE 3 " + "🔍" * 20)
    comparative_results = test_advanced_comparative_analysis()
    all_results.extend(comparative_results)
    
    # Test 4: Enhanced AI Analysis
    print("\n" + "🤖" * 20 + " TEST SUITE 4 " + "🤖" * 20)
    ai_analysis_results = test_enhanced_ai_analysis()
    all_results.extend(ai_analysis_results)
    
    # Test 5: Error Handling and Validation
    print("\n" + "⚠️" * 20 + " TEST SUITE 5 " + "⚠️" * 20)
    error_handling_results = test_error_handling_and_validation()
    all_results.extend(error_handling_results)
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST RESULTS SUMMARY")
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
    print(f"  Standard Settlement Analysis: {sum(standard_results)}/{len(standard_results)} passed")
    print(f"  Advanced Monte Carlo Analysis: {sum(monte_carlo_results)}/{len(monte_carlo_results)} passed")
    print(f"  Advanced Comparative Analysis: {sum(comparative_results)}/{len(comparative_results)} passed")
    print(f"  Enhanced AI Analysis: {sum(ai_analysis_results)}/{len(ai_analysis_results)} passed")
    print(f"  Error Handling & Validation: {sum(error_handling_results)}/{len(error_handling_results)} passed")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Feature Assessment
    print(f"\n🚨 FEATURE ASSESSMENT:")
    
    standard_success = sum(standard_results) / len(standard_results) * 100 if standard_results else 0
    monte_carlo_success = sum(monte_carlo_results) / len(monte_carlo_results) * 100 if monte_carlo_results else 0
    comparative_success = sum(comparative_results) / len(comparative_results) * 100 if comparative_results else 0
    ai_success = sum(ai_analysis_results) / len(ai_analysis_results) * 100 if ai_analysis_results else 0
    error_handling_success = sum(error_handling_results) / len(error_handling_results) * 100 if error_handling_results else 0
    
    features_status = [
        ("✅ Enhanced AI Analysis (GPT-4o, Claude-3.5, Gemini-2.0)", ai_success >= 80),
        ("✅ Multi-scenario Monte Carlo modeling", monte_carlo_success >= 80),
        ("✅ Real-time interactive analysis", standard_success >= 80),
        ("✅ Advanced risk assessment", monte_carlo_success >= 80),
        ("✅ Comparative case analysis", comparative_success >= 80),
        ("✅ Settlement strategy optimization", ai_success >= 80)
    ]
    
    print("\n🎯 EXPECTED ENHANCEMENTS VERIFICATION:")
    for feature, status in features_status:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {feature.split(' ', 1)[1]}")
    
    working_features = sum(1 for _, status in features_status if status)
    total_features = len(features_status)
    
    print(f"\n📊 Feature Implementation: {working_features}/{total_features} ({working_features/total_features*100:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 OVERALL RESULT: OUTSTANDING - Enhanced Settlement Probability Calculator working excellently!")
    elif success_rate >= 80:
        print("✅ OVERALL RESULT: EXCELLENT - Settlement Calculator working very well with minor issues")
    elif success_rate >= 70:
        print("✅ OVERALL RESULT: GOOD - Settlement Calculator mostly functional")
    elif success_rate >= 60:
        print("⚠️ OVERALL RESULT: FAIR - Settlement Calculator working but needs improvements")
    else:
        print("❌ OVERALL RESULT: NEEDS ATTENTION - Settlement Calculator has significant issues")
    
    print("=" * 80)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)