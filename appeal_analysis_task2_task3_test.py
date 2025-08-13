#!/usr/bin/env python3
"""
Appeal Analysis TASK 2 & TASK 3 Testing - Evidence/Complexity Correlation & Cost Estimation Fix
===============================================================================================

Testing the newly implemented TASK 2 (Evidence/Complexity Correlation Fix) and TASK 3 
(Cost Estimation improvements) for Appeal Probability Analysis.

CRITICAL TEST SCENARIOS:

**USER'S EXACT SCENARIO:**
- Case Type: Civil
- Jurisdiction: Federal  
- Case Value: $250,000
- Judge Name: Judge Rebecca Morgan
- Evidence Strength: 7/10
- Case Complexity: 65%
- Case Facts: "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."

**TASK 2 - Evidence/Complexity AI Correlation:**
- Verify new `case_facts_analysis` field is present in response
- Check AI suggests reasonable evidence strength (should analyze contract/docs = strong evidence, expect ~6-8/10)
- Check AI suggests reasonable complexity (supply chain + international = moderate-high complexity, expect ~50-70%)
- Verify reasoning fields explain the suggestions
- Confirm key evidence factors and complexity factors are identified

**TASK 3 - Enhanced Cost Estimation:**
- For $250k federal case, cost should be ~$27,500 (11% of case value) 
- Cost should NOT exceed 18% of case value ($45,000 maximum)
- Verify cost is much more reasonable than previous $97,500 estimate
- Check backend logs show cost calculation details

Test endpoint: POST /api/litigation/appeal-analysis
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://legalmate-research.preview.emergentagent.com/api"

def test_user_exact_scenario_task2_task3():
    """Test the user's exact scenario to verify TASK 2 and TASK 3 fixes"""
    print("🎯 TESTING USER'S EXACT SCENARIO - TASK 2 & TASK 3 FIXES")
    print("=" * 80)
    
    test_results = []
    
    # User's exact scenario from the review request
    user_case = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 7.0,
        "case_complexity": 0.65,
        "case_facts": "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    print(f"\n📋 USER'S EXACT TEST CASE:")
    print(f"Case Type: {user_case['case_type']}")
    print(f"Jurisdiction: {user_case['jurisdiction']}")
    print(f"Case Value: ${user_case['case_value']:,}")
    print(f"Judge Name: {user_case['judge_name']}")
    print(f"Evidence Strength: {user_case['evidence_strength']}/10")
    print(f"Case Complexity: {user_case['case_complexity']*100}%")
    print(f"Case Facts Length: {len(user_case['case_facts'])} characters")
    print(f"Case Facts Preview: {user_case['case_facts'][:100]}...")
    
    try:
        url = f"{BACKEND_URL}/litigation/appeal-analysis"
        print(f"\nRequest URL: {url}")
        
        response = requests.post(url, json=user_case, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Appeal analysis endpoint working")
            
            # =================================================================
            # TASK 2 TESTING: Evidence/Complexity AI Correlation
            # =================================================================
            print("\n" + "="*60)
            print("🔍 TASK 2 TESTING: Evidence/Complexity AI Correlation")
            print("="*60)
            
            task2_results = []
            
            # Check for new case_facts_analysis field
            case_facts_analysis = data.get('case_facts_analysis')
            if case_facts_analysis:
                print("✅ TASK 2: case_facts_analysis field is present")
                task2_results.append(True)
                
                print(f"\n📊 Case Facts Analysis Structure:")
                if isinstance(case_facts_analysis, dict):
                    for key, value in case_facts_analysis.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {len(value)} characters")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Type: {type(case_facts_analysis)}")
                    print(f"  Content: {str(case_facts_analysis)[:200]}...")
            else:
                print("❌ TASK 2: case_facts_analysis field is MISSING")
                task2_results.append(False)
            
            # Check AI suggested evidence strength
            ai_suggested_evidence = data.get('ai_suggested_evidence_strength')
            if ai_suggested_evidence is not None:
                print(f"\n📊 AI Suggested Evidence Strength: {ai_suggested_evidence}/10")
                
                # Should analyze contract/docs = strong evidence, expect ~6-8/10
                if 6.0 <= ai_suggested_evidence <= 8.0:
                    print("✅ TASK 2: AI suggested evidence strength in expected range (6-8/10)")
                    print("   Analysis: Contract documents + delivery logs + email correspondence = strong evidence")
                    task2_results.append(True)
                else:
                    print(f"❌ TASK 2: AI suggested evidence strength {ai_suggested_evidence}/10 outside expected range (6-8/10)")
                    task2_results.append(False)
            else:
                print("❌ TASK 2: ai_suggested_evidence_strength field is MISSING")
                task2_results.append(False)
            
            # Check AI suggested complexity
            ai_suggested_complexity = data.get('ai_suggested_complexity')
            if ai_suggested_complexity is not None:
                complexity_percentage = ai_suggested_complexity * 100 if ai_suggested_complexity <= 1 else ai_suggested_complexity
                print(f"\n📊 AI Suggested Complexity: {complexity_percentage}%")
                
                # Supply chain + international = moderate-high complexity, expect ~50-70%
                if 50 <= complexity_percentage <= 70:
                    print("✅ TASK 2: AI suggested complexity in expected range (50-70%)")
                    print("   Analysis: Supply chain disruptions + international shipping = moderate-high complexity")
                    task2_results.append(True)
                else:
                    print(f"❌ TASK 2: AI suggested complexity {complexity_percentage}% outside expected range (50-70%)")
                    task2_results.append(False)
            else:
                print("❌ TASK 2: ai_suggested_complexity field is MISSING")
                task2_results.append(False)
            
            # Check for reasoning fields
            evidence_reasoning = data.get('evidence_strength_reasoning')
            complexity_reasoning = data.get('complexity_reasoning')
            
            if evidence_reasoning:
                print(f"\n📝 Evidence Strength Reasoning: {len(evidence_reasoning)} characters")
                print(f"   Preview: {evidence_reasoning[:150]}...")
                
                # Check if reasoning mentions key evidence factors
                key_evidence_terms = ['contract', 'document', 'delivery', 'log', 'email', 'correspondence']
                mentioned_terms = [term for term in key_evidence_terms if term.lower() in evidence_reasoning.lower()]
                
                if len(mentioned_terms) >= 2:
                    print(f"✅ TASK 2: Evidence reasoning mentions key factors: {mentioned_terms}")
                    task2_results.append(True)
                else:
                    print(f"❌ TASK 2: Evidence reasoning lacks key evidence factors (found: {mentioned_terms})")
                    task2_results.append(False)
            else:
                print("❌ TASK 2: evidence_strength_reasoning field is MISSING")
                task2_results.append(False)
            
            if complexity_reasoning:
                print(f"\n📝 Complexity Reasoning: {len(complexity_reasoning)} characters")
                print(f"   Preview: {complexity_reasoning[:150]}...")
                
                # Check if reasoning mentions key complexity factors
                key_complexity_terms = ['supply chain', 'international', 'shipping', 'disruption', 'contractor', 'federal']
                mentioned_complexity = [term for term in key_complexity_terms if term.lower() in complexity_reasoning.lower()]
                
                if len(mentioned_complexity) >= 2:
                    print(f"✅ TASK 2: Complexity reasoning mentions key factors: {mentioned_complexity}")
                    task2_results.append(True)
                else:
                    print(f"❌ TASK 2: Complexity reasoning lacks key complexity factors (found: {mentioned_complexity})")
                    task2_results.append(False)
            else:
                print("❌ TASK 2: complexity_reasoning field is MISSING")
                task2_results.append(False)
            
            # =================================================================
            # TASK 3 TESTING: Enhanced Cost Estimation
            # =================================================================
            print("\n" + "="*60)
            print("💰 TASK 3 TESTING: Enhanced Cost Estimation")
            print("="*60)
            
            task3_results = []
            
            # Check appeal cost estimate
            appeal_cost = data.get('appeal_cost_estimate', 0)
            case_value = user_case['case_value']
            
            print(f"\n📊 Appeal Cost Analysis:")
            print(f"   Case Value: ${case_value:,}")
            print(f"   Appeal Cost Estimate: ${appeal_cost:,.2f}")
            
            if appeal_cost > 0:
                cost_percentage = (appeal_cost / case_value) * 100
                print(f"   Cost as % of Case Value: {cost_percentage:.1f}%")
                
                # For $250k federal case, cost should be ~$27,500 (11% of case value)
                expected_cost = case_value * 0.11  # 11%
                max_acceptable_cost = case_value * 0.18  # 18% maximum
                
                print(f"   Expected Cost (~11%): ${expected_cost:,.2f}")
                print(f"   Maximum Acceptable (18%): ${max_acceptable_cost:,.2f}")
                
                # Test 1: Cost should be reasonable (~11% of case value)
                if abs(appeal_cost - expected_cost) <= (expected_cost * 0.5):  # Within 50% of expected
                    print("✅ TASK 3: Appeal cost is reasonable (~11% of case value)")
                    task3_results.append(True)
                else:
                    print(f"❌ TASK 3: Appeal cost ${appeal_cost:,.2f} not close to expected ${expected_cost:,.2f}")
                    task3_results.append(False)
                
                # Test 2: Cost should NOT exceed 18% of case value ($45,000 maximum)
                if appeal_cost <= max_acceptable_cost:
                    print("✅ TASK 3: Appeal cost does not exceed 18% maximum")
                    task3_results.append(True)
                else:
                    print(f"❌ TASK 3: Appeal cost ${appeal_cost:,.2f} exceeds 18% maximum ${max_acceptable_cost:,.2f}")
                    task3_results.append(False)
                
                # Test 3: Cost should be much more reasonable than previous $97,500 estimate
                previous_problematic_cost = 97500
                if appeal_cost < previous_problematic_cost:
                    improvement = ((previous_problematic_cost - appeal_cost) / previous_problematic_cost) * 100
                    print(f"✅ TASK 3: Cost improved by {improvement:.1f}% from previous ${previous_problematic_cost:,}")
                    task3_results.append(True)
                else:
                    print(f"❌ TASK 3: Cost ${appeal_cost:,.2f} not improved from previous ${previous_problematic_cost:,}")
                    task3_results.append(False)
                
            else:
                print("❌ TASK 3: appeal_cost_estimate is 0 or missing")
                task3_results.extend([False, False, False])
            
            # Check for cost calculation details
            cost_breakdown = data.get('cost_breakdown') or data.get('appeal_cost_breakdown')
            if cost_breakdown:
                print(f"\n📊 Cost Breakdown Details:")
                if isinstance(cost_breakdown, dict):
                    for component, amount in cost_breakdown.items():
                        if isinstance(amount, (int, float)):
                            print(f"   {component}: ${amount:,.2f}")
                        else:
                            print(f"   {component}: {amount}")
                    print("✅ TASK 3: Cost calculation details are provided")
                    task3_results.append(True)
                else:
                    print(f"   Type: {type(cost_breakdown)}")
                    print(f"   Content: {cost_breakdown}")
                    task3_results.append(True)
            else:
                print("⚠️ TASK 3: Cost breakdown details not provided (may be in backend logs)")
                task3_results.append(False)
            
            # =================================================================
            # OVERALL ASSESSMENT
            # =================================================================
            print("\n" + "="*60)
            print("📊 OVERALL TASK 2 & TASK 3 ASSESSMENT")
            print("="*60)
            
            task2_success_rate = (sum(task2_results) / len(task2_results)) * 100 if task2_results else 0
            task3_success_rate = (sum(task3_results) / len(task3_results)) * 100 if task3_results else 0
            
            print(f"\n📈 TASK 2 (Evidence/Complexity Correlation): {sum(task2_results)}/{len(task2_results)} tests passed ({task2_success_rate:.1f}%)")
            print(f"📈 TASK 3 (Cost Estimation): {sum(task3_results)}/{len(task3_results)} tests passed ({task3_success_rate:.1f}%)")
            
            # Overall success criteria
            task2_success = task2_success_rate >= 80
            task3_success = task3_success_rate >= 75
            
            if task2_success and task3_success:
                print("\n🎉 BOTH TASKS SUCCESSFUL: Evidence/Complexity correlation and Cost estimation fixes working!")
                test_results.append(("User Scenario TASK 2 & 3", True, f"T2: {task2_success_rate:.1f}%, T3: {task3_success_rate:.1f}%"))
            elif task2_success:
                print("\n✅ TASK 2 SUCCESSFUL, TASK 3 NEEDS ATTENTION")
                test_results.append(("User Scenario TASK 2", True, f"Evidence/Complexity: {task2_success_rate:.1f}%"))
                test_results.append(("User Scenario TASK 3", False, f"Cost Estimation: {task3_success_rate:.1f}%"))
            elif task3_success:
                print("\n✅ TASK 3 SUCCESSFUL, TASK 2 NEEDS ATTENTION")
                test_results.append(("User Scenario TASK 2", False, f"Evidence/Complexity: {task2_success_rate:.1f}%"))
                test_results.append(("User Scenario TASK 3", True, f"Cost Estimation: {task3_success_rate:.1f}%"))
            else:
                print("\n❌ BOTH TASKS NEED ATTENTION")
                test_results.append(("User Scenario TASK 2", False, f"Evidence/Complexity: {task2_success_rate:.1f}%"))
                test_results.append(("User Scenario TASK 3", False, f"Cost Estimation: {task3_success_rate:.1f}%"))
            
            # Display other key metrics for context
            print(f"\n📊 OTHER KEY METRICS:")
            appeal_prob = data.get('appeal_probability', 0)
            appeal_success = data.get('appeal_success_probability', 0)
            print(f"   Appeal Probability: {appeal_prob:.1%}")
            print(f"   Appeal Success Probability: {appeal_success:.1%}")
            
            # Check for AI analysis quality
            ai_insights = data.get('ai_insights', '') or data.get('ai_analysis', '')
            if ai_insights and len(ai_insights) > 100:
                print(f"   AI Analysis Length: {len(ai_insights)} characters")
                print("✅ Substantial AI analysis provided")
            else:
                print("⚠️ Limited AI analysis provided")
                
        elif response.status_code == 422:
            print(f"❌ VALIDATION ERROR (422)")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw error response: {response.text}")
            test_results.append(("User Scenario", False, "Validation error"))
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            test_results.append(("User Scenario", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        test_results.append(("User Scenario", False, str(e)))
    
    print("-" * 80)
    return test_results

def test_additional_evidence_complexity_scenarios():
    """Test additional scenarios to verify evidence/complexity correlation works across different cases"""
    print("\n🧪 TESTING ADDITIONAL EVIDENCE/COMPLEXITY SCENARIOS")
    print("=" * 80)
    
    test_results = []
    
    # Scenario 1: Simple case with strong evidence (should suggest low complexity)
    simple_case = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 100000,
        "evidence_strength": 9.0,
        "case_complexity": 0.3,
        "case_facts": "Simple breach of contract case. Clear written agreement with specific terms. Defendant failed to deliver goods as specified. Strong documentary evidence including signed contract, delivery receipts, and payment records."
    }
    
    print(f"\n📋 Scenario 1: Simple Case with Strong Evidence")
    print(f"Expected: High AI evidence suggestion (~8-9), Low AI complexity suggestion (~20-40%)")
    
    try:
        response = requests.post(f"{BACKEND_URL}/litigation/appeal-analysis", json=simple_case, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            ai_evidence = data.get('ai_suggested_evidence_strength', 0)
            ai_complexity = data.get('ai_suggested_complexity', 0)
            complexity_pct = ai_complexity * 100 if ai_complexity <= 1 else ai_complexity
            
            print(f"📊 AI Suggested Evidence: {ai_evidence}/10")
            print(f"📊 AI Suggested Complexity: {complexity_pct}%")
            
            # Validate expectations
            evidence_ok = ai_evidence >= 8.0  # Should be high for strong documentary evidence
            complexity_ok = complexity_pct <= 40  # Should be low for simple case
            
            if evidence_ok and complexity_ok:
                print("✅ Simple case correctly analyzed: High evidence, Low complexity")
                test_results.append(("Simple Case Analysis", True, f"Evidence: {ai_evidence}, Complexity: {complexity_pct}%"))
            else:
                print(f"❌ Simple case analysis incorrect: Evidence {ai_evidence} (expected ≥8), Complexity {complexity_pct}% (expected ≤40%)")
                test_results.append(("Simple Case Analysis", False, f"Evidence: {ai_evidence}, Complexity: {complexity_pct}%"))
        else:
            test_results.append(("Simple Case Analysis", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        test_results.append(("Simple Case Analysis", False, str(e)))
    
    # Scenario 2: Complex case with weak evidence (should suggest high complexity, low evidence)
    complex_case = {
        "case_type": "commercial",
        "jurisdiction": "federal",
        "case_value": 2000000,
        "evidence_strength": 3.0,
        "case_complexity": 0.9,
        "case_facts": "Multi-party intellectual property dispute involving patent infringement claims across multiple jurisdictions. Complex licensing agreements with ambiguous terms. Conflicting expert testimony on technical specifications. Missing key documentation due to data breach. International regulatory compliance issues."
    }
    
    print(f"\n📋 Scenario 2: Complex Case with Weak Evidence")
    print(f"Expected: Low AI evidence suggestion (~2-4), High AI complexity suggestion (~80-95%)")
    
    try:
        response = requests.post(f"{BACKEND_URL}/litigation/appeal-analysis", json=complex_case, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            ai_evidence = data.get('ai_suggested_evidence_strength', 0)
            ai_complexity = data.get('ai_suggested_complexity', 0)
            complexity_pct = ai_complexity * 100 if ai_complexity <= 1 else ai_complexity
            
            print(f"📊 AI Suggested Evidence: {ai_evidence}/10")
            print(f"📊 AI Suggested Complexity: {complexity_pct}%")
            
            # Validate expectations
            evidence_ok = ai_evidence <= 4.0  # Should be low for weak/missing evidence
            complexity_ok = complexity_pct >= 80  # Should be high for multi-party IP dispute
            
            if evidence_ok and complexity_ok:
                print("✅ Complex case correctly analyzed: Low evidence, High complexity")
                test_results.append(("Complex Case Analysis", True, f"Evidence: {ai_evidence}, Complexity: {complexity_pct}%"))
            else:
                print(f"❌ Complex case analysis incorrect: Evidence {ai_evidence} (expected ≤4), Complexity {complexity_pct}% (expected ≥80%)")
                test_results.append(("Complex Case Analysis", False, f"Evidence: {ai_evidence}, Complexity: {complexity_pct}%"))
        else:
            test_results.append(("Complex Case Analysis", False, f"HTTP {response.status_code}"))
            
    except Exception as e:
        test_results.append(("Complex Case Analysis", False, str(e)))
    
    return test_results

def test_cost_estimation_across_case_values():
    """Test cost estimation improvements across different case values"""
    print("\n💰 TESTING COST ESTIMATION ACROSS DIFFERENT CASE VALUES")
    print("=" * 80)
    
    test_results = []
    
    # Test different case values to ensure cost scaling is reasonable
    test_cases = [
        {"value": 50000, "expected_pct_min": 8, "expected_pct_max": 20},    # Small case: 8-20%
        {"value": 250000, "expected_pct_min": 9, "expected_pct_max": 18},   # Medium case: 9-18% (user's case)
        {"value": 1000000, "expected_pct_min": 10, "expected_pct_max": 16}, # Large case: 10-16%
        {"value": 5000000, "expected_pct_min": 8, "expected_pct_max": 14},  # Very large case: 8-14%
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        case_value = test_case["value"]
        min_pct = test_case["expected_pct_min"]
        max_pct = test_case["expected_pct_max"]
        
        print(f"\n📋 Cost Test {i}: ${case_value:,} case")
        print(f"Expected cost range: {min_pct}%-{max_pct}% (${case_value * min_pct/100:,.0f} - ${case_value * max_pct/100:,.0f})")
        
        test_data = {
            "case_type": "civil",
            "jurisdiction": "federal",
            "case_value": case_value,
            "evidence_strength": 6.0,
            "case_complexity": 0.6,
            "case_facts": f"Standard civil litigation case with ${case_value:,} in damages"
        }
        
        try:
            response = requests.post(f"{BACKEND_URL}/litigation/appeal-analysis", json=test_data, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                appeal_cost = data.get('appeal_cost_estimate', 0)
                
                if appeal_cost > 0:
                    cost_pct = (appeal_cost / case_value) * 100
                    print(f"📊 Appeal Cost: ${appeal_cost:,.2f} ({cost_pct:.1f}% of case value)")
                    
                    # Check if cost is within expected range
                    if min_pct <= cost_pct <= max_pct:
                        print(f"✅ Cost percentage within expected range ({min_pct}%-{max_pct}%)")
                        test_results.append((f"Cost Test ${case_value:,}", True, f"{cost_pct:.1f}% of case value"))
                    else:
                        print(f"❌ Cost percentage {cost_pct:.1f}% outside expected range ({min_pct}%-{max_pct}%)")
                        test_results.append((f"Cost Test ${case_value:,}", False, f"{cost_pct:.1f}% outside range"))
                else:
                    print("❌ No appeal cost provided")
                    test_results.append((f"Cost Test ${case_value:,}", False, "No cost estimate"))
            else:
                test_results.append((f"Cost Test ${case_value:,}", False, f"HTTP {response.status_code}"))
                
        except Exception as e:
            test_results.append((f"Cost Test ${case_value:,}", False, str(e)))
    
    return test_results

def main():
    """Main test execution function"""
    print("🎯 APPEAL ANALYSIS TASK 2 & TASK 3 COMPREHENSIVE TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 FOCUS: Testing Evidence/Complexity Correlation (TASK 2) and Cost Estimation (TASK 3)")
    print("USER SCENARIO: Civil case, Federal jurisdiction, $250k value, Judge Rebecca Morgan")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: User's Exact Scenario - TASK 2 & TASK 3
    print("\n" + "🎯" * 25 + " TEST 1 " + "🎯" * 25)
    user_results = test_user_exact_scenario_task2_task3()
    all_results.extend(user_results)
    
    # Test 2: Additional Evidence/Complexity Scenarios
    print("\n" + "🧪" * 25 + " TEST 2 " + "🧪" * 25)
    additional_results = test_additional_evidence_complexity_scenarios()
    all_results.extend(additional_results)
    
    # Test 3: Cost Estimation Across Different Case Values
    print("\n" + "💰" * 25 + " TEST 3 " + "💰" * 25)
    cost_results = test_cost_estimation_across_case_values()
    all_results.extend(cost_results)
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("🎯 TASK 2 & TASK 3 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, passed, _ in all_results if passed)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Detailed breakdown
    print(f"\n📋 Test Suite Breakdown:")
    user_tests = [r for r in all_results if 'User Scenario' in r[0]]
    evidence_tests = [r for r in all_results if 'Evidence' in r[0] or 'Complex' in r[0] or 'Simple' in r[0]]
    cost_tests = [r for r in all_results if 'Cost' in r[0]]
    
    if user_tests:
        print(f"  User's Exact Scenario: {sum(1 for _, passed, _ in user_tests if passed)}/{len(user_tests)} passed")
    if evidence_tests:
        print(f"  Evidence/Complexity Analysis: {sum(1 for _, passed, _ in evidence_tests if passed)}/{len(evidence_tests)} passed")
    if cost_tests:
        print(f"  Cost Estimation: {sum(1 for _, passed, _ in cost_tests if passed)}/{len(cost_tests)} passed")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Task-specific Assessment
    print(f"\n🔍 TASK-SPECIFIC ASSESSMENT:")
    
    # TASK 2 Assessment
    task2_tests = [r for r in all_results if 'TASK 2' in r[0] or 'Evidence' in r[0] or 'Complex' in r[0] or 'Simple' in r[0]]
    if task2_tests:
        task2_success = sum(1 for _, passed, _ in task2_tests if passed)
        task2_rate = (task2_success / len(task2_tests)) * 100
        print(f"📊 TASK 2 (Evidence/Complexity Correlation): {task2_success}/{len(task2_tests)} tests passed ({task2_rate:.1f}%)")
        
        if task2_rate >= 80:
            print("🎉 TASK 2 SUCCESS: Evidence/Complexity AI correlation working excellently!")
            print("✅ case_facts_analysis field present with AI suggestions")
            print("✅ AI provides reasonable evidence strength and complexity suggestions")
            print("✅ Reasoning fields explain the AI analysis")
        elif task2_rate >= 60:
            print("✅ TASK 2 MOSTLY SUCCESSFUL: Evidence/Complexity correlation working with minor issues")
        else:
            print("❌ TASK 2 NEEDS ATTENTION: Evidence/Complexity correlation has significant issues")
    
    # TASK 3 Assessment
    task3_tests = [r for r in all_results if 'TASK 3' in r[0] or 'Cost' in r[0]]
    if task3_tests:
        task3_success = sum(1 for _, passed, _ in task3_tests if passed)
        task3_rate = (task3_success / len(task3_tests)) * 100
        print(f"📊 TASK 3 (Cost Estimation): {task3_success}/{len(task3_tests)} tests passed ({task3_rate:.1f}%)")
        
        if task3_rate >= 75:
            print("🎉 TASK 3 SUCCESS: Cost estimation improvements working excellently!")
            print("✅ $250k federal case costs ~$27,500 (11% of case value)")
            print("✅ Costs do not exceed 18% maximum threshold")
            print("✅ Much more reasonable than previous $97,500 estimate")
        elif task3_rate >= 60:
            print("✅ TASK 3 MOSTLY SUCCESSFUL: Cost estimation improved with minor issues")
        else:
            print("❌ TASK 3 NEEDS ATTENTION: Cost estimation still has significant issues")
    
    print(f"\n📊 OVERALL ASSESSMENT:")
    if success_rate >= 80:
        print("🎉 BOTH TASKS HIGHLY SUCCESSFUL: Evidence/Complexity correlation and Cost estimation fixes working!")
        overall_status = "HIGHLY_SUCCESSFUL"
    elif success_rate >= 70:
        print("✅ TASKS MOSTLY SUCCESSFUL: Most functionality working with minor issues")
        overall_status = "MOSTLY_SUCCESSFUL"
    else:
        print("❌ TASKS NEED ATTENTION: Significant issues remain")
        overall_status = "NEEDS_ATTENTION"
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, passed, details in all_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name} - {details}")
    
    print(f"\n📊 FINAL STATUS: {overall_status}")
    print("=" * 80)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)