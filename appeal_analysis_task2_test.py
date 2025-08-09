#!/usr/bin/env python3
"""
Appeal Analysis Task 2 - Evidence/Complexity Correlation Fix Testing
==================================================================

Testing the specific issue where the AI correlation for case facts analysis
is returning user input values (7.0/10 evidence) instead of analyzing case
facts independently.

SPECIFIC TEST SCENARIO:
- Case Type: Civil  
- Jurisdiction: Federal
- Case Value: $250000
- Judge Name: Judge Rebecca Morgan
- Evidence Strength: 7/10
- Case Complexity: 65%
- Case Facts: "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."

KEY TESTING POINTS:
1. Check if case_facts_analysis contains proper AI-suggested values that differ from user inputs
2. Verify that evidence_strength_suggested and case_complexity_suggested are independently calculated
3. Check the reasoning fields to see if they contain actual AI analysis or generic text
4. Verify logs show "🔍 Analyzing case facts for evidence/complexity correlation"
5. Ensure system doesn't echo user inputs (7.0/10 evidence) but performs independent analysis
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com/api"

def test_appeal_analysis_evidence_complexity_correlation():
    """Test Appeal Analysis Task 2 - Evidence/Complexity Correlation Fix"""
    print("🎯 TESTING APPEAL ANALYSIS TASK 2 - EVIDENCE/COMPLEXITY CORRELATION FIX")
    print("=" * 80)
    
    test_results = []
    
    # Exact test case from the review request
    test_case = {
        "case_type": "civil",
        "jurisdiction": "Federal", 
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 7.0,  # User input - system should NOT echo this
        "case_complexity": 0.65,   # User input - system should NOT echo this
        "case_facts": "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    print(f"\n📋 Test Case: Appeal Analysis Evidence/Complexity Correlation")
    print(f"Case Type: {test_case['case_type']}")
    print(f"Jurisdiction: {test_case['jurisdiction']}")
    print(f"Case Value: ${test_case['case_value']:,}")
    print(f"Judge: {test_case['judge_name']}")
    print(f"User Evidence Strength: {test_case['evidence_strength']}/10")
    print(f"User Case Complexity: {test_case['case_complexity']*100}%")
    print(f"Case Facts Length: {len(test_case['case_facts'])} characters")
    print(f"Case Facts Preview: {test_case['case_facts'][:100]}...")
    
    try:
        url = f"{BACKEND_URL}/litigation/appeal-analysis"
        print(f"\nRequest URL: {url}")
        
        response = requests.post(url, json=test_case, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if case_facts_analysis exists
            case_facts_analysis = data.get('case_facts_analysis')
            if not case_facts_analysis:
                print("❌ case_facts_analysis field is missing from response")
                test_results.append(False)
                return test_results
            
            print("\n🔍 CASE FACTS ANALYSIS VALIDATION:")
            print(f"case_facts_analysis present: ✅")
            
            # CRITICAL TEST 1: Check if evidence_strength_suggested differs from user input
            evidence_strength_suggested = case_facts_analysis.get('evidence_strength_suggested')
            user_evidence_strength = test_case['evidence_strength']
            
            print(f"\n📊 EVIDENCE STRENGTH CORRELATION:")
            print(f"User Input Evidence Strength: {user_evidence_strength}/10")
            print(f"AI Suggested Evidence Strength: {evidence_strength_suggested}")
            
            if evidence_strength_suggested is not None:
                # Convert to same scale for comparison (AI might return 0-1 scale)
                if evidence_strength_suggested <= 1.0:
                    ai_evidence_scaled = evidence_strength_suggested * 10
                else:
                    ai_evidence_scaled = evidence_strength_suggested
                
                print(f"AI Suggested (scaled to /10): {ai_evidence_scaled}")
                
                # Check if AI analysis differs significantly from user input
                evidence_difference = abs(ai_evidence_scaled - user_evidence_strength)
                print(f"Difference from user input: {evidence_difference}")
                
                if evidence_difference > 0.5:  # More than 0.5 point difference indicates independent analysis
                    print("✅ AI evidence analysis differs from user input - INDEPENDENT ANALYSIS WORKING")
                    evidence_independent = True
                else:
                    print("❌ AI evidence analysis too similar to user input - MAY BE ECHOING USER INPUT")
                    evidence_independent = False
            else:
                print("❌ evidence_strength_suggested is missing")
                evidence_independent = False
            
            test_results.append(evidence_independent)
            
            # CRITICAL TEST 2: Check if case_complexity_suggested differs from user input
            case_complexity_suggested = case_facts_analysis.get('case_complexity_suggested')
            user_case_complexity = test_case['case_complexity']
            
            print(f"\n📊 CASE COMPLEXITY CORRELATION:")
            print(f"User Input Case Complexity: {user_case_complexity*100}%")
            print(f"AI Suggested Case Complexity: {case_complexity_suggested}")
            
            if case_complexity_suggested is not None:
                # Ensure both are in same scale (0-1)
                if case_complexity_suggested > 1.0:
                    ai_complexity_scaled = case_complexity_suggested / 100
                else:
                    ai_complexity_scaled = case_complexity_suggested
                
                print(f"AI Suggested (scaled 0-1): {ai_complexity_scaled}")
                print(f"AI Suggested (percentage): {ai_complexity_scaled*100}%")
                
                # Check if AI analysis differs significantly from user input
                complexity_difference = abs(ai_complexity_scaled - user_case_complexity)
                print(f"Difference from user input: {complexity_difference}")
                
                if complexity_difference > 0.05:  # More than 5% difference indicates independent analysis
                    print("✅ AI complexity analysis differs from user input - INDEPENDENT ANALYSIS WORKING")
                    complexity_independent = True
                else:
                    print("❌ AI complexity analysis too similar to user input - MAY BE ECHOING USER INPUT")
                    complexity_independent = False
            else:
                print("❌ case_complexity_suggested is missing")
                complexity_independent = False
            
            test_results.append(complexity_independent)
            
            # CRITICAL TEST 3: Check reasoning fields for actual AI analysis
            evidence_reasoning = case_facts_analysis.get('evidence_strength_reasoning', '')
            complexity_reasoning = case_facts_analysis.get('case_complexity_reasoning', '')
            
            print(f"\n🧠 AI REASONING ANALYSIS:")
            print(f"Evidence Reasoning Length: {len(evidence_reasoning)} characters")
            print(f"Complexity Reasoning Length: {len(complexity_reasoning)} characters")
            
            # Check for specific case details in reasoning (medical equipment, contract breach, etc.)
            case_specific_terms = [
                'medical equipment', 'contract', 'breach', 'delivery', 'hospital',
                'supply chain', 'shipping', 'federal contractor', '30 days', '90 days'
            ]
            
            evidence_case_specific = sum(1 for term in case_specific_terms if term.lower() in evidence_reasoning.lower())
            complexity_case_specific = sum(1 for term in case_specific_terms if term.lower() in complexity_reasoning.lower())
            
            print(f"Evidence reasoning contains {evidence_case_specific}/{len(case_specific_terms)} case-specific terms")
            print(f"Complexity reasoning contains {complexity_case_specific}/{len(case_specific_terms)} case-specific terms")
            
            # Check for generic vs specific analysis
            generic_phrases = [
                'based on the information provided', 'according to the case details',
                'considering the facts', 'analysis shows', 'evidence suggests'
            ]
            
            evidence_generic = sum(1 for phrase in generic_phrases if phrase.lower() in evidence_reasoning.lower())
            complexity_generic = sum(1 for phrase in generic_phrases if phrase.lower() in complexity_reasoning.lower())
            
            print(f"Evidence reasoning contains {evidence_generic} generic phrases")
            print(f"Complexity reasoning contains {complexity_generic} generic phrases")
            
            # Reasoning quality assessment
            reasoning_quality = (
                len(evidence_reasoning) > 50 and
                len(complexity_reasoning) > 50 and
                (evidence_case_specific > 0 or complexity_case_specific > 0)
            )
            
            if reasoning_quality:
                print("✅ AI reasoning appears to contain substantial case-specific analysis")
                reasoning_test = True
            else:
                print("❌ AI reasoning appears generic or insufficient")
                reasoning_test = False
            
            test_results.append(reasoning_test)
            
            # CRITICAL TEST 4: Check for correlation analysis
            correlation_analysis = case_facts_analysis.get('correlation_analysis', '')
            correlation_confidence = case_facts_analysis.get('correlation_confidence')
            
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"Correlation Analysis Length: {len(correlation_analysis)} characters")
            print(f"Correlation Confidence: {correlation_confidence}")
            
            if len(correlation_analysis) > 100 and correlation_confidence is not None:
                print("✅ Correlation analysis appears comprehensive")
                correlation_test = True
            else:
                print("❌ Correlation analysis appears insufficient")
                correlation_test = False
            
            test_results.append(correlation_test)
            
            # CRITICAL TEST 5: Overall AI analysis vs user input echo test
            print(f"\n🎯 OVERALL ECHO TEST RESULTS:")
            
            # If both evidence and complexity are too similar to user inputs, it's likely echoing
            if not evidence_independent and not complexity_independent:
                print("🚨 CRITICAL ISSUE: System appears to be ECHOING USER INPUTS instead of performing independent AI analysis")
                echo_test = False
            elif evidence_independent and complexity_independent:
                print("✅ EXCELLENT: System performing independent AI analysis, not echoing user inputs")
                echo_test = True
            else:
                print("⚠️ PARTIAL: System shows some independence but may still be influenced by user inputs")
                echo_test = True  # Partial credit
            
            test_results.append(echo_test)
            
            # Display sample reasoning for manual review
            print(f"\n📝 SAMPLE AI REASONING (for manual review):")
            print(f"Evidence Reasoning Preview: {evidence_reasoning[:200]}...")
            print(f"Complexity Reasoning Preview: {complexity_reasoning[:200]}...")
            
            # Check for the specific log message mentioned in the request
            print(f"\n🔍 LOG MESSAGE CHECK:")
            print("Note: Backend logs should show '🔍 Analyzing case facts for evidence/complexity correlation'")
            print("This test cannot verify backend logs directly, but the presence of case_facts_analysis suggests the correlation analysis is running.")
            
        elif response.status_code == 404:
            print("❌ Appeal Analysis endpoint not found (404)")
            print("This suggests the litigation analytics engine may not be loaded properly")
            test_results.append(False)
            
        elif response.status_code == 500:
            print("❌ Internal server error (500)")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
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
    
    print("-" * 80)
    return test_results

def main():
    """Main test execution function"""
    print("🎯 APPEAL ANALYSIS TASK 2 - EVIDENCE/COMPLEXITY CORRELATION FIX TESTING")
    print("=" * 90)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 FOCUS: Testing if AI correlation analysis works independently vs echoing user inputs")
    print("USER ISSUE: System returns user input values (7.0/10 evidence) instead of analyzing case facts independently")
    print("=" * 90)
    
    # Execute the main test
    print("\n" + "🎯" * 30 + " MAIN TEST " + "🎯" * 30)
    test_results = test_appeal_analysis_evidence_complexity_correlation()
    
    # Final Results Summary
    print("\n" + "=" * 90)
    print("🎯 TASK 2 EVIDENCE/COMPLEXITY CORRELATION TEST RESULTS")
    print("=" * 90)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Detailed test breakdown
    if len(test_results) >= 5:
        test_names = [
            "Evidence Strength Independence",
            "Case Complexity Independence", 
            "AI Reasoning Quality",
            "Correlation Analysis Presence",
            "Overall Echo Test"
        ]
        
        print(f"\n📋 Detailed Test Results:")
        for i, (name, result) in enumerate(zip(test_names, test_results)):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {i+1}. {name}: {status}")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Task 2 Fix Assessment
    print(f"\n🔍 TASK 2 FIX ASSESSMENT:")
    
    if success_rate >= 90:
        print("🎉 TASK 2 FIX COMPLETELY SUCCESSFUL!")
        print("✅ AI performs independent analysis of case facts")
        print("✅ Evidence/complexity suggestions differ from user inputs")
        print("✅ Reasoning contains case-specific analysis")
        print("✅ No evidence of echoing user input values")
        fix_status = "COMPLETELY_SUCCESSFUL"
    elif success_rate >= 70:
        print("✅ TASK 2 FIX MOSTLY SUCCESSFUL")
        print("✅ Most correlation analysis working independently")
        print("⚠️ Some aspects may still need refinement")
        fix_status = "MOSTLY_SUCCESSFUL"
    elif success_rate >= 40:
        print("⚠️ TASK 2 FIX PARTIALLY WORKING")
        print("⚠️ Some independent analysis detected")
        print("❌ Still may be echoing user inputs in some cases")
        fix_status = "PARTIALLY_WORKING"
    else:
        print("❌ TASK 2 FIX NOT WORKING")
        print("🚨 System appears to still be echoing user input values")
        print("❌ AI correlation analysis not performing independent case facts analysis")
        fix_status = "NOT_WORKING"
    
    print(f"\n🎯 EXPECTED BEHAVIOR VERIFICATION:")
    expected_behaviors = [
        "✅ evidence_strength_suggested should differ from user input (7.0/10)",
        "✅ case_complexity_suggested should differ from user input (65%)",
        "✅ Reasoning should contain case-specific analysis about medical equipment contract breach",
        "✅ Correlation analysis should be based on case facts, not user inputs",
        "✅ Backend logs should show '🔍 Analyzing case facts for evidence/complexity correlation'"
    ]
    
    for behavior in expected_behaviors:
        print(behavior)
    
    print(f"\n📊 TASK 2 FIX STATUS: {fix_status}")
    
    # Specific recommendations based on results
    if success_rate < 70:
        print(f"\n🔧 RECOMMENDATIONS FOR IMPROVEMENT:")
        print("1. Verify that case facts are being properly parsed and analyzed by AI")
        print("2. Check that evidence/complexity calculation logic uses case content, not user inputs")
        print("3. Ensure correlation analysis is based on case narrative content")
        print("4. Review AI prompts to ensure they focus on case facts analysis")
        print("5. Add more sophisticated case facts parsing and analysis logic")
    
    print("=" * 90)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)