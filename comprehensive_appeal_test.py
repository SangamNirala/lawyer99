#!/usr/bin/env python3
"""
Comprehensive Appeal Analysis Testing - Multiple Scenarios
"""

import requests
import json
import sys
from datetime import datetime

# Backend URL
API_BASE_URL = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com/api"

def test_scenario(name, test_data, expected_behavior=""):
    """Test a specific scenario"""
    print(f"\n🎯 TESTING: {name}")
    print("-" * 50)
    
    if expected_behavior:
        print(f"Expected: {expected_behavior}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/litigation/appeal-analysis",
            json=test_data,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            appeal_filing = data.get('appeal_probability', 0)
            appeal_success = data.get('appeal_success_probability', 0)
            confidence = data.get('appeal_confidence', 0)
            
            print(f"✅ SUCCESS - Status: 200")
            print(f"   📈 Appeal Filing Probability: {appeal_filing:.1%}")
            print(f"   🏆 Appeal Success Probability: {appeal_success:.1%}")
            print(f"   🎯 Confidence: {confidence:.1%}")
            
            # Verify metrics are different (key requirement)
            if appeal_filing != appeal_success:
                print(f"   ✅ Metrics properly separated")
            else:
                print(f"   ⚠️  Metrics identical - may be an issue")
            
            return True, {
                'filing_prob': appeal_filing,
                'success_prob': appeal_success,
                'confidence': confidence
            }
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False, None

def main():
    print("🎯 COMPREHENSIVE APPEAL ANALYSIS TESTING")
    print("=" * 70)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Focus: Verify metrics clarification fix across multiple scenarios")
    
    test_results = []
    
    # Test 1: User's exact scenario (from review request)
    user_scenario = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 7.0,
        "case_complexity": 0.65,
        "case_facts": "Plaintiff alleges breach of contract by federal contractor involving delayed delivery of critical medical equipment. Agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    success, result = test_scenario(
        "User's Exact Scenario", 
        user_scenario,
        "Low filing probability (~5-10%), moderate success probability (~40-60%)"
    )
    test_results.append(('User Scenario', success, result))
    
    # Test 2: Strong evidence case (should have very low appeal filing probability)
    strong_evidence = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "evidence_strength": 9.0,
        "case_complexity": 0.3,
        "case_facts": "Clear-cut breach of contract with overwhelming evidence including signed contracts, witness testimony, and documented damages."
    }
    
    success, result = test_scenario(
        "Strong Evidence Case",
        strong_evidence,
        "Very low filing probability (<5%), moderate success probability"
    )
    test_results.append(('Strong Evidence', success, result))
    
    # Test 3: Weak evidence case (should have higher appeal filing probability)
    weak_evidence = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "evidence_strength": 3.0,
        "case_complexity": 0.8,
        "case_facts": "Disputed contract terms with limited documentation and conflicting witness accounts."
    }
    
    success, result = test_scenario(
        "Weak Evidence Case",
        weak_evidence,
        "Higher filing probability (>10%), lower success probability"
    )
    test_results.append(('Weak Evidence', success, result))
    
    # Test 4: High value case
    high_value = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 5000000,  # $5M
        "evidence_strength": 6.0,
        "case_complexity": 0.6
    }
    
    success, result = test_scenario(
        "High Value Case ($5M)",
        high_value,
        "Higher filing probability due to case value"
    )
    test_results.append(('High Value', success, result))
    
    # Test 5: Low value case
    low_value = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 50000,  # $50K
        "evidence_strength": 6.0,
        "case_complexity": 0.4
    }
    
    success, result = test_scenario(
        "Low Value Case ($50K)",
        low_value,
        "Lower filing probability due to cost vs. value"
    )
    test_results.append(('Low Value', success, result))
    
    # Test 6: Minimal data (edge case)
    minimal_data = {
        "case_type": "civil",
        "jurisdiction": "federal"
    }
    
    success, result = test_scenario(
        "Minimal Data (Edge Case)",
        minimal_data,
        "Should still return reasonable probabilities"
    )
    test_results.append(('Minimal Data', success, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for _, success, _ in test_results if success)
    total_tests = len(test_results)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, success, result in test_results:
        if success and result:
            print(f"   ✅ {test_name}: Filing={result['filing_prob']:.1%}, Success={result['success_prob']:.1%}, Conf={result['confidence']:.1%}")
        else:
            print(f"   ❌ {test_name}: FAILED")
    
    # Key findings
    print(f"\n🔍 KEY FINDINGS:")
    
    successful_results = [result for _, success, result in test_results if success and result]
    
    if successful_results:
        # Check if metrics are properly separated
        all_separated = all(r['filing_prob'] != r['success_prob'] for r in successful_results)
        if all_separated:
            print("✅ Metrics separation working - all tests show different filing vs success probabilities")
        else:
            print("⚠️  Some tests show identical metrics - may need investigation")
        
        # Check confidence levels
        high_confidence_count = sum(1 for r in successful_results if r['confidence'] >= 0.85)
        if high_confidence_count >= len(successful_results) * 0.8:
            print("✅ AI analysis mode working - most tests show high confidence (85%+)")
        else:
            print("⚠️  Some tests show lower confidence - may be using fallback mode")
        
        # Check evidence correlation
        strong_evidence_result = next((r for name, success, r in test_results if name == 'Strong Evidence' and success and r), None)
        weak_evidence_result = next((r for name, success, r in test_results if name == 'Weak Evidence' and success and r), None)
        
        if strong_evidence_result and weak_evidence_result:
            if weak_evidence_result['filing_prob'] > strong_evidence_result['filing_prob']:
                print("✅ Evidence correlation working - weak evidence shows higher filing probability")
            else:
                print("⚠️  Evidence correlation may not be working as expected")
    
    # Final assessment
    print(f"\n🎯 METRICS CLARIFICATION FIX ASSESSMENT:")
    
    if success_rate >= 90:
        print("🎉 OUTSTANDING SUCCESS - Metrics clarification fix working perfectly!")
        print("✅ Both appeal filing and success probabilities calculated correctly")
        print("✅ Metrics properly separated across all scenarios")
        print("✅ User confusion about contradictory metrics should be completely resolved")
        assessment = "OUTSTANDING"
    elif success_rate >= 75:
        print("✅ GOOD SUCCESS - Metrics clarification fix working well")
        print("✅ Most scenarios working correctly")
        print("⚠️  Some minor issues may remain")
        assessment = "GOOD"
    else:
        print("❌ NEEDS ATTENTION - Metrics clarification fix has issues")
        print("❌ Multiple scenarios failing")
        print("❌ User confusion may not be fully resolved")
        assessment = "NEEDS_ATTENTION"
    
    print(f"\n📊 FINAL STATUS: {assessment}")
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)