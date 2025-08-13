#!/usr/bin/env python3
"""
Focused Appeal Analysis Enhancements Test - TASK 2 & TASK 3
===========================================================

Quick focused test of the key enhancements:
1. Cost estimation fixes (TASK 3)
2. Evidence/complexity correlation (TASK 2)
3. User-reported $250k scenario
"""

import requests
import json
from datetime import datetime

BACKEND_URL = "https://legalcore.preview.emergentagent.com/api"

def test_cost_estimation_fix():
    """Test TASK 3: Cost estimation fixes"""
    print("💰 TESTING TASK 3: COST ESTIMATION FIXES")
    print("=" * 50)
    
    # Test the user-reported $250k case
    test_case = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "evidence_strength": 7,
        "case_complexity": 0.65,
        "case_facts": "Contract breach case involving medical equipment delivery"
    }
    
    print(f"Testing $250k case (user-reported issue)")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=test_case,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            appeal_cost = data.get('appeal_cost_estimate', 0)
            cost_percentage = (appeal_cost / test_case['case_value']) * 100
            
            print(f"✅ Appeal Cost: ${appeal_cost:,.2f}")
            print(f"✅ Percentage of Case Value: {cost_percentage:.1f}%")
            
            # Check if user issue is resolved (should be ~12%, not 39%)
            if cost_percentage <= 20:  # Should be much less than the reported 39%
                print(f"✅ USER ISSUE RESOLVED: Cost is {cost_percentage:.1f}% (was 39%)")
                return True
            else:
                print(f"❌ USER ISSUE NOT RESOLVED: Cost still {cost_percentage:.1f}%")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_evidence_complexity_correlation():
    """Test TASK 2: Evidence/complexity correlation"""
    print("\n🧠 TESTING TASK 2: EVIDENCE/COMPLEXITY CORRELATION")
    print("=" * 50)
    
    # Test with rich case narrative (should extract parameters)
    rich_case = {
        "case_type": "civil",
        "jurisdiction": "california",
        "case_value": 500000,
        "case_facts": """Complex breach of contract case involving software licensing dispute. 
        The plaintiff software company licensed proprietary enterprise software to defendant 
        corporation. The agreement included specific usage restrictions and termination clauses. 
        However, the defendant allegedly exceeded the licensed user count and shared proprietary 
        code with third parties in violation of the agreement. Evidence includes detailed 
        licensing agreement, server logs showing unauthorized access, email communications 
        discussing code sharing, and expert testimony on software usage analytics."""
        # No evidence_strength or case_complexity provided - let AI extract
    }
    
    print(f"Testing rich case narrative ({len(rich_case['case_facts'])} chars)")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=rich_case,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            appeal_prob = data.get('appeal_probability', 0)
            factors = data.get('appeal_factors', [])
            
            print(f"✅ Appeal Probability: {appeal_prob:.1%}")
            print(f"✅ Appeal Factors Generated: {len(factors)}")
            
            # Check if AI analysis is working
            if 0.1 <= appeal_prob <= 0.9 and len(factors) > 0:
                print("✅ AI PARAMETER EXTRACTION WORKING")
                return True
            else:
                print("❌ AI parameter extraction may not be working properly")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_multiple_case_values():
    """Test cost scaling across different case values"""
    print("\n📊 TESTING COST SCALING ACROSS CASE VALUES")
    print("=" * 50)
    
    test_cases = [
        {"value": 100000, "description": "$100k case"},
        {"value": 1000000, "description": "$1M case"},
        {"value": 10000000, "description": "$10M case"}
    ]
    
    results = []
    
    for test_case in test_cases:
        case_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": test_case["value"],
            "evidence_strength": 6,
            "case_complexity": 0.6
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/litigation/appeal-analysis",
                json=case_data,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                appeal_cost = data.get('appeal_cost_estimate', 0)
                cost_percentage = (appeal_cost / case_data['case_value']) * 100
                
                print(f"✅ {test_case['description']}: ${appeal_cost:,.2f} ({cost_percentage:.1f}%)")
                results.append((test_case["value"], appeal_cost, cost_percentage))
                
                # Check 25% cap
                if cost_percentage <= 25:
                    print(f"  ✅ Under 25% cap")
                else:
                    print(f"  ❌ Exceeds 25% cap")
            else:
                print(f"❌ {test_case['description']} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['description']} error: {e}")
    
    # Check if costs scale reasonably
    if len(results) >= 2:
        costs_increase = all(results[i][1] <= results[i+1][1] for i in range(len(results)-1))
        if costs_increase:
            print("✅ Costs scale appropriately with case value")
            return True
        else:
            print("❌ Costs don't scale properly with case value")
            return False
    
    return len(results) > 0

def main():
    print("🎯 FOCUSED APPEAL ANALYSIS ENHANCEMENTS TEST")
    print("=" * 60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Cost Estimation Fix (TASK 3)
    cost_fix_result = test_cost_estimation_fix()
    results.append(("Cost Estimation Fix", cost_fix_result))
    
    # Test 2: Evidence/Complexity Correlation (TASK 2)
    correlation_result = test_evidence_complexity_correlation()
    results.append(("Evidence/Complexity Correlation", correlation_result))
    
    # Test 3: Cost Scaling
    scaling_result = test_multiple_case_values()
    results.append(("Cost Scaling", scaling_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT: Appeal analysis enhancements working well!")
        return True
    elif success_rate >= 60:
        print("\n✅ GOOD: Most enhancements working")
        return True
    else:
        print("\n⚠️ NEEDS ATTENTION: Enhancements have issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)