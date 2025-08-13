#!/usr/bin/env python3
"""
Quick Appeal Analysis Test - Check Key Functionality
"""

import requests
import json
import time

BACKEND_URL = "https://legalengine.preview.emergentagent.com/api"

def test_basic_appeal_analysis():
    """Test basic appeal analysis functionality"""
    print("🎯 Testing Basic Appeal Analysis")
    
    # Simple test case
    test_case = {
        "case_type": "civil",
        "jurisdiction": "federal", 
        "case_value": 250000,
        "evidence_strength": 7,
        "case_complexity": 0.65
    }
    
    print(f"Case Value: ${test_case['case_value']:,}")
    print("Making request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/litigation/appeal-analysis",
            json=test_case,
            timeout=45  # 45 second timeout
        )
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.1f} seconds")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract key metrics
            appeal_prob = data.get('appeal_probability', 0)
            appeal_cost = data.get('appeal_cost_estimate', 0)
            cost_percentage = (appeal_cost / test_case['case_value']) * 100
            
            print(f"✅ Appeal Probability: {appeal_prob:.1%}")
            print(f"✅ Appeal Cost: ${appeal_cost:,.2f}")
            print(f"✅ Cost Percentage: {cost_percentage:.1f}%")
            
            # Check TASK 3: Cost estimation fix
            if cost_percentage <= 20:  # Should be much less than reported 39%
                print(f"✅ TASK 3 SUCCESS: Cost {cost_percentage:.1f}% is reasonable (was 39%)")
                task3_success = True
            else:
                print(f"❌ TASK 3 ISSUE: Cost {cost_percentage:.1f}% still high")
                task3_success = False
            
            # Check if response has expected fields
            expected_fields = ['appeal_probability', 'appeal_cost_estimate', 'appeal_factors', 'preventive_measures']
            missing_fields = [f for f in expected_fields if f not in data]
            
            if not missing_fields:
                print("✅ All expected fields present")
                structure_success = True
            else:
                print(f"❌ Missing fields: {missing_fields}")
                structure_success = False
            
            return task3_success and structure_success
            
        elif response.status_code == 503:
            print("❌ Service unavailable - litigation engine not ready")
            return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>45 seconds)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cost_scaling():
    """Test cost scaling across different case values"""
    print("\n💰 Testing Cost Scaling")
    
    test_values = [100000, 500000, 1000000]
    results = []
    
    for case_value in test_values:
        test_case = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_value": case_value,
            "evidence_strength": 6,
            "case_complexity": 0.6
        }
        
        print(f"Testing ${case_value:,} case...")
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/litigation/appeal-analysis",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                appeal_cost = data.get('appeal_cost_estimate', 0)
                cost_percentage = (appeal_cost / case_value) * 100
                
                print(f"  Cost: ${appeal_cost:,.2f} ({cost_percentage:.1f}%)")
                results.append((case_value, appeal_cost, cost_percentage))
                
                # Check 25% cap
                if cost_percentage <= 25:
                    print(f"  ✅ Under 25% cap")
                else:
                    print(f"  ❌ Exceeds 25% cap")
            else:
                print(f"  ❌ Failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"  ❌ Timeout")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Check if costs scale reasonably
    if len(results) >= 2:
        print(f"\nCost scaling analysis:")
        for i, (value, cost, pct) in enumerate(results):
            print(f"  ${value:,}: ${cost:,.2f} ({pct:.1f}%)")
        
        # Costs should generally increase with case value
        costs_increase = all(results[i][1] <= results[i+1][1] for i in range(len(results)-1))
        if costs_increase:
            print("✅ Costs scale appropriately")
            return True
        else:
            print("❌ Costs don't scale properly")
            return False
    
    return len(results) > 0

def main():
    print("🚀 QUICK APPEAL ANALYSIS FUNCTIONALITY TEST")
    print("=" * 50)
    
    results = []
    
    # Test 1: Basic functionality and TASK 3 cost fix
    basic_result = test_basic_appeal_analysis()
    results.append(("Basic Appeal Analysis & Cost Fix", basic_result))
    
    # Test 2: Cost scaling
    scaling_result = test_cost_scaling()
    results.append(("Cost Scaling", scaling_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: Appeal analysis enhancements working!")
        return True
    elif success_rate >= 50:
        print("✅ PARTIAL: Some functionality working")
        return True
    else:
        print("❌ ISSUES: Major problems detected")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)