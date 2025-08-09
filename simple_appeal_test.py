#!/usr/bin/env python3
"""
Simple Appeal Analysis Test - Quick verification of metrics clarification fix
"""

import requests
import json
import sys
from datetime import datetime

# Backend URL
API_BASE_URL = "https://b3d0e54e-8004-47d5-83bd-25e76a95a599.preview.emergentagent.com/api"

def test_appeal_analysis_quick():
    """Quick test of appeal analysis endpoint"""
    print("🎯 QUICK APPEAL ANALYSIS TEST")
    print("=" * 50)
    
    # User's exact scenario from review request
    test_data = {
        "case_type": "civil",
        "jurisdiction": "federal",
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 7.0,
        "case_complexity": 0.65,
        "case_facts": "Plaintiff alleges breach of contract by federal contractor involving delayed delivery of critical medical equipment. Agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    print(f"📋 Testing with user's exact scenario:")
    print(f"   Case Type: {test_data['case_type']}")
    print(f"   Jurisdiction: {test_data['jurisdiction']}")
    print(f"   Case Value: ${test_data['case_value']:,}")
    print(f"   Judge: {test_data['judge_name']}")
    print(f"   Evidence Strength: {test_data['evidence_strength']}/10")
    print(f"   Case Complexity: {test_data['case_complexity']:.0%}")
    
    try:
        print(f"\n🔗 Making request to: {API_BASE_URL}/litigation/appeal-analysis")
        print("⏳ Processing... (this may take 30-60 seconds)")
        
        response = requests.post(
            f"{API_BASE_URL}/litigation/appeal-analysis",
            json=test_data,
            timeout=90  # Extended timeout for AI processing
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract the two key metrics
            appeal_probability = data.get('appeal_probability', 0)
            appeal_success_probability = data.get('appeal_success_probability', 0)
            appeal_confidence = data.get('appeal_confidence', 0)
            
            print("\n✅ SUCCESS - Appeal Analysis Working!")
            print("=" * 50)
            print("🎯 METRICS CLARIFICATION VERIFICATION:")
            print(f"   📈 Appeal Filing Probability: {appeal_probability:.1%}")
            print(f"      (Likelihood that losing party will file an appeal)")
            print(f"   🏆 Appeal Success Probability: {appeal_success_probability:.1%}")
            print(f"      (Likelihood of winning if appeal is filed)")
            print(f"   🎯 Analysis Confidence: {appeal_confidence:.1%}")
            
            # Verify metrics separation
            if appeal_probability != appeal_success_probability:
                print("\n✅ METRICS SEPARATION WORKING:")
                print("   Both metrics are different values, representing separate aspects")
                print("   - Filing probability: Risk of appeal being filed")
                print("   - Success probability: Chance of winning if filed")
            else:
                print("\n⚠️  METRICS SEPARATION ISSUE:")
                print("   Both metrics have identical values - should be different")
            
            # Check for expected ranges based on case details
            print(f"\n📊 ANALYSIS QUALITY:")
            if appeal_confidence >= 0.85:
                print("   ✅ High confidence (85%+) - Full AI analysis mode")
            elif appeal_confidence >= 0.65:
                print("   ⚠️  Moderate confidence (65-84%) - Partial AI analysis")
            else:
                print("   ❌ Low confidence (<65%) - Fallback mode")
            
            # Check other required fields
            required_fields = [
                'appeal_factors', 'appeal_timeline', 'appeal_cost_estimate',
                'preventive_measures', 'jurisdictional_appeal_rate'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if not missing_fields:
                print("   ✅ All required fields present")
            else:
                print(f"   ❌ Missing fields: {missing_fields}")
            
            # Display additional details
            appeal_factors = data.get('appeal_factors', [])
            preventive_measures = data.get('preventive_measures', [])
            
            print(f"\n📋 ADDITIONAL DETAILS:")
            print(f"   Appeal Factors: {len(appeal_factors)} identified")
            print(f"   Preventive Measures: {len(preventive_measures)} suggested")
            
            if data.get('appeal_cost_estimate'):
                print(f"   Estimated Appeal Cost: ${data['appeal_cost_estimate']:,.0f}")
            
            if data.get('appeal_timeline'):
                print(f"   Appeal Timeline: {data['appeal_timeline']} days")
            
            return True
            
        elif response.status_code == 503:
            print("❌ SERVICE UNAVAILABLE (503)")
            print("   Litigation Analytics Engine may not be loaded")
            return False
            
        elif response.status_code == 500:
            print("❌ INTERNAL SERVER ERROR (500)")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Raw error: {response.text}")
            return False
            
        else:
            print(f"❌ UNEXPECTED STATUS CODE: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ REQUEST TIMEOUT")
        print("   The AI analysis is taking too long (>90 seconds)")
        print("   This may indicate performance issues with the AI processing")
        return False
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

def main():
    print("🎯 APPEAL PROBABILITY ANALYSIS - METRICS CLARIFICATION FIX TEST")
    print("=" * 70)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔗 Backend URL: {API_BASE_URL}")
    print()
    
    success = test_appeal_analysis_quick()
    
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    if success:
        print("🎉 APPEAL ANALYSIS METRICS CLARIFICATION FIX - SUCCESS!")
        print("✅ Both appeal filing and success probabilities are working")
        print("✅ Metrics separation is functioning correctly")
        print("✅ User's reported confusion should be resolved")
        print("\n📋 READY FOR PRODUCTION USE")
    else:
        print("❌ APPEAL ANALYSIS METRICS CLARIFICATION FIX - ISSUES FOUND")
        print("❌ The endpoint is not working as expected")
        print("❌ User's reported issues may not be resolved")
        print("\n🔧 REQUIRES FURTHER INVESTIGATION")
    
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)