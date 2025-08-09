#!/usr/bin/env python3

import requests
import json

# Test the user's exact scenario
user_case = {
    "case_type": "civil",
    "jurisdiction": "federal", 
    "case_value": 250000,
    "judge_name": "Judge Rebecca Morgan",
    "evidence_strength": 7.0,
    "case_complexity": 0.65,
    "case_facts": "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
}

url = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com/api/litigation/appeal-analysis"
response = requests.post(url, json=user_case, timeout=120)

print("🎯 APPEAL ANALYSIS TASK 2 & TASK 3 TESTING RESULTS")
print("=" * 80)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print("✅ SUCCESS: Appeal analysis endpoint working")
    print()
    
    # TASK 2 Analysis
    print("🔍 TASK 2: Evidence/Complexity AI Correlation")
    print("-" * 50)
    
    case_facts_analysis = data.get("case_facts_analysis")
    if case_facts_analysis:
        print("✅ case_facts_analysis field is present")
        print(f"📊 Full Analysis: {json.dumps(case_facts_analysis, indent=2)}")
        
        ai_evidence = case_facts_analysis.get("evidence_strength_suggested")
        ai_complexity = case_facts_analysis.get("case_complexity_suggested")
        
        if ai_evidence is not None:
            print(f"📊 AI Suggested Evidence: {ai_evidence}/10 (User input: 7.0/10)")
            if 6.0 <= ai_evidence <= 8.0:
                print("✅ Evidence suggestion in expected range (6-8/10)")
            else:
                print(f"❌ Evidence suggestion {ai_evidence}/10 outside expected range")
        
        if ai_complexity is not None:
            complexity_pct = ai_complexity * 100 if ai_complexity <= 1 else ai_complexity
            print(f"📊 AI Suggested Complexity: {complexity_pct}% (User input: 65%)")
            if 50 <= complexity_pct <= 70:
                print("✅ Complexity suggestion in expected range (50-70%)")
            else:
                print(f"❌ Complexity suggestion {complexity_pct}% outside expected range")
        
        print(f"📝 Evidence Reasoning: {case_facts_analysis.get('evidence_reasoning', 'Missing')}")
        print(f"📝 Complexity Reasoning: {case_facts_analysis.get('complexity_reasoning', 'Missing')}")
    else:
        print("❌ case_facts_analysis field is MISSING")
    
    print()
    
    # TASK 3 Analysis
    print("💰 TASK 3: Enhanced Cost Estimation")
    print("-" * 50)
    
    appeal_cost = data.get("appeal_cost_estimate", 0)
    case_value = 250000
    
    print(f"📊 Case Value: ${case_value:,}")
    print(f"📊 Appeal Cost: ${appeal_cost:,.2f}")
    
    if appeal_cost > 0:
        cost_pct = (appeal_cost / case_value) * 100
        expected_cost = case_value * 0.11  # 11%
        max_cost = case_value * 0.18  # 18%
        
        print(f"📊 Cost Percentage: {cost_pct:.1f}%")
        print(f"📊 Expected (~11%): ${expected_cost:,.2f}")
        print(f"📊 Maximum (18%): ${max_cost:,.2f}")
        
        if abs(appeal_cost - expected_cost) <= (expected_cost * 0.5):
            print("✅ Cost is reasonable (~11% of case value)")
        else:
            print(f"❌ Cost not close to expected 11%")
        
        if appeal_cost <= max_cost:
            print("✅ Cost does not exceed 18% maximum")
        else:
            print(f"❌ Cost exceeds 18% maximum")
        
        if appeal_cost < 97500:
            improvement = ((97500 - appeal_cost) / 97500) * 100
            print(f"✅ Cost improved by {improvement:.1f}% from previous $97,500")
        else:
            print("❌ Cost not improved from previous estimate")
    else:
        print("❌ No appeal cost estimate provided")
    
    print()
    print("📊 OTHER KEY METRICS:")
    print(f"   Appeal Probability: {data.get('appeal_probability', 0):.1%}")
    print(f"   Appeal Success Probability: {data.get('appeal_success_probability', 0):.1%}")
    print(f"   Appeal Timeline: {data.get('appeal_timeline', 0)} days")
    
else:
    print(f"❌ Request failed: {response.status_code}")
    print(f"Response: {response.text}")
