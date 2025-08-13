#!/usr/bin/env python3

import requests
import json

def test_scenario(name, case_data, expected_evidence_range, expected_complexity_range):
    print(f"\n🧪 TESTING: {name}")
    print("-" * 60)
    
    url = "https://legalcore.preview.emergentagent.com/api/litigation/appeal-analysis"
    response = requests.post(url, json=case_data, timeout=60)
    
    if response.status_code == 200:
        data = response.json()
        
        # TASK 2 Analysis
        case_facts_analysis = data.get("case_facts_analysis", {})
        ai_evidence = case_facts_analysis.get("evidence_strength_suggested", 0)
        ai_complexity = case_facts_analysis.get("case_complexity_suggested", 0)
        complexity_pct = ai_complexity * 100 if ai_complexity <= 1 else ai_complexity
        
        print(f"📊 AI Evidence: {ai_evidence}/10 (Expected: {expected_evidence_range[0]}-{expected_evidence_range[1]}/10)")
        print(f"📊 AI Complexity: {complexity_pct}% (Expected: {expected_complexity_range[0]}-{expected_complexity_range[1]}%)")
        
        evidence_ok = expected_evidence_range[0] <= ai_evidence <= expected_evidence_range[1]
        complexity_ok = expected_complexity_range[0] <= complexity_pct <= expected_complexity_range[1]
        
        if evidence_ok:
            print("✅ Evidence suggestion in expected range")
        else:
            print("❌ Evidence suggestion outside expected range")
            
        if complexity_ok:
            print("✅ Complexity suggestion in expected range")
        else:
            print("❌ Complexity suggestion outside expected range")
        
        # TASK 3 Analysis
        appeal_cost = data.get("appeal_cost_estimate", 0)
        case_value = case_data["case_value"]
        cost_pct = (appeal_cost / case_value) * 100 if appeal_cost > 0 else 0
        
        print(f"📊 Cost: ${appeal_cost:,.2f} ({cost_pct:.1f}% of ${case_value:,})")
        
        # Cost should be reasonable (8-18% range)
        if 8 <= cost_pct <= 18:
            print("✅ Cost percentage in reasonable range (8-18%)")
        else:
            print("❌ Cost percentage outside reasonable range")
        
        return evidence_ok and complexity_ok and (8 <= cost_pct <= 18)
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

# Test scenarios
print("🎯 ADDITIONAL APPEAL ANALYSIS TESTING")
print("=" * 80)

results = []

# Scenario 1: Simple case with strong evidence
simple_case = {
    "case_type": "civil",
    "jurisdiction": "federal",
    "case_value": 100000,
    "evidence_strength": 9.0,
    "case_complexity": 0.3,
    "case_facts": "Simple breach of contract case. Clear written agreement with specific terms. Defendant failed to deliver goods as specified. Strong documentary evidence including signed contract, delivery receipts, and payment records."
}
results.append(test_scenario("Simple Case with Strong Evidence", simple_case, [8, 9], [20, 40]))

# Scenario 2: Complex case with weak evidence
complex_case = {
    "case_type": "commercial",
    "jurisdiction": "federal",
    "case_value": 2000000,
    "evidence_strength": 3.0,
    "case_complexity": 0.9,
    "case_facts": "Multi-party intellectual property dispute involving patent infringement claims across multiple jurisdictions. Complex licensing agreements with ambiguous terms. Conflicting expert testimony on technical specifications. Missing key documentation due to data breach."
}
results.append(test_scenario("Complex Case with Weak Evidence", complex_case, [2, 4], [80, 95]))

# Scenario 3: Medium complexity case
medium_case = {
    "case_type": "employment",
    "jurisdiction": "california",
    "case_value": 500000,
    "evidence_strength": 6.0,
    "case_complexity": 0.5,
    "case_facts": "Employment discrimination case with mixed evidence. Some documentation supporting claims, witness testimony available, but some key evidence is circumstantial. Standard employment law complexity."
}
results.append(test_scenario("Medium Complexity Employment Case", medium_case, [5, 7], [40, 60]))

# Summary
print(f"\n📊 SUMMARY")
print("=" * 40)
passed = sum(results)
total = len(results)
success_rate = (passed / total) * 100 if total > 0 else 0

print(f"Tests Passed: {passed}/{total}")
print(f"Success Rate: {success_rate:.1f}%")

if success_rate >= 80:
    print("🎉 EXCELLENT: TASK 2 & TASK 3 working across different scenarios!")
elif success_rate >= 60:
    print("✅ GOOD: TASK 2 & TASK 3 mostly working with minor issues")
else:
    print("❌ NEEDS ATTENTION: TASK 2 & TASK 3 have significant issues")
