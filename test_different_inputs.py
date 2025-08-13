#!/usr/bin/env python3
"""
Test with different user input values to see if AI analysis is truly independent
"""

import requests
import json
import sys

# Backend URL from environment
BACKEND_URL = "https://legalengine.preview.emergentagent.com/api"

def test_with_different_inputs():
    """Test with different user input values"""
    print("🧪 TESTING WITH DIFFERENT USER INPUT VALUES")
    print("=" * 60)
    
    # Test cases with different user inputs but same case facts
    test_cases = [
        {
            "name": "Low User Inputs",
            "evidence_strength": 3.0,  # Low user input
            "case_complexity": 0.2,    # Low user input
        },
        {
            "name": "High User Inputs", 
            "evidence_strength": 9.0,  # High user input
            "case_complexity": 0.9,    # High user input
        },
        {
            "name": "Medium User Inputs",
            "evidence_strength": 5.0,  # Medium user input
            "case_complexity": 0.5,    # Medium user input
        }
    ]
    
    # Same case facts for all tests
    case_facts = "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🎯 TEST: {test_case['name']}")
        print(f"User Evidence Input: {test_case['evidence_strength']}/10")
        print(f"User Complexity Input: {test_case['case_complexity']*100}%")
        
        request_data = {
            "case_type": "civil",
            "jurisdiction": "federal", 
            "case_value": 250000,
            "judge_name": "Judge Rebecca Morgan",
            "evidence_strength": test_case['evidence_strength'],
            "case_complexity": test_case['case_complexity'],
            "case_facts": case_facts
        }
        
        try:
            url = f"{BACKEND_URL}/litigation/appeal-analysis"
            response = requests.post(url, json=request_data, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                case_facts_analysis = data.get('case_facts_analysis', {})
                
                ai_evidence = case_facts_analysis.get('evidence_strength_suggested')
                ai_complexity = case_facts_analysis.get('case_complexity_suggested')
                
                print(f"AI Evidence Suggested: {ai_evidence}/10")
                print(f"AI Complexity Suggested: {ai_complexity*100 if ai_complexity else 0}%")
                
                # Calculate differences
                evidence_diff = abs(ai_evidence - test_case['evidence_strength']) if ai_evidence else 0
                complexity_diff = abs(ai_complexity - test_case['case_complexity']) if ai_complexity else 0
                
                print(f"Evidence Difference: {evidence_diff}")
                print(f"Complexity Difference: {complexity_diff}")
                
                # Check if AI is independent
                evidence_independent = evidence_diff > 0.5
                complexity_independent = complexity_diff > 0.05
                
                print(f"Evidence Independent: {'✅' if evidence_independent else '❌'}")
                print(f"Complexity Independent: {'✅' if complexity_independent else '❌'}")
                
                results.append({
                    'name': test_case['name'],
                    'user_evidence': test_case['evidence_strength'],
                    'ai_evidence': ai_evidence,
                    'user_complexity': test_case['case_complexity'],
                    'ai_complexity': ai_complexity,
                    'evidence_independent': evidence_independent,
                    'complexity_independent': complexity_independent
                })
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 INDEPENDENCE TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  User Evidence: {result['user_evidence']}/10 → AI: {result['ai_evidence']}/10")
        print(f"  User Complexity: {result['user_complexity']*100}% → AI: {result['ai_complexity']*100 if result['ai_complexity'] else 0}%")
        print(f"  Evidence Independent: {'✅' if result['evidence_independent'] else '❌'}")
        print(f"  Complexity Independent: {'✅' if result['complexity_independent'] else '❌'}")
    
    # Check if AI values are consistent across different user inputs
    if len(results) >= 2:
        ai_evidence_values = [r['ai_evidence'] for r in results if r['ai_evidence']]
        ai_complexity_values = [r['ai_complexity'] for r in results if r['ai_complexity']]
        
        evidence_consistent = len(set(ai_evidence_values)) <= 2  # Allow some variation
        complexity_consistent = len(set([round(c, 1) for c in ai_complexity_values])) <= 2
        
        print(f"\n🔍 CONSISTENCY CHECK:")
        print(f"AI Evidence Values: {ai_evidence_values}")
        print(f"AI Complexity Values: {[round(c*100) for c in ai_complexity_values]}%")
        print(f"Evidence Consistent: {'✅' if evidence_consistent else '❌'}")
        print(f"Complexity Consistent: {'✅' if complexity_consistent else '❌'}")
        
        if evidence_consistent and complexity_consistent:
            print("\n🎉 SUCCESS: AI analysis appears to be based on case facts, not user inputs!")
        else:
            print("\n⚠️ WARNING: AI analysis may still be influenced by user inputs")

if __name__ == "__main__":
    test_with_different_inputs()