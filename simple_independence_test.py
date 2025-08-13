#!/usr/bin/env python3
"""
Simple test to check if AI analysis is independent of user inputs
"""

import requests
import json

# Backend URL from environment
BACKEND_URL = "https://legal-engine-check.preview.emergentagent.com/api"

def test_independence():
    """Test with very different user input to see if AI is independent"""
    print("🧪 TESTING AI INDEPENDENCE WITH DIFFERENT USER INPUT")
    print("=" * 60)
    
    # Test with very low user inputs but same strong case facts
    test_case = {
        "case_type": "civil",
        "jurisdiction": "federal", 
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 2.0,  # Very low user input
        "case_complexity": 0.1,    # Very low user input
        "case_facts": "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    print(f"User Evidence Input: {test_case['evidence_strength']}/10 (Very Low)")
    print(f"User Complexity Input: {test_case['case_complexity']*100}% (Very Low)")
    print("Case Facts: Strong evidence (signed contracts, delivery logs, email correspondence)")
    
    try:
        url = f"{BACKEND_URL}/litigation/appeal-analysis"
        response = requests.post(url, json=test_case, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            case_facts_analysis = data.get('case_facts_analysis', {})
            
            ai_evidence = case_facts_analysis.get('evidence_strength_suggested')
            ai_complexity = case_facts_analysis.get('case_complexity_suggested')
            
            print(f"\nAI Evidence Suggested: {ai_evidence}/10")
            print(f"AI Complexity Suggested: {ai_complexity*100 if ai_complexity else 0}%")
            
            # Check if AI ignored low user inputs and analyzed case facts independently
            if ai_evidence and ai_evidence > 5.0:  # Should be higher than user input of 2.0
                print("✅ AI Evidence Analysis: INDEPENDENT (ignored low user input)")
            else:
                print("❌ AI Evidence Analysis: MAY BE INFLUENCED BY USER INPUT")
                
            if ai_complexity and ai_complexity > 0.3:  # Should be higher than user input of 0.1
                print("✅ AI Complexity Analysis: INDEPENDENT (ignored low user input)")
            else:
                print("❌ AI Complexity Analysis: MAY BE INFLUENCED BY USER INPUT")
                
            # Show reasoning
            evidence_reasoning = case_facts_analysis.get('evidence_reasoning', '')
            print(f"\nEvidence Reasoning: {evidence_reasoning[:100]}...")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_independence()