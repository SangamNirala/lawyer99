#!/usr/bin/env python3
"""
Debug AI Analysis - Check what Gemini is actually returning
"""

import requests
import json
import sys

# Backend URL from environment
BACKEND_URL = "https://legalmate-research.preview.emergentagent.com/api"

def debug_ai_analysis():
    """Debug the AI analysis response"""
    print("🔍 DEBUGGING AI ANALYSIS RESPONSE")
    print("=" * 50)
    
    # Test case
    test_case = {
        "case_type": "civil",
        "jurisdiction": "federal", 
        "case_value": 250000,
        "judge_name": "Judge Rebecca Morgan",
        "evidence_strength": 7.0,
        "case_complexity": 0.65,
        "case_facts": "Plaintiff alleges breach of contract by a federal contractor involving delayed delivery of critical medical equipment. The agreement stipulated delivery within 30 days, but actual delivery occurred after 90 days, causing significant financial loss to the plaintiff's hospital operations. Defendant claims delays were due to unforeseen supply chain disruptions caused by international shipping restrictions. Evidence includes signed contracts, delivery logs, and email correspondence between parties."
    }
    
    try:
        url = f"{BACKEND_URL}/litigation/appeal-analysis"
        response = requests.post(url, json=test_case, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            case_facts_analysis = data.get('case_facts_analysis', {})
            
            print("🔍 CASE FACTS ANALYSIS DETAILED INSPECTION:")
            print(f"Evidence Strength Suggested: {case_facts_analysis.get('evidence_strength_suggested')}")
            print(f"Case Complexity Suggested: {case_facts_analysis.get('case_complexity_suggested')}")
            print(f"Analysis Confidence: {case_facts_analysis.get('analysis_confidence')}")
            
            print(f"\n📝 REASONING FIELDS:")
            evidence_reasoning = case_facts_analysis.get('evidence_reasoning', '')
            complexity_reasoning = case_facts_analysis.get('complexity_reasoning', '')
            
            print(f"Evidence Reasoning: '{evidence_reasoning}'")
            print(f"Evidence Reasoning Length: {len(evidence_reasoning)}")
            print(f"Complexity Reasoning: '{complexity_reasoning}'")
            print(f"Complexity Reasoning Length: {len(complexity_reasoning)}")
            
            print(f"\n🔑 KEY FACTORS:")
            key_evidence_factors = case_facts_analysis.get('key_evidence_factors', [])
            complexity_factors = case_facts_analysis.get('complexity_factors', [])
            
            print(f"Key Evidence Factors: {key_evidence_factors}")
            print(f"Complexity Factors: {complexity_factors}")
            
            # Check if reasoning is actually populated
            if evidence_reasoning and len(evidence_reasoning) > 10:
                print("✅ Evidence reasoning is populated")
            else:
                print("❌ Evidence reasoning is empty or too short")
                
            if complexity_reasoning and len(complexity_reasoning) > 10:
                print("✅ Complexity reasoning is populated")
            else:
                print("❌ Complexity reasoning is empty or too short")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    debug_ai_analysis()