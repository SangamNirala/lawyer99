#!/usr/bin/env python3
"""
Debug test for litigation strategy endpoint
"""

import requests
import json
import sys

BACKEND_URL = "https://legal-research-api.preview.emergentagent.com/api"

def test_simple_litigation_strategy():
    """Test with minimal data to debug the issue"""
    print("🔍 DEBUGGING LITIGATION STRATEGY ENDPOINT")
    print("=" * 60)
    
    # Very simple test case
    simple_case = {
        "case_type": "Civil - Breach of Contract",
        "jurisdiction": "California",
        "case_value": 750000,
        "evidence_strength": 7
    }
    
    print(f"Test Data: {json.dumps(simple_case, indent=2)}")
    
    try:
        url = f"{BACKEND_URL}/litigation/strategy-recommendations"
        print(f"URL: {url}")
        
        response = requests.post(url, json=simple_case, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"Response keys: {list(data.keys())}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_litigation_strategy()
    sys.exit(0 if success else 1)