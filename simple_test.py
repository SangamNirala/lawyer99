#!/usr/bin/env python3
"""
Simple test to check if endpoints respond at all
"""

import requests
import json

def test_endpoint(url, method="GET", data=None):
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data, timeout=5)
        
        print(f"{method} {url}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"{method} {url}")
        print(f"❌ Error: {e}")
        return False

# Test basic endpoints
print("Testing Advanced Legal Research Engine endpoints...")

base_url = "http://localhost:8001/api/legal-research-engine"

# Test stats endpoint
test_endpoint(f"{base_url}/stats")

print("\n" + "="*50)

# Test research endpoint with minimal data
research_data = {
    "query_text": "test query",
    "research_type": "precedent_search",
    "jurisdiction": "US",
    "legal_domain": "general",
    "priority": "medium"
}

test_endpoint(f"{base_url}/research", "POST", research_data)

print("\n" + "="*50)

# Test memo generation with minimal data
memo_data = {
    "memo_data": {
        "research_query": "test memo",
        "jurisdiction": "US",
        "legal_domain": "general"
    },
    "memo_type": "brief",
    "format_style": "professional"
}

test_endpoint(f"{base_url}/generate-memo", "POST", memo_data)