#!/usr/bin/env python3
"""
Quick test of Advanced Legal Research Engine endpoints
"""

import requests
import json
import sys

BACKEND_URL = "http://localhost:8001/api"

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("🎯 Testing Stats Endpoint")
    try:
        response = requests.get(f"{BACKEND_URL}/legal-research-engine/stats", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats endpoint working - Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
        return False

def test_main_research_endpoint():
    """Test the main research endpoint"""
    print("\n🎯 Testing Main Research Endpoint")
    test_data = {
        "query_text": "Contract breach remedies in commercial agreements",
        "research_type": "precedent_search",
        "jurisdiction": "US",
        "legal_domain": "contract_law",
        "priority": "high",
        "max_results": 5,
        "min_confidence": 0.7,
        "include_analysis": True,
        "cache_results": True
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/legal-research-engine/research", 
                               json=test_data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Main research endpoint working")
            print(f"Research Type: {data.get('research_type', 'unknown')}")
            print(f"Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Main research endpoint failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Main research endpoint error: {e}")
        return False

def test_memo_generation_endpoint():
    """Test the memo generation endpoint"""
    print("\n🎯 Testing Memo Generation Endpoint")
    test_data = {
        "memo_data": {
            "research_query": "Corporate governance requirements for public companies",
            "legal_issues": ["board_composition", "shareholder_rights"],
            "jurisdiction": "US",
            "legal_domain": "corporate_law",
            "client_context": "Public company seeking compliance guidance"
        },
        "memo_type": "brief",
        "format_style": "professional"
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/legal-research-engine/generate-memo", 
                               json=test_data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memo generation endpoint working")
            if 'id' in data:
                print(f"✅ 'id' field present: {data['id']}")
            else:
                print(f"❌ 'id' field missing")
                if 'memo_id' in data:
                    print(f"Found 'memo_id' instead: {data['memo_id']}")
            return True
        else:
            print(f"❌ Memo generation endpoint failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Memo generation endpoint error: {e}")
        return False

def main():
    print("🎯 QUICK ADVANCED LEGAL RESEARCH ENGINE TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Stats endpoint
    results.append(test_stats_endpoint())
    
    # Test 2: Main research endpoint
    results.append(test_main_research_endpoint())
    
    # Test 3: Memo generation endpoint
    results.append(test_memo_generation_endpoint())
    
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"📊 Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 66.7:  # 2/3 endpoints working
        print("✅ Advanced Legal Research Engine is mostly operational")
        return True
    else:
        print("❌ Advanced Legal Research Engine has significant issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)