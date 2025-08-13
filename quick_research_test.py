#!/usr/bin/env python3
"""
Quick Advanced Legal Research Engine Test
========================================

Quick test to verify the Advanced Legal Research Engine endpoints are working
and the MemoFormat fix is implemented.
"""

import requests
import json
import sys
from datetime import datetime

# Backend URL from environment
BACKEND_URL = "https://legal-research-api.preview.emergentagent.com/api"

def test_stats_endpoint():
    """Quick test of stats endpoint"""
    print("🏥 TESTING STATS ENDPOINT")
    print("=" * 50)
    
    try:
        url = f"{BACKEND_URL}/legal-research-engine/stats"
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats endpoint working")
            
            # Check for system status
            system_status = data.get('system_status', 'unknown')
            modules_loaded = data.get('modules_loaded', {})
            
            print(f"System Status: {system_status}")
            print(f"Modules loaded: {len(modules_loaded)}")
            
            # Check if advanced research engine is loaded
            engine_loaded = any('research' in key.lower() for key in modules_loaded.keys())
            if engine_loaded:
                print("✅ Advanced Legal Research Engine modules detected")
                return True
            else:
                print("⚠️ Advanced Legal Research Engine modules not clearly detected")
                return False
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_research_queries_endpoint():
    """Quick test of research queries endpoint"""
    print("\n📚 TESTING RESEARCH QUERIES ENDPOINT")
    print("=" * 50)
    
    try:
        url = f"{BACKEND_URL}/legal-research-engine/research-queries"
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Research queries endpoint working")
            
            if isinstance(data, dict):
                total_count = data.get('total_count', 0)
                queries = data.get('queries', [])
                print(f"Total queries: {total_count}")
                print(f"Queries returned: {len(queries)}")
            elif isinstance(data, list):
                print(f"Queries found: {len(data)}")
            
            return True
        else:
            print(f"❌ Research queries endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_research_memos_endpoint():
    """Quick test of research memos endpoint"""
    print("\n📝 TESTING RESEARCH MEMOS ENDPOINT")
    print("=" * 50)
    
    try:
        url = f"{BACKEND_URL}/legal-research-engine/research-memos"
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Research memos endpoint working")
            
            if isinstance(data, dict):
                total_count = data.get('total_count', 0)
                memos = data.get('memos', [])
                print(f"Total memos: {total_count}")
                print(f"Memos returned: {len(memos)}")
            elif isinstance(data, list):
                print(f"Memos found: {len(data)}")
            
            return True
        else:
            print(f"❌ Research memos endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_memo_generation_endpoint():
    """Test memo generation with traditional format (MemoFormat fix)"""
    print("\n📄 TESTING MEMO GENERATION - MEMOFORMAT FIX")
    print("=" * 50)
    
    # Simple test request with traditional format
    test_request = {
        "memo_data": {
            "research_query": "Simple contract law analysis",
            "legal_issues": ["contract_interpretation"],
            "jurisdiction": "US"
        },
        "memo_type": "brief",
        "format_style": "traditional"  # CRITICAL: Testing the fix
    }
    
    print(f"Format Style: {test_request['format_style']} (Testing MemoFormat fix)")
    
    try:
        url = f"{BACKEND_URL}/legal-research-engine/generate-memo"
        response = requests.post(url, json=test_request, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Memo generation working")
            print("✅ MEMOFORMAT FIX VERIFIED: 'traditional' format accepted")
            
            # Check memo content
            generated_memo = data.get('generated_memo', '')
            word_count = data.get('word_count', 0)
            
            print(f"Generated memo length: {len(generated_memo)} characters")
            print(f"Word count: {word_count}")
            
            if len(generated_memo) > 100:
                print("✅ Substantial memo content generated")
                return True
            else:
                print("⚠️ Limited memo content")
                return False
                
        elif response.status_code == 422:
            print("❌ VALIDATION ERROR - MemoFormat issue may persist!")
            try:
                error_data = response.json()
                error_detail = str(error_data)
                if 'MemoFormat' in error_detail or 'traditional' in error_detail:
                    print("🚨 CRITICAL: MemoFormat validation error with 'traditional'!")
                elif 'professional' in error_detail:
                    print("🚨 CRITICAL: Still seeing 'professional' format error!")
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw error: {response.text}")
            return False
            
        else:
            print(f"❌ Memo generation failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Main test execution"""
    print("🎯 QUICK ADVANCED LEGAL RESEARCH ENGINE TEST")
    print("=" * 60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Stats endpoint
    results.append(test_stats_endpoint())
    
    # Test 2: Research queries endpoint
    results.append(test_research_queries_endpoint())
    
    # Test 3: Research memos endpoint
    results.append(test_research_memos_endpoint())
    
    # Test 4: Memo generation (MemoFormat fix)
    results.append(test_memo_generation_endpoint())
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 QUICK TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Assessment
    if success_rate >= 75:
        print("\n🎉 ADVANCED LEGAL RESEARCH ENGINE WORKING!")
        print("✅ Core endpoints operational")
        if results[-1]:  # Memo generation test
            print("✅ MEMOFORMAT FIX VERIFIED: 'traditional' format working")
        status = "OPERATIONAL"
    elif success_rate >= 50:
        print("\n✅ ADVANCED LEGAL RESEARCH ENGINE MOSTLY WORKING")
        print("⚠️ Some endpoints may need attention")
        status = "MOSTLY_OPERATIONAL"
    else:
        print("\n❌ ADVANCED LEGAL RESEARCH ENGINE NEEDS ATTENTION")
        print("🚨 Multiple endpoints have issues")
        status = "NEEDS_ATTENTION"
    
    print(f"\n📊 FINAL STATUS: {status}")
    print("=" * 60)
    
    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)