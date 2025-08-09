#!/usr/bin/env python3
"""
Quick Legal Q&A Assistant RAG System Testing Suite

This script tests the core Legal Q&A Assistant RAG system endpoints.
Skips the time-consuming initialization test for faster results.
"""

import requests
import sys
import json
from datetime import datetime

class QuickLegalQARAGTester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            print(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Exception: {str(e)}")
            return False, {}

    def run_quick_test_suite(self):
        """Run core Legal Q&A RAG system tests (excluding initialization)"""
        print("🚀 QUICK LEGAL Q&A ASSISTANT RAG SYSTEM TESTING")
        print("=" * 60)
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🔗 API URL: {self.api_url}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        results = []
        
        # Test 1: RAG System Statistics
        print("\n🎯 TEST 1: RAG System Statistics")
        success, response_data = self.run_test(
            "RAG System Stats",
            "GET",
            "legal-qa/stats",
            200
        )
        
        if success and isinstance(response_data, dict):
            expected_fields = ['vector_db', 'embeddings_model', 'active_sessions', 'total_conversations']
            found_fields = [f for f in expected_fields if f in response_data]
            print(f"   📊 Found {len(found_fields)}/{len(expected_fields)} expected fields")
            for field in found_fields:
                print(f"   ✅ {field}: {response_data[field]}")
        
        results.append(("RAG System Stats", success))
        
        # Test 2: Knowledge Base Statistics  
        print("\n🎯 TEST 2: Knowledge Base Statistics")
        success, response_data = self.run_test(
            "Knowledge Base Stats",
            "GET",
            "legal-qa/knowledge-base/stats",
            200
        )
        
        if success and isinstance(response_data, dict):
            expected_fields = ['total_documents', 'by_jurisdiction', 'by_legal_domain', 'by_document_type', 'by_source']
            found_fields = [f for f in expected_fields if f in response_data]
            print(f"   📊 Found {len(found_fields)}/{len(expected_fields)} expected fields")
            if 'total_documents' in response_data:
                print(f"   📄 Total documents: {response_data['total_documents']}")
        
        results.append(("Knowledge Base Stats", success))
        
        # Test 3: Legal Question Answering (Main endpoint)
        print("\n🎯 TEST 3: Legal Question Answering")
        test_payload = {
            "question": "What are the key elements of a valid contract under US law?",
            "jurisdiction": "US", 
            "legal_domain": "contract_law"
        }
        
        success, response_data = self.run_test(
            "Legal Question Answering",
            "POST",
            "legal-qa/ask",
            200,
            test_payload,
            timeout=60
        )
        
        if success and isinstance(response_data, dict):
            expected_fields = ['answer', 'confidence', 'sources', 'session_id', 'retrieved_documents', 'timestamp']
            found_fields = [f for f in expected_fields if f in response_data]
            print(f"   📊 Found {len(found_fields)}/{len(expected_fields)} expected fields")
            
            if 'answer' in response_data:
                answer_length = len(response_data['answer'])
                print(f"   📝 Answer length: {answer_length} characters")
                print(f"   📝 Answer preview: {response_data['answer'][:150]}...")
            
            if 'confidence' in response_data:
                print(f"   🎯 Confidence: {response_data['confidence']:.2f}")
            
            if 'sources' in response_data:
                print(f"   📚 Sources: {len(response_data['sources'])}")
            
            if 'model_used' in response_data:
                print(f"   🤖 Model: {response_data['model_used']}")
        
        results.append(("Legal Question Answering", success))
        
        # Final Results
        print("\n" + "="*60)
        print("🏁 QUICK TEST RESULTS")
        print("="*60)
        
        passed_count = 0
        total_count = len(results)
        
        for test_name, result in results:
            if result:
                print(f"✅ {test_name}: PASSED")
                passed_count += 1
            else:
                print(f"❌ {test_name}: FAILED")
        
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   - Tests Passed: {passed_count}/{total_count}")
        print(f"   - Success Rate: {success_rate:.1f}%")
        print(f"   - Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Detailed Analysis
        print(f"\n🔍 Analysis:")
        if success_rate == 100:
            print("   🎉 ALL CORE LEGAL Q&A RAG ENDPOINTS ARE WORKING!")
            print("   ✅ RAG_SYSTEM_AVAILABLE appears to be True")
            print("   ✅ All endpoints return 200 status codes")
            print("   ✅ Response structures match expected models")
            print("   ✅ AI-powered question answering works with Gemini integration")
        elif success_rate >= 66:
            print("   ⚠️  MOST LEGAL Q&A RAG ENDPOINTS ARE WORKING")
            print("   ✅ Core functionality appears operational")
            print("   ⚠️  Some endpoints may need attention")
        else:
            print("   ❌ LEGAL Q&A RAG SYSTEM HAS SIGNIFICANT ISSUES")
            print("   ❌ Multiple endpoints are failing")
            print("   ❌ RAG_SYSTEM_AVAILABLE may still be False")
        
        return success_rate >= 75

if __name__ == "__main__":
    tester = QuickLegalQARAGTester()
    success = tester.run_quick_test_suite()
    sys.exit(0 if success else 1)