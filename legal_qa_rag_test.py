#!/usr/bin/env python3
"""
Legal Q&A Assistant RAG System Testing Suite

This script tests the Legal Q&A Assistant RAG system endpoints that were recently fixed.
The issue was that RAG_SYSTEM_AVAILABLE was False due to missing dependencies, 
but now it should be True and all endpoints should be working.

Test Coverage:
1. GET /api/legal-qa/stats - System statistics
2. GET /api/legal-qa/knowledge-base/stats - Knowledge base statistics  
3. POST /api/legal-qa/ask - Main question answering endpoint
4. POST /api/legal-qa/initialize-knowledge-base - Knowledge base initialization
"""

import requests
import sys
import json
import time
from datetime import datetime

class LegalQARAGTester:
    def __init__(self, base_url="https://7eb15b8a-98e2-46d7-83f4-aba058ad34e0.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=60):
        """Run a single API test with extended timeout for RAG operations"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
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
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'List with ' + str(len(response_data)) + ' items'}")
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

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Exception: {str(e)}")
            return False, {}

    def test_rag_system_stats(self):
        """Test GET /api/legal-qa/stats - Should return system statistics"""
        print("\n" + "="*60)
        print("🎯 TESTING RAG SYSTEM STATISTICS ENDPOINT")
        print("="*60)
        
        success, response_data = self.run_test(
            "RAG System Stats",
            "GET",
            "legal-qa/stats",
            200
        )
        
        if success and isinstance(response_data, dict):
            # Verify expected fields in response
            expected_fields = ['vector_db', 'embeddings_model', 'active_sessions', 'total_conversations']
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in response_data:
                    found_fields.append(field)
                    print(f"   ✅ {field}: {response_data[field]}")
                else:
                    missing_fields.append(field)
                    print(f"   ❌ Missing field: {field}")
            
            if 'indexed_documents' in response_data:
                print(f"   ✅ indexed_documents: {response_data['indexed_documents']}")
            
            print(f"\n   📊 Stats Summary:")
            print(f"   - Found fields: {len(found_fields)}/{len(expected_fields)}")
            if missing_fields:
                print(f"   - Missing fields: {missing_fields}")
            
            return success and len(missing_fields) == 0
        
        return success

    def test_knowledge_base_stats(self):
        """Test GET /api/legal-qa/knowledge-base/stats - Should return knowledge base statistics"""
        print("\n" + "="*60)
        print("🎯 TESTING KNOWLEDGE BASE STATISTICS ENDPOINT")
        print("="*60)
        
        success, response_data = self.run_test(
            "Knowledge Base Stats",
            "GET",
            "legal-qa/knowledge-base/stats",
            200
        )
        
        if success and isinstance(response_data, dict):
            # Verify expected fields in response
            expected_fields = ['total_documents', 'by_jurisdiction', 'by_legal_domain', 'by_document_type', 'by_source']
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in response_data:
                    found_fields.append(field)
                    if field == 'total_documents':
                        print(f"   ✅ {field}: {response_data[field]}")
                    else:
                        # For dictionary fields, show count of categories
                        field_data = response_data[field]
                        if isinstance(field_data, dict):
                            print(f"   ✅ {field}: {len(field_data)} categories")
                            # Show top 3 categories
                            sorted_items = sorted(field_data.items(), key=lambda x: x[1], reverse=True)[:3]
                            for category, count in sorted_items:
                                print(f"      - {category}: {count}")
                        else:
                            print(f"   ✅ {field}: {field_data}")
                else:
                    missing_fields.append(field)
                    print(f"   ❌ Missing field: {field}")
            
            print(f"\n   📊 Knowledge Base Summary:")
            print(f"   - Found fields: {len(found_fields)}/{len(expected_fields)}")
            if missing_fields:
                print(f"   - Missing fields: {missing_fields}")
            
            return success and len(missing_fields) == 0
        
        return success

    def test_legal_question_answering(self):
        """Test POST /api/legal-qa/ask - Main question answering endpoint"""
        print("\n" + "="*60)
        print("🎯 TESTING LEGAL QUESTION ANSWERING ENDPOINT")
        print("="*60)
        
        # Test with the specific payload from the review request
        test_payload = {
            "question": "What are the key elements of a valid contract under US law?",
            "jurisdiction": "US", 
            "legal_domain": "contract_law"
        }
        
        print(f"   📝 Test Question: {test_payload['question']}")
        print(f"   🏛️ Jurisdiction: {test_payload['jurisdiction']}")
        print(f"   ⚖️ Legal Domain: {test_payload['legal_domain']}")
        
        success, response_data = self.run_test(
            "Legal Question Answering",
            "POST",
            "legal-qa/ask",
            200,
            test_payload,
            timeout=90  # Extended timeout for AI processing
        )
        
        if success and isinstance(response_data, dict):
            # Verify expected fields in response
            expected_fields = ['answer', 'confidence', 'sources', 'session_id', 'retrieved_documents', 'timestamp']
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in response_data:
                    found_fields.append(field)
                    if field == 'answer':
                        answer_length = len(response_data[field])
                        print(f"   ✅ {field}: {answer_length} characters")
                        # Show first 200 characters of answer
                        print(f"      Preview: {response_data[field][:200]}...")
                    elif field == 'confidence':
                        confidence = response_data[field]
                        print(f"   ✅ {field}: {confidence:.2f}")
                    elif field == 'sources':
                        sources = response_data[field]
                        print(f"   ✅ {field}: {len(sources)} sources")
                        # Show first source if available
                        if sources and len(sources) > 0:
                            first_source = sources[0]
                            if isinstance(first_source, dict):
                                print(f"      First source keys: {list(first_source.keys())}")
                    elif field == 'session_id':
                        self.session_id = response_data[field]  # Store for potential future tests
                        print(f"   ✅ {field}: {response_data[field]}")
                    else:
                        print(f"   ✅ {field}: {response_data[field]}")
                else:
                    missing_fields.append(field)
                    print(f"   ❌ Missing field: {field}")
            
            # Check optional fields
            if 'model_used' in response_data:
                print(f"   ✅ model_used: {response_data['model_used']}")
            
            print(f"\n   📊 Response Summary:")
            print(f"   - Found fields: {len(found_fields)}/{len(expected_fields)}")
            if missing_fields:
                print(f"   - Missing fields: {missing_fields}")
            
            # Validate response quality
            quality_checks = []
            if 'answer' in response_data and len(response_data['answer']) > 100:
                quality_checks.append("✅ Comprehensive answer (>100 chars)")
            else:
                quality_checks.append("❌ Answer too short")
            
            if 'confidence' in response_data and response_data['confidence'] > 0.5:
                quality_checks.append("✅ Good confidence score (>0.5)")
            else:
                quality_checks.append("❌ Low confidence score")
            
            if 'sources' in response_data and len(response_data['sources']) > 0:
                quality_checks.append("✅ Sources provided")
            else:
                quality_checks.append("❌ No sources provided")
            
            print(f"\n   🔍 Quality Checks:")
            for check in quality_checks:
                print(f"   {check}")
            
            return success and len(missing_fields) == 0
        
        return success

    def test_knowledge_base_initialization(self):
        """Test POST /api/legal-qa/initialize-knowledge-base - Knowledge base initialization"""
        print("\n" + "="*60)
        print("🎯 TESTING KNOWLEDGE BASE INITIALIZATION ENDPOINT")
        print("="*60)
        print("⚠️  WARNING: This operation may take several minutes to complete...")
        
        success, response_data = self.run_test(
            "Knowledge Base Initialization",
            "POST",
            "legal-qa/initialize-knowledge-base",
            200,
            {},  # Empty payload
            timeout=300  # 5 minutes timeout for initialization
        )
        
        if success and isinstance(response_data, dict):
            # Verify expected fields in response
            expected_fields = ['message', 'rag_system_ready', 'knowledge_base_stats', 'timestamp']
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in response_data:
                    found_fields.append(field)
                    if field == 'message':
                        print(f"   ✅ {field}: {response_data[field]}")
                    elif field == 'rag_system_ready':
                        ready_status = response_data[field]
                        print(f"   ✅ {field}: {ready_status}")
                        if ready_status:
                            print("   🎉 RAG system is ready!")
                        else:
                            print("   ⚠️ RAG system not ready")
                    elif field == 'knowledge_base_stats':
                        stats = response_data[field]
                        if isinstance(stats, dict):
                            print(f"   ✅ {field}: {len(stats)} stat categories")
                            if 'total_documents' in stats:
                                print(f"      - Total documents: {stats['total_documents']}")
                        else:
                            print(f"   ✅ {field}: {stats}")
                    else:
                        print(f"   ✅ {field}: {response_data[field]}")
                else:
                    missing_fields.append(field)
                    print(f"   ❌ Missing field: {field}")
            
            print(f"\n   📊 Initialization Summary:")
            print(f"   - Found fields: {len(found_fields)}/{len(expected_fields)}")
            if missing_fields:
                print(f"   - Missing fields: {missing_fields}")
            
            return success and len(missing_fields) == 0
        
        return success

    def run_comprehensive_test_suite(self):
        """Run all Legal Q&A RAG system tests"""
        print("🚀 LEGAL Q&A ASSISTANT RAG SYSTEM TESTING SUITE")
        print("=" * 80)
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🔗 API URL: {self.api_url}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Test 1: RAG System Statistics
        stats_success = self.test_rag_system_stats()
        
        # Test 2: Knowledge Base Statistics  
        kb_stats_success = self.test_knowledge_base_stats()
        
        # Test 3: Legal Question Answering (Main endpoint)
        qa_success = self.test_legal_question_answering()
        
        # Test 4: Knowledge Base Initialization (may take time)
        print("\n" + "="*60)
        print("⚠️  KNOWLEDGE BASE INITIALIZATION TEST")
        print("Running knowledge base initialization test (may take several minutes)...")
        user_input = 'yes'  # Auto-run for testing
        
        init_success = self.test_knowledge_base_initialization()
        
        # Final Results
        print("\n" + "="*80)
        print("🏁 FINAL TEST RESULTS")
        print("="*80)
        
        results = [
            ("RAG System Stats", stats_success),
            ("Knowledge Base Stats", kb_stats_success), 
            ("Legal Question Answering", qa_success),
            ("Knowledge Base Initialization", init_success)
        ]
        
        passed_count = 0
        total_count = 0
        
        for test_name, result in results:
            if result == "Skipped":
                print(f"⏭️  {test_name}: SKIPPED")
            elif result:
                print(f"✅ {test_name}: PASSED")
                passed_count += 1
                total_count += 1
            else:
                print(f"❌ {test_name}: FAILED")
                total_count += 1
        
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   - Tests Passed: {passed_count}/{total_count}")
        print(f"   - Success Rate: {success_rate:.1f}%")
        print(f"   - Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success_rate >= 75:
            print("\n🎉 LEGAL Q&A RAG SYSTEM IS WORKING WELL!")
            return True
        else:
            print("\n⚠️  LEGAL Q&A RAG SYSTEM HAS ISSUES THAT NEED ATTENTION")
            return False

if __name__ == "__main__":
    tester = LegalQARAGTester()
    success = tester.run_comprehensive_test_suite()
    sys.exit(0 if success else 1)